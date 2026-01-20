"""EFS 因子生成与进化：LLM 生成 -> 回测评估 -> 进化筛选。"""

import json
import logging
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from django.db import transaction

from markets.models import Stock
from ..config.settings import EfsConfig
from ..models import EfsDataPoint
from ..models import EfsAlphaFactor, EfsAlphaEvaluation
from ..utils.cache import cached_result, CacheConfig
from .ai import call_deepseek
from .efs import (
    EFS_FEATURE_KEYS,
    get_feature_snapshot,
    get_month_end_dates,
)

logger = logging.getLogger(__name__)


ALLOWED_FEATURES = {
    *EFS_FEATURE_KEYS,
    "rsi",
    "ret_20",
    "vol",
    "vol_ratio",
    "bb_upper",
    "bb_lower",
    "bb_middle",
    "adx_raw",
    "atr_raw",
    "atr_pct",
    "stoch_d",
    "sma_mid",
    "sma_short",
    "ma_mid",
}


@dataclass
class FactorEvalResult:
    ic: Optional[float]
    top_return: Optional[float]
    per_as_of: List[Tuple[date, Optional[float], Optional[float]]]


def _safe_eval(expression: str, features: Dict[str, Optional[float]]) -> Optional[float]:
    allowed_vars = {k: (features.get(k) or 0.0) for k in ALLOWED_FEATURES}
    try:
        return float(eval(expression, {"__builtins__": {}}, allowed_vars))
    except Exception:
        return None


def _validate_expression(expression: str) -> bool:
    # 只允许使用白名单变量与基本运算符
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression)
    for token in tokens:
        if token not in ALLOWED_FEATURES:
            return False
    return True


def _pick_eval_dates(
    window_months: int = EfsConfig.DEFAULT_WINDOW_MONTHS,
    future_days: int = EfsConfig.DEFAULT_FUTURE_DAYS,
    eval_frequency: str = EfsConfig.DEFAULT_EVAL_FREQUENCY,
    step_days: int = EfsConfig.DEFAULT_STEP_DAYS,
) -> List[date]:
    """选择评估日期"""
    try:
        if eval_frequency == "monthly":
            month_ends = get_month_end_dates()
            if not month_ends:
                return []
            usable = month_ends[:-1] if len(month_ends) > 1 else month_ends
            return usable[-window_months:] if len(usable) >= window_months else usable

        # biweekly: use trading dates sampled every ~10 trading days
        dates = list(
            EfsDataPoint.objects.values_list("date", flat=True)
            .order_by("date")
            .distinct()
        )
        if not dates:
            return []
        # last ~12 months of trading days (roughly 252)
        recent = dates[-252:] if len(dates) > 252 else dates
        usable = recent[:-1] if len(recent) > 1 else recent
        return usable[:: max(1, step_days)]
    except Exception as e:
        logger.error(f"Error picking eval dates: {e}", exc_info=True)
        return []


def _get_cached_feature_snapshot(stock: Stock, target_date: date) -> Optional[Dict]:
    """获取缓存的特征快照"""
    from ..utils.cache import get_cached, set_cached
    
    cache_key = f"{CacheConfig.KEY_PREFIX_FEATURE}{stock.id}:{target_date}"
    cached = get_cached(cache_key)
    if cached is not None:
        return cached
    
    result = get_feature_snapshot(stock, target_date)
    if result is not None:
        set_cached(cache_key, result, timeout=CacheConfig.FEATURE_SNAPSHOT_TIMEOUT)
    return result


def evaluate_factor(
    factor: EfsAlphaFactor,
    top_m: int = EfsConfig.DEFAULT_TOP_M,
    future_days: int = EfsConfig.DEFAULT_FUTURE_DAYS,
    min_samples: int = EfsConfig.DEFAULT_MIN_SAMPLES,
    window_months: int = EfsConfig.DEFAULT_WINDOW_MONTHS,
    eval_frequency: str = EfsConfig.DEFAULT_EVAL_FREQUENCY,
    step_days: int = EfsConfig.DEFAULT_STEP_DAYS,
) -> FactorEvalResult:
    """评估因子性能"""
    if not factor or not factor.expression:
        logger.warning("Invalid factor provided for evaluation")
        return FactorEvalResult(ic=None, top_return=None, per_as_of=[])
    
    try:
        as_of_dates = _pick_eval_dates(
            window_months=window_months,
            future_days=future_days,
            eval_frequency=eval_frequency,
            step_days=step_days,
        )
        if not as_of_dates:
            return FactorEvalResult(ic=None, top_return=None, per_as_of=[])
        
        # 优化：批量加载所有股票，使用select_related优化查询
        stocks = list(Stock.objects.all().select_related().only("id", "symbol"))
        if not stocks:
            logger.warning("No stocks available for factor evaluation")
            return FactorEvalResult(ic=None, top_return=None, per_as_of=[])
        
        ic_list: List[float] = []
        top_returns: List[float] = []
        per_as_of: List[Tuple[date, Optional[float], Optional[float]]] = []
        
        for as_of in as_of_dates:
            try:
                values = []
                returns = []
                
                # 批量获取特征快照（使用缓存）
                for stock in stocks:
                    try:
                        snapshot = _get_cached_feature_snapshot(stock, as_of)
                        if not snapshot:
                            continue
                        
                        val = _safe_eval(factor.expression, snapshot)
                        if val is None:
                            continue
                        
                        # 未来收益：用 snapshot 的 ret_20 作为近似（若缺失则跳过）
                        future_return = snapshot.get("ret_20")
                        if future_return is None:
                            continue
                        
                        values.append(val)
                        returns.append(float(future_return))
                    except Exception as e:
                        logger.debug(f"Error processing stock {stock.symbol} at {as_of}: {e}")
                        continue

                universe_size = len(values)
                if universe_size < min_samples:
                    per_as_of.append((as_of, None, None))
                    continue

                # 验证数据有效性
                if len(values) != len(returns) or len(values) < 2:
                    per_as_of.append((as_of, None, None))
                    continue

                try:
                    values_std = np.std(values)
                    returns_std = np.std(returns)
                    if values_std == 0 or returns_std == 0:
                        per_as_of.append((as_of, None, None))
                        continue
                    
                    ic = np.corrcoef(values, returns)[0, 1]
                    ic_val = float(ic) if not np.isnan(ic) else None
                    if ic_val is not None:
                        ic_list.append(ic_val)
                    
                    ranked = sorted(zip(values, returns), key=lambda x: x[0], reverse=True)
                    top_slice = ranked[: max(1, min(top_m, len(ranked)))]
                    top_val = None
                    if top_slice:
                        top_val = float(sum(r for _, r in top_slice) / len(top_slice))
                        top_returns.append(top_val)
                    
                    per_as_of.append((as_of, ic_val, top_val))
                except Exception as e:
                    logger.debug(f"Error calculating IC for date {as_of}: {e}")
                    per_as_of.append((as_of, None, None))
                    continue
                    
            except Exception as e:
                logger.warning(f"Error evaluating factor for date {as_of}: {e}")
                per_as_of.append((as_of, None, None))
                continue

        if not ic_list or not top_returns:
            return FactorEvalResult(ic=None, top_return=None, per_as_of=per_as_of)
        
        return FactorEvalResult(
            ic=float(sum(ic_list) / len(ic_list)),
            top_return=float(sum(top_returns) / len(top_returns)),
            per_as_of=per_as_of,
        )
    except Exception as e:
        logger.error(f"Error evaluating factor {factor.id}: {e}", exc_info=True)
        return FactorEvalResult(ic=None, top_return=None, per_as_of=[])


def generate_factors_via_llm(
    count: int = EfsConfig.DEFAULT_GENERATION_SIZE,
    temperature: float = EfsConfig.LLM_TEMPERATURE,
) -> List[EfsAlphaFactor]:
    """通过LLM生成因子"""
    if count <= 0 or count > 20:
        logger.warning(f"Invalid count {count}, using default")
        count = EfsConfig.DEFAULT_GENERATION_SIZE
    
    try:
        features_desc = ", ".join(sorted(ALLOWED_FEATURES))
        prompt = f"""
        You are a quantitative researcher. Generate {count} alpha factors.
        Each factor must be a single expression using ONLY these variables:
        {features_desc}
        Output JSON list only: [{{"name": "...", "description": "...", "expression": "..."}}]
        Expressions must be simple arithmetic, no functions, no conditionals.
        Avoid divide-by-zero; use additions/multiplications only if unsure.
        """
        text = call_deepseek("You are a quantitative researcher.", prompt, temperature=temperature)
        if not text:
            logger.warning("LLM returned empty response")
            return []
        
        cleaned = text.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response as JSON: {e}")
            return []
        
        if not isinstance(data, list):
            logger.warning(f"LLM response is not a list: {type(data)}")
            return []
        
        factors = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                name = str(item.get("name") or "").strip() or "Alpha"
                desc = str(item.get("description") or "").strip()
                expression = str(item.get("expression") or "").strip()
                
                if not expression:
                    continue
                
                if not _validate_expression(expression):
                    logger.debug(f"Invalid expression: {expression}")
                    continue
                
                factors.append(
                    EfsAlphaFactor(
                        name=name[:128],
                        description=desc,
                        expression=expression,
                        status="candidate"
                    )
                )
            except Exception as e:
                logger.debug(f"Error processing factor item: {e}")
                continue
        
        return factors
    except Exception as e:
        logger.error(f"Error generating factors via LLM: {e}", exc_info=True)
        return []


def evolve_factors(
    generation_size: int = EfsConfig.DEFAULT_GENERATION_SIZE,
    top_k: int = EfsConfig.DEFAULT_TOP_K,
    mutate_count: int = EfsConfig.DEFAULT_MUTATE_COUNT,
    top_m: int = EfsConfig.DEFAULT_TOP_M,
    future_days: int = EfsConfig.DEFAULT_FUTURE_DAYS,
    min_samples: int = EfsConfig.DEFAULT_MIN_SAMPLES,
    window_months: int = EfsConfig.DEFAULT_WINDOW_MONTHS,
    eval_frequency: str = EfsConfig.DEFAULT_EVAL_FREQUENCY,
    step_days: int = EfsConfig.DEFAULT_STEP_DAYS,
) -> Dict[str, int]:
    """执行因子进化流程"""
    created = 0
    evaluated = 0
    promoted = 0
    
    try:
        as_of_dates = _pick_eval_dates(
            window_months=window_months,
            future_days=future_days,
            eval_frequency=eval_frequency,
            step_days=step_days,
        )
        if not as_of_dates:
            logger.warning("No eval dates available")
            return {"created": created, "evaluated": evaluated, "promoted": promoted}

        # 1) 如果没有因子，先生成一批
        if not EfsAlphaFactor.objects.exists():
            batch = generate_factors_via_llm(count=generation_size)
            if batch:
                try:
                    EfsAlphaFactor.objects.bulk_create(batch)
                    created += len(batch)
                except Exception as e:
                    logger.error(f"Error bulk creating factors: {e}", exc_info=True)

        # 2) 评估所有候选因子（优化：批量查询）
        candidates = list(
            EfsAlphaFactor.objects.filter(status="candidate")
            .only("id", "expression", "name", "last_score")
        )
        
        for factor in candidates:
            try:
                result = evaluate_factor(
                    factor,
                    top_m=top_m,
                    future_days=future_days,
                    min_samples=min_samples,
                    window_months=window_months,
                    eval_frequency=eval_frequency,
                    step_days=step_days,
                )
                evaluated += 1
                
                try:
                    with transaction.atomic():
                        # 批量创建评估记录
                        evaluations = [
                            EfsAlphaEvaluation(
                                factor=factor,
                                as_of=as_of,
                                ic=ic_val,
                                top_return=top_val,
                            )
                            for as_of, ic_val, top_val in result.per_as_of
                            if ic_val is not None or top_val is not None
                        ]
                        if evaluations:
                            EfsAlphaEvaluation.objects.bulk_create(evaluations, ignore_conflicts=True)
                        
                        # 计算并更新分数
                        score = (
                            (result.ic or 0) * EfsConfig.IC_WEIGHT +
                            (result.top_return or 0) * EfsConfig.TOP_RETURN_WEIGHT
                        )
                        factor.last_score = score
                        factor.save(update_fields=["last_score", "updated_at"])
                except Exception as e:
                    logger.error(f"Error saving evaluation for factor {factor.id}: {e}", exc_info=True)
                    continue
                    
            except Exception as e:
                logger.error(f"Error evaluating factor {factor.id}: {e}", exc_info=True)
                continue

        # 3) 选出 top_k，标记为 active，其余归档（保留因子，不删除）
        try:
            ranked = list(
                EfsAlphaFactor.objects.exclude(last_score__isnull=True)
                .order_by("-last_score")
                .only("id", "last_score")
            )
            top_factors = ranked[:top_k]
            active_ids = {f.id for f in top_factors}
            
            with transaction.atomic():
                # 将 top_k 设为 active
                EfsAlphaFactor.objects.filter(id__in=active_ids).update(status="active")
                # 将非 top_k 的候选因子设为 archived（保留它们，不删除）
                EfsAlphaFactor.objects.exclude(id__in=active_ids).filter(status="candidate").update(status="archived")
                promoted = len(active_ids)
        except Exception as e:
            logger.error(f"Error promoting factors: {e}", exc_info=True)

        # 4) 生成新的候选因子
        new_batch = generate_factors_via_llm(count=mutate_count)
        if new_batch:
            try:
                EfsAlphaFactor.objects.bulk_create(new_batch)
                created += len(new_batch)
            except Exception as e:
                logger.error(f"Error bulk creating new factors: {e}", exc_info=True)

    except Exception as e:
        logger.error(f"Error in evolve_factors: {e}", exc_info=True)

    return {"created": created, "evaluated": evaluated, "promoted": promoted}
