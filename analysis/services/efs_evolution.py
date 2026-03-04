"""
EFS Factor Evolution: Alpha158 base -> LLM mutation -> evaluation -> top-1 selection.

- Base: 158+ archived Alpha158 factors (qlib-style)
- New factors: LLM mutates from base; 3 per evolve run
- States: active (1) | archived (all others)
- Each run: re-evaluate all factors; promote top 1 to active
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from django.db import transaction

from markets.models import Stock
from ..config.settings import EfsConfig
from ..models import EfsAlphaFactor, EfsAlphaEvaluation, EfsDataPoint
from ..utils.cache import CacheConfig
from .ai import call_deepseek
from .alpha158 import ALPHA158_FEATURE_NAMES, ALPHA158_INIT_FACTORS
from .efs import get_feature_snapshot, get_month_end_dates

logger = logging.getLogger(__name__)

ALLOWED_FEATURES = set(ALPHA158_FEATURE_NAMES)

# Per-run constants
NEW_FACTORS_PER_EVOLVE = 3


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
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expression)
    return all(t in ALLOWED_FEATURES for t in tokens)


def _pick_eval_dates(
    window_months: int = EfsConfig.DEFAULT_WINDOW_MONTHS,
    future_days: int = EfsConfig.DEFAULT_FUTURE_DAYS,
    eval_frequency: str = EfsConfig.DEFAULT_EVAL_FREQUENCY,
    step_days: int = EfsConfig.DEFAULT_STEP_DAYS,
) -> List:
    """Select evaluation dates (month-end or biweekly)."""
    try:
        if eval_frequency == "monthly":
            month_ends = get_month_end_dates()
            if not month_ends:
                return []
            usable = month_ends[:-1] if len(month_ends) > 1 else month_ends
            return usable[-window_months:] if len(usable) >= window_months else usable

        dates = list(
            EfsDataPoint.objects.values_list("date", flat=True)
            .order_by("date")
            .distinct()
        )
        if not dates:
            return []
        recent = dates[-252:] if len(dates) > 252 else dates
        usable = recent[:-1] if len(recent) > 1 else recent
        return usable[:: max(1, step_days)]
    except Exception as e:
        logger.error(f"Error picking eval dates: {e}", exc_info=True)
        return []


def _get_cached_feature_snapshot(stock: Stock, target_date) -> Optional[Dict]:
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
    """Evaluate factor performance: IC and top-M return vs forward returns."""
    if not factor or not factor.expression:
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

        stocks = list(Stock.objects.all().only("id", "symbol"))
        if not stocks:
            return FactorEvalResult(ic=None, top_return=None, per_as_of=[])

        ic_list: List[float] = []
        top_returns: List[float] = []
        per_as_of: List[Tuple] = []

        for as_of in as_of_dates:
            values, returns = [], []
            for stock in stocks:
                snapshot = _get_cached_feature_snapshot(stock, as_of)
                if not snapshot:
                    continue
                val = _safe_eval(factor.expression, snapshot)
                if val is None:
                    continue
                future_return = snapshot.get("forward_ret_20") or snapshot.get("ret_20")
                if future_return is None:
                    continue
                values.append(val)
                returns.append(float(future_return))

            if len(values) < min_samples or len(values) != len(returns) or len(values) < 2:
                per_as_of.append((as_of, None, None))
                continue

            try:
                if np.std(values) == 0 or np.std(returns) == 0:
                    per_as_of.append((as_of, None, None))
                    continue
                ic = np.corrcoef(values, returns)[0, 1]
                ic_val = float(ic) if not np.isnan(ic) else None
                if ic_val is not None:
                    ic_list.append(ic_val)
                ranked = sorted(zip(values, returns), key=lambda x: x[0], reverse=True)
                top_slice = ranked[: max(1, min(top_m, len(ranked)))]
                top_val = sum(r for _, r in top_slice) / len(top_slice) if top_slice else None
                if top_val is not None:
                    top_returns.append(top_val)
                per_as_of.append((as_of, ic_val, top_val))
            except Exception as e:
                logger.debug(f"IC calc error at {as_of}: {e}")
                per_as_of.append((as_of, None, None))

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
    count: int = NEW_FACTORS_PER_EVOLVE,
    temperature: float = EfsConfig.LLM_TEMPERATURE,
    existing_expressions: Optional[Set[str]] = None,
    init_expressions: Optional[List[str]] = None,
) -> List[EfsAlphaFactor]:
    """Generate new factors by mutating from Alpha158 base; avoid duplicates."""
    if count <= 0 or count > 20:
        count = NEW_FACTORS_PER_EVOLVE

    existing_expressions = existing_expressions or set()
    init_expressions = init_expressions or [e for _, e in ALPHA158_INIT_FACTORS]

    try:
        features_desc = ", ".join(sorted(ALLOWED_FEATURES)[:80])  # truncate for prompt
        existing_str = ", ".join(sorted(existing_expressions)[:40]) if existing_expressions else "(none)"
        init_sample = ", ".join(init_expressions[:20])

        prompt = f"""You are a quantitative researcher. Generate {count} NEW alpha factors by MUTATING from these qlib Alpha158 base factors:
{init_sample}

Each factor must be a single expression using ONLY these variables: {features_desc}

CRITICAL: Do NOT generate expressions identical or equivalent to: {existing_str}

Mutate by combining factors (e.g. 0.5*ROC5 + 0.5*MA10), scaling, or arithmetic.
Output JSON list only: [{{"name": "...", "description": "...", "expression": "..."}}]
Expressions: simple arithmetic only. No functions, no conditionals.
Each expression MUST be unique and different from all existing.
"""
        text = call_deepseek("You are a quantitative researcher.", prompt, temperature=temperature)
        if not text:
            return []

        cleaned = text.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.warning(f"LLM JSON parse error: {e}")
            return []
        if not isinstance(data, list):
            return []

        factors = []
        seen = set(existing_expressions)
        for item in data:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip() or "Alpha"
            desc = str(item.get("description") or "").strip()
            expression = str(item.get("expression") or "").strip()
            if not expression or expression in seen or not _validate_expression(expression):
                continue
            seen.add(expression)
            factors.append(
                EfsAlphaFactor(
                    name=name[:128],
                    description=desc,
                    expression=expression,
                    status=EfsAlphaFactor.STATUS_ARCHIVED,
                )
            )
        return factors
    except Exception as e:
        logger.error(f"Error generating factors: {e}", exc_info=True)
        return []


def evolve_factors(
    mutate_count: int = NEW_FACTORS_PER_EVOLVE,
    top_k: int = EfsConfig.DEFAULT_TOP_K,
    top_m: int = EfsConfig.DEFAULT_TOP_M,
    future_days: int = EfsConfig.DEFAULT_FUTURE_DAYS,
    min_samples: int = EfsConfig.DEFAULT_MIN_SAMPLES,
    window_months: int = EfsConfig.DEFAULT_WINDOW_MONTHS,
    eval_frequency: str = EfsConfig.DEFAULT_EVAL_FREQUENCY,
    step_days: int = EfsConfig.DEFAULT_STEP_DAYS,
) -> Dict[str, int]:
    """
    Run one EFS evolution cycle:
    1. Seed Alpha158 base if empty (all archived)
    2. Re-evaluate all factors (active + archived)
    3. Promote top 1 to active, rest to archived
    4. Generate 3 new factors via LLM mutation
    """
    created = evaluated = promoted = 0

    try:
        # 1) Seed with full Alpha158 base if empty (all archived)
        if not EfsAlphaFactor.objects.exists():
            init_factors = [
                EfsAlphaFactor(
                    name=name,
                    description=f"Alpha158 {name}",
                    expression=expr,
                    status=EfsAlphaFactor.STATUS_ARCHIVED,
                )
                for name, expr in ALPHA158_INIT_FACTORS
            ]
            EfsAlphaFactor.objects.bulk_create(init_factors)
            created += len(init_factors)
            logger.info(f"Seeded {len(init_factors)} Alpha158 base factors")

        as_of_dates = _pick_eval_dates(
            window_months=window_months,
            future_days=future_days,
            eval_frequency=eval_frequency,
            step_days=step_days,
        )
        if not as_of_dates:
            logger.warning("No eval dates available; seeded factors only")
            return {"created": created, "evaluated": evaluated, "promoted": promoted}

        # 2) Re-evaluate ALL factors (active + archived)
        to_evaluate = list(
            EfsAlphaFactor.objects.all().only("id", "expression", "name", "last_score")
        )
        for factor in to_evaluate:
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
                score = (
                    (result.ic or 0) * EfsConfig.IC_WEIGHT
                    + (result.top_return or 0) * EfsConfig.TOP_RETURN_WEIGHT
                )
                with transaction.atomic():
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
                    factor.last_score = score
                    factor.save(update_fields=["last_score", "updated_at"])
            except Exception as e:
                logger.error(f"Error evaluating factor {factor.id}: {e}", exc_info=True)

        # 3) Top 1 -> active, rest -> archived
        try:
            ranked = list(
                EfsAlphaFactor.objects.exclude(last_score__isnull=True)
                .order_by("-last_score")
                .only("id", "last_score")[:top_k]
            )
            active_ids = {f.id for f in ranked}
            with transaction.atomic():
                EfsAlphaFactor.objects.filter(status=EfsAlphaFactor.STATUS_ACTIVE).update(
                    status=EfsAlphaFactor.STATUS_ARCHIVED
                )
                EfsAlphaFactor.objects.filter(id__in=active_ids).update(
                    status=EfsAlphaFactor.STATUS_ACTIVE
                )
            promoted = len(active_ids)
        except Exception as e:
            logger.error(f"Error promoting factors: {e}", exc_info=True)

        # 4) Generate 3 new factors (archived)
        existing_exprs = set(
            EfsAlphaFactor.objects.exclude(expression__isnull=True).values_list(
                "expression", flat=True
            )
        )
        init_exprs = [e for _, e in ALPHA158_INIT_FACTORS]
        new_batch = generate_factors_via_llm(
            count=mutate_count,
            existing_expressions=existing_exprs,
            init_expressions=init_exprs,
        )
        if new_batch:
            EfsAlphaFactor.objects.bulk_create(new_batch)
            created += len(new_batch)

    except Exception as e:
        logger.error(f"Error in evolve_factors: {e}", exc_info=True)

    return {"created": created, "evaluated": evaluated, "promoted": promoted}
