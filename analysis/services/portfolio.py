"""组合优化服务：基于得分与风险惩罚进行稀疏选股。"""

import logging
from collections import defaultdict
from datetime import timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np

from django.db.models import Max

from markets.models import Stock
from ..config.constants import ETF_SYMBOLS
from ..config.settings import PortfolioConfig, ValidationConfig
from ..models import AnalysisResult, EfsDataPoint
from ..utils.validators import (
    ValidationError,
    validate_horizon,
    validate_risk_level,
    validate_score,
    validate_top_n,
)

logger = logging.getLogger(__name__)


def _get_returns(stock: Stock, lookback: int) -> List[float]:
    """获取回看窗口内的日收益序列"""
    if not stock or lookback <= 0:
        return []
    
    try:
        # 优化：只查询需要的字段
        data = list(
            EfsDataPoint.objects.filter(stock=stock)
            .order_by("date")
            .values_list("close", flat=True)
        )
        if len(data) < lookback + 1:
            return []
        
        closes = [float(d) for d in data[-(lookback + 1):]]
        returns = [
            (closes[i] / closes[i - 1]) - 1
            for i in range(1, len(closes))
            if closes[i - 1] != 0
        ]
        return returns
    except Exception as e:
        logger.debug(f"Error getting returns for {stock.symbol}: {e}")
        return []


def _load_price_series(stocks: List[Stock], lookback: int) -> Dict[str, List[Tuple]]:
    """批量加载价格序列（优化查询）"""
    if not stocks or lookback <= 0:
        return {}
    
    try:
        latest = EfsDataPoint.objects.aggregate(max_date=Max("date")).get("max_date")
        if not latest:
            return {}
        
        cutoff = latest - timedelta(days=lookback * 3)
        stock_ids = [s.id for s in stocks]
        
        # 优化：使用select_related减少查询次数，只查询需要的字段
        rows = (
            EfsDataPoint.objects.filter(stock_id__in=stock_ids, date__gte=cutoff)
            .select_related("stock")
            .order_by("date")
            .values_list("stock__symbol", "date", "close")
        )
        
        series_map: Dict[str, List[Tuple]] = defaultdict(list)
        for symbol, day, close in rows:
            if close is not None and close > 0:
                series_map[symbol].append((day, float(close)))
        
        return series_map
    except Exception as e:
        logger.error(f"Error loading price series: {e}", exc_info=True)
        return {}


def _returns_from_series(series: List[Tuple], lookback: int) -> List[float]:
    if len(series) < lookback + 1:
        return []
    window = series[-(lookback + 1) :]
    closes = [c for _, c in window]
    returns = [
        (closes[i] / closes[i - 1]) - 1 for i in range(1, len(closes)) if closes[i - 1] != 0
    ]
    return returns


def _returns_map_from_series(
    series_map: Dict[str, List[Tuple]], lookback: int
) -> Dict[str, List[float]]:
    returns_map: Dict[str, List[float]] = {}
    for symbol, series in series_map.items():
        returns = _returns_from_series(series, lookback)
        if returns:
            returns_map[symbol] = returns
    return returns_map


def _align_returns_from_series(
    symbols: List[str], series_map: Dict[str, List[Tuple]], lookback: int
) -> Optional[np.ndarray]:
    if not symbols:
        return None
    date_sets = []
    close_maps: Dict[str, Dict] = {}
    for symbol in symbols:
        series = series_map.get(symbol, [])
        if len(series) < lookback + 1:
            return None
        window = series[-(lookback + 1) :]
        dates = [d for d, _ in window]
        date_sets.append(set(dates))
        close_maps[symbol] = {d: c for d, c in window}
    common_dates = sorted(set.intersection(*date_sets)) if date_sets else []
    if len(common_dates) < lookback + 1:
        return None
    common_dates = common_dates[-(lookback + 1) :]
    returns_matrix = []
    for symbol in symbols:
        closes = [close_maps[symbol][d] for d in common_dates]
        returns = [
            (closes[i] / closes[i - 1]) - 1
            for i in range(1, len(closes))
            if closes[i - 1] != 0
        ]
        if len(returns) < 5:
            return None
        returns_matrix.append(returns)
    return np.array(returns_matrix, dtype=float)


def _cvar(returns: List[float], alpha: float = 0.05) -> float:
    """条件在险价值（CVaR）"""
    if not returns or alpha <= 0 or alpha >= 1:
        return 0.0
    
    try:
        cut = max(1, int(len(returns) * alpha))
        worst = sorted(returns)[:cut]
        if not worst:
            return 0.0
        return abs(sum(worst) / len(worst))
    except Exception as e:
        logger.debug(f"Error calculating CVaR: {e}")
        return 0.0


def _avg_positive_corr(returns: List[float], selected: List[List[float]]) -> float:
    # 仅惩罚正相关，降低组合相关性
    if not selected:
        return 0.0
    corrs = []
    for sel in selected:
        if len(sel) != len(returns):
            min_len = min(len(sel), len(returns))
            sel = sel[-min_len:]
            ret = returns[-min_len:]
        else:
            ret = returns
        if len(ret) < 2:
            continue
        corr = np.corrcoef(ret, sel)[0, 1]
        corrs.append(max(corr, 0))
    return float(sum(corrs) / len(corrs)) if corrs else 0.0


def _align_returns(returns_list: List[List[float]]) -> Optional[np.ndarray]:
    if not returns_list:
        return None
    min_len = min(len(r) for r in returns_list)
    if min_len < 5:
        return None
    aligned = np.array([r[-min_len:] for r in returns_list], dtype=float)
    return aligned


def _project_weights(weights: np.ndarray, max_weight: float) -> np.ndarray:
    clipped = np.clip(weights, 0.0, max_weight)
    total = clipped.sum()
    if total <= 0:
        return clipped
    return clipped / total


def _optimize_weights(
    returns_matrix: np.ndarray,
    max_weight: float,
    risk_aversion: float,
) -> Optional[np.ndarray]:
    """简化的均值-方差优化（投影梯度）"""
    if returns_matrix is None or returns_matrix.size == 0:
        return None
    
    try:
        if returns_matrix.shape[0] < 2 or returns_matrix.shape[1] < 5:
            return None
        
        mu = returns_matrix.mean(axis=1)
        cov = np.cov(returns_matrix)
        
        # 验证协方差矩阵
        if np.any(np.isnan(cov)) or np.any(np.isinf(cov)):
            return None
        
        n = len(mu)
        if n == 0:
            return None
        
        weights = np.ones(n) / n
        for _ in range(PortfolioConfig.OPTIMIZATION_ITERATIONS):
            grad = mu - 2 * risk_aversion * (cov @ weights)
            weights = weights + PortfolioConfig.OPTIMIZATION_STEP_SIZE * grad
            weights = _project_weights(weights, max_weight)
        
        return weights
    except Exception as e:
        logger.debug(f"Error optimizing weights: {e}")
        return None


def _empty_portfolio_result(
    top_n: int,
    horizon: str,
    risk_profile: str,
) -> Dict[str, object]:
    """返回空组合结果（错误情况）"""
    return {
        "inputs": {
            "top_n": top_n,
            "horizon": horizon,
            "use_hedge": False,
            "risk_level": risk_profile,
        },
        "portfolio": [],
        "allocation_analysis": {
            "sector_distribution": {},
            "is_concentrated": False,
            "dominant_sector": "Unknown",
        },
        "analysis": {
            "scenario_analysis": {"bull": 0.0, "base": 0.0, "bear": 0.0},
        },
        "summary": {
            "cvar_5pct": 0.0,
            "hedge_symbol": None,
            "portfolio_risk": "Unknown",
            "explanation": "No valid candidates found for portfolio construction.",
        },
    }


def _market_volatility(lookback: int = 60) -> Optional[float]:
    """计算市场波动率（基于SPY）"""
    if lookback <= 0:
        return None
    
    try:
        spy = Stock.objects.filter(symbol="SPY").first()
        if not spy:
            return None
        
        returns = _get_returns(spy, lookback)
        if len(returns) < 5:
            return None
        
        vol = float(np.std(returns)) * (252 ** 0.5)
        if np.isnan(vol) or np.isinf(vol):
            return None
        
        return vol
    except Exception as e:
        logger.debug(f"Error calculating market volatility: {e}")
        return None


def build_sparse_portfolio(
    top_n: int,
    horizon: str = "mid",
    use_hedge: bool = True,
    hedge_symbol: str = "TLT",
    risk_level: Optional[str] = None,
) -> Dict[str, object]:
    """主入口：基于 overall_score + 风险惩罚选股"""
    # 输入验证
    is_valid, error_msg = validate_top_n(top_n)
    if not is_valid:
        raise ValidationError(error_msg or "Invalid top_n")
    
    is_valid, error_msg = validate_horizon(horizon)
    if not is_valid:
        raise ValidationError(error_msg or "Invalid horizon")
    
    is_valid, error_msg = validate_risk_level(risk_level)
    if not is_valid:
        raise ValidationError(error_msg or "Invalid risk_level")
    
    # 标准化输入
    horizon = (horizon or "mid").lower()
    if horizon not in PortfolioConfig.HORIZON_LOOKBACK:
        horizon = "mid"
    
    lookback = PortfolioConfig.HORIZON_LOOKBACK[horizon]
    risk_lambda = PortfolioConfig.RISK_LAMBDA[horizon]
    corr_lambda = PortfolioConfig.CORR_LAMBDA[horizon]

    # 根据风险偏好调整惩罚强度
    risk_profile = (risk_level or "mid").lower().strip()
    if risk_profile not in PortfolioConfig.RISK_PROFILE_MULTIPLIERS:
        risk_profile = "mid"
    
    multipliers = PortfolioConfig.RISK_PROFILE_MULTIPLIERS[risk_profile]
    risk_lambda = int(risk_lambda * multipliers["risk"])
    corr_lambda = int(corr_lambda * multipliers["corr"])

    candidates: List[Tuple[Stock, int, List[float], float]] = []
    
    try:
        # 优化：批量查询，使用select_related和only减少查询
        results = (
            AnalysisResult.objects.select_related("stock")
            .only("stock", "result", "stock__symbol", "stock__sector")
            .all()
        )
        
        candidate_stocks = [
            res.stock for res in results
            if res.stock and res.stock.symbol not in ETF_SYMBOLS
        ]
        
        if not candidate_stocks:
            logger.warning("No candidate stocks available")
            return _empty_portfolio_result(top_n, horizon, risk_profile)
        
        series_map = _load_price_series(candidate_stocks, lookback)
        returns_map = _returns_map_from_series(series_map, lookback)
        
        min_returns_length = max(
            PortfolioConfig.MIN_ABSOLUTE_RETURNS,
            int(lookback * PortfolioConfig.MIN_RETURNS_RATIO)
        )
        
        for res in results:
            try:
                stock = res.stock
                if not stock or stock.symbol in ETF_SYMBOLS:
                    continue
                
                # 支持新格式 (scores.overall_score) 和旧格式 (overall_score)
                scores = res.result.get("scores", {}) if isinstance(res.result, dict) else {}
                overall_score = int(
                    scores.get("overall_score") or res.result.get("overall_score", 50)
                )
                
                # 验证分数
                is_valid, _ = validate_score(overall_score)
                if not is_valid:
                    continue
                
                returns = returns_map.get(stock.symbol, [])
                if len(returns) < min_returns_length:
                    continue
                
                cvar = _cvar(returns)
                candidates.append((stock, overall_score, returns, cvar))
            except Exception as e:
                logger.debug(f"Error processing candidate {res.stock.symbol if res.stock else 'unknown'}: {e}")
                continue
    except Exception as e:
        logger.error(f"Error building candidates list: {e}", exc_info=True)
        return _empty_portfolio_result(top_n, horizon, risk_profile)

    if not candidates:
        logger.warning("No valid candidates found")
        return _empty_portfolio_result(top_n, horizon, risk_profile)
    
    candidates.sort(key=lambda item: item[1], reverse=True)
    effective_top_n = top_n - 1 if use_hedge and top_n > 1 else top_n
    pool = candidates[:max(effective_top_n * 3, 15)]

    selected: List[Tuple[Stock, int, List[float], float]] = []
    selected_returns: List[List[float]] = []
    sector_counts: Dict[str, int] = {}
    sector_cap = max(1, int(round(top_n * PortfolioConfig.SECTOR_CAP_RATIO)))

    for stock, score, returns, cvar in pool:
        if len(selected) >= effective_top_n:
            break
        sector = (stock.sector or "Unknown").strip() or "Unknown"
        if sector_counts.get(sector, 0) >= sector_cap:
            continue
        if not selected:
            selected.append((stock, score, returns, cvar))
            selected_returns.append(returns)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
            continue
        avg_corr = _avg_positive_corr(returns, selected_returns)
        selection_score = (score * 0.6) - (avg_corr * corr_lambda) - (cvar * risk_lambda)
        if selection_score > 0:
            selected.append((stock, score, returns, cvar))
            selected_returns.append(returns)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

    if len(selected) < effective_top_n:
        for stock, score, returns, cvar in pool:
            if len(selected) >= effective_top_n:
                break
            if stock in {s for s, *_ in selected}:
                continue
            sector = (stock.sector or "Unknown").strip() or "Unknown"
            if sector_counts.get(sector, 0) >= sector_cap:
                continue
            selected.append((stock, score, returns, cvar))
            selected_returns.append(returns)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1

    # 如果整体相关性过高，尝试替换一只相关性最低的备选股
    symbols_selected = [stock.symbol for stock, *_ in selected]
    returns_matrix = _align_returns_from_series(symbols_selected, series_map, lookback)
    if returns_matrix is not None and returns_matrix.shape[0] >= 2:
        corr = np.corrcoef(returns_matrix)
        upper = corr[np.triu_indices_from(corr, k=1)]
        avg_corr = float(np.mean(np.abs(upper))) if upper.size else 0.0
        if avg_corr > 0.8:
            # 找出风险贡献最高的持仓（近似）
            cov = np.cov(returns_matrix)
            contrib = np.diag(cov) if cov.ndim == 2 else np.zeros(len(selected))
            idx = int(np.argmax(contrib)) if contrib.size else 0
            current = selected[idx]
            # 选择与组合平均相关性最低的候选
            best_candidate = None
            best_score = None
            for cand in pool:
                if cand in selected:
                    continue
                cand_returns = cand[2]
                cand_vec = np.array(cand_returns[-returns_matrix.shape[1]:], dtype=float)
                if cand_vec.size != returns_matrix.shape[1]:
                    continue
                cand_corr = np.mean(np.abs(np.corrcoef(np.vstack([returns_matrix, cand_vec]))[-1, :-1]))
                if best_score is None or cand_corr < best_score:
                    best_score = cand_corr
                    best_candidate = cand
            if best_candidate:
                selected[idx] = best_candidate
                selected_returns[idx] = best_candidate[2]

    def _hedge_candidates() -> List[str]:
        if hedge_symbol:
            return [hedge_symbol, "TLT", "IEF", "BIL"]
        if risk_profile == "low":
            return ["BIL", "IEF", "TLT"]
        if risk_profile == "high":
            return ["TLT", "IEF", "BIL"]
        if horizon == "short":
            return ["BIL", "IEF", "TLT"]
        if horizon == "long":
            return ["TLT", "IEF", "BIL"]
        return ["IEF", "TLT", "BIL"]

    hedge = None
    if use_hedge:
        for candidate in _hedge_candidates():
            hedge = Stock.objects.filter(symbol=candidate).first()
            if hedge:
                break

    total_score = sum(max(s, 1) for _, s, _, _ in selected) or 1
    if hedge:
        try:
            market_vol = _market_volatility(lookback=lookback) or 0.0
            hedge_ratio = PortfolioConfig.HEDGE_RATIO_BASE
            
            if market_vol > PortfolioConfig.MARKET_VOL_THRESHOLD_CRISIS:
                hedge_ratio = PortfolioConfig.HEDGE_RATIO_CRISIS
            elif market_vol > PortfolioConfig.MARKET_VOL_THRESHOLD_HIGH:
                hedge_ratio = PortfolioConfig.HEDGE_RATIO_HIGH_VOL
            
            if risk_profile == "low":
                hedge_ratio = max(hedge_ratio, PortfolioConfig.HEDGE_RATIO_HIGH_VOL)
            elif risk_profile == "high":
                hedge_ratio = min(hedge_ratio, PortfolioConfig.HEDGE_RATIO_BASE)
            
            equity_weight_total = max(0.6, min(0.95, 1.0 - hedge_ratio))
        except Exception as e:
            logger.debug(f"Error calculating hedge ratio: {e}")
            equity_weight_total = 1.0
    else:
        equity_weight_total = 1.0
    portfolio = []

    # 单只股票权重上限（风险越低越严格）
    max_weight = PortfolioConfig.MAX_WEIGHT[risk_profile]

    weights = None
    if returns_matrix is not None:
        risk_aversion = PortfolioConfig.RISK_AVERSION[risk_profile]
        weights = _optimize_weights(returns_matrix, max_weight=max_weight, risk_aversion=risk_aversion)
    
    if weights is None:
        try:
            weights = np.array(
                [max(score, 1) / total_score for _, score, _, _ in selected],
                dtype=float
            )
            weights = _project_weights(weights, max_weight=max_weight)
        except Exception as e:
            logger.error(f"Error creating fallback weights: {e}", exc_info=True)
            return _empty_portfolio_result(top_n, horizon, risk_profile)
    raw_weights = [float(w) * equity_weight_total for w in weights.tolist()]
    rounded_weights = [round(w, 2) for w in raw_weights]
    if rounded_weights:
        target_total = round(equity_weight_total, 2)
        delta = round(target_total - sum(rounded_weights), 2)
        if abs(delta) >= 0.01:
            rounded_weights[-1] = round(max(0, rounded_weights[-1] + delta), 2)

    def _role_for_stock(symbol: str, sector: str, score: int) -> str:
        if hedge and symbol == hedge.symbol:
            return "Hedge"
        sector_lower = (sector or "").lower()
        if sector_lower in {"consumer staples", "utilities", "healthcare"}:
            return "Stability"
        if score >= 75:
            return "Alpha Generator"
        return "Core"

    for (stock, score, _, _), weight in zip(selected, rounded_weights):
        portfolio.append(
            {
                "symbol": stock.symbol,
                "weight": weight,
                "sector": stock.sector or "Unknown",
                "role": _role_for_stock(stock.symbol, stock.sector or "", score)
                or "Unclassified",
            }
        )

    if hedge:
        hedge_weight = round(1.0 - equity_weight_total, 2)
        portfolio.append(
            {
                "symbol": hedge.symbol,
                "weight": hedge_weight,
                "sector": hedge.sector or "Fixed Income",
                "role": "Hedge",
            }
        )

    # Portfolio risk summary (simple proxy)
    try:
        if selected_returns:
            min_len = min(len(r) for r in selected_returns)
            if min_len > 0:
                aligned = [r[-min_len:] for r in selected_returns]
                weights_list = [p["weight"] for p in portfolio if p["symbol"] not in ETF_SYMBOLS]
                if len(weights_list) == len(aligned):
                    port_returns = []
                    for i in range(min_len):
                        r = sum(aligned[j][i] * weights_list[j] for j in range(len(aligned)))
                        port_returns.append(r)
                    port_cvar = _cvar(port_returns)
                    port_mean = sum(port_returns) / len(port_returns) if port_returns else 0.0
                    port_vol = float(np.std(port_returns)) if len(port_returns) > 1 else 0.0
                else:
                    port_cvar = 0.0
                    port_mean = 0.0
                    port_vol = 0.0
            else:
                port_cvar = 0.0
                port_mean = 0.0
                port_vol = 0.0
        else:
            port_cvar = 0.0
            port_mean = 0.0
            port_vol = 0.0
    except Exception as e:
        logger.error(f"Error calculating portfolio risk: {e}", exc_info=True)
        port_cvar = 0.0
        port_mean = 0.0
        port_vol = 0.0

    avg_score = int(sum(s for _, s, _, _ in selected) / len(selected)) if selected else 50
    sector_weights: Dict[str, float] = {}
    for item in portfolio:
        sector = item.get("sector", "Unknown")
        sector_weights[sector] = sector_weights.get(sector, 0.0) + float(item["weight"])
    top_sector = max(sector_weights.items(), key=lambda x: x[1]) if sector_weights else ("Unknown", 0)
    concentration_note = ""
    if top_sector[1] >= 0.5 and top_sector[0] != "Unknown":
        concentration_note = (
            f" The portfolio is concentrated in {top_sector[0]} (~{int(top_sector[1]*100)}%), "
            "so sector drawdowns could have outsized impact."
        )
    explanation = (
        f"Avg score {avg_score}/100 with CVaR-aware selection. "
        f"Horizon '{horizon}' sets the risk penalty."
        f"{' A defensive allocation is included to smooth volatility.' if hedge else ''}"
        f"{concentration_note}"
        f"{' Sector caps were applied to reduce concentration risk.' if sector_cap else ''}"
    )
    explanation_words = explanation.split()
    if len(explanation_words) > 50:
        explanation = " ".join(explanation_words[:50]).rstrip(".") + "."

    annualized_mean = port_mean * 252
    annualized_vol = port_vol * (252 ** 0.5)
    expected_return = annualized_mean * 0.5
    expected_range = (
        max(expected_return - annualized_vol, -0.1),
        min(expected_return + annualized_vol, 0.35),
    )
    scenario_analysis = {
        "bull": round(min(expected_return + annualized_vol, 0.35) * 100, 2),
        "base": round(expected_return * 100, 2),
        "bear": round(max(expected_return - annualized_vol, -0.1) * 100, 2),
    }

    # 组合标签（用于前端展示）
    strategy_label = (
        "Aggressive Growth with Hedge"
        if horizon == "short"
        else "Balanced Growth with Hedge"
    )
    if not hedge:
        strategy_label = strategy_label.replace(" with Hedge", "")

    if annualized_vol > 0.35 or port_cvar > 0.04:
        portfolio_risk_label = "High"
    elif annualized_vol > 0.25 or port_cvar > 0.025:
        portfolio_risk_label = "Medium-High"
    else:
        portfolio_risk_label = "Medium"

    is_concentrated = top_sector[1] >= 0.5 and top_sector[0] != "Unknown"

    # 相关矩阵输入（只基于组合成分）
    return {
        "inputs": {
            "top_n": top_n,
            "horizon": horizon,
            "use_hedge": use_hedge,
            "risk_level": risk_profile,
        },
        "portfolio": portfolio,
        "allocation_analysis": {
            "sector_distribution": {
                sector: round(weight, 2) for sector, weight in sector_weights.items()
            },
            "is_concentrated": is_concentrated,
            "dominant_sector": top_sector[0],
        },
        "analysis": {
            "scenario_analysis": scenario_analysis,
        },
        "summary": {
            "cvar_5pct": round(port_cvar * 100, 2),
            "hedge_symbol": hedge.symbol if hedge else None,
            "portfolio_risk": portfolio_risk_label,
            "explanation": explanation,
        },
    }
