"""Walk-forward backtesting service for dynamic factor-only ranking."""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, timedelta
from statistics import pstdev
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from markets.models import Stock
from ..config.constants import BENCHMARK_ETF, ETF_SYMBOLS
from ..config.settings import EfsConfig
from ..models import EfsAlphaFactor, EfsDataPoint
from .alpha158 import compute_alpha158_features


MIN_FEATURE_BARS = 61
DEFAULT_INITIAL_CAPITAL = 1_000_000.0
RANKING_MODE_FACTOR_ONLY = "factor_only"
RANKING_MODE_FACTOR_RISK = "factor_risk"


@dataclass
class PriceSeries:
    """In-memory OHLCV series with date lookup helpers."""

    stock: Stock
    dates: List[date]
    opens: List[float]
    highs: List[float]
    lows: List[float]
    closes: List[float]
    volumes: List[float]

    @classmethod
    def from_rows(cls, stock: Stock, rows: Sequence[EfsDataPoint]) -> "PriceSeries":
        return cls(
            stock=stock,
            dates=[row.date for row in rows],
            opens=[float(row.open) for row in rows],
            highs=[float(row.high) for row in rows],
            lows=[float(row.low) for row in rows],
            closes=[float(row.close) for row in rows],
            volumes=[float(row.volume or 0) for row in rows],
        )

    def last_index_on_or_before(self, target_date: date) -> Optional[int]:
        idx = bisect_right(self.dates, target_date) - 1
        return idx if idx >= 0 else None

    def price_on_or_before(self, target_date: date) -> Optional[float]:
        idx = self.last_index_on_or_before(target_date)
        if idx is None:
            return None
        price = self.closes[idx]
        return price if price > 0 else None

    def factor_value(self, as_of: date, expression: str) -> Optional[float]:
        idx = self.last_index_on_or_before(as_of)
        if idx is None or idx + 1 < MIN_FEATURE_BARS:
            return None
        features = compute_alpha158_features(
            self.opens,
            self.highs,
            self.lows,
            self.closes,
            self.volumes,
            idx,
        )
        if not features:
            return None
        return _safe_eval_expression(expression, features)

    def forward_return(
        self,
        start_date: date,
        end_date: date,
    ) -> Optional[float]:
        start_price = self.price_on_or_before(start_date)
        end_price = self.price_on_or_before(end_date)
        if start_price is None or end_price is None or start_price <= 0:
            return None
        return (end_price / start_price) - 1.0


@dataclass
class FactorEvaluationResult:
    """Historical factor performance known by a rebalance date."""

    score: float
    ic_mean: float
    top_return_mean: float
    eval_dates_count: int
    sample_count_mean: float


def _safe_eval_expression(
    expression: str,
    features: Dict[str, Optional[float]],
) -> Optional[float]:
    allowed = {key: (0.0 if value is None else float(value)) for key, value in features.items()}
    try:
        value = float(eval(expression, {"__builtins__": {}}, allowed))
    except Exception:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _load_series_map(
    stocks: Sequence[Stock],
    start_date: date,
    end_date: date,
) -> Dict[str, PriceSeries]:
    rows = (
        EfsDataPoint.objects.filter(
            stock__in=stocks,
            date__gte=start_date,
            date__lte=end_date,
        )
        .select_related("stock")
        .order_by("stock_id", "date")
    )

    grouped: Dict[int, List[EfsDataPoint]] = {}
    stock_lookup: Dict[int, Stock] = {}
    for row in rows:
        grouped.setdefault(row.stock_id, []).append(row)
        stock_lookup[row.stock_id] = row.stock

    series_map: Dict[str, PriceSeries] = {}
    for stock_id, stock_rows in grouped.items():
        stock = stock_lookup.get(stock_id)
        if stock and stock_rows:
            series_map[stock.symbol] = PriceSeries.from_rows(stock, stock_rows)
    return series_map


def _candidate_factors(top_k: int) -> List[EfsAlphaFactor]:
    if top_k <= 0:
        raise ValueError("candidate_factor_top_k must be greater than 0")

    factors = list(
        EfsAlphaFactor.objects.exclude(expression__isnull=True)
        .exclude(expression="")
        .exclude(last_score__isnull=True)
        .order_by("-last_score", "-updated_at")
        .only("id", "name", "expression", "last_score")[:top_k]
    )
    if factors:
        return factors

    return list(
        EfsAlphaFactor.objects.exclude(expression__isnull=True)
        .exclude(expression="")
        .order_by("-updated_at")
        .only("id", "name", "expression", "last_score")[:top_k]
    )


def _build_rebalance_dates(
    benchmark_dates: Sequence[date],
    start_date: date,
    end_date: date,
    rebalance_every: int,
) -> List[date]:
    window_dates = [d for d in benchmark_dates if start_date <= d <= end_date]
    if len(window_dates) < 2:
        raise ValueError("Not enough benchmark trading dates in the selected window")

    rebalance_dates = window_dates[::rebalance_every]
    if rebalance_dates[-1] != window_dates[-1]:
        rebalance_dates.append(window_dates[-1])

    if len(rebalance_dates) < 2:
        raise ValueError("Not enough rebalance dates for backtesting")
    return rebalance_dates


def _eligible_eval_dates(
    benchmark_dates: Sequence[date],
    rebalance_date: date,
    factor_lookback_days: int,
    future_days: int,
    eval_step_days: int,
) -> List[date]:
    if eval_step_days <= 0:
        raise ValueError("eval_step_days must be greater than 0")

    try:
        rebalance_idx = benchmark_dates.index(rebalance_date)
    except ValueError as exc:
        raise ValueError(f"Rebalance date {rebalance_date} not in benchmark calendar") from exc

    end_idx = rebalance_idx - future_days
    if end_idx < 0:
        return []

    start_idx = max(0, rebalance_idx - factor_lookback_days + 1)
    return list(benchmark_dates[start_idx : end_idx + 1 : eval_step_days])


def _evaluate_factor_as_of(
    expression: str,
    rebalance_date: date,
    benchmark_dates: Sequence[date],
    universe_series: Sequence[PriceSeries],
    factor_lookback_days: int,
    future_days: int,
    eval_step_days: int,
    top_m: int,
    min_samples: int,
) -> Optional[FactorEvaluationResult]:
    eval_dates = _eligible_eval_dates(
        benchmark_dates=benchmark_dates,
        rebalance_date=rebalance_date,
        factor_lookback_days=factor_lookback_days,
        future_days=future_days,
        eval_step_days=eval_step_days,
    )
    if not eval_dates:
        return None

    benchmark_index = {day: idx for idx, day in enumerate(benchmark_dates)}
    ic_list: List[float] = []
    top_return_list: List[float] = []
    sample_counts: List[int] = []

    for eval_date in eval_dates:
        future_idx = benchmark_index[eval_date] + future_days
        if future_idx >= len(benchmark_dates):
            continue
        future_date = benchmark_dates[future_idx]

        values: List[float] = []
        returns: List[float] = []

        for series in universe_series:
            factor_value = series.factor_value(eval_date, expression)
            if factor_value is None:
                continue
            forward_ret = series.forward_return(eval_date, future_date)
            if forward_ret is None:
                continue
            values.append(factor_value)
            returns.append(forward_ret)

        if len(values) < min_samples:
            continue

        value_std = float(np.std(values))
        ret_std = float(np.std(returns))
        if value_std <= 0 or ret_std <= 0:
            continue

        ic = float(np.corrcoef(values, returns)[0, 1])
        if math.isnan(ic) or math.isinf(ic):
            continue

        ranked = sorted(zip(values, returns), key=lambda item: item[0], reverse=True)
        top_slice = ranked[: max(1, min(top_m, len(ranked)))]
        top_return = float(sum(ret for _, ret in top_slice) / len(top_slice))

        ic_list.append(ic)
        top_return_list.append(top_return)
        sample_counts.append(len(values))

    if not ic_list or not top_return_list:
        return None

    ic_mean = float(sum(ic_list) / len(ic_list))
    top_return_mean = float(sum(top_return_list) / len(top_return_list))
    score = (ic_mean * EfsConfig.IC_WEIGHT) + (top_return_mean * EfsConfig.TOP_RETURN_WEIGHT)
    sample_count_mean = float(sum(sample_counts) / len(sample_counts)) if sample_counts else 0.0

    return FactorEvaluationResult(
        score=score,
        ic_mean=ic_mean,
        top_return_mean=top_return_mean,
        eval_dates_count=len(ic_list),
        sample_count_mean=sample_count_mean,
    )


def _select_best_factor_as_of(
    rebalance_date: date,
    benchmark_dates: Sequence[date],
    candidate_factors: Sequence[EfsAlphaFactor],
    universe_series: Sequence[PriceSeries],
    factor_lookback_days: int,
    future_days: int,
    eval_step_days: int,
    top_m: int,
    min_samples: int,
) -> Tuple[EfsAlphaFactor, FactorEvaluationResult]:
    best_factor: Optional[EfsAlphaFactor] = None
    best_result: Optional[FactorEvaluationResult] = None

    for factor in candidate_factors:
        result = _evaluate_factor_as_of(
            expression=factor.expression,
            rebalance_date=rebalance_date,
            benchmark_dates=benchmark_dates,
            universe_series=universe_series,
            factor_lookback_days=factor_lookback_days,
            future_days=future_days,
            eval_step_days=eval_step_days,
            top_m=top_m,
            min_samples=min_samples,
        )
        if result is None:
            continue
        if best_result is None or result.score > best_result.score:
            best_factor = factor
            best_result = result

    if best_factor is None or best_result is None:
        raise ValueError(f"No valid factor could be selected as of {rebalance_date}")

    return best_factor, best_result


def _normalize_rank_weights(
    factor_weight: float,
    risk_weight: float,
) -> Tuple[float, float]:
    if factor_weight < 0 or risk_weight < 0:
        raise ValueError("factor_weight and risk_weight must be non-negative")
    total = factor_weight + risk_weight
    if total <= 0:
        raise ValueError("factor_weight and risk_weight cannot both be zero")
    return factor_weight / total, risk_weight / total


def _percentile_score(values: Sequence[float], current: float) -> float:
    if not values:
        return 50.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return 100.0
    rank = sum(1 for value in ordered if value <= current)
    return ((rank - 1) / (len(ordered) - 1)) * 100.0


def _historical_risk_score(series: PriceSeries, as_of: date) -> float:
    idx = series.last_index_on_or_before(as_of)
    if idx is None or idx + 1 < 30:
        return 50.0

    closes = series.closes[: idx + 1]
    if len(closes) < 30:
        return 50.0

    vol_window = min(60, len(closes))
    vol_closes = closes[-vol_window:]
    daily_returns = [
        (vol_closes[i] / vol_closes[i - 1]) - 1
        for i in range(1, len(vol_closes))
        if vol_closes[i - 1] != 0
    ]
    vol = pstdev(daily_returns) if len(daily_returns) >= 2 else 0.0

    dd_window = min(120, len(closes))
    dd_closes = closes[-dd_window:]
    peak = dd_closes[0] if dd_closes else 0.0
    max_drawdown = 0.0
    for close in dd_closes[1:]:
        if close > peak:
            peak = close
        if peak > 0:
            drawdown = (peak - close) / peak
            max_drawdown = max(max_drawdown, drawdown)

    vol_score = _clamp(100 - (vol * 1000), 0, 100)
    drawdown_score = _clamp(100 - (max_drawdown * 100), 0, 100)
    return (vol_score * 0.6) + (drawdown_score * 0.4)


def _rank_stocks_as_of(
    expression: str,
    rebalance_date: date,
    universe_series: Sequence[PriceSeries],
    top_n: int,
    ranking_mode: str = RANKING_MODE_FACTOR_ONLY,
    factor_weight: float = 0.7,
    risk_weight: float = 0.3,
) -> List[Dict[str, object]]:
    if ranking_mode not in {RANKING_MODE_FACTOR_ONLY, RANKING_MODE_FACTOR_RISK}:
        raise ValueError(f"Unsupported ranking_mode: {ranking_mode}")

    normalized_factor_weight, normalized_risk_weight = _normalize_rank_weights(
        factor_weight,
        risk_weight,
    )

    ranked: List[Dict[str, object]] = []
    raw_factor_values: List[float] = []
    for series in universe_series:
        value = series.factor_value(rebalance_date, expression)
        if value is None:
            continue
        risk_score = _historical_risk_score(series, rebalance_date)
        ranked.append(
            {
                "series": series,
                "factor_value": value,
                "risk_score": risk_score,
            }
        )
        raw_factor_values.append(value)

    for item in ranked:
        factor_score = _percentile_score(raw_factor_values, float(item["factor_value"]))
        risk_score = float(item["risk_score"])
        if ranking_mode == RANKING_MODE_FACTOR_RISK:
            combined_score = (
                factor_score * normalized_factor_weight
                + risk_score * normalized_risk_weight
            )
        else:
            combined_score = factor_score
        item["factor_score"] = factor_score
        item["combined_score"] = combined_score

    ranked.sort(key=lambda item: float(item["combined_score"]), reverse=True)
    if len(ranked) < top_n:
        raise ValueError(
            f"Only {len(ranked)} stocks could be ranked on {rebalance_date}, fewer than top_n={top_n}"
        )
    return ranked[:top_n]


def _compute_summary_metrics(
    initial_capital: float,
    final_value: float,
    daily_returns: Sequence[float],
    daily_values: Sequence[float],
) -> Dict[str, float]:
    if initial_capital <= 0:
        raise ValueError("initial_capital must be greater than 0")

    if not daily_values:
        total_return = (final_value / initial_capital) - 1.0
        return {
            "total_return": total_return,
            "annualized_return": total_return,
            "max_drawdown": 0.0,
            "volatility": 0.0,
            "sharpe": 0.0,
        }

    total_return = (final_value / initial_capital) - 1.0
    periods = len(daily_returns)
    annualized_return = (final_value / initial_capital) ** (252 / periods) - 1.0 if periods > 0 else 0.0
    volatility = pstdev(daily_returns) * (252 ** 0.5) if len(daily_returns) >= 2 else 0.0
    sharpe = annualized_return / volatility if volatility > 1e-12 else 0.0

    running_peak = daily_values[0]
    max_drawdown = 0.0
    for value in daily_values:
        running_peak = max(running_peak, value)
        drawdown = (value / running_peak) - 1.0 if running_peak > 0 else 0.0
        max_drawdown = min(max_drawdown, drawdown)

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "max_drawdown": max_drawdown,
        "volatility": volatility,
        "sharpe": sharpe,
    }


def _history_buffer_start(
    start_date: date,
    factor_lookback_days: int,
    future_days: int,
) -> date:
    trading_days_needed = factor_lookback_days + future_days + MIN_FEATURE_BARS + 20
    return start_date - timedelta(days=max(400, trading_days_needed * 3))


def backtest_dynamic_factor_strategy(
    start_date: date,
    end_date: date,
    top_n: int = 5,
    rebalance_every: int = 20,
    factor_lookback_days: int = 120,
    future_days: int = EfsConfig.DEFAULT_FUTURE_DAYS,
    candidate_factor_top_k: int = 10,
    eval_step_days: int = EfsConfig.DEFAULT_STEP_DAYS,
    top_m: int = EfsConfig.DEFAULT_TOP_M,
    min_samples: int = EfsConfig.DEFAULT_MIN_SAMPLES,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    ranking_mode: str = RANKING_MODE_FACTOR_ONLY,
    factor_weight: float = 0.7,
    risk_weight: float = 0.3,
) -> Dict[str, object]:
    """
    Dynamic factor walk-forward backtest.

    On each rebalance date, the service:
    1. Selects the best factor using only historical windows fully known by that date.
    2. Ranks stocks using factor-only or factor-plus-risk scoring.
    3. Buys top_n stocks with equal weight on the next trading day.
    4. Holds until the next rebalance date.
    """
    if start_date >= end_date:
        raise ValueError("start_date must be earlier than end_date")
    if top_n <= 0:
        raise ValueError("top_n must be greater than 0")
    if rebalance_every <= 0:
        raise ValueError("rebalance_every must be greater than 0")
    if factor_lookback_days <= 0:
        raise ValueError("factor_lookback_days must be greater than 0")
    if future_days <= 0:
        raise ValueError("future_days must be greater than 0")
    if initial_capital <= 0:
        raise ValueError("initial_capital must be greater than 0")
    if ranking_mode not in {RANKING_MODE_FACTOR_ONLY, RANKING_MODE_FACTOR_RISK}:
        raise ValueError(
            f"ranking_mode must be '{RANKING_MODE_FACTOR_ONLY}' or '{RANKING_MODE_FACTOR_RISK}'"
        )

    normalized_factor_weight, normalized_risk_weight = _normalize_rank_weights(
        factor_weight,
        risk_weight,
    )

    benchmark_stock = Stock.objects.filter(symbol=BENCHMARK_ETF).first()
    if not benchmark_stock:
        raise ValueError(f"Benchmark {BENCHMARK_ETF} was not found")

    universe = list(Stock.objects.exclude(symbol__in=ETF_SYMBOLS).order_by("symbol"))
    if not universe:
        raise ValueError("No eligible stocks available for backtesting")

    candidate_factors = _candidate_factors(candidate_factor_top_k)
    if not candidate_factors:
        raise ValueError("No factor candidates available for backtesting")

    load_start = _history_buffer_start(start_date, factor_lookback_days, future_days)
    series_map = _load_series_map(universe + [benchmark_stock], load_start, end_date)
    benchmark_series = series_map.get(BENCHMARK_ETF)
    if benchmark_series is None or not benchmark_series.dates:
        raise ValueError(f"No benchmark OHLCV data found for {BENCHMARK_ETF}")

    benchmark_dates = [d for d in benchmark_series.dates if start_date <= d <= end_date]
    rebalance_dates = _build_rebalance_dates(
        benchmark_dates=benchmark_series.dates,
        start_date=start_date,
        end_date=end_date,
        rebalance_every=rebalance_every,
    )

    universe_series = [series for symbol, series in series_map.items() if symbol not in ETF_SYMBOLS]
    if not universe_series:
        raise ValueError("No stock price series found for the backtest universe")

    benchmark_index = {day: idx for idx, day in enumerate(benchmark_series.dates)}

    portfolio_value = float(initial_capital)
    benchmark_value = float(initial_capital)
    daily_portfolio_values: List[float] = [portfolio_value]
    daily_benchmark_values: List[float] = [benchmark_value]
    daily_portfolio_returns: List[float] = []
    daily_benchmark_returns: List[float] = []
    rebalance_log: List[Dict[str, object]] = []

    for idx in range(len(rebalance_dates) - 1):
        rebalance_date = rebalance_dates[idx]
        next_rebalance_date = rebalance_dates[idx + 1]

        best_factor, factor_result = _select_best_factor_as_of(
            rebalance_date=rebalance_date,
            benchmark_dates=benchmark_series.dates,
            candidate_factors=candidate_factors,
            universe_series=universe_series,
            factor_lookback_days=factor_lookback_days,
            future_days=future_days,
            eval_step_days=eval_step_days,
            top_m=top_m,
            min_samples=min_samples,
        )

        ranked_stocks = _rank_stocks_as_of(
            expression=best_factor.expression,
            rebalance_date=rebalance_date,
            universe_series=universe_series,
            top_n=top_n,
            ranking_mode=ranking_mode,
            factor_weight=normalized_factor_weight,
            risk_weight=normalized_risk_weight,
        )

        trade_start_idx = benchmark_index[rebalance_date] + 1
        if trade_start_idx >= len(benchmark_series.dates):
            break
        trade_start_date = benchmark_series.dates[trade_start_idx]
        trade_end_date = next_rebalance_date
        valuation_dates = [
            d
            for d in benchmark_series.dates
            if trade_start_date <= d <= trade_end_date
        ]
        if len(valuation_dates) < 2:
            continue

        valid_ranked: List[Dict[str, object]] = []
        for item in ranked_stocks:
            series = item["series"]
            entry_price = series.price_on_or_before(trade_start_date)
            if entry_price is None or entry_price <= 0:
                continue
            valid_ranked.append(item)

        if len(valid_ranked) < top_n:
            raise ValueError(
                f"Only {len(valid_ranked)} ranked stocks had tradable prices on {trade_start_date}"
            )

        selected = valid_ranked[:top_n]
        weight = 1.0 / len(selected)
        portfolio_value_start = portfolio_value
        benchmark_value_start = benchmark_value
        allocation_per_stock = portfolio_value_start * weight

        holdings = []
        for rank, item in enumerate(selected, start=1):
            series = item["series"]
            entry_price = series.price_on_or_before(trade_start_date)
            if entry_price is None or entry_price <= 0:
                raise ValueError(f"Missing entry price for {series.stock.symbol} on {trade_start_date}")
            shares = allocation_per_stock / entry_price
            holdings.append(
                {
                    "series": series,
                    "symbol": series.stock.symbol,
                    "rank": rank,
                    "factor_value": float(item["factor_value"]),
                    "factor_score": float(item["factor_score"]),
                    "risk_score": float(item["risk_score"]),
                    "combined_score": float(item["combined_score"]),
                    "weight": weight,
                    "allocation_usd": allocation_per_stock,
                    "entry_price": entry_price,
                    "shares": shares,
                }
            )

        benchmark_entry_price = benchmark_series.price_on_or_before(trade_start_date)
        if benchmark_entry_price is None or benchmark_entry_price <= 0:
            raise ValueError(f"Missing benchmark entry price on {trade_start_date}")
        benchmark_shares = benchmark_value_start / benchmark_entry_price

        period_start_value = portfolio_value_start
        period_benchmark_start_value = benchmark_value_start

        for day_position, valuation_date in enumerate(valuation_dates):
            current_value = sum(
                holding["shares"] * (holding["series"].price_on_or_before(valuation_date) or 0.0)
                for holding in holdings
            )
            current_benchmark_value = benchmark_shares * (
                benchmark_series.price_on_or_before(valuation_date) or 0.0
            )

            if day_position == 0:
                portfolio_value = current_value
                benchmark_value = current_benchmark_value
                daily_portfolio_values.append(portfolio_value)
                daily_benchmark_values.append(benchmark_value)
                continue

            previous_portfolio_value = portfolio_value
            previous_benchmark_value = benchmark_value
            portfolio_value = current_value
            benchmark_value = current_benchmark_value

            portfolio_return = (
                (portfolio_value / previous_portfolio_value) - 1.0
                if previous_portfolio_value > 0
                else 0.0
            )
            benchmark_return = (
                (benchmark_value / previous_benchmark_value) - 1.0
                if previous_benchmark_value > 0
                else 0.0
            )

            daily_portfolio_returns.append(portfolio_return)
            daily_benchmark_returns.append(benchmark_return)
            daily_portfolio_values.append(portfolio_value)
            daily_benchmark_values.append(benchmark_value)

        period_return = (
            (portfolio_value / period_start_value) - 1.0 if period_start_value > 0 else 0.0
        )
        benchmark_period_return = (
            (benchmark_value / period_benchmark_start_value) - 1.0
            if period_benchmark_start_value > 0
            else 0.0
        )

        log_holdings = []
        for holding in holdings:
            exit_price = holding["series"].price_on_or_before(trade_end_date)
            exit_value = holding["shares"] * exit_price if exit_price else 0.0
            log_holdings.append(
                {
                    "symbol": holding["symbol"],
                    "rank": holding["rank"],
                    "factor_value": round(float(holding["factor_value"]), 6),
                    "factor_score": round(float(holding["factor_score"]), 4),
                    "risk_score": round(float(holding["risk_score"]), 4),
                    "combined_score": round(float(holding["combined_score"]), 4),
                    "weight": round(float(holding["weight"]), 6),
                    "allocation_usd": round(float(holding["allocation_usd"]), 2),
                    "entry_price": round(float(holding["entry_price"]), 4),
                    "shares": round(float(holding["shares"]), 6),
                    "exit_price": round(float(exit_price), 4) if exit_price else None,
                    "exit_value_usd": round(float(exit_value), 2),
                }
            )

        eval_dates = _eligible_eval_dates(
            benchmark_dates=benchmark_series.dates,
            rebalance_date=rebalance_date,
            factor_lookback_days=factor_lookback_days,
            future_days=future_days,
            eval_step_days=eval_step_days,
        )
        training_window_start = eval_dates[0].isoformat() if eval_dates else None
        training_window_end = eval_dates[-1].isoformat() if eval_dates else None

        rebalance_log.append(
            {
                "rebalance_date": rebalance_date.isoformat(),
                "trade_start_date": trade_start_date.isoformat(),
                "next_rebalance_date": next_rebalance_date.isoformat(),
                "training_window_start": training_window_start,
                "training_window_end": training_window_end,
                "selected_factor": {
                    "name": best_factor.name,
                    "expression": best_factor.expression,
                    "historical_score": round(float(factor_result.score), 6),
                    "ic_mean": round(float(factor_result.ic_mean), 6),
                    "top_return_mean": round(float(factor_result.top_return_mean), 6),
                    "eval_dates_count": factor_result.eval_dates_count,
                    "sample_count_mean": round(float(factor_result.sample_count_mean), 2),
                    "candidate_factor_top_k": candidate_factor_top_k,
                },
                "ranking_mode": ranking_mode,
                "ranking_weights": {
                    "factor_weight": round(float(normalized_factor_weight), 4),
                    "risk_weight": round(float(normalized_risk_weight), 4),
                },
                "holdings": log_holdings,
                "portfolio_value_start": round(float(period_start_value), 2),
                "portfolio_value_end": round(float(portfolio_value), 2),
                "period_return": round(float(period_return), 6),
                "spy_period_return": round(float(benchmark_period_return), 6),
                "excess_return": round(float(period_return - benchmark_period_return), 6),
            }
        )

    strategy_metrics = _compute_summary_metrics(
        initial_capital=initial_capital,
        final_value=portfolio_value,
        daily_returns=daily_portfolio_returns,
        daily_values=daily_portfolio_values,
    )
    benchmark_metrics = _compute_summary_metrics(
        initial_capital=initial_capital,
        final_value=benchmark_value,
        daily_returns=daily_benchmark_returns,
        daily_values=daily_benchmark_values,
    )

    return {
        "summary": {
            "Initial Capital": round(float(initial_capital), 2),
            "Final Portfolio Value": round(float(portfolio_value), 2),
            "Final Benchmark Value": round(float(benchmark_value), 2),
            "Total Return": round(strategy_metrics["total_return"] * 100, 2),
            "Annualized Return": round(strategy_metrics["annualized_return"] * 100, 2),
            "Max Drawdown": round(strategy_metrics["max_drawdown"] * 100, 2),
            "Volatility": round(strategy_metrics["volatility"] * 100, 2),
            "Sharpe": round(strategy_metrics["sharpe"], 3),
            "SPY Return": round(benchmark_metrics["total_return"] * 100, 2),
        },
        "rebalance_log": rebalance_log,
        "config": {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "initial_capital": round(float(initial_capital), 2),
            "top_n": top_n,
            "rebalance_every": rebalance_every,
            "factor_lookback_days": factor_lookback_days,
            "future_days": future_days,
            "candidate_factor_top_k": candidate_factor_top_k,
            "eval_step_days": eval_step_days,
            "top_m": top_m,
            "min_samples": min_samples,
            "ranking_mode": ranking_mode,
            "factor_weight": round(float(normalized_factor_weight), 4),
            "risk_weight": round(float(normalized_risk_weight), 4),
            "benchmark": BENCHMARK_ETF,
        },
        # "portfolio_nav": [round(value, 2) for value in daily_portfolio_values],
        # "benchmark_nav": [round(value, 2) for value in daily_benchmark_values],
        # "portfolio_daily_returns": [round(value, 8) for value in daily_portfolio_returns],
        # "benchmark_daily_returns": [round(value, 8) for value in daily_benchmark_returns],
    }
