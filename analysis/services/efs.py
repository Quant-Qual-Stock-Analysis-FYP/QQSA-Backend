"""EFS 数据抓取、特征计算与打分逻辑。使用 qlib Alpha158 因子体系。"""

import calendar
import logging
from datetime import date, datetime, timedelta
from statistics import pstdev
from typing import Dict, Iterable, List, Optional, Tuple

from django.db import transaction
from yahooquery import Ticker

from markets.models import Stock
from ..models import EfsAlphaFactor, EfsDataPoint
from .alpha158 import ALPHA158_FEATURE_NAMES, compute_alpha158_features

logger = logging.getLogger(__name__)

EFS_LOOKBACK_DAYS = 1095
EFS_LOOKBACK_PERIOD = "3y"


def _normalize_symbol(symbol: str) -> str:
    # 统一 symbol 格式，兼容 Binance 形式
    if "BINANCE:" in symbol:
        return symbol.replace("BINANCE:", "").replace("USDT", "-USD")
    return symbol


def fetch_efs_data(symbol: str, period: str = "1y") -> List[dict]:
    """
    Fetch OHLCV for a ticker; returns list of dict rows.
    Default 1y used for initial backfill; shorter periods for incremental.
    """
    if not symbol or not symbol.strip():
        return []
    
    try:
        normalized_symbol = _normalize_symbol(symbol)
        ticker = Ticker(normalized_symbol, asynchronous=True)
        df = ticker.history(period=period, interval="1d")
    except Exception as e:
        logger.debug(f"Error fetching data for {symbol}: {e}")
        return []

    if df is None or df.empty:
        return []

    try:
        frame = df.reset_index()

        # Handle multi-symbol responses
        if "symbol" in frame.columns:
            frame = frame[frame["symbol"] == normalized_symbol]

        rows = []
        for _, row in frame.iterrows():
            try:
                date_val = row.get("date")
                if date_val is None:
                    continue
                
                date_obj = (
                    date_val.date()
                    if hasattr(date_val, "date")
                    else datetime.strptime(str(date_val), "%Y-%m-%d").date()
                )
                
                open_val = row.get("open")
                high_val = row.get("high")
                low_val = row.get("low")
                close_val = row.get("close")
                volume_val = row.get("volume")
                
                # 验证数据有效性
                if close_val is None or close_val <= 0:
                    continue

                rows.append(
                    {
                        "date": date_obj,
                        "open": float(open_val) if open_val is not None else float(close_val),
                        "high": float(high_val) if high_val is not None else float(close_val),
                        "low": float(low_val) if low_val is not None else float(close_val),
                        "close": float(close_val),
                        "volume": int(volume_val or 0),
                    }
                )
            except Exception as e:
                logger.debug(f"Error processing row for {symbol}: {e}")
                continue
        
        return rows
    except Exception as e:
        logger.error(f"Error processing dataframe for {symbol}: {e}", exc_info=True)
        return []


def upsert_efs_timeseries(stock: Stock, rows: Iterable[dict]) -> None:
    """持久化获取的OHLCV数据到EfsDataPoint（按日期幂等）"""
    if not stock or not rows:
        return
    
    payload = []
    for row in rows:
        try:
            if not isinstance(row, dict):
                continue
            
            date_val = row.get("date")
            if not date_val:
                continue
            
            payload.append(
                EfsDataPoint(
                    stock=stock,
                    date=date_val,
                    open=row.get("open") or 0,
                    high=row.get("high") or 0,
                    low=row.get("low") or 0,
                    close=row.get("close") or 0,
                    volume=row.get("volume") or 0,
                )
            )
        except Exception as e:
            logger.debug(f"Error creating EfsDataPoint: {e}")
            continue

    if not payload:
        return
    
    try:
        with transaction.atomic():
            for item in payload:
                EfsDataPoint.objects.update_or_create(
                    stock=item.stock,
                    date=item.date,
                    defaults={
                        "open": item.open,
                        "high": item.high,
                        "low": item.low,
                        "close": item.close,
                        "volume": item.volume,
                    },
                )
    except Exception as e:
        logger.error(f"Error upserting timeseries for {stock.symbol}: {e}", exc_info=True)


def _filter_rows_by_window(rows: List[dict], start_date: date, end_date: date) -> List[dict]:
    """仅保留时间窗口内的数据，避免窗口外数据污染。"""
    filtered = []
    for row in rows:
        try:
            d = row.get("date")
            if d and start_date <= d <= end_date:
                filtered.append(row)
        except Exception:
            continue
    return filtered


def _prune_efs_timeseries(stock: Stock, start_date: date, end_date: date) -> None:
    """删除窗口外数据：只保留 [start_date, end_date]。"""
    try:
        EfsDataPoint.objects.filter(stock=stock, date__lt=start_date).delete()
        EfsDataPoint.objects.filter(stock=stock, date__gt=end_date).delete()
    except Exception as e:
        logger.error(f"Error pruning timeseries for {stock.symbol}: {e}", exc_info=True)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# Alpha158 feature names (for backward compat with efs_evolution)
EFS_FEATURE_KEYS = ALPHA158_FEATURE_NAMES


def _month_end(day: date) -> date:
    # 取月末日期
    last_day = calendar.monthrange(day.year, day.month)[1]
    return date(day.year, day.month, last_day)


def _get_month_end_dates() -> List[date]:
    # 从数据中提取每月最后一个交易日
    dates = list(EfsDataPoint.objects.values_list("date", flat=True).order_by("date"))
    if not dates:
        return []
    month_max: Dict[Tuple[int, int], date] = {}
    for d in dates:
        key = (d.year, d.month)
        if key not in month_max or d > month_max[key]:
            month_max[key] = d
    return sorted(month_max.values())


def get_month_end_dates() -> List[date]:
    # 对外暴露：EFS 月末序列
    return _get_month_end_dates()


def get_feature_snapshot(stock: Stock, target_date: date) -> Optional[Dict[str, Optional[float]]]:
    """获取指定日期的 Alpha158 特征快照（用于因子评估/演化）"""
    series = list(EfsDataPoint.objects.filter(stock=stock).order_by("date"))
    if not series:
        return None
    dates = [dp.date for dp in series]
    if target_date not in dates:
        fallback_dates = [d for d in dates if d <= target_date]
        if not fallback_dates:
            return None
        target_date = fallback_dates[-1]
    idx = dates.index(target_date)
    opens = [float(dp.open) for dp in series]
    highs = [float(dp.high) for dp in series]
    lows = [float(dp.low) for dp in series]
    closes = [float(dp.close) for dp in series]
    volumes = [float(dp.volume or 0) for dp in series]
    return compute_alpha158_features(opens, highs, lows, closes, volumes, idx)


# 无 active 因子时的兜底表达式（Alpha158 MA20）
DEFAULT_FALLBACK_EXPRESSION = "MA20"


def _safe_eval_expression(expression: str, features: Dict[str, Optional[float]]) -> Optional[float]:
    allowed = {k: (features.get(k) or 0.0) for k in features.keys()}
    try:
        return float(eval(expression, {"__builtins__": {}}, allowed))
    except Exception:
        return None


def _calculate_factor_rank_score(stock: Stock, target_date: date) -> Optional[Tuple[int, str, str]]:
    """使用唯一 active 因子进行排名打分；无因子时用 MA20 兜底"""
    factor = (
        EfsAlphaFactor.objects.filter(status="active")
        .exclude(expression__isnull=True)
        .order_by("-last_score", "-updated_at")
        .first()
    )
    expression = factor.expression if factor else DEFAULT_FALLBACK_EXPRESSION
    factor_name = factor.name if factor else "MA20"
    values = []
    stock_value = None
    for s in Stock.objects.all():
        snapshot = get_feature_snapshot(s, target_date)
        if not snapshot:
            continue
        val = _safe_eval_expression(expression, snapshot)
        if val is None:
            continue
        values.append((s.symbol, val))
        if s.id == stock.id:
            stock_value = val
    if stock_value is None or not values:
        return None
    ranked = sorted(values, key=lambda x: x[1], reverse=True)
    symbols = [sym for sym, _ in ranked]
    rank = symbols.index(stock.symbol) + 1 if stock.symbol in symbols else len(symbols)
    score = int(round((1 - (rank - 1) / max(1, len(symbols) - 1)) * 100))
    reason = f"EFS alpha '{factor_name}' rank {rank}/{len(symbols)}"
    return score, reason, expression


def calculate_risk_score(data: List[EfsDataPoint]) -> Tuple[int, str, str]:
    """
    Compute a risk score using recent volatility and max drawdown.
    Higher score implies lower risk.
    """
    if not data or len(data) < 30:
        return 50, "Insufficient data", "N/A"

    closes = [float(dp.close) for dp in data]
    if len(closes) < 30:
        return 50, "Insufficient data", "N/A"

    vol_window = 60 if len(closes) >= 60 else len(closes)
    vol_closes = closes[-vol_window:]
    daily_returns = [
        (vol_closes[i] / vol_closes[i - 1]) - 1
        for i in range(1, len(vol_closes))
        if vol_closes[i - 1] != 0
    ]
    vol = pstdev(daily_returns) if len(daily_returns) >= 2 else 0

    dd_window = 120 if len(closes) >= 120 else len(closes)
    dd_closes = closes[-dd_window:]
    peak = dd_closes[0] if dd_closes else 0
    max_drawdown = 0.0
    for close in dd_closes[1:]:
        if close > peak:
            peak = close
        if peak:
            drawdown = (peak - close) / peak
            max_drawdown = max(max_drawdown, drawdown)

    vol_score = _clamp(100 - (vol * 1000), 0, 100)
    drawdown_score = _clamp(100 - (max_drawdown * 100), 0, 100)
    risk_score = int((vol_score * 0.6) + (drawdown_score * 0.4))

    description = (
        f"Risk: {vol * 100:.2f}% daily volatility, "
        f"{max_drawdown * 100:.1f}% max drawdown"
    )
    formula = (
        f"VolStd{len(daily_returns)}={vol:.4f}, "
        f"MaxDD{len(dd_closes)}={max_drawdown:.4f}"
    )
    return risk_score, description, formula


def run_efs_analysis(symbol: str) -> Tuple[int, str, str]:
    """
    EFS pipeline: ensure timeseries, compute score, persist factor.
    Returns (score, reason, formula).
    """
    if not symbol or not symbol.strip():
        return 50, "Invalid symbol", "N/A"
    
    try:
        stock = Stock.objects.filter(symbol=symbol.upper()).first()
        if not stock:
            return 50, "Stock not found", "N/A"

        today = date.today()
        start_date = today - timedelta(days=EFS_LOOKBACK_DAYS)

        # 每次同步「今天往前2年」窗口，补齐中间遗漏交易日
        rows = fetch_efs_data(stock.symbol, period=EFS_LOOKBACK_PERIOD)
        rows = _filter_rows_by_window(rows, start_date, today)
        if rows:
            upsert_efs_timeseries(stock, rows)

        # 保持数据库仅存2年窗口
        _prune_efs_timeseries(stock, start_date, today)

        existing = list(
            EfsDataPoint.objects.filter(stock=stock, date__gte=start_date, date__lte=today)
            .order_by("date")
            .only("date", "close", "high", "low", "volume")
        )

        if not existing:
            return 50, "Insufficient data", "N/A"

        try:
            factor_score = _calculate_factor_rank_score(stock, existing[-1].date)
            if factor_score:
                score, reason, formula = factor_score
            else:
                score, reason, formula = 50, "Insufficient data for factor ranking", "N/A"
            return score, reason, formula
        except Exception as e:
            logger.error(f"Error calculating score for {symbol}: {e}", exc_info=True)
            return 50, f"Error: {str(e)}", "N/A"
            
    except Exception as e:
        logger.error(f"Error in run_efs_analysis for {symbol}: {e}", exc_info=True)
        return 50, f"Error: {str(e)}", "N/A"

