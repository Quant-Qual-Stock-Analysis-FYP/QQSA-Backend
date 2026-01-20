"""EFS 数据抓取、特征计算与打分逻辑（含滚动权重更新）。"""

import calendar
import logging
from datetime import date, datetime
from statistics import pstdev
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from django.db import transaction
from yahooquery import Ticker

from markets.models import Stock
from ..models import EfsAlphaFactor, EfsDataPoint

logger = logging.getLogger(__name__)

try:
    import talib  # type: ignore

    TA_LIB_AVAILABLE = True
except Exception:
    talib = None
    TA_LIB_AVAILABLE = False


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


def _calculate_rsi(closes: List[float], period: int = 14) -> Optional[float]:
    # 轻量 RSI 计算（TA-Lib 不可用时兜底）
    if len(closes) <= period:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(delta, 0) for delta in deltas]
    losses = [abs(min(delta, 0)) for delta in deltas]
    recent_gains = gains[-period:]
    recent_losses = losses[-period:]
    avg_gain = sum(recent_gains) / period
    avg_loss = sum(recent_losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# EFS 特征集合与默认权重（权重会被滚动训练覆盖）
EFS_FEATURE_KEYS = [
    "trend",
    "momentum",
    "volatility",
    "volume",
    "macd",
    "bbands",
    "adx",
    "atr",
    "stoch",
]

def _safe_array_value(array: Optional[np.ndarray], idx: int) -> Optional[float]:
    # 读取 TA-Lib 输出并处理 NaN/None
    if array is None or idx >= len(array):
        return None
    value = array[idx]
    if value is None:
        return None
    try:
        if np.isnan(value):
            return None
    except Exception:
        pass
    return float(value)


def _build_indicator_cache(
    closes: List[float], highs: List[float], lows: List[float]
) -> Dict[str, Optional[np.ndarray]]:
    # 统一计算 TA-Lib 指标，避免重复计算
    if not TA_LIB_AVAILABLE or len(closes) < 20:
        return {
            "macd_hist": None,
            "bb_upper": None,
            "bb_middle": None,
            "bb_lower": None,
            "adx": None,
            "atr": None,
            "stoch_k": None,
            "stoch_d": None,
            "rsi": None,
        }
    close_arr = np.array(closes, dtype=float)
    high_arr = np.array(highs, dtype=float)
    low_arr = np.array(lows, dtype=float)
    macd, macd_signal, macd_hist = talib.MACD(close_arr, fastperiod=12, slowperiod=26, signalperiod=9)
    bb_upper, bb_middle, bb_lower = talib.BBANDS(close_arr, timeperiod=20, nbdevup=2, nbdevdn=2)
    adx = talib.ADX(high_arr, low_arr, close_arr, timeperiod=14)
    atr = talib.ATR(high_arr, low_arr, close_arr, timeperiod=14)
    stoch_k, stoch_d = talib.STOCH(
        high_arr, low_arr, close_arr, fastk_period=14, slowk_period=3, slowd_period=3
    )
    rsi = talib.RSI(close_arr, timeperiod=14)
    return {
        "macd_hist": macd_hist,
        "bb_upper": bb_upper,
        "bb_middle": bb_middle,
        "bb_lower": bb_lower,
        "adx": adx,
        "atr": atr,
        "stoch_k": stoch_k,
        "stoch_d": stoch_d,
        "rsi": rsi,
    }


def _compute_raw_features(
    closes: List[float],
    highs: List[float],
    lows: List[float],
    volumes: List[float],
    idx: int,
    indicator_cache: Dict[str, Optional[np.ndarray]],
) -> Dict[str, Optional[float]]:
    # 计算单日特征（用于打分与IC训练）
    last_close = closes[idx]
    window_closes = closes[: idx + 1]
    window_volumes = volumes[: idx + 1]
    ma_short = 20 if len(window_closes) >= 20 else max(10, len(window_closes))
    ma_mid = 50 if len(window_closes) >= 50 else ma_short
    sma_short = sum(window_closes[-ma_short:]) / ma_short
    sma_mid = sum(window_closes[-ma_mid:]) / ma_mid if ma_mid else 0
    price_vs_ma = (last_close / sma_mid) - 1 if sma_mid else 0
    slope = (sma_short / sma_mid) - 1 if sma_mid else 0

    rsi = _safe_array_value(indicator_cache.get("rsi"), idx)
    if rsi is None:
        rsi = _calculate_rsi(window_closes, period=14)

    ret_20 = (window_closes[-1] / window_closes[-21]) - 1 if len(window_closes) >= 21 else 0
    daily_returns = [
        (window_closes[i] / window_closes[i - 1]) - 1
        for i in range(1, len(window_closes))
        if window_closes[i - 1] != 0
    ]
    vol = pstdev(daily_returns) if len(daily_returns) >= 2 else 0
    vol_window = 20 if len(window_volumes) >= 20 else len(window_volumes)
    avg_vol = sum(window_volumes[-vol_window:]) / vol_window if vol_window else 0
    vol_ratio = (window_volumes[-1] / avg_vol) if avg_vol else 1

    macd_hist = _safe_array_value(indicator_cache.get("macd_hist"), idx)
    bb_upper = _safe_array_value(indicator_cache.get("bb_upper"), idx)
    bb_lower = _safe_array_value(indicator_cache.get("bb_lower"), idx)
    bb_middle = _safe_array_value(indicator_cache.get("bb_middle"), idx)
    bb_pos = None
    if bb_upper is not None and bb_lower is not None and bb_upper != bb_lower:
        bb_pos = (last_close - bb_lower) / (bb_upper - bb_lower)

    adx = _safe_array_value(indicator_cache.get("adx"), idx)
    atr = _safe_array_value(indicator_cache.get("atr"), idx)
    atr_pct = (atr / last_close) if atr and last_close else None
    stoch_k = _safe_array_value(indicator_cache.get("stoch_k"), idx)
    stoch_d = _safe_array_value(indicator_cache.get("stoch_d"), idx)

    return {
        "trend": price_vs_ma + slope,
        "momentum": ret_20,
        "volatility": -vol,
        "volume": vol_ratio,
        "macd": macd_hist,
        "bbands": bb_pos,
        "adx": adx,
        "atr": -atr_pct if atr_pct is not None else None,
        "stoch": stoch_k,
        "rsi": rsi,
        "ret_20": ret_20,
        "vol": vol,
        "vol_ratio": vol_ratio,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "bb_middle": bb_middle,
        "adx_raw": adx,
        "atr_raw": atr,
        "atr_pct": atr_pct,
        "stoch_d": stoch_d,
        "sma_mid": sma_mid,
        "ma_mid": ma_mid,
        "sma_short": sma_short,
    }


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
    # 获取指定日期（或最近交易日）的特征快照（用于因子评估/演化）
    series = list(EfsDataPoint.objects.filter(stock=stock).order_by("date"))
    if not series:
        return None
    dates = [dp.date for dp in series]
    if target_date not in dates:
        # 回退到最近一个不晚于 target_date 的交易日
        fallback_dates = [d for d in dates if d <= target_date]
        if not fallback_dates:
            return None
        target_date = fallback_dates[-1]
    idx = dates.index(target_date)
    closes = [float(dp.close) for dp in series]
    highs = [float(dp.high) for dp in series]
    lows = [float(dp.low) for dp in series]
    volumes = [float(dp.volume or 0) for dp in series]
    cache = _build_indicator_cache(closes, highs, lows)
    return _compute_raw_features(closes, highs, lows, volumes, idx, cache)


def _safe_eval_expression(expression: str, features: Dict[str, Optional[float]]) -> Optional[float]:
    allowed = {k: (features.get(k) or 0.0) for k in features.keys()}
    try:
        return float(eval(expression, {"__builtins__": {}}, allowed))
    except Exception:
        return None


def _calculate_factor_rank_score(stock: Stock, target_date: date) -> Optional[Tuple[int, str, str]]:
    # 使用 LLM 进化因子进行排名打分
    factor = (
        EfsAlphaFactor.objects.filter(status="active")
        .exclude(expression__isnull=True)
        .order_by("-last_score", "-updated_at")
        .first()
    )
    if not factor:
        return None
    values = []
    stock_value = None
    for s in Stock.objects.all():
        snapshot = get_feature_snapshot(s, target_date)
        if not snapshot:
            continue
        val = _safe_eval_expression(factor.expression, snapshot)
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
    reason = f"EFS alpha '{factor.name}' rank {rank}/{len(symbols)}"
    formula = factor.expression
    return score, reason, formula


def _calculate_score(data: List[EfsDataPoint]) -> Tuple[int, str, str]:
    """Compute a multi-factor technical score and explanation."""
    if not data:
        return 50, "Insufficient data", "N/A"

    closes = [float(dp.close) for dp in data]
    highs = [float(dp.high) for dp in data]
    lows = [float(dp.low) for dp in data]
    volumes = [float(dp.volume or 0) for dp in data]
    if len(closes) < 20:
        return 50, "Insufficient data", "N/A"

    # 计算 TA-Lib/自研指标
    indicator_cache = _build_indicator_cache(closes, highs, lows)
    raw = _compute_raw_features(closes, highs, lows, volumes, len(closes) - 1, indicator_cache)

    last_close = closes[-1]
    sma_mid = raw.get("sma_mid") or 0
    ma_mid = raw.get("ma_mid") or 50
    sma_short = raw.get("sma_short") or sma_mid
    if sma_mid == 0:
        return 50, "Insufficient data", "N/A"

    price_vs_ma = (last_close / sma_mid) - 1
    slope = (sma_short / sma_mid) - 1 if sma_mid else 0
    trend_score = int(_clamp((price_vs_ma + slope) * 200 + 50, 0, 100))

    rsi = raw.get("rsi")
    ret_20 = raw.get("ret_20") or 0
    if rsi is None:
        mom_score = 50
    elif rsi < 30:
        mom_score = 85
    elif rsi > 70:
        mom_score = 35
    else:
        mom_score = 60
    mom_score = _clamp(mom_score + (ret_20 * 200), 0, 100)

    vol = raw.get("vol") or 0
    vol_score = _clamp(90 - (vol * 1000), 20, 90)

    vol_ratio = raw.get("vol_ratio") or 1
    vol_score_2 = _clamp(50 + (vol_ratio - 1) * 40, 10, 90)

    macd_hist = raw.get("macd")
    macd_ratio = (macd_hist / last_close) if macd_hist is not None and last_close else 0
    macd_score = _clamp(50 + macd_ratio * 1000, 0, 100)

    bb_pos = raw.get("bbands")
    if bb_pos is None:
        bb_score = 50
    else:
        bb_score = _clamp(50 + ((bb_pos - 0.5) * 100), 0, 100)

    adx = raw.get("adx")
    adx_score = _clamp(30 + (adx or 0), 0, 100)

    atr_pct = raw.get("atr_pct")
    atr_score = _clamp(90 - ((atr_pct or 0) * 500), 10, 90)

    stoch_k = raw.get("stoch")
    if stoch_k is None:
        stoch_score = 50
    elif stoch_k < 20:
        stoch_score = 80
    elif stoch_k > 80:
        stoch_score = 40
    else:
        stoch_score = 60

    # 使用滚动权重聚合得分
    final_score = int(
        (trend_score * 0.25)
        + (mom_score * 0.2)
        + (vol_score * 0.15)
        + (vol_score_2 * 0.1)
        + (macd_score * 0.1)
        + (bb_score * 0.08)
        + (adx_score * 0.05)
        + (atr_score * 0.04)
        + (stoch_score * 0.03)
    )

    trend_desc = f"Trend: price {'above' if last_close > sma_mid else 'below'} {int(ma_mid)}-day average"
    if rsi is None:
        mom_desc = "Momentum: RSI unavailable"
    elif rsi < 30:
        mom_desc = f"Momentum: RSI {int(rsi)} (oversold)"
    elif rsi > 70:
        mom_desc = f"Momentum: RSI {int(rsi)} (overbought)"
    else:
        mom_desc = f"Momentum: RSI {int(rsi)} (neutral)"

    vol_desc = f"Volatility: {vol * 100:.1f}% daily"
    volume_desc = f"Volume: {vol_ratio:.2f}x 20-day avg"
    extra_desc = []
    if macd_hist is not None:
        extra_desc.append(f"MACD hist {macd_hist:.3f}")
    if bb_pos is not None:
        extra_desc.append(f"Bollinger pos {bb_pos:.2f}")
    if adx is not None:
        extra_desc.append(f"ADX {adx:.1f}")
    if atr_pct is not None:
        extra_desc.append(f"ATR {atr_pct * 100:.2f}%")
    if stoch_k is not None:
        extra_desc.append(f"StochK {stoch_k:.1f}")
    description = "; ".join([trend_desc, mom_desc, vol_desc, volume_desc] + extra_desc)
    def _fmt(value: Optional[float], digits: int = 4) -> str:
        if value is None:
            return "N/A"
        return f"{value:.{digits}f}"

    formula = (
        f"SMA20/{int(ma_mid)}, RSI14, VolStd={vol:.4f}, VolRatio={vol_ratio:.2f}, "
        f"MACDHist={_fmt(macd_hist)} BBPos={_fmt(bb_pos, 2)} ADX={_fmt(adx, 2)} "
        f"ATR%={_fmt(atr_pct)} StochK={_fmt(stoch_k, 2)}"
    )

    return final_score, description, formula


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

        existing = list(
            EfsDataPoint.objects.filter(stock=stock)
            .order_by("date")
            .only("date", "close", "high", "low", "volume")
        )
        
        if not existing:
            rows = fetch_efs_data(stock.symbol, period="1y")
            if rows:
                upsert_efs_timeseries(stock, rows)
                existing = list(
                    EfsDataPoint.objects.filter(stock=stock)
                    .order_by("date")
                    .only("date", "close", "high", "low", "volume")
                )
        else:
            try:
                last_date = existing[-1].date
                rows = fetch_efs_data(stock.symbol, period="5d")
                rows = [row for row in rows if row.get("date") and row["date"] > last_date]
                if rows:
                    upsert_efs_timeseries(stock, rows)
                    existing = list(
                        EfsDataPoint.objects.filter(stock=stock)
                        .order_by("date")
                        .only("date", "close", "high", "low", "volume")
                    )
            except Exception as e:
                logger.debug(f"Error updating timeseries for {symbol}: {e}")

        if not existing:
            return 50, "Insufficient data", "N/A"

        try:
            factor_score = _calculate_factor_rank_score(stock, existing[-1].date)
            if factor_score:
                score, reason, formula = factor_score
            else:
                # 兜底：使用原始技术指标打分
                score, reason, formula = _calculate_score(existing)
            
            return score, reason, formula
        except Exception as e:
            logger.error(f"Error calculating score for {symbol}: {e}", exc_info=True)
            return 50, f"Error: {str(e)}", "N/A"
            
    except Exception as e:
        logger.error(f"Error in run_efs_analysis for {symbol}: {e}", exc_info=True)
        return 50, f"Error: {str(e)}", "N/A"

