"""
Qlib Alpha158 factor set: computable from OHLCV without qlib dependency.

Reference: https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py
Implements the full Alpha158 factor expressions as Python computations.
"""

import math
from statistics import pstdev
from typing import Dict, List, Optional, Tuple

# Rolling windows (qlib default)
ALPHA158_WINDOWS = [5, 10, 20, 30, 60]


def _safe_div(a: float, b: float, eps: float = 1e-12) -> Optional[float]:
    """Safe division; returns None if divisor is zero or too small."""
    if b is None or abs(b) < eps:
        return None
    return a / b


def _linear_regression_slope(x: List[float]) -> Optional[float]:
    """Slope of linear regression y = ax + b where x = [0,1,...,n-1], y = values."""
    n = len(x)
    if n < 2:
        return None
    x_mean = (n - 1) / 2
    y_mean = sum(x) / n
    num = sum((i - x_mean) * (x[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if den == 0:
        return None
    return num / den


def _linear_regression_rsquare(y: List[float]) -> Optional[float]:
    """R-squared of linear regression."""
    n = len(y)
    if n < 3:
        return None
    slope = _linear_regression_slope(y)
    if slope is None:
        return None
    x_mean = (n - 1) / 2
    y_mean = sum(y) / n
    y_hat = [y_mean + slope * (i - x_mean) for i in range(n)]
    ss_res = sum((y[i] - y_hat[i]) ** 2 for i in range(n))
    ss_tot = sum((y[i] - y_mean) ** 2 for i in range(n))
    if ss_tot == 0:
        return None
    return 1 - (ss_res / ss_tot)


def _linear_regression_residual(y: List[float]) -> Optional[float]:
    """Residual at last point: y[-1] - y_hat[-1], normalized."""
    n = len(y)
    if n < 2 or y[-1] == 0:
        return None
    slope = _linear_regression_slope(y)
    if slope is None:
        return None
    x_mean = (n - 1) / 2
    y_mean = sum(y) / n
    y_hat_last = y_mean + slope * ((n - 1) - x_mean)
    return (y[-1] - y_hat_last) / y[-1]


def _quantile(values: List[float], q: float) -> Optional[float]:
    """Quantile at q (0-1)."""
    if not values:
        return None
    sorted_v = sorted(values)
    idx = (len(sorted_v) - 1) * q
    i, f = int(idx), idx - int(idx)
    if i + 1 >= len(sorted_v):
        return sorted_v[-1]
    return sorted_v[i] * (1 - f) + sorted_v[i + 1] * f


def _rank(value: float, values: List[float]) -> Optional[float]:
    """Percentile rank of value in values (0-1)."""
    if not values:
        return None
    return sum(1 for v in values if v <= value) / len(values)


def _correlation(x: List[float], y: List[float]) -> Optional[float]:
    """Pearson correlation."""
    n = len(x)
    if n != len(y) or n < 2:
        return None
    mx, my = sum(x) / n, sum(y) / n
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    den_x = sum((x[i] - mx) ** 2 for i in range(n)) ** 0.5
    den_y = sum((y[i] - my) ** 2 for i in range(n)) ** 0.5
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def compute_alpha158_features(
    opens: List[float],
    highs: List[float],
    lows: List[float],
    closes: List[float],
    volumes: List[float],
    idx: int,
) -> Dict[str, Optional[float]]:
    """
    Compute Alpha158-style features at index idx.
    All expressions normalized by $close for unit-free values.
    """
    out: Dict[str, Optional[float]] = {}
    o = opens[idx] if idx < len(opens) else 0
    h = highs[idx] if idx < len(highs) else 0
    l_ = lows[idx] if idx < len(lows) else 0
    c = closes[idx] if idx < len(closes) else 0
    v = volumes[idx] if idx < len(volumes) else 0
    if c <= 0:
        return out

    # --- kbar (9 factors) ---
    hl = h - l_
    oc = c - o
    out["KMID"] = _safe_div(oc, o) if o else None
    out["KLEN"] = _safe_div(hl, o) if o else None
    out["KMID2"] = _safe_div(oc, hl) if hl and abs(hl) > 1e-12 else None
    up = h - max(o, c)
    out["KUP"] = _safe_div(up, o) if o else None
    out["KUP2"] = _safe_div(up, hl) if hl and abs(hl) > 1e-12 else None
    low_body = min(o, c) - l_
    out["KLOW"] = _safe_div(low_body, o) if o else None
    out["KLOW2"] = _safe_div(low_body, hl) if hl and abs(hl) > 1e-12 else None
    body2 = 2 * c - h - l_
    out["KSFT"] = _safe_div(body2, o) if o else None
    out["KSFT2"] = _safe_div(body2, hl) if hl and abs(hl) > 1e-12 else None

    # --- price: windows [0,1,2,3,4], features [OPEN, HIGH, LOW, VWAP] ---
    for d in range(5):
        if idx < d:
            continue
        ref_idx = idx - d
        ref_o = opens[ref_idx] if ref_idx < len(opens) else c
        ref_h = highs[ref_idx] if ref_idx < len(highs) else c
        ref_l = lows[ref_idx] if ref_idx < len(lows) else c
        ref_c = closes[ref_idx] if ref_idx < len(closes) else c
        ref_vwap = (ref_h + ref_l + ref_c) / 3 if (ref_h + ref_l + ref_c) else ref_c
        out[f"OPEN{d}"] = ref_o / c if c else None
        out[f"HIGH{d}"] = ref_h / c if c else None
        out[f"LOW{d}"] = ref_l / c if c else None
        out[f"VWAP{d}"] = ref_vwap / c if c else None

    # --- volume: windows [0,1,2,3,4] ---
    for d in range(5):
        if idx < d:
            continue
        ref_idx = idx - d
        ref_v = float(volumes[ref_idx] or 0) if ref_idx < len(volumes) else 0
        out[f"VOLUME{d}"] = ref_v / (v + 1e-12) if v is not None else None

    # --- ret_20, forward_ret_20 (for evaluation) ---
    if idx >= 20 and closes[idx - 20] and closes[idx - 20] > 0:
        out["ret_20"] = (c / closes[idx - 20]) - 1
    else:
        out["ret_20"] = None
    if idx + 20 < len(closes) and closes[idx + 20] and c > 0:
        out["forward_ret_20"] = (closes[idx + 20] / c) - 1
    else:
        out["forward_ret_20"] = None

    # --- rolling factors ---
    for d in ALPHA158_WINDOWS:
        if idx < d:
            continue
        win_closes = closes[idx - d : idx + 1]
        win_highs = highs[idx - d : idx + 1]
        win_lows = lows[idx - d : idx + 1]
        win_vols = [
            float(volumes[i] or 0) if i < len(volumes) else 0.0
            for i in range(idx - d, idx + 1)
        ]
        if not win_closes or win_closes[-1] <= 0:
            continue

        # ROC, MA, STD
        out[f"ROC{d}"] = win_closes[0] / c if c else None
        ma = sum(win_closes) / len(win_closes)
        out[f"MA{d}"] = ma / c if c else None
        out[f"STD{d}"] = (pstdev(win_closes) / c) if len(win_closes) >= 2 and c else None

        # BETA, RSQR, RESI
        slope = _linear_regression_slope(win_closes)
        out[f"BETA{d}"] = slope / c if slope is not None and c else None
        out[f"RSQR{d}"] = _linear_regression_rsquare(win_closes)
        out[f"RESI{d}"] = _linear_regression_residual(win_closes)

        # MAX, MIN
        out[f"MAX{d}"] = max(win_highs) / c if c else None
        out[f"MIN{d}"] = min(win_lows) / c if c else None

        # QTLU, QTLD
        out[f"QTLU{d}"] = _safe_div(_quantile(win_closes, 0.8), c)
        out[f"QTLD{d}"] = _safe_div(_quantile(win_closes, 0.2), c)

        # RANK, RSV
        out[f"RANK{d}"] = _rank(c, win_closes)
        mn, mx = min(win_lows), max(win_highs)
        denom = mx - mn
        out[f"RSV{d}"] = (c - mn) / (denom + 1e-12) if denom and abs(denom) > 1e-12 else None

        # IMAX, IMIN, IMXD
        imax_idx = max(range(len(win_highs)), key=lambda i: win_highs[i])
        imin_idx = min(range(len(win_lows)), key=lambda i: win_lows[i])
        out[f"IMAX{d}"] = (d - imax_idx) / d if d else None
        out[f"IMIN{d}"] = (d - imin_idx) / d if d else None
        out[f"IMXD{d}"] = (imax_idx - imin_idx) / d if d else None

        # CNTP, CNTN, CNTD
        up_days = sum(1 for i in range(1, len(win_closes)) if win_closes[i] > win_closes[i - 1])
        down_days = sum(1 for i in range(1, len(win_closes)) if win_closes[i] < win_closes[i - 1])
        n_changes = len(win_closes) - 1
        out[f"CNTP{d}"] = up_days / n_changes if n_changes else None
        out[f"CNTN{d}"] = down_days / n_changes if n_changes else None
        out[f"CNTD{d}"] = (up_days - down_days) / n_changes if n_changes else None

        # SUMP, SUMN, SUMD
        gains = [max(win_closes[i] - win_closes[i - 1], 0) for i in range(1, len(win_closes))]
        losses = [max(win_closes[i - 1] - win_closes[i], 0) for i in range(1, len(win_closes))]
        total_abs = sum(abs(win_closes[i] - win_closes[i - 1]) for i in range(1, len(win_closes))) + 1e-12
        total_gain, total_loss = sum(gains), sum(losses)
        out[f"SUMP{d}"] = total_gain / total_abs if total_abs else None
        out[f"SUMN{d}"] = total_loss / total_abs if total_abs else None
        out[f"SUMD{d}"] = (total_gain - total_loss) / total_abs if total_abs else None

        # VMA, VSTD
        avg_vol = sum(win_vols) / len(win_vols) if win_vols else 0
        out[f"VMA{d}"] = avg_vol / (v + 1e-12) if v is not None else None
        out[f"VSTD{d}"] = (pstdev(win_vols) / (v + 1e-12)) if len(win_vols) >= 2 and v else None

        # CORR (close vs log(volume+1))
        log_vols = [math.log(x + 1) for x in win_vols]
        out[f"CORR{d}"] = _correlation(win_closes, log_vols)

        # CORD (price change ratio vs volume change ratio)
        if len(win_closes) >= 2 and len(win_vols) >= 2:
            price_chg = [
                (win_closes[i] / win_closes[i - 1] - 1) if win_closes[i - 1] else 0
                for i in range(1, len(win_closes))
            ]
            vol_chg = [
                math.log(win_vols[i] / (win_vols[i - 1] or 1) + 1)
                for i in range(1, len(win_vols))
            ]
            out[f"CORD{d}"] = _correlation(price_chg, vol_chg)
        else:
            out[f"CORD{d}"] = None

        # WVMA, VSUMP, VSUMN, VSUMD
        abs_ret_vol = [
            abs(win_closes[i] / (win_closes[i - 1] or 1) - 1) * (win_vols[i] or 0)
            for i in range(1, len(win_closes))
        ]
        mean_arv = sum(abs_ret_vol) / len(abs_ret_vol) + 1e-12 if abs_ret_vol else 1e-12
        std_arv = pstdev(abs_ret_vol) if len(abs_ret_vol) >= 2 else 0
        out[f"WVMA{d}"] = std_arv / mean_arv if mean_arv else None
        v_gains = [max((win_vols[i] or 0) - (win_vols[i - 1] or 0), 0) for i in range(1, len(win_vols))]
        v_losses = [max((win_vols[i - 1] or 0) - (win_vols[i] or 0), 0) for i in range(1, len(win_vols))]
        v_total = sum(abs((win_vols[i] or 0) - (win_vols[i - 1] or 0)) for i in range(1, len(win_vols))) + 1e-12
        out[f"VSUMP{d}"] = sum(v_gains) / v_total if v_total else None
        out[f"VSUMN{d}"] = sum(v_losses) / v_total if v_total else None
        out[f"VSUMD{d}"] = (sum(v_gains) - sum(v_losses)) / v_total if v_total else None

    return out


def _build_alpha158_feature_names() -> List[str]:
    """Build ordered list of all Alpha158 feature names (excluding ret_20, forward_ret_20)."""
    names = [
        "KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2",
    ]
    for d in range(5):
        names.extend([f"OPEN{d}", f"HIGH{d}", f"LOW{d}", f"VWAP{d}", f"VOLUME{d}"])
    for d in ALPHA158_WINDOWS:
        names.extend([
            f"ROC{d}", f"MA{d}", f"STD{d}", f"BETA{d}", f"RSQR{d}", f"RESI{d}",
            f"MAX{d}", f"MIN{d}", f"QTLU{d}", f"QTLD{d}", f"RANK{d}", f"RSV{d}",
            f"IMAX{d}", f"IMIN{d}", f"IMXD{d}", f"CORR{d}", f"CORD{d}",
            f"CNTP{d}", f"CNTN{d}", f"CNTD{d}", f"SUMP{d}", f"SUMN{d}", f"SUMD{d}",
            f"VMA{d}", f"VSTD{d}", f"WVMA{d}", f"VSUMP{d}", f"VSUMN{d}", f"VSUMD{d}",
        ])
    return names


def _build_alpha158_init_factors() -> List[Tuple[str, str]]:
    """Build full Alpha158 base factors: (name, expression) for EFS seed."""
    factors: List[Tuple[str, str]] = []
    # kbar
    for name in ["KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2"]:
        factors.append((name, name))
    # price
    for d in range(5):
        for f in ["OPEN", "HIGH", "LOW", "VWAP"]:
            factors.append((f"{f}{d}", f"{f}{d}"))
    # volume
    for d in range(5):
        factors.append((f"VOLUME{d}", f"VOLUME{d}"))
    # rolling
    for d in ALPHA158_WINDOWS:
        for op in ["ROC", "MA", "STD", "BETA", "RSQR", "RESI", "MAX", "MIN",
                   "QTLU", "QTLD", "RANK", "RSV", "IMAX", "IMIN", "IMXD",
                   "CORR", "CORD", "CNTP", "CNTN", "CNTD", "SUMP", "SUMN", "SUMD",
                   "VMA", "VSTD", "WVMA", "VSUMP", "VSUMN", "VSUMD"]:
            factors.append((f"{op}{d}", f"{op}{d}"))
    return factors


ALPHA158_FEATURE_NAMES = _build_alpha158_feature_names()
ALPHA158_INIT_FACTORS = _build_alpha158_init_factors()
