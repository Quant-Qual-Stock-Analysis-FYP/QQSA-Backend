"""AI 分析主流程：融合 EFS、RAG 与大模型输出。"""

import json
import logging
from statistics import pstdev
from typing import Any, Dict, List, Optional, Tuple

from markets.models import Stock
from ..config.constants import (
    ETF_SYMBOLS,
    EQUITY_ETFS,
    BOND_ETFS,
    CASH_ETFS,
    COMMODITY_ETFS,
    GROWTH_ETF,
    VALUE_ETF,
    BENCHMARK_ETF,
    LONG_BOND_ETF,
    MID_BOND_ETF,
    SHORT_BOND_ETF,
)
from ..models import AnalysisResult, EfsAlphaFactor, EfsDataPoint
from .ai import call_deepseek
from .efs import calculate_risk_score, run_efs_analysis
from .rag import get_rag_fundamental, get_rag_sentiment
from .utils import (
    coerce_score,
    coerce_tags,
    ensure_word_range,
    is_etf_symbol,
    is_low_confidence_reason,
    parse_ai_json,
    shrink_words,
    truncate_chars,
)

logger = logging.getLogger(__name__)


def _calculate_rsi(closes: List[float], period: int = 14) -> Optional[float]:
    # 轻量 RSI 计算（用于标签与波动性评估）
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


def _calculate_volatility(closes: List[float], window: int = 60) -> float:
    # 近似年化波动（%）
    if len(closes) < 2:
        return 0.0
    window = min(window, len(closes))
    subset = closes[-window:]
    daily_returns = [
        (subset[i] / subset[i - 1]) - 1 for i in range(1, len(subset)) if subset[i - 1] != 0
    ]
    if len(daily_returns) < 2:
        return 0.0
    return float(pstdev(daily_returns)) * 100


def _rag_score(score: int) -> int:
    # RAG sentiment now returns numeric score (0-100) directly
    # This function is kept for backward compatibility but just returns the score
    return max(0, min(100, score))


def _fundamental_score(score: int) -> int:
    # Fundamental analysis now returns numeric score (0-100) directly
    # This function is kept for backward compatibility but just returns the score
    return max(0, min(100, score))


def _score_to_sentiment_label(score: int) -> str:
    """Convert numeric score (0-100) to sentiment label for backward compatibility."""
    if score >= 70:
        return "Positive"
    elif score <= 30:
        return "Negative"
    else:
        return "Neutral"


def _signal_from_score(overall_score: int) -> str:
    # 总分 -> 行动信号
    if overall_score >= 80:
        return "Strong Buy"
    if overall_score >= 60:
        return "Buy"
    if overall_score >= 40:
        return "Hold"
    return "Sell"


def _shorten_reason(text: str) -> str:
    """缩短原因文本到第一句"""
    if not text:
        return ""
    parts = text.split(".")
    return (parts[0].strip() + ".") if parts else text


def _fallback_fundamental_reason(score: int) -> str:
    return (
        f"Fundamental view is neutral at {score}/100 because verified fundamentals are limited in "
        "the current dataset. Without clear visibility into margins, cash flow quality, or balance "
        "sheet strength, the assessment leans on business scale and sector position rather than "
        "hard financial evidence. That keeps the score anchored near neutral rather than high or "
        "low. To move higher, the company would need clearer proof of durable earnings power, "
        "pricing strength, and disciplined capital allocation. Until then, fundamentals support "
        "patience but do not justify aggressive accumulation."
    )


def _fallback_technical_reason(score: int, detail: str) -> str:
    trend_label = "constructive" if score >= 60 else "mixed"
    return (
        "Technical outlook uses the EFS trend and momentum signals. "
        f"The technical score of {score}/100 suggests price action is {trend_label}, "
        "so the chart signal is not fully one-sided. Recent volatility and volume are embedded, "
        "which means the score can change quickly if participation fades or a reversal appears. "
        "Overall, the setup is workable but not decisive, so sizing should respect nearby support "
        "and resistance and avoid chasing extended moves. "
        f"Signal details: {detail}."
    )


def _fallback_sentiment_reason(score: int, sentiment: str, reason: str) -> str:
    return (
        f"Sentiment score of {score}/100 reflects the latest news tone, currently {sentiment}. "
        f"The key headlines point to {reason}. This suggests the narrative is leaning that "
        "direction in the near term, which can influence short-term price action. If the tone is "
        "positive, momentum may persist; if neutral, catalysts may be muted; if negative, the stock "
        "could face hesitancy around upcoming events. Treat sentiment as a tactical tailwind or "
        "headwind rather than a standalone thesis."
    )


def _fallback_risk_reason(score: int, detail: str) -> str:
    return (
        f"Safety score of {score}/100 reflects recent volatility and drawdown behavior ({detail}). "
        "Higher scores mean a steadier return path and lower observed turbulence, while lower "
        "scores signal sharper swings or deeper pullbacks. This snapshot helps with position sizing "
        "and risk budgeting: lower safety suggests smaller sizing or tighter stops, while higher "
        "safety can support a more patient hold. Treat it as a stability gauge, not a guarantee."
    )


def _fallback_summary(
    fundamental_score: int,
    technical_score: int,
    sentiment_score: int,
    risk_score: int,
    overall_score: int,
    signal: str,
    strategy: str,
) -> str:
    return (
        "Summary: The analysis blends fundamental, technical, sentiment, and safety angles into an "
        "overall score. "
        f"The fundamental score is {fundamental_score}/100, reflecting current visibility into "
        "business quality and durability. "
        f"The technical score of {technical_score}/100 reflects current trend and momentum, which "
        "may or may not be aligned with the broader view. "
        f"Sentiment sits at {sentiment_score}/100 based on recent news tone, shaping short-term "
        "demand. "
        f"Safety is {risk_score}/100, indicating recent volatility and drawdown behavior that "
        "should influence position sizing. "
        f"After applying the market regime bias, the overall score is {overall_score}/100 with a "
        f"{signal} signal. The suggested strategy is {strategy}. "
        + (
            "Despite softer technicals, the signal leans positive because fundamentals or sentiment "
            "carry more weight in the current mix, so the trade-off should be made explicit in "
            "position sizing. "
            if technical_score < 60 and signal in {"Buy", "Strong Buy"}
            else ""
        )
        + "Reassess after major earnings, guidance changes, or macro shocks, as these can quickly "
        "shift both sentiment and safety."
    )


def _build_tags(
    technical_score: int,
    sentiment_score: int,
    risk_score: int,
    fundamental_score: int,
    volatility: float,
    rsi: Optional[float],
    price_above_ma50: Optional[bool],
    rag_reason: str,
    sector: str,
    signal: str,
    strategy: str,
) -> List[str]:
    # 规则标签（英文短标签，便于前端展示）
    tags: List[str] = []

    def add(label: str) -> None:
        if label and label not in tags:
            tags.append(label)

    # Risk & Profile
    if risk_score > 80:
        add("Defensive")
    if risk_score < 40 or volatility > 3:
        add("High Volatility")
    if fundamental_score > 80 and sector.lower() in {"consumer", "consumer staples", "utilities"}:
        add("Cash Cow")
    if risk_score < 30 and technical_score > 70:
        add("Speculative")
    if volatility < 1.5:
        add("Low Volatility")

    # Sentiment & News
    if sentiment_score > 85:
        add("Market Hotspot")
    elif 60 <= sentiment_score <= 85:
        add("Positive News")
    if sentiment_score < 30:
        add("Sentiment Low")
    lowered_reason = rag_reason.lower() if rag_reason else ""
    if sentiment_score < 40 and any(k in lowered_reason for k in ["scandal", "lawsuit", "fraud", "investigation"]):
        add("News Alert")

    # Technical & Momentum
    if price_above_ma50 and technical_score > 60:
        add("Uptrend")
    if rsi is not None and rsi > 75:
        add("Overbought")
    if rsi is not None and rsi < 30 and technical_score >= 50:
        add("Rebound")
    if price_above_ma50 and technical_score < 50:
        add("Pullback")
    if 40 <= technical_score <= 60 and volatility < 1.5:
        add("Consolidation")

    # Fundamental & Valuation
    if fundamental_score > 80:
        add("Sector Leader")
    if fundamental_score > 70 and any(k in lowered_reason for k in ["low pe", "undervalued", "cheap valuation"]):
        add("Undervalued")
    if any(k in lowered_reason for k in ["high pe", "overpriced", "overvalued"]):
        add("Overvalued")
    if fundamental_score == 50:
        add("Data Limited")
    if any(k in lowered_reason for k in ["revenue growth", "growth >20%", "growth above 20%"]):
        add("High Growth")

    # Strategy Action
    strategy_lower = strategy.lower() if strategy else ""
    if signal in {"Buy", "Strong Buy"} and "long" in strategy_lower:
        add("Long-term Hold")
    if "short" in strategy_lower and technical_score > 70:
        add("Short-term Play")
    if signal == "Hold":
        add("Wait & See")
    if signal in {"Buy", "Strong Buy"} and rsi is not None and rsi > 70:
        add("DCA Strategy")
    if signal == "Sell":
        add("Take Profit")

    return tags


def _build_one_liner(
    symbol: str,
    technical_score: int,
    sentiment_score: int,
    risk_score: int,
    fundamental_score: int,
    signal: str,
) -> str:
    parts = [
        f"{symbol.upper()} is rated {signal}",
        f"fundamentals {fundamental_score}/100",
        f"technicals {technical_score}/100",
        f"sentiment {sentiment_score}/100",
        f"safety {risk_score}/100",
    ]
    return ", ".join(parts) + "."


def _get_etf_scores_and_prices() -> Dict[str, Dict[str, Any]]:
    """獲取所有 ETF 的技術分數和價格數據"""
    etf_data = {}
    
    for symbol in ETF_SYMBOLS:
        stock = Stock.objects.filter(symbol=symbol).first()
        if not stock:
            continue
        
        try:
            # 獲取技術分數
            efs_score, _, _ = run_efs_analysis(symbol)
            
            # 獲取價格數據（用於計算相對強弱）
            data_points = list(
                EfsDataPoint.objects.filter(stock=stock)
                .order_by("-date")[:252]  # 約1年的數據
            )
            
            if not data_points:
                continue
            
            closes = [float(dp.close) for dp in data_points]
            current_price = closes[0] if closes else None
            
            # 計算相對價格變化（相對於基準）
            price_change_20d = None
            price_change_60d = None
            if len(closes) >= 20:
                price_change_20d = ((closes[0] / closes[min(19, len(closes)-1)]) - 1) * 100
            if len(closes) >= 60:
                price_change_60d = ((closes[0] / closes[min(59, len(closes)-1)]) - 1) * 100
            
            etf_data[symbol] = {
                "score": efs_score,
                "price": current_price,
                "closes": closes,  # 保存完整價格歷史用於計算相對均線
                "change_20d": price_change_20d,
                "change_60d": price_change_60d,
                "data_points": len(data_points),
            }
        except Exception as e:
            logger.debug(f"Error processing ETF {symbol}: {e}")
            continue
    
    return etf_data


def _calculate_relative_strength(etf_data: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """
    計算相對強弱比率（使用相對均線方法）
    
    改進：不再使用絕對比率（Price_A / Price_B），而是使用相對均線
    - 計算當前比率 vs 20日均線比率
    - 如果當前比率 > 20日均線 → 強勢
    - 如果當前比率 < 20日均線 → 弱勢
    """
    ratios = {}
    ratio_ma20 = {}  # 存儲20日均線比率，用於判斷相對強弱
    
    # QQQ/SPY: 成長 vs 基準
    if GROWTH_ETF in etf_data and BENCHMARK_ETF in etf_data:
        qqq_closes = etf_data[GROWTH_ETF].get("closes", [])
        spy_closes = etf_data[BENCHMARK_ETF].get("closes", [])
        if qqq_closes and spy_closes and len(qqq_closes) >= 20 and len(spy_closes) >= 20:
            # 計算歷史比率序列（取較短的長度，確保對齊）
            ratio_history = []
            min_len = min(len(qqq_closes), len(spy_closes))
            for i in range(min_len):
                if spy_closes[i] > 0:
                    ratio_history.append(qqq_closes[i] / spy_closes[i])
            
            if ratio_history and len(ratio_history) >= 20:
                current_ratio = ratio_history[0]
                # 計算20日均線
                ma20_ratio = sum(ratio_history[:20]) / 20
                ratios["qqq_spy"] = current_ratio
                ratio_ma20["qqq_spy"] = ma20_ratio
    
    # SPY/TLT: 股票 vs 長期債券
    if BENCHMARK_ETF in etf_data and LONG_BOND_ETF in etf_data:
        spy_closes = etf_data[BENCHMARK_ETF].get("closes", [])
        tlt_closes = etf_data[LONG_BOND_ETF].get("closes", [])
        if spy_closes and tlt_closes and len(spy_closes) >= 20 and len(tlt_closes) >= 20:
            ratio_history = []
            min_len = min(len(spy_closes), len(tlt_closes))
            for i in range(min_len):
                if tlt_closes[i] > 0:
                    ratio_history.append(spy_closes[i] / tlt_closes[i])
            
            if ratio_history and len(ratio_history) >= 20:
                current_ratio = ratio_history[0]
                ma20_ratio = sum(ratio_history[:20]) / 20
                ratios["spy_tlt"] = current_ratio
                ratio_ma20["spy_tlt"] = ma20_ratio
    
    # DIA/QQQ: 價值 vs 成長
    if VALUE_ETF in etf_data and GROWTH_ETF in etf_data:
        dia_closes = etf_data[VALUE_ETF].get("closes", [])
        qqq_closes = etf_data[GROWTH_ETF].get("closes", [])
        if dia_closes and qqq_closes and len(dia_closes) >= 20 and len(qqq_closes) >= 20:
            ratio_history = []
            min_len = min(len(dia_closes), len(qqq_closes))
            for i in range(min_len):
                if qqq_closes[i] > 0:
                    ratio_history.append(dia_closes[i] / qqq_closes[i])
            
            if ratio_history and len(ratio_history) >= 20:
                current_ratio = ratio_history[0]
                ma20_ratio = sum(ratio_history[:20]) / 20
                ratios["dia_qqq"] = current_ratio
                ratio_ma20["dia_qqq"] = ma20_ratio
    
    # TLT/BIL: 長期債券 vs 現金
    if LONG_BOND_ETF in etf_data and SHORT_BOND_ETF in etf_data:
        tlt_closes = etf_data[LONG_BOND_ETF].get("closes", [])
        bil_closes = etf_data[SHORT_BOND_ETF].get("closes", [])
        if tlt_closes and bil_closes and len(tlt_closes) >= 20 and len(bil_closes) >= 20:
            ratio_history = []
            min_len = min(len(tlt_closes), len(bil_closes))
            for i in range(min_len):
                if bil_closes[i] > 0:
                    ratio_history.append(tlt_closes[i] / bil_closes[i])
            
            if ratio_history and len(ratio_history) >= 20:
                current_ratio = ratio_history[0]
                ma20_ratio = sum(ratio_history[:20]) / 20
                ratios["tlt_bil"] = current_ratio
                ratio_ma20["tlt_bil"] = ma20_ratio
    
    # SPY/BIL: 股票 vs 現金
    if BENCHMARK_ETF in etf_data and SHORT_BOND_ETF in etf_data:
        spy_closes = etf_data[BENCHMARK_ETF].get("closes", [])
        bil_closes = etf_data[SHORT_BOND_ETF].get("closes", [])
        if spy_closes and bil_closes and len(spy_closes) >= 20 and len(bil_closes) >= 20:
            ratio_history = []
            min_len = min(len(spy_closes), len(bil_closes))
            for i in range(min_len):
                if bil_closes[i] > 0:
                    ratio_history.append(spy_closes[i] / bil_closes[i])
            
            if ratio_history and len(ratio_history) >= 20:
                current_ratio = ratio_history[0]
                ma20_ratio = sum(ratio_history[:20]) / 20
                ratios["spy_bil"] = current_ratio
                ratio_ma20["spy_bil"] = ma20_ratio
    
    # 將均線數據附加到 ratios 字典中（用於後續判斷）
    ratios["_ma20"] = ratio_ma20
    
    return ratios


def build_market_context() -> Dict[str, Any]:
    """
    使用跨市場分析 (Intermarket Analysis) 構建市場週期信號
    
    分析邏輯：
    - Bull Market: QQQ > SPY, SPY > TLT, 股票 > 債券
    - Bear Market: TLT > SPY, DIA > QQQ, 債券 > 股票
    - Base/Rotation: 橫盤整理，類股輪動
    """
    # 獲取所有 ETF 的技術分數和價格數據
    etf_data = _get_etf_scores_and_prices()
    
    if not etf_data:
        return {
            "market_cycle": "Base",
            "market_score": 50,
            "market_bias": 0,
            "market_reason": ["ETF 數據不足"],
        }
    
    # 計算相對強弱比率
    ratios = _calculate_relative_strength(etf_data)
    
    # 獲取情感分析數據
    sentiments: List[str] = []
    news_reasons: List[str] = []  # 純新聞原因
    technical_reasons: List[str] = []  # 技術分析原因
    any_news = False
    
    for symbol in ETF_SYMBOLS:
        if symbol not in etf_data:
            continue
        try:
            rag_score, rag_reason, _ = get_rag_sentiment(symbol)
            sentiment_label = _score_to_sentiment_label(rag_score)
            sentiments.append(sentiment_label)
            if rag_reason and "no news" not in rag_reason.lower():
                news_reasons.append(_shorten_reason(rag_reason))
                any_news = True
        except Exception as e:
            logger.debug(f"Error getting sentiment for {symbol}: {e}")
            continue
    
    # 計算各類別的平均分數
    equity_scores = [
        etf_data[symbol]["score"]
        for symbol in EQUITY_ETFS
        if symbol in etf_data
    ]
    bond_scores = [
        etf_data[symbol]["score"]
        for symbol in BOND_ETFS
        if symbol in etf_data
    ]
    equity_avg = int(sum(equity_scores) / len(equity_scores)) if equity_scores else 50
    bond_avg = int(sum(bond_scores) / len(bond_scores)) if bond_scores else 50
    
    # 獲取關鍵 ETF 的分數
    spy_score = etf_data.get(BENCHMARK_ETF, {}).get("score", 50)
    qqq_score = etf_data.get(GROWTH_ETF, {}).get("score", 50)
    tlt_score = etf_data.get(LONG_BOND_ETF, {}).get("score", 50)
    
    # 計算市場總分（加權平均：股票權重更高）
    market_score = int((equity_avg * 0.7 + bond_avg * 0.3))
    
    # 判斷市場週期和偏差（引入緩衝區和平滑過渡）
    market_cycle = "Base"
    market_bias = 0
    
    # 獲取相對均線數據（如果可用）
    ratio_ma20 = ratios.get("_ma20", {})
    
    # 使用相對均線判斷比率強弱（如果可用）
    qqq_spy_ma20 = ratio_ma20.get("qqq_spy")
    spy_tlt_ma20 = ratio_ma20.get("spy_tlt")
    dia_qqq_ma20 = ratio_ma20.get("dia_qqq")
    
    qqq_spy_current = ratios.get("qqq_spy")
    spy_tlt_current = ratios.get("spy_tlt")
    dia_qqq_current = ratios.get("dia_qqq")
    
    # 判斷 QQQ/SPY 是否強勢（使用相對均線或絕對值）
    qqq_strong = False
    if qqq_spy_ma20 and qqq_spy_current:
        qqq_strong = qqq_spy_current > qqq_spy_ma20 * 1.02  # 當前比率 > 均線 2%
    elif qqq_spy_current:
        qqq_strong = qqq_spy_current > 1.02
    
    # 判斷 SPY/TLT 是否強勢
    spy_strong_vs_tlt = False
    if spy_tlt_ma20 and spy_tlt_current:
        spy_strong_vs_tlt = spy_tlt_current > spy_tlt_ma20 * 1.05  # 當前比率 > 均線 5%
    elif spy_tlt_current:
        spy_strong_vs_tlt = spy_tlt_current > 1.05
    
    # 判斷 DIA/QQQ（價值 vs 成長）
    dia_strong_vs_qqq = False
    if dia_qqq_ma20 and dia_qqq_current:
        dia_strong_vs_qqq = dia_qqq_current > dia_qqq_ma20 * 1.05
    elif dia_qqq_current:
        dia_strong_vs_qqq = dia_qqq_current > 1.05
    
    # 計算股債差距（用於緩衝區判定）
    gap = equity_avg - bond_avg
    gap_threshold = 5  # 緩衝區閾值：5 分
    
    # 1. 檢查是否為牛市 (Bull Market)
    # 條件：股票平均分數高，且有以下任一信號
    if equity_avg >= 60 and spy_score >= 55:
        # QQQ 領漲 SPY（成長領跑）- 強牛市信號
        if qqq_strong or qqq_score > spy_score + 8:
            market_cycle = "Bull"
            market_bias = 8
            technical_reasons.append(f"牛市信號：成長股領漲（QQQ {qqq_score} vs SPY {spy_score}），風險偏好強勁")
        # SPY 明顯強於 TLT（風險偏好）- 牛市信號
        elif spy_strong_vs_tlt or equity_avg > bond_avg + 15:
            market_cycle = "Bull"
            market_bias = 8
            if gap >= gap_threshold:
                technical_reasons.append(f"牛市信號：股票顯著強於債券（股票 {equity_avg} vs 債券 {bond_avg}），風險偏好上升")
            else:
                technical_reasons.append(f"牛市信號：股票強於債券（股票 {equity_avg} vs 債券 {bond_avg}），風險偏好上升")
        # 股票整體強勢但無明顯領跑者
        elif equity_avg >= 65:
            market_cycle = "Bull"
            market_bias = 5
            technical_reasons.append(f"偏多信號：股票整體強勢（平均分數 {equity_avg}），市場情緒積極")
    
    # 2. 檢查是否為熊市 (Bear Market)
    # 條件：股票平均分數低，或有以下避險信號
    elif equity_avg <= 45 or spy_score <= 45:
        # TLT 明顯強於 SPY（避險情緒）- 強熊市信號
        if (spy_tlt_ma20 and spy_tlt_current and spy_tlt_current < spy_tlt_ma20 * 0.92) or \
           (spy_tlt_current and spy_tlt_current < 0.92) or \
           tlt_score > spy_score + 15:
            market_cycle = "Bear"
            market_bias = -8
            if gap <= -gap_threshold:
                technical_reasons.append(f"熊市信號：債券顯著強於股票（TLT {tlt_score} vs SPY {spy_score}），避險情緒濃厚")
            else:
                technical_reasons.append(f"熊市信號：債券強於股票（TLT {tlt_score} vs SPY {spy_score}），避險情緒強烈")
        # DIA 跑贏 QQQ（防禦性輪動）- 熊市信號
        elif dia_strong_vs_qqq or (
            VALUE_ETF in etf_data and GROWTH_ETF in etf_data and
            etf_data[VALUE_ETF]["score"] > etf_data[GROWTH_ETF]["score"] + 10
        ):
            market_cycle = "Bear"
            market_bias = -8
            technical_reasons.append(f"熊市信號：價值股跑贏成長股（DIA {etf_data.get(VALUE_ETF, {}).get('score', 50)} vs QQQ {qqq_score}），防禦性輪動")
        # 股票整體弱勢
        elif equity_avg <= 40:
            market_cycle = "Bear"
            market_bias = -5
            technical_reasons.append(f"偏空信號：股票整體弱勢（平均分數 {equity_avg}），市場情緒謹慎")
    
    # 3. 檢查是否為築底/輪動 (Base/Rotation)
    # 條件：市場分數在中等範圍（45 < equity_avg < 60），需要更細緻的判斷
    else:
        market_cycle = "Base"
        # 引入緩衝區和平滑過渡（僅在 Base 週期內）
        # Base 週期的 equity_avg 範圍是 45-60（不包括邊界，因為邊界屬於 Bull/Bear）
        if equity_avg >= 55:  # 55-60 分：偏多緩衝區
            market_bias = 4
        elif equity_avg <= 45:  # 40-45 分：偏空緩衝區（理論上不會進入這裡，因為 <=45 會觸發 Bear）
            market_bias = -4
        else:  # 45-55 分：中性
            market_bias = 0
        
        # 檢查相對強弱來判斷趨勢
        spy_vs_tlt = ratios.get("spy_tlt", 1.0)
        qqq_vs_spy = ratios.get("qqq_spy", 1.0)
        
        # 如果股票略強於債券，但分數不高，可能是築底
        if spy_vs_tlt > 1.0 and equity_avg > bond_avg:
            if abs(gap) < gap_threshold:
                # 差距小於 5 分，視為「膠著」或「平衡」
                if abs(qqq_score - spy_score) > 8:
                    technical_reasons.append(f"市場橫盤整理：股債評分接近（{equity_avg} vs {bond_avg}），類股輪動明顯（QQQ {qqq_score} vs SPY {spy_score}），資金無明顯流向")
                else:
                    technical_reasons.append(f"市場橫盤整理：股債評分接近（{equity_avg} vs {bond_avg}），資金無明顯流向")
            elif gap >= gap_threshold:
                # 股票顯著強於債券
                if abs(qqq_score - spy_score) > 8:
                    technical_reasons.append(f"市場築底中：股票顯著強於債券（{equity_avg} vs {bond_avg}），類股輪動明顯（QQQ {qqq_score} vs SPY {spy_score}），風險偏好上升")
                else:
                    technical_reasons.append(f"市場橫盤整理：股票顯著強於債券（{equity_avg} vs {bond_avg}），風險偏好上升")
        # 如果債券略強，可能是防禦性配置
        elif spy_vs_tlt < 1.0:
            if abs(gap) < gap_threshold:
                # 差距小於 5 分，視為「膠著」或「平衡」
                technical_reasons.append(f"市場橫盤整理：股債評分接近（{equity_avg} vs {bond_avg}），資金無明顯流向")
            elif gap <= -gap_threshold:
                # 債券顯著強於股票
                technical_reasons.append(f"市場謹慎：債券顯著強於股票（債券 {bond_avg} vs 股票 {equity_avg}），避險情緒濃厚")
            else:
                # 債券略強但差距不大
                technical_reasons.append(f"市場謹慎：債券略強於股票（債券 {bond_avg} vs 股票 {equity_avg}），資金尋求避險")
        # 其他情況：真正的橫盤
        else:
            if abs(gap) < gap_threshold:
                # 股債評分接近
                if abs(qqq_score - (etf_data.get(VALUE_ETF, {}).get("score", 50))) > 10:
                    technical_reasons.append(f"市場橫盤整理：股債評分接近（{equity_avg} vs {bond_avg}），類股輪動明顯（QQQ {qqq_score} vs DIA {etf_data.get(VALUE_ETF, {}).get('score', 50)}），資金無明顯流向")
                else:
                    technical_reasons.append(f"市場橫盤整理：股債評分接近（{equity_avg} vs {bond_avg}），各類資產表現均衡，資金無明顯流向")
            else:
                if abs(qqq_score - (etf_data.get(VALUE_ETF, {}).get("score", 50))) > 10:
                    technical_reasons.append(f"市場橫盤整理：類股輪動明顯（QQQ {qqq_score} vs DIA {etf_data.get(VALUE_ETF, {}).get('score', 50)}），方向不明")
                else:
                    technical_reasons.append(f"市場橫盤整理：各類資產表現均衡（股票 {equity_avg}，債券 {bond_avg}），等待催化劑")
    
    # 生成 AI 綜合解釋，說明市場週期判斷和偏差的原因
    ai_explanation = _generate_market_cycle_explanation(
        market_cycle=market_cycle,
        market_score=market_score,
        market_bias=market_bias,
        equity_avg=equity_avg,
        bond_avg=bond_avg,
        spy_score=spy_score,
        qqq_score=qqq_score,
        tlt_score=tlt_score,
        ratios=ratios,
        news_reasons=news_reasons[:3] if news_reasons else [],  # 只使用新聞原因
        technical_reasons=technical_reasons[:2] if technical_reasons else [],  # 技術分析原因作為補充
    )
    
    # 過濾新聞原因：移除以 "Analysis:" 開頭的 ETF 情感分析
    filtered_news_reasons = [
        reason for reason in news_reasons
        if not reason.strip().startswith("Analysis:")
    ]
    
    # 組合最終原因：AI 解釋 + 技術原因 + 過濾後的新聞原因
    final_reasons = [ai_explanation]
    if technical_reasons:
        final_reasons.extend(technical_reasons[:1])  # 添加第一個技術原因
    if filtered_news_reasons:
        final_reasons.extend(filtered_news_reasons[:3])  # 添加前3個過濾後的新聞原因
    elif not any_news:
        final_reasons.append("當前市場無顯著宏觀消息")
    
    return {
        "market_cycle": market_cycle,
        "market_score": market_score,
        "market_bias": market_bias,
        "market_reason": final_reasons[:5],  # 限制最多5個原因
    }


def _interpret_ratio_signal(
    ratio_name: str, 
    ratio_value: Optional[float], 
    ratio_ma20: Optional[float] = None
) -> Tuple[str, str]:
    """
    解釋相對強弱比率的含義（使用相對均線方法）
    
    改進：不再使用絕對閾值（如 1.0），而是使用相對均線
    - 如果當前比率 > 20日均線 → 強勢
    - 如果當前比率 < 20日均線 → 弱勢
    
    Args:
        ratio_name: 比率名稱
        ratio_value: 當前比率值
        ratio_ma20: 20日均線比率值（可選）
    
    Returns:
        (status_icon, interpretation) - 狀態圖標和解釋
    """
    if ratio_value is None:
        return "🟡", "數據不足"
    
    # 如果有均線數據，使用相對均線判斷
    if ratio_ma20 is not None and ratio_ma20 > 0:
        deviation = (ratio_value - ratio_ma20) / ratio_ma20  # 偏離度（百分比）
        
        if ratio_name == "qqq_spy":
            if deviation > 0.02:  # 當前比率 > 均線 2%
                return "🟢", f"成長股領跑（當前 {ratio_value:.3f} vs 均線 {ratio_ma20:.3f}）"
            elif deviation < -0.02:  # 當前比率 < 均線 2%
                return "🔴", f"成長股落後（當前 {ratio_value:.3f} vs 均線 {ratio_ma20:.3f}）"
            else:
                return "🟡", f"成長與基準均衡（當前 {ratio_value:.3f} vs 均線 {ratio_ma20:.3f}）"
        
        elif ratio_name == "spy_tlt":
            if deviation > 0.05:  # 當前比率 > 均線 5%
                return "🟢", f"風險偏好強勁（當前 {ratio_value:.3f} vs 均線 {ratio_ma20:.3f}）"
            elif deviation < -0.05:  # 當前比率 < 均線 5%
                return "🔴", f"避險情緒上升（當前 {ratio_value:.3f} vs 均線 {ratio_ma20:.3f}）"
            else:
                return "🟡", f"股債膠著（當前 {ratio_value:.3f} vs 均線 {ratio_ma20:.3f}）"
        
        elif ratio_name == "dia_qqq":
            if deviation > 0.05:  # 當前比率 > 均線 5%
                return "🟢", f"價值股領跑（當前 {ratio_value:.3f} vs 均線 {ratio_ma20:.3f}）"
            elif deviation < -0.05:  # 當前比率 < 均線 5%
                return "🔴", f"成長股領跑（當前 {ratio_value:.3f} vs 均線 {ratio_ma20:.3f}）"
            else:
                return "🟡", f"價值成長均衡（當前 {ratio_value:.3f} vs 均線 {ratio_ma20:.3f}）"
    
    # 如果沒有均線數據，回退到絕對閾值（向後兼容）
    if ratio_name == "qqq_spy":
        if ratio_value > 1.02:
            return "🟢", f"成長股領跑（{ratio_value:.3f}）"
        elif ratio_value < 0.98:
            return "🔴", f"成長股落後（{ratio_value:.3f}）"
        else:
            return "🟡", f"成長與基準均衡（{ratio_value:.3f}）"
    
    elif ratio_name == "spy_tlt":
        if ratio_value > 1.05:
            return "🟢", f"風險偏好強勁（{ratio_value:.3f}）"
        elif ratio_value < 0.95:
            return "🔴", f"避險情緒上升（{ratio_value:.3f}）"
        else:
            return "🟡", f"股債膠著（{ratio_value:.3f}）"
    
    elif ratio_name == "dia_qqq":
        if ratio_value > 1.05:
            return "🟢", f"價值股領跑（{ratio_value:.3f}）"
        elif ratio_value < 0.95:
            return "🔴", f"成長股領跑（{ratio_value:.3f}）"
        else:
            return "🟡", f"價值成長均衡（{ratio_value:.3f}）"
    
    return "🟡", f"比率 {ratio_value:.3f}"


def _generate_market_cycle_explanation(
    market_cycle: str,
    market_score: int,
    market_bias: int,
    equity_avg: int,
    bond_avg: int,
    spy_score: int,
    qqq_score: int,
    tlt_score: int,
    ratios: Dict[str, float],
    news_reasons: List[str],
    technical_reasons: List[str],
) -> str:
    """
    使用 AI 生成 Bloomberg 風格的市場週期解釋
    
    Returns:
        Bloomberg Terminal 風格的市場簡報
    """
    # 解釋相對強弱比率（使用相對均線）
    ratio_ma20 = ratios.get("_ma20", {})
    qqq_spy_icon, qqq_spy_desc = _interpret_ratio_signal(
        "qqq_spy", 
        ratios.get("qqq_spy"),
        ratio_ma20.get("qqq_spy")
    )
    spy_tlt_icon, spy_tlt_desc = _interpret_ratio_signal(
        "spy_tlt", 
        ratios.get("spy_tlt"),
        ratio_ma20.get("spy_tlt")
    )
    dia_qqq_icon, dia_qqq_desc = _interpret_ratio_signal(
        "dia_qqq", 
        ratios.get("dia_qqq"),
        ratio_ma20.get("dia_qqq")
    )
    
    # 判斷資產強弱
    asset_strength = ""
    if equity_avg > bond_avg + 5:
        asset_strength = f"股票明顯強於債券（{equity_avg} vs {bond_avg}）"
    elif bond_avg > equity_avg + 5:
        asset_strength = f"債券明顯強於股票（{bond_avg} vs {equity_avg}）"
    else:
        asset_strength = f"股債膠著（股票 {equity_avg} vs 債券 {bond_avg}）"
    
    # 構建數據摘要（預處理，讓 AI 更容易理解）
    data_summary = f"""
**市場數據摘要：**
- 市場週期：{market_cycle}（Bull=牛市, Bear=熊市, Base=震盪整理）
- 市場總分：{market_score}/100（{'偏強' if market_score >= 60 else '偏弱' if market_score <= 40 else '中性'}）
- 市場偏差：{market_bias}（{'偏多' if market_bias > 0 else '偏空' if market_bias < 0 else '中性'}）

**資產強弱：**
- {asset_strength}

**關鍵訊號解讀：**
- {qqq_spy_icon} **成長 vs 基準 (QQQ/SPY):** {qqq_spy_desc}
- {spy_tlt_icon} **風險偏好 (SPY/TLT):** {spy_tlt_desc}
- {dia_qqq_icon} **風格輪動 (DIA/QQQ):** {dia_qqq_desc}

**個別 ETF 分數：**
- SPY (基準): {spy_score}/100
- QQQ (成長): {qqq_score}/100
- TLT (債券): {tlt_score}/100
"""
    
    # 構建技術分析關鍵信號
    technical_signals = ""
    if technical_reasons:
        technical_signals = "\n**技術分析關鍵信號：**\n" + "\n".join(f"- {r}" for r in technical_reasons[:2])
    
    # 構建 prompt
    prompt = f"""# Role
You are a senior market analyst at a top-tier investment bank (like Bloomberg or Goldman Sachs). Your job is to interpret technical market data into a concise, professional, and actionable summary using Bloomberg Terminal style.

# Input Data
{data_summary}
{technical_signals}

# Task
Generate a "Bloomberg Terminal Style" market brief in Traditional Chinese (繁體中文).

# Output Rules (Strictly Follow)
1. **Tone:** Professional, objective, concise. No "Hello user" or fluffy intros.
2. **Structure (必須包含以下部分):**
   - **Header:** 📊 Emoji + Cycle Name (e.g., "📊 市場週期診斷：震盪整理 (Base / Neutral)")
   - **Core View:** One sentence summarizing the "Conflict" (e.g., "市場進入方向不明的「拉鋸戰」。儘管資金尚未恐慌性逃離股市，但缺乏關鍵領漲板塊，導致大盤上攻無力。")
   - **Key Signals:** Use bullet points with status icons:
     * 🔴 (Bearish/Weak): 負面訊號，拖累市場
     * 🟢 (Bullish/Strong): 正面訊號，支持市場
     * 🟡 (Neutral/Mixed): 中性訊號，膠著狀態
   - **Conclusion:** One actionable sentence with strategy recommendation (e.g., "策略建議：觀望 (Hold)。在 QQQ 重回強勢或市場總分突破 60 之前，不宜激進加倉。")
3. **Data Interpretation Logic:**
   - Do NOT just list the numbers (e.g., "Ratio is 0.896").
   - INSTEAD, explain the meaning: "QQQ < SPY" means "Risk appetite is fading" or "Tech is dragging the market".
   - "Stocks (55) ≈ Bonds (52)" means "No clear asset class dominance".
   - Focus on CAUSAL relationships: Why is the cycle Base? What causes market_bias to be {market_bias}?
4. **Logic Consistency Check (一致性檢查):**
   - Before generating output, check for contradictions in the signals:
     * If QQQ/SPY implies "Weak Tech" BUT DIA/QQQ implies "Strong Tech" (成長股領跑), explicitly mention this divergence as "市場分歧 (Market Confusion)" or "風格輪動矛盾".
     * If SPY/TLT shows "Strong Risk Appetite" BUT equity_avg is low, explain this as "資金配置異常" or "技術面與基本面背離".
     * Do NOT blindly trust individual ratios; interpret the *conflict* or *divergence* as the main insight when signals contradict.
   - If signals are consistent, explain the unified trend clearly.
   - If signals conflict, prioritize explaining WHY there is confusion (e.g., "市場內部輪動混亂" or "資金流向不明確").
5. **Length:** Keep total output under 250 characters (繁體中文字符).
6. **Format:** Use markdown formatting with **bold** for emphasis.

# Generate Output (in Traditional Chinese 繁體中文)
"""
    
    try:
        explanation = call_deepseek(
            "你是一名頂級投資銀行的資深市場分析師，擅長使用 Bloomberg Terminal 風格撰寫簡潔專業的市場簡報。",
            prompt,
            temperature=0.4
        )
        # 清理和截斷
        explanation = explanation.strip()
        # 移除可能的 markdown 代碼塊標記
        if explanation.startswith("```"):
            explanation = explanation.split("```")[1]
            if explanation.startswith("markdown") or explanation.startswith("md"):
                explanation = explanation.split("\n", 1)[1] if "\n" in explanation else explanation
        explanation = explanation.strip()
        
        # 確保長度合理（Bloomberg 風格應該簡潔）
        if len(explanation) > 300:
            # 嘗試截斷到最後一個完整句子
            sentences = explanation.split("。")
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence + "。") <= 300:
                    truncated += sentence + "。"
                else:
                    break
            explanation = truncated if truncated else explanation[:297] + "..."
        
        return explanation if explanation else f"📊 市場週期診斷：{market_cycle}。市場總分 {market_score}/100，偏差 {market_bias}。{asset_strength}，{qqq_spy_desc}，{spy_tlt_desc}。"
    except Exception as e:
        logger.warning(f"Error generating AI explanation: {e}")
        # 回退到簡單的 Bloomberg 風格解釋
        cycle_name_map = {"Bull": "牛市", "Bear": "熊市", "Base": "震盪整理"}
        bias_desc = "偏多" if market_bias > 0 else "偏空" if market_bias < 0 else "中性"
        return f"📊 市場週期診斷：{cycle_name_map.get(market_cycle, market_cycle)}。核心觀點：{asset_strength}，市場總分 {market_score}/100，偏差 {market_bias}（{bias_desc}）。關鍵訊號：{qqq_spy_desc}，{spy_tlt_desc}。策略建議：{'積極' if market_bias > 0 else '謹慎' if market_bias < 0 else '觀望'}。"


def generate_analysis(symbol: str, market_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Unified analysis generator used by API and daily job.
    """
    # 入口：融合技术面、消息面与 AI 文本分析
    if market_context is None:
        market_context = build_market_context()
    efs_score, efs_detail, efs_formula = run_efs_analysis(symbol)
    rag_score, rag_reason, rag_source = get_rag_sentiment(symbol)
    fund_score, fund_reason, fund_source = get_rag_fundamental(symbol)

    stock = Stock.objects.filter(symbol=symbol.upper()).first()
    efs_series = list(EfsDataPoint.objects.filter(stock=stock).order_by("date")) if stock else []
    risk_score, risk_detail, risk_formula = calculate_risk_score(efs_series)
    closes = [float(dp.close) for dp in efs_series]
    volatility = _calculate_volatility(closes)
    rsi = _calculate_rsi(closes)
    price_above_ma50 = None
    if len(closes) >= 50:
        ma50 = sum(closes[-50:]) / 50
        price_above_ma50 = closes[-1] > ma50

    # RAG functions now return numeric scores directly
    rag_score = _rag_score(rag_score)
    fundamental_score = _fundamental_score(fund_score)
    applied_market_bias = (
        int(market_context.get("market_bias", 0))
        if market_context and not is_etf_symbol(symbol)
        else 0
    )

    market_prompt = ""
    if market_context and not is_etf_symbol(symbol):
        market_prompt = (
            f"- Market Cycle: {market_context.get('market_cycle')} "
            f"(Score {market_context.get('market_score')}/100, Bias {applied_market_bias})\n"
        )

    prompt_efs_detail = shrink_words(efs_detail, reduction=0.45)
    prompt_risk_detail = shrink_words(risk_detail, reduction=0.45)
    prompt_rag_reason = shrink_words(rag_reason, reduction=0.45)
    prompt_fund_reason = shrink_words(fund_reason, reduction=0.45)
    prompt_efs_formula = truncate_chars(efs_formula, limit=120)
    prompt_risk_formula = truncate_chars(risk_formula, limit=120)
    active_factor = (
        EfsAlphaFactor.objects.filter(status="active")
        .order_by("-last_score", "-updated_at")
        .first()
    )
    alpha_prompt = ""
    if active_factor:
        alpha_prompt = (
            f"- EFS Alpha Factor: {active_factor.name} "
            f"(Score {active_factor.last_score})\n"
            f"- Alpha Expression: {active_factor.expression}\n"
        )

    # 获取股票信息用于个性化
    stock_info = Stock.objects.filter(symbol=symbol.upper()).first()
    sector_info = stock_info.sector if stock_info else "Unknown"
    
    # 给 LLM 的结构化提示词（改进版：更个性化、专业但易懂）
    ai_prompt = f"""
    You are a senior equity analyst writing for investors with financial knowledge but not professional-level expertise.
    
    **Target Audience:** Investors who understand basic financial terms but need clear, actionable insights.
    **Writing Style:** Use professional terminology (e.g., "momentum", "volatility", "drawdown", "RSI") but immediately follow with simple explanations in parentheses or short phrases. Balance technical accuracy with accessibility.
    
    **Example Style:**
    - Good: "The stock shows strong momentum (upward price trend) with low volatility (price stability), making it suitable for risk-averse investors."
    - Bad: "The stock shows strong momentum with low volatility." (too technical)
    - Bad: "The stock is going up and not moving much." (too simple)
    
    [Data Profile for {symbol}]
    - Symbol: {symbol}
    - Sector: {sector_info}
    - Technical Score (EFS): {efs_score}/100 ({prompt_efs_detail})
    - Technical Formula: {prompt_efs_formula}
    {alpha_prompt}
    - Safety Score (Risk Stability): {risk_score}/100 ({prompt_risk_detail})
    - Risk Formula: {prompt_risk_formula}
    - Fundamental RAG Score: {fund_score}/100
    - Fundamental Context: {prompt_fund_reason}
    - News Sentiment Score (RAG): {rag_score}/100
    - Sentiment Context: {prompt_rag_reason}
    - Key News Context: {prompt_rag_reason}
    {market_prompt}
    
    [Critical Requirements]
    1. **Be Specific to {symbol}**: Reference actual data points, news, or technical indicators unique to this stock. Avoid generic templates.
    2. **Vary Your Language**: Each reason should sound different. Use different sentence structures, different starting phrases, and different emphasis points.
    3. **Match Scores to Reasons**: If technical score is {efs_score}, explain WHY it's {efs_score} for THIS stock specifically. Mention specific indicators, trends, or patterns.
    4. **Professional + Accessible**: Use terms like "momentum", "RSI", "volatility", "drawdown" but explain them briefly in context.
    5. **No Repetition**: Each reason (fundamental, technical, sentiment, risk) should have a distinct voice and focus.
    6. **Consistency Check**: Reasons must match scores. If data is limited, keep the score near 50.
    7. **Conflict Resolution**: If Technical is Bearish but the final signal is Buy, explicitly explain why.
    8. **Score Polarity**: All scores are positive polarity (higher = better). The Risk score is a Safety/Stability score.
    
    [Your Task]
    Return JSON only with these fields:
    1. fundamental_score: integer 0-100
    2. fundamental_reason: 70-90 words, SPECIFIC to {symbol}. Mention actual data points, business metrics, or sector context. Use professional terms with simple explanations.
    3. technical_reason: 70-90 words, SPECIFIC to {symbol}. Reference actual indicators, trends, or patterns. Explain WHY the score is what it is for THIS stock.
    4. sentiment_reason: 70-90 words, SPECIFIC to {symbol}. Reference actual news headlines or events. Explain how sentiment affects THIS stock specifically.
    5. risk_reason: 70-90 words, SPECIFIC to {symbol}. Reference actual volatility or drawdown numbers. Explain what this means for position sizing for THIS stock.
    6. signal: one of "Strong Buy", "Buy", "Hold", "Sell"
    7. Time Horizon (cycle of investment): 1-2 sentences, short and actionable, specific to {symbol}
    8. is_ai_concept: boolean (true if {symbol} is meaningfully tied to AI themes)
    9. tags: array of 3-6 short English strings (no emojis)
    10. summary: 140-160 words with a clear conflict note if signals disagree. Be specific to {symbol}, reference actual scores and data points.
    
    **Remember:** Each reason must be UNIQUE to {symbol}. Do not use generic templates. Reference specific numbers, news, or indicators.
    """

    # 调用模型，解析结构化结果
    ai_text = call_deepseek("You are a senior investment advisor.", ai_prompt)
    ai_data = parse_ai_json(ai_text)
    if not ai_data:
        ai_data = {}

    raw_fundamental_reason = ai_data.get("fundamental_reason")
    low_confidence_fundamental = is_low_confidence_reason(raw_fundamental_reason) or is_low_confidence_reason(fund_reason)
    # 各分数加权得到总分（数据不足时分数设为50，权重保持不变）
    # 权重分配：fundamental 40%, technical 25%, sentiment 20%, risk 15%
    fundamental_weight = 0.4
    sentiment_weight = 0.2
    base_score = int(
        (fundamental_score * fundamental_weight)
        + (efs_score * 0.25)
        + (rag_score * sentiment_weight)
        + (risk_score * 0.15)
    )
    overall_score = base_score
    if market_context and not is_etf_symbol(symbol):
        overall_score = max(0, min(100, base_score + applied_market_bias))
    signal = _signal_from_score(overall_score)

    default_strategy = "Long-term Hold" if overall_score >= 60 else "Short-term Trade"
    strategy = ensure_word_range(
        ai_data.get("strategy")
        or ai_data.get("investment_strategy")
        or ai_data.get("time_horizon")
        or ai_data.get("Time Horizon (cycle of investment)"),
        2,
        60,
        default_strategy,
    )

    fundamental_reason = ensure_word_range(
        raw_fundamental_reason if not low_confidence_fundamental else fund_reason,
        70,
        90,
        _fallback_fundamental_reason(fundamental_score),
    )
    # Convert scores to sentiment labels for compatibility checks
    fund_sentiment_label = _score_to_sentiment_label(fund_score)
    rag_sentiment_label = _score_to_sentiment_label(rag_score)
    
    if fund_score == 50 and ("not available" in fund_reason.lower() or "no fundamental" in fund_reason.lower()):
        fundamental_reason = (
            "Fundamental documents are not available in the current dataset, so the "
            "fundamental score is set to 50 (neutral) to reflect limited visibility into "
            "business quality and durability. The score maintains its weight in the overall assessment."
        )
    technical_reason = ensure_word_range(
        ai_data.get("technical_reason"),
        70,
        90,
        _fallback_technical_reason(efs_score, efs_detail),
    )
    sentiment_reason = ensure_word_range(
        ai_data.get("sentiment_reason"),
        70,
        90,
        _fallback_sentiment_reason(rag_score, rag_sentiment_label, rag_reason),
    )
    if rag_score == 50 and ("not available" in rag_reason.lower() or "no news" in rag_reason.lower()):
        sentiment_reason = (
            "Sentiment documents are not available in the current dataset, so the "
            "sentiment score is set to 50 (neutral) to reflect limited visibility into "
            "recent news tone and market narrative. The score maintains its weight in the overall assessment."
        )
    risk_reason = ensure_word_range(
        ai_data.get("risk_reason"),
        70,
        90,
        _fallback_risk_reason(risk_score, risk_detail),
    )

    summary = ensure_word_range(
        ai_data.get("summary"),
        140,
        160,
        _fallback_summary(
            fundamental_score,
            efs_score,
            rag_score,
            risk_score,
            overall_score,
            signal,
            strategy,
        ),
    )

    conflict_note = ""
    if efs_score < 60 and signal in {"Buy", "Strong Buy"}:
        conflict_note = (
            "Technical signals are weaker or overextended, but the overall rating remains positive "
            "due to stronger fundamentals or sentiment. Consider smaller sizing or staged entries."
        )

    ai_tags = coerce_tags(ai_data.get("tags"))
    is_ai_concept = bool(ai_data.get("is_ai_concept"))
    # tags 优先使用 LLM 输出，缺失时用规则兜底
    tags = ai_tags or _build_tags(
        technical_score=efs_score,
        sentiment_score=rag_score,
        risk_score=risk_score,
        fundamental_score=fundamental_score,
        volatility=volatility,
        rsi=rsi,
        price_above_ma50=price_above_ma50,
        rag_reason=rag_reason,
        sector=stock.sector if stock else "",
        signal=signal,
        strategy=strategy,
    )
    if is_ai_concept and "AI Play" not in tags:
        tags.append("AI Play")
    
    # 判断是否为 ETF
    is_etf = is_etf_symbol(symbol)
    asset_type = "etf" if is_etf else "stock"
    
    # 重新组织的 payload 结构（按用户要求的格式）
    payload = {
        # ========== 标签和基本信息 ==========
        "tags": tags,
        "label": asset_type,  # "stock" 或 "etf"
        "conflict_note": conflict_note if conflict_note else None,
        
        # ========== 评分（包含 overall_score） ==========
        "scores": {
            "stability": risk_score,
            "sentiment": rag_score,
            "technical": efs_score,
            "fundamental": fundamental_score,
            "overall_score": overall_score,
        },
        
        # ========== 信号和符号 ==========
        "signal": signal,
        "symbol": symbol.upper(),
        
        # ========== 综合总结 ==========
        "summary": summary,
        
        # ========== 各维度详细分析 ==========
        "analysis": {
            "stability": {
                "score": risk_score,
                "reason": risk_reason,
                "data": {
                    "score": risk_score,
                    "detail": risk_detail,
                    "formula": risk_formula,
                }
            },
            "sentiment": {
                "score": rag_score,
                "reason": sentiment_reason,
                "data": {
                    "sentiment": rag_sentiment_label,
                    "score": rag_score,
                    "source": rag_source,
                    "reason": rag_reason,
                }
            },
            "technical": {
                "score": efs_score,
                "reason": technical_reason,
                "data": {
                    "score": efs_score,
                    "status": "Bullish" if efs_score >= 60 else "Bearish",
                    "indicators": efs_detail,
                    "formula": efs_formula,
                    "alpha": {
                        "name": active_factor.name if active_factor else None,
                        "expression": active_factor.expression if active_factor else None,
                        "score": active_factor.last_score if active_factor else None,
                    }
                }
            },
            "fundamental": {
                "score": fundamental_score,
                "reason": fundamental_reason,
                "data": {
                    "sentiment": fund_sentiment_label,
                    "score": fund_score,
                    "source": fund_source,
                    "reason": fund_reason,
                }
            }
        },
        
        # ========== 投资策略 ==========
        "strategy": strategy,
    }
    return payload


def generate_and_store_analysis(
    stock: Stock, market_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generate analysis and persist to AnalysisResult for quick API retrieval.
    """
    # 写入缓存结果，便于 API 快速返回
    payload = generate_analysis(stock.symbol, market_context=market_context)
    AnalysisResult.objects.update_or_create(
        stock=stock,
        defaults={"result": payload},
    )
    return payload

