"""AI 分析主流程：融合 EFS、RAG 与大模型输出。"""

import json
import logging
from statistics import pstdev
from typing import Any, Dict, List, Optional, Tuple

from markets.models import Stock
from ..config.constants import ETF_SYMBOLS
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


def build_market_context() -> Dict[str, Any]:
    """
    Build a market regime signal from ETF universe (SPY/QQQ/DIA).
    """
    efs_scores: List[int] = []
    sentiments: List[str] = []
    reasons: List[str] = []
    any_news = False

    for symbol in ETF_SYMBOLS:
        stock = Stock.objects.filter(symbol=symbol).first()
        if not stock:
            continue
        efs_score, _, _ = run_efs_analysis(symbol)
        rag_score, rag_reason, _ = get_rag_sentiment(symbol)
        efs_scores.append(efs_score)
        # Convert numeric score to sentiment label for compatibility
        sentiment_label = _score_to_sentiment_label(rag_score)
        sentiments.append(sentiment_label)
        if rag_reason and "no news" not in rag_reason.lower():
            reasons.append(_shorten_reason(rag_reason))
            any_news = True

    if not efs_scores:
        return {
            "market_regime": "Neutral",
            "market_cycle": "Base",
            "market_score": 50,
            "market_bias": 0,
            "market_reason": [],
        }

    if not any_news:
        return {
            "market_regime": "Neutral",
            "market_cycle": "Base",
            "market_score": 50,
            "market_bias": 0,
            "market_reason": ["当前市场无显著宏观消息"],
        }

    avg_score = int(sum(efs_scores) / len(efs_scores))
    pos = sentiments.count("Positive")
    neg = sentiments.count("Negative")

    if avg_score >= 70 or pos >= 2:
        regime = "Bullish"
        bias = 8
    elif avg_score <= 45 or neg >= 2:
        regime = "Bearish"
        bias = -8
    else:
        regime = "Neutral"
        bias = 0

    return {
        "market_regime": regime,
        "market_cycle": "Bull" if regime == "Bullish" else "Bear" if regime == "Bearish" else "Base",
        "market_score": avg_score,
        "market_bias": bias,
        "market_reason": [r for r in reasons if r],
    }


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
            f"- Market Regime: {market_context.get('market_regime')} "
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

