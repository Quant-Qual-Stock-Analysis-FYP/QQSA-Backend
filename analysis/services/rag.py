"""RAG服务：文档检索与情感分析。"""

import logging
import re
from typing import Iterable, List, Tuple, Optional

import numpy as np
from django.db import transaction

from markets.models import Stock, StockNews
from ..models import RagDocument, RagDocumentFundamental
from .ai import call_deepseek, get_embedding
from .scoring_rubric import (
    SENTIMENT_RUBRIC,
    FUNDAMENTAL_RUBRIC,
    SENTIMENT_FEW_SHOT_EXAMPLES,
    FUNDAMENTAL_FEW_SHOT_EXAMPLES,
)

logger = logging.getLogger(__name__)


def _build_embedding(text: str) -> List[float]:
    """构建文本嵌入向量"""
    if not text or not text.strip():
        return []
    
    try:
        embedding = get_embedding(text)
        return embedding if embedding else []
    except Exception as e:
        logger.debug(f"Error building embedding: {e}")
        return []


def _news_to_docs(stock: Stock, news_items: Iterable[StockNews]) -> List[RagDocument]:
    payload = []
    for news in news_items:
        text = f"{news.title}. {news.summary or ''}".strip()
        payload.append(
            RagDocument(
                stock=stock,
                content=text,
                source=news.publisher or "news",
                is_manual=False,
                embedding=_build_embedding(text),
            )
        )
    return payload


def ensure_rag_data(stock: Stock, cold_limit: int = 10, incremental_limit: int = 5) -> None:
    """
    Cold-start: hydrate from existing StockNews if empty.
    Incremental: append latest news (no deletion), keep manual docs intact.
    """
    if not stock:
        return
    
    try:
        existing_docs = RagDocument.objects.filter(stock=stock)
        if not existing_docs.exists():
            news_items = (
                StockNews.objects.filter(stock=stock)
                .order_by("-created_at")
                .select_related("stock")[:cold_limit]
            )
            if not news_items:
                return
            try:
                with transaction.atomic():
                    RagDocument.objects.bulk_create(
                        _news_to_docs(stock, news_items),
                        ignore_conflicts=True
                    )
            except Exception as e:
                logger.error(f"Error bulk creating RAG documents: {e}", exc_info=True)
            return

        # incremental: fetch recent news and append if not already present by title
        latest_titles = set(existing_docs.values_list("content", flat=True))
        news_items = (
            StockNews.objects.filter(stock=stock)
            .order_by("-created_at")
            .select_related("stock")[:incremental_limit]
        )
        new_payload = []
        for news in news_items:
            try:
                candidate_text = f"{news.title}. {news.summary or ''}".strip()
                if not candidate_text or candidate_text in latest_titles:
                    continue
                new_payload.append(
                    RagDocument(
                        stock=stock,
                        content=candidate_text,
                        source=news.publisher or "news",
                        is_manual=False,
                        embedding=_build_embedding(candidate_text),
                    )
                )
            except Exception as e:
                logger.debug(f"Error processing news item: {e}")
                continue
        
        if new_payload:
            try:
                with transaction.atomic():
                    RagDocument.objects.bulk_create(new_payload, ignore_conflicts=True)
            except Exception as e:
                logger.error(f"Error bulk creating incremental RAG documents: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Error ensuring RAG data for {stock.symbol}: {e}", exc_info=True)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _rank_documents(model, stock: Stock, query_text: str, k: int = 5):
    query_vec = np.array(_build_embedding(query_text), dtype=float)
    docs = list(model.objects.filter(stock=stock))
    if query_vec.size == 0:
        return sorted(docs, key=lambda d: d.created_at, reverse=True)[:k]

    scored = []
    for doc in docs:
        vec = np.array(doc.embedding or [], dtype=float)
        scored.append((doc, _cosine_similarity(query_vec, vec)))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [doc for doc, _ in scored[:k]]


def _extract_score_from_response(response: str) -> Optional[int]:
    """Extract numeric score (0-100) from LLM response."""
    if not response:
        return None
    
    # Try to find a number between 0-100 in the response
    # Look for patterns like "Score: 75", "75", "score is 82", etc.
    # Also handle Chinese patterns like "评分: 75", "75分"
    patterns = [
        r'score[:\s]+(\d{1,2}|100)',  # English: "Score: 75"
        r'(\d{1,2}|100)\s*分',  # Chinese: "75分"
        r'评分[:\s]+(\d{1,2}|100)',  # Chinese: "评分: 75"
        r'分数[:\s]+(\d{1,2}|100)',  # Chinese: "分数: 75"
        r'(\d{1,2}|100)\s*$',  # Number at end of line
        r'\b(\d{1,2}|100)\b',  # Any standalone number 0-100 (last resort)
    ]
    
    # First, try to find explicit score markers
    for pattern in patterns[:-1]:  # Exclude the last catch-all pattern initially
        matches = re.findall(pattern, response, re.IGNORECASE | re.MULTILINE)
        if matches:
            try:
                score = int(matches[-1])  # Take the last match
                if 0 <= score <= 100:
                    return score
            except (ValueError, IndexError):
                continue
    
    # If no explicit marker found, look for numbers in the last few lines
    lines = response.strip().split('\n')
    for line in reversed(lines[-3:]):  # Check last 3 lines
        numbers = re.findall(r'\b(\d{1,2}|100)\b', line)
        if numbers:
            try:
                score = int(numbers[-1])
                if 0 <= score <= 100:
                    return score
            except (ValueError, IndexError):
                continue
    
    return None


def _build_sentiment_prompt(symbol: str, context: str) -> str:
    """Build improved sentiment analysis prompt with rubric and Chain of Thought."""
    return f"""Analyze the sentiment of stock news for {symbol} using the following information:

{context}

**Task:** Assign a sentiment score from 0 to 100 using the detailed rubric below.

{SENTIMENT_RUBRIC}

**Few-Shot Examples:**
{SENTIMENT_FEW_SHOT_EXAMPLES}

**Instructions:**
1. First, identify the key positive and negative factors in the news (Chain of Thought reasoning)
2. Compare the news against the rubric to determine the appropriate score range
3. Provide a brief 1-2 sentence justification explaining your reasoning
4. Output the numeric score (0-100) at the end

**Response Format:**
Analysis: [Your 1-2 sentence reasoning]
Score: [Number between 0-100]

**Important:** Use the full 0-100 range. Do NOT default to extremes (0 or 100) unless the news is truly exceptional or catastrophic. Be granular and precise."""


def get_rag_sentiment(symbol: str) -> Tuple[int, str, str]:
    """
    获取RAG情感分析结果，返回数值分数 (0-100)
    
    Returns:
        Tuple[int, str, str]: (score, reason, sources)
        - score: 0-100 的数值分数
        - reason: 分析理由
        - sources: 数据来源
    """
    if not symbol or not symbol.strip():
        return 50, "Invalid symbol", ""
    
    try:
        stock = Stock.objects.filter(symbol=symbol.upper()).first()
        if not stock:
            return 50, "Stock not found", ""

        ensure_rag_data(stock)

        docs = _rank_documents(RagDocument, stock, "market sentiment and growth outlook", k=5)
        if not docs:
            return 50, "No news available", ""

        # Clean context: extract key sentences if documents are too long
        context_items = []
        for doc in docs:
            content = doc.content.strip()
            # If content is very long, try to extract key sentences
            if len(content) > 500:
                # Simple heuristic: take first 300 chars and last 200 chars
                content = content[:300] + "..." + content[-200:]
            context_items.append(f"- {content}")
        
        context = "\n".join(context_items)
        prompt = _build_sentiment_prompt(stock.symbol, context)
        
        try:
            # Use slightly higher temperature (0.4) for more nuanced scoring
            response = call_deepseek(
                "You are a strict, quantitative Wall Street sentiment analyst. "
                "You are skeptical of corporate PR fluff and avoid extreme scores unless justified. "
                "Your job is to provide granular, precise sentiment scores.",
                prompt,
                temperature=0.4
            )
        except Exception as e:
            logger.warning(f"Error calling LLM for sentiment: {e}")
            return 50, "LLM service unavailable", ""

        # Extract numeric score from response
        score = _extract_score_from_response(response)
        if score is None:
            # Fallback: try to infer from sentiment keywords
            lowered = response.lower() if isinstance(response, str) else ""
            if any(word in lowered for word in ["extremely bullish", "record", "breakthrough", "exceptional"]):
                score = 90
            elif any(word in lowered for word in ["very bullish", "strong", "beat", "exceed"]):
                score = 80
            elif any(word in lowered for word in ["bullish", "positive", "growth"]):
                score = 70
            elif any(word in lowered for word in ["neutral", "stable", "in line"]):
                score = 50
            elif any(word in lowered for word in ["bearish", "miss", "decline", "concern"]):
                score = 30
            elif any(word in lowered for word in ["very bearish", "significant", "loss", "crisis"]):
                score = 20
            elif any(word in lowered for word in ["extremely bearish", "bankruptcy", "catastrophic"]):
                score = 10
            else:
                score = 50  # Default neutral if cannot determine
        
        # Ensure score is in valid range
        score = max(0, min(100, score))
        
        top_sources = ", ".join({doc.source or "news" for doc in docs if doc.source})
        return score, response, top_sources
    except Exception as e:
        logger.error(f"Error getting RAG sentiment for {symbol}: {e}", exc_info=True)
        return 50, f"Error: {str(e)}", ""


def _build_fundamental_prompt(symbol: str, context: str) -> str:
    """Build improved fundamental analysis prompt with rubric and Chain of Thought."""
    return f"""Analyze the fundamental financial health for {symbol} using the following information:

{context}

**Task:** Assign a fundamental score from 0 to 100 using the detailed rubric below.

{FUNDAMENTAL_RUBRIC}

**Few-Shot Examples:**
{FUNDAMENTAL_FEW_SHOT_EXAMPLES}

**Instructions:**
1. First, identify key financial metrics mentioned (profitability, leverage, liquidity, growth) - Chain of Thought reasoning
2. Compare the metrics against the rubric to determine the appropriate score range
3. Consider industry context and company lifecycle stage
4. Provide a brief 1-2 sentence justification explaining your reasoning
5. Output the numeric score (0-100) at the end

**Response Format:**
Analysis: [Your 1-2 sentence reasoning]
Score: [Number between 0-100]

**Important:** Use the full 0-100 range. Do NOT default to extremes. Consider trends, not just absolute values. Be granular and precise."""


def get_rag_fundamental(symbol: str) -> Tuple[int, str, str]:
    """
    获取RAG基本面分析结果，返回数值分数 (0-100)
    
    Returns:
        Tuple[int, str, str]: (score, reason, sources)
        - score: 0-100 的数值分数
        - reason: 分析理由
        - sources: 数据来源
    """
    if not symbol or not symbol.strip():
        return 50, "Invalid symbol", ""
    
    try:
        stock = Stock.objects.filter(symbol=symbol.upper()).first()
        if not stock:
            return 50, "Stock not found", ""

        docs = _rank_documents(
            RagDocumentFundamental,
            stock,
            "fundamentals profitability balance sheet cash flow valuation",
            k=5,
        )
        if not docs:
            return 50, "No fundamental documents", ""

        # Clean context: extract key information if documents are too long
        context_items = []
        for doc in docs:
            content = doc.content.strip()
            # If content is very long, try to extract key sentences
            if len(content) > 500:
                # Simple heuristic: take first 300 chars and last 200 chars
                content = content[:300] + "..." + content[-200:]
            context_items.append(f"- {content}")
        
        context = "\n".join(context_items)
        prompt = _build_fundamental_prompt(stock.symbol, context)
        
        try:
            # Use slightly higher temperature (0.4) for more nuanced scoring
            response = call_deepseek(
                "You are a strict, quantitative fundamental analyst. "
                "You analyze financial metrics objectively and avoid extreme scores unless justified. "
                "Your job is to provide granular, precise fundamental scores based on financial data.",
                prompt,
                temperature=0.4
            )
        except Exception as e:
            logger.warning(f"Error calling LLM for fundamental: {e}")
            return 50, "LLM service unavailable", ""

        # Extract numeric score from response
        score = _extract_score_from_response(response)
        if score is None:
            # Fallback: try to infer from sentiment keywords
            lowered = response.lower() if isinstance(response, str) else ""
            if any(word in lowered for word in ["exceptional", "excellent", "strong", "roe >20", "roe > 20"]):
                score = 90
            elif any(word in lowered for word in ["very strong", "good", "roe 15", "roe 18"]):
                score = 80
            elif any(word in lowered for word in ["strong", "solid", "roe 10", "adequate"]):
                score = 70
            elif any(word in lowered for word in ["moderate", "acceptable", "average", "neutral"]):
                score = 55
            elif any(word in lowered for word in ["weak", "concern", "declining", "poor"]):
                score = 35
            elif any(word in lowered for word in ["very weak", "distress", "loss", "crisis"]):
                score = 20
            elif any(word in lowered for word in ["critical", "bankruptcy", "failure", "collapse"]):
                score = 10
            else:
                score = 50  # Default neutral if cannot determine
        
        # Ensure score is in valid range
        score = max(0, min(100, score))
        
        top_sources = ", ".join({doc.source or "manual" for doc in docs if doc.source})
        return score, response, top_sources
    except Exception as e:
        logger.error(f"Error getting RAG fundamental for {symbol}: {e}", exc_info=True)
        return 50, f"Error: {str(e)}", ""

