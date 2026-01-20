"""RAG服务：文档检索与情感分析。"""

import logging
from typing import Iterable, List, Tuple

import numpy as np
from django.db import transaction

from markets.models import Stock, StockNews
from ..models import RagDocument, RagDocumentFundamental
from .ai import call_deepseek, get_embedding

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


def get_rag_sentiment(symbol: str) -> Tuple[str, str, str]:
    """获取RAG情感分析结果"""
    if not symbol or not symbol.strip():
        return "Unknown", "Invalid symbol", ""
    
    try:
        stock = Stock.objects.filter(symbol=symbol.upper()).first()
        if not stock:
            return "Unknown", "Stock not found", ""

        ensure_rag_data(stock)

        docs = _rank_documents(RagDocument, stock, "market sentiment and growth outlook", k=5)
        if not docs:
            return "Unknown", "No news available", ""

        context = "\n".join(f"- {doc.content}" for doc in docs)
        prompt = (
            f"Analyze sentiment for {stock.symbol} using the bullet points:\n{context}\n"
            "Respond with a short justification and end with the single word "
            "sentiment label (Positive/Neutral/Negative)."
        )
        
        try:
            response = call_deepseek("You are a financial sentiment analyst.", prompt)
        except Exception as e:
            logger.warning(f"Error calling LLM for sentiment: {e}")
            return "Unknown", "LLM service unavailable", ""

        # Heuristic parsing: if label not found, default to Unknown.
        sentiment = "Unknown"
        if isinstance(response, str):
            lowered = response.lower()
            if "positive" in lowered:
                sentiment = "Positive"
            elif "negative" in lowered:
                sentiment = "Negative"

        top_sources = ", ".join({doc.source or "news" for doc in docs if doc.source})
        return sentiment, response, top_sources
    except Exception as e:
        logger.error(f"Error getting RAG sentiment for {symbol}: {e}", exc_info=True)
        return "Unknown", f"Error: {str(e)}", ""


def get_rag_fundamental(symbol: str) -> Tuple[str, str, str]:
    """获取RAG基本面分析结果"""
    if not symbol or not symbol.strip():
        return "Unknown", "Invalid symbol", ""
    
    try:
        stock = Stock.objects.filter(symbol=symbol.upper()).first()
        if not stock:
            return "Unknown", "Stock not found", ""

        docs = _rank_documents(
            RagDocumentFundamental,
            stock,
            "fundamentals profitability balance sheet cash flow valuation",
            k=5,
        )
        if not docs:
            return "Unknown", "No fundamental documents", ""

        context = "\n".join(f"- {doc.content}" for doc in docs)
        prompt = (
            f"Analyze fundamentals for {stock.symbol} using the bullet points:\n{context}\n"
            "Respond with a short justification and end with the single word "
            "sentiment label (Positive/Neutral/Negative)."
        )
        
        try:
            response = call_deepseek("You are a fundamental analyst.", prompt)
        except Exception as e:
            logger.warning(f"Error calling LLM for fundamental: {e}")
            return "Unknown", "LLM service unavailable", ""

        sentiment = "Unknown"
        if isinstance(response, str):
            lowered = response.lower()
            if "positive" in lowered:
                sentiment = "Positive"
            elif "negative" in lowered:
                sentiment = "Negative"

        top_sources = ", ".join({doc.source or "manual" for doc in docs if doc.source})
        return sentiment, response, top_sources
    except Exception as e:
        logger.error(f"Error getting RAG fundamental for {symbol}: {e}", exc_info=True)
        return "Unknown", f"Error: {str(e)}", ""

