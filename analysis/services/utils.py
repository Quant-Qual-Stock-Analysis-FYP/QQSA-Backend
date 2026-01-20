"""通用工具函数：文本处理、数据转换等"""

import json
import logging
from typing import Any, Dict, List, Optional

from ..config.constants import ETF_SYMBOLS

logger = logging.getLogger(__name__)


def shrink_words(text: str, reduction: float = 0.2) -> str:
    """缩短文本，保留前 N% 的单词"""
    if not text:
        return ""
    words = str(text).split()
    if not words:
        return ""
    keep_count = max(1, int(len(words) * (1 - reduction)))
    if keep_count >= len(words):
        return str(text)
    return " ".join(words[:keep_count]) + "..."


def truncate_chars(text: str, limit: int = 140) -> str:
    """截断文本到指定字符数"""
    if not text:
        return ""
    cleaned = str(text)
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


def ensure_word_range(text: str, min_words: int, max_words: int, fallback: str) -> str:
    """确保文本在指定的单词数范围内"""
    if not text:
        return fallback
    cleaned = str(text).strip()
    if not cleaned:
        return fallback
    words = cleaned.split()
    if len(words) < min_words:
        return fallback
    if len(words) > max_words:
        return " ".join(words[:max_words]) + "..."
    return cleaned


def coerce_score(value: Any, default: int = 50) -> int:
    """将值转换为 0-100 的分数"""
    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        return default
    return max(0, min(100, score))


def parse_ai_json(text: str) -> Dict[str, Any]:
    """解析 LLM 返回的 JSON（去掉代码块）"""
    if not text:
        return {}
    cleaned = text.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return {}


def coerce_tags(raw_tags: Any, limit: int = 8) -> List[str]:
    """清洗 tags：仅保留英文、去重、限量"""
    if not isinstance(raw_tags, list):
        return []
    cleaned: List[str] = []
    for item in raw_tags:
        if item is None:
            continue
        tag = str(item).strip()
        if not tag:
            continue
        tag = "".join(ch for ch in tag if ch.isascii())
        if not tag or tag in cleaned:
            continue
        cleaned.append(tag)
        if len(cleaned) >= limit:
            break
    return cleaned


def is_etf_symbol(symbol: str) -> bool:
    """检查是否为 ETF 符号"""
    return symbol.upper() in ETF_SYMBOLS


def is_low_confidence_reason(text: Optional[str]) -> bool:
    """检查文本是否表示低置信度"""
    if not text:
        return False
    lowered = str(text).lower()
    keywords = [
        "limited",
        "insufficient",
        "no data",
        "not enough",
        "uncertain",
        "uncertainty",
        "lack of",
        "missing",
        "incomplete",
        "low visibility",
        "no visibility",
        "unknown",
        "not available",
    ]
    return any(keyword in lowered for keyword in keywords)
