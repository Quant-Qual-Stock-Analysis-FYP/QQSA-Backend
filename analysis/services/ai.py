"""AI服务：LLM调用和嵌入向量生成。"""

import logging
from functools import lru_cache
from typing import List, Optional

from django.conf import settings

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency at runtime
    OpenAI = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency at runtime
    SentenceTransformer = None

from ..utils.ai_client import get_openai_client

logger = logging.getLogger(__name__)


def _get_client() -> Optional["OpenAI"]:
    """Create a cached DeepSeek/OpenAI compatible client."""
    return get_openai_client()


@lru_cache(maxsize=1)
def _get_embedder() -> Optional["SentenceTransformer"]:
    """Load a lightweight local embedding model."""
    if not SentenceTransformer:
        return None
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception:
        return None


def call_deepseek(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
    """调用DeepSeek聊天完成接口（带防御性降级）"""
    if not system_prompt or not user_prompt:
        logger.warning("Empty prompt provided to call_deepseek")
        return "Empty prompt provided"
    
    client = _get_client()
    if not client:
        logger.warning("DeepSeek client unavailable")
        return "DeepSeek client unavailable. Please set DEEPSEEK_API_KEY."

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": str(system_prompt)},
                {"role": "user", "content": str(user_prompt)},
            ],
            temperature=max(0.0, min(2.0, temperature)),  # 限制temperature范围
            stream=False,
        )
        
        if not response or not response.choices:
            logger.warning("Empty response from DeepSeek")
            return "Empty response from DeepSeek"
        
        content = response.choices[0].message.content
        return content if content else "Empty response content"
    except Exception as exc:
        logger.error(f"Error calling DeepSeek: {exc}", exc_info=True)
        return f"Error calling DeepSeek: {exc}"


def get_embedding(text: str) -> List[float]:
    """将文本转换为嵌入向量列表；如果不可用则返回空列表"""
    if not text or not text.strip():
        return []
    
    embedder = _get_embedder()
    if not embedder:
        logger.debug("Embedder not available")
        return []
    
    try:
        embedding = embedder.encode(str(text))
        if embedding is None:
            return []
        result = embedding.tolist()
        return result if isinstance(result, list) else []
    except Exception as e:
        logger.debug(f"Error generating embedding: {e}")
        return []

