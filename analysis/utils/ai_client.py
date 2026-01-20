"""AI客户端工具：提供统一的OpenAI/DeepSeek客户端初始化。"""

import logging
import os
from functools import lru_cache
from typing import Optional

from django.conf import settings

try:
    from openai import OpenAI
    import httpx
except Exception:
    OpenAI = None
    httpx = None

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_openai_client() -> Optional["OpenAI"]:
    """
    创建并缓存 DeepSeek/OpenAI 兼容客户端
    
    该函数会：
    1. 临时清除代理环境变量，避免 OpenAI/httpx 库自动读取
    2. 创建显式的 httpx 客户端，禁用从环境变量读取代理设置
    3. 将 httpx 客户端传递给 OpenAI 客户端
    
    Returns:
        OpenAI 客户端实例，如果初始化失败则返回 None
    """
    if not OpenAI or not settings.DEEPSEEK_API_KEY:
        return None
    
    try:
        timeout = float(getattr(settings, "DEEPSEEK_TIMEOUT", 300))
        
        # 临时清除代理环境变量，避免 OpenAI/httpx 库自动读取
        original_proxies = {}
        proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY', 'all_proxy']
        for var in proxy_vars:
            if var in os.environ:
                original_proxies[var] = os.environ.pop(var)
        
        try:
            # 创建显式的 httpx 客户端，禁用从环境变量读取代理设置
            if httpx:
                # 使用 trust_env=False 禁用从环境变量读取代理配置
                http_client = httpx.Client(
                    timeout=timeout,
                    follow_redirects=True,
                    trust_env=False,  # 禁用从环境变量读取代理设置
                )
                client = OpenAI(
                    api_key=settings.DEEPSEEK_API_KEY,
                    base_url=getattr(settings, "DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                    http_client=http_client,
                )
            else:
                # 如果没有 httpx，使用默认方式
                client = OpenAI(
                    api_key=settings.DEEPSEEK_API_KEY,
                    base_url=getattr(settings, "DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
                    timeout=timeout,
                )
        finally:
            # 恢复原始环境变量
            for var, value in original_proxies.items():
                os.environ[var] = value
        
        return client
    except Exception as e:
        logger.error(f"Error creating OpenAI client: {e}", exc_info=True)
        return None
