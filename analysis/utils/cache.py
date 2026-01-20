"""缓存工具：提供统一的缓存接口。"""

import hashlib
import json
from functools import wraps
from typing import Any, Callable, Optional, TypeVar

from django.core.cache import cache

from ..config.settings import CacheConfig

T = TypeVar("T")


def _make_cache_key(prefix: str, *args, **kwargs) -> str:
    """生成缓存键"""
    key_parts = [prefix]
    if args:
        key_parts.append(str(hash(str(args))))
    if kwargs:
        sorted_kwargs = sorted(kwargs.items())
        key_parts.append(str(hash(str(sorted_kwargs))))
    key_string = ":".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()


def cached_result(
    timeout: int = CacheConfig.FEATURE_SNAPSHOT_TIMEOUT,
    key_prefix: str = "",
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    缓存函数结果的装饰器
    
    Args:
        timeout: 缓存超时时间（秒）
        key_prefix: 缓存键前缀
    
    Returns:
        装饰器函数
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            cache_key = _make_cache_key(key_prefix or func.__name__, *args, **kwargs)
            cached_value = cache.get(cache_key)
            
            if cached_value is not None:
                return cached_value
            
            result = func(*args, **kwargs)
            cache.set(cache_key, result, timeout=timeout)
            return result
        
        return wrapper
    return decorator


def get_cached(key: str, default: Any = None) -> Any:
    """获取缓存值"""
    return cache.get(key, default)


def set_cached(key: str, value: Any, timeout: int = CacheConfig.FEATURE_SNAPSHOT_TIMEOUT) -> None:
    """设置缓存值"""
    cache.set(key, value, timeout=timeout)


def delete_cached(key: str) -> None:
    """删除缓存值"""
    cache.delete(key)


def clear_pattern(pattern: str) -> None:
    """清除匹配模式的缓存（需要Redis支持）"""
    try:
        from django.core.cache import caches
        redis_cache = caches["default"]
        if hasattr(redis_cache, "delete_pattern"):
            redis_cache.delete_pattern(pattern)
    except Exception:
        pass  # 如果Redis不可用，忽略
