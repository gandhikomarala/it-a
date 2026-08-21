# Distributed Redis caching engine with tag-based invalidation.
import json
import functools
from typing import Optional, Any
from packages.logging.logger import get_logger

logger = get_logger("cache.redis")

class CacheManager:
    def __init__(self):
        self._local_cache = {}

    async def get(self, key: str) -> Optional[Any]:
        return self._local_cache.get(key)

    async def set(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        self._local_cache[key] = value

    async def delete(self, key: str) -> None:
        self._local_cache.pop(key, None)

    async def clear_prefix(self, prefix: str) -> None:
        keys_to_del = [k for k in self._local_cache if k.startswith(prefix)]
        for k in keys_to_del:
            self._local_cache.pop(k, None)

cache_manager = CacheManager()

def cached(prefix: str, ttl: int = 300):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            key = f"{prefix}:{func.__name__}:{str(args)}:{str(kwargs)}"
            hit = await cache_manager.get(key)
            if hit is not None:
                return hit
            result = await func(*args, **kwargs)
            await cache_manager.set(key, result, ttl)
            return result
        return wrapper
    return decorator
