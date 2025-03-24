"""
Hybrid caching implementation for LLM service.

This module provides a multi-level caching system combining memory and disk storage
for optimal performance and persistence.
"""

import json
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, cast, Literal

from enterprise_ai.llm.cache import LLMCache, MemoryCache, DiskCache
from enterprise_ai.schema import Message
from enterprise_ai.logger import get_logger

# Initialize logger
logger = get_logger("llm.service.caching")


class HybridCache(LLMCache):
    """Multi-level cache combining memory and disk storage."""

    def __init__(
        self,
        memory_ttl: Optional[int] = 3600,  # 1 hour
        disk_ttl: Optional[int] = 86400,  # 24 hours
        memory_max_entries: int = 1000,
        disk_cache_dir: Union[str, Path] = "cache",
        disk_max_size_mb: int = 500,
        promotion_policy: Literal["read", "write", "both"] = "both",
        synchronize_writes: bool = True,
    ):
        """Initialize the hybrid cache.

        Args:
            memory_ttl: Time-to-live for memory cache
            disk_ttl: Time-to-live for disk cache
            memory_max_entries: Maximum entries in memory cache
            disk_cache_dir: Directory for disk cache
            disk_max_size_mb: Maximum disk cache size in MB
            promotion_policy: When to promote items from disk to memory
            synchronize_writes: Whether to wait for disk writes to complete
        """
        super().__init__(ttl=memory_ttl)
        self.memory_cache = MemoryCache(ttl=memory_ttl, max_entries=memory_max_entries)
        self.disk_cache = DiskCache(
            cache_dir=disk_cache_dir, ttl=disk_ttl, max_size_mb=disk_max_size_mb
        )
        self.promotion_policy = promotion_policy
        self.synchronize_writes = synchronize_writes
        self._stats = {
            "memory_hits": 0,
            "disk_hits": 0,
            "misses": 0,
            "memory_sets": 0,
            "disk_sets": 0,
        }
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get an item from the cache with multi-level lookup.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found
        """
        # Try memory cache first (fastest)
        result = self.memory_cache.get(key)
        if result is not None:
            with self._lock:
                self._stats["memory_hits"] += 1
            return result

        # Try disk cache second (slower)
        result = self.disk_cache.get(key)
        if result is not None:
            with self._lock:
                self._stats["disk_hits"] += 1

            # Promote to memory cache if policy allows
            if self.promotion_policy in ["read", "both"]:
                self.memory_cache.set(key, result)

            return result

        # Not found in any cache
        with self._lock:
            self._stats["misses"] += 1
        return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        """Store an item in the cache with multi-level strategy.

        Args:
            key: Cache key
            value: Value to store
        """
        # Always store in memory for fast access
        self.memory_cache.set(key, value)
        with self._lock:
            self._stats["memory_sets"] += 1

        # Store in disk for persistence based on policy
        if self.promotion_policy in ["write", "both"]:
            if self.synchronize_writes:
                # Synchronous write
                self.disk_cache.set(key, value)
                with self._lock:
                    self._stats["disk_sets"] += 1
            else:
                # Asynchronous write
                def _background_set() -> None:
                    self.disk_cache.set(key, value)
                    with self._lock:
                        self._stats["disk_sets"] += 1

                self._executor.submit(_background_set)

    def invalidate(self, key: str) -> bool:
        """Invalidate a cache entry.

        Args:
            key: Cache key

        Returns:
            True if entry was found and removed
        """
        memory_result = self.memory_cache.invalidate(key)
        disk_result = self.disk_cache.invalidate(key)
        return memory_result or disk_result

    def clear(self) -> None:
        """Clear all items from the cache."""
        self.memory_cache.clear()
        self.disk_cache.clear()

        with self._lock:
            self._stats = {
                "memory_hits": 0,
                "disk_hits": 0,
                "misses": 0,
                "memory_sets": 0,
                "disk_sets": 0,
            }

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics.

        Returns:
            Dictionary of cache statistics
        """
        with self._lock:
            return self._stats.copy()

    def is_expired(self, timestamp: float) -> bool:
        """Check if a timestamp is expired.

        Args:
            timestamp: UNIX timestamp

        Returns:
            True if expired, False otherwise
        """
        # Use memory cache TTL for expiration check
        return self.memory_cache.is_expired(timestamp)


def generate_cache_key(model: str, messages: List[Message], request_params: Dict[str, Any]) -> str:
    """Generate a cache key for LLM requests.

    Args:
        model: Model name
        messages: List of messages
        request_params: Request parameters

    Returns:
        Cache key
    """
    # Create a serializable representation of the request
    serializable_messages = [msg.to_dict() for msg in messages]

    # Filter out non-serializable params
    filtered_params = {}
    for k, v in request_params.items():
        if k in ("stream", "tools", "tool_choice"):
            # Skip non-cacheable params
            continue

        try:
            # Test if serializable
            json.dumps({k: v})
            filtered_params[k] = v
        except (TypeError, OverflowError):
            # Skip non-serializable params
            logger.debug(f"Skipping non-serializable parameter: {k}")

    # Create a dictionary to hash
    request_dict = {"model": model, "messages": serializable_messages, "params": filtered_params}

    # Calculate hash
    request_json = json.dumps(request_dict, sort_keys=True)
    return hashlib.sha256(request_json.encode()).hexdigest()
