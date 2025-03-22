"""
Caching system for LLM responses.

This module provides caching mechanisms to improve performance and reduce
API calls by reusing results for identical inputs.
"""

import json
import hashlib
import time
from typing import Any, Dict, List, Optional, Tuple, Union, cast
from pathlib import Path
from datetime import datetime, timedelta

from enterprise_ai.schema import Message
from enterprise_ai.logger import get_logger

# Initialize logger
logger = get_logger("llm.cache")


class LLMCache:
    """Base class for LLM response caching."""

    def __init__(self, ttl: Optional[int] = None):
        """Initialize the cache.

        Args:
            ttl: Time-to-live in seconds (None for no expiration)
        """
        self.ttl = ttl

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get an item from the cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
        """
        raise NotImplementedError("Subclasses must implement get()")

    def set(self, key: str, value: Dict[str, Any]) -> None:
        """Store an item in the cache.

        Args:
            key: Cache key
            value: Value to store
        """
        raise NotImplementedError("Subclasses must implement set()")

    def invalidate(self, key: str) -> bool:
        """Invalidate a cache entry.

        Args:
            key: Cache key

        Returns:
            True if the entry was removed, False if it wasn't found
        """
        raise NotImplementedError("Subclasses must implement invalidate()")

    def clear(self) -> None:
        """Clear all items from the cache."""
        raise NotImplementedError("Subclasses must implement clear()")

    def is_expired(self, timestamp: float) -> bool:
        """Check if a timestamp is expired.

        Args:
            timestamp: UNIX timestamp

        Returns:
            True if expired, False otherwise
        """
        if self.ttl is None:
            return False

        return time.time() > timestamp + self.ttl


class MemoryCache(LLMCache):
    """In-memory cache for LLM responses."""

    def __init__(self, ttl: Optional[int] = None, max_entries: int = 1000):
        """Initialize the memory cache.

        Args:
            ttl: Time-to-live in seconds (None for no expiration)
            max_entries: Maximum number of cache entries
        """
        super().__init__(ttl)
        self.max_entries = max_entries
        self.cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}  # key -> (timestamp, value)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get an item from the cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
        """
        if key not in self.cache:
            return None

        timestamp, value = self.cache[key]

        # Check if the entry is expired
        if self.is_expired(timestamp):
            # Remove expired entry
            del self.cache[key]
            return None

        return value

    def set(self, key: str, value: Dict[str, Any]) -> None:
        """Store an item in the cache.

        Args:
            key: Cache key
            value: Value to store
        """
        # Enforce maximum entries limit
        if len(self.cache) >= self.max_entries and key not in self.cache:
            # Remove the oldest entry
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][0])
            del self.cache[oldest_key]

        # Store value with current timestamp
        self.cache[key] = (time.time(), value)

    def invalidate(self, key: str) -> bool:
        """Invalidate a cache entry.

        Args:
            key: Cache key

        Returns:
            True if the entry was removed, False if it wasn't found
        """
        if key in self.cache:
            del self.cache[key]
            return True

        return False

    def clear(self) -> None:
        """Clear all items from the cache."""
        self.cache.clear()


class DiskCache(LLMCache):
    """Disk-based cache for LLM responses."""

    def __init__(
        self, cache_dir: Union[str, Path], ttl: Optional[int] = None, max_size_mb: int = 100
    ):
        """Initialize the disk cache.

        Args:
            cache_dir: Directory for cache files
            ttl: Time-to-live in seconds (None for no expiration)
            max_size_mb: Maximum cache size in MB
        """
        super().__init__(ttl)
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = max_size_mb * 1024 * 1024  # Convert to bytes

        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create index file if it doesn't exist
        self.index_path = self.cache_dir / "index.json"
        if not self.index_path.exists():
            self._write_index({})

    def _read_index(self) -> Dict[str, Dict[str, Any]]:
        """Read the cache index."""
        if not self.index_path.exists():
            return {}

        try:
            with open(self.index_path, "r") as f:
                result = json.load(f)
                return cast(Dict[str, Dict[str, Any]], result)
        except Exception as e:
            logger.error(f"Error reading cache index: {e}")
            return {}

    def _write_index(self, index: Dict[str, Dict[str, Any]]) -> None:
        """Write the cache index.

        Args:
            index: Cache index
        """
        try:
            with open(self.index_path, "w") as f:
                json.dump(index, f)
        except Exception as e:
            logger.error(f"Error writing cache index: {e}")

    def _get_cache_path(self, key: str) -> Path:
        """Get the path for a cache entry.

        Args:
            key: Cache key

        Returns:
            Path to the cache file
        """
        return self.cache_dir / f"{key}.json"

    def _get_cache_size(self) -> int:
        """Get the total size of the cache in bytes.

        Returns:
            Cache size in bytes
        """
        total_size = 0
        for file_path in self.cache_dir.glob("*.json"):
            if file_path.name != "index.json":
                total_size += file_path.stat().st_size

        return total_size

    def _enforce_size_limit(self) -> None:
        """Enforce the maximum cache size limit by removing old entries."""
        total_size = self._get_cache_size()

        if total_size <= self.max_size_bytes:
            return

        # Read index to get timestamps
        index = self._read_index()

        # Sort entries by timestamp (oldest first)
        sorted_entries = sorted(
            [(k, meta.get("timestamp", 0)) for k, meta in index.items()], key=lambda x: x[1]
        )

        # Remove entries until we're under the limit
        for key, _ in sorted_entries:
            if self._get_cache_size() <= self.max_size_bytes:
                break

            # Remove entry
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()

            # Remove from index
            if key in index:
                del index[key]

        # Write updated index
        self._write_index(index)

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get an item from the cache."""
        # Check the index first
        index = self._read_index()
        if key not in index:
            return None

        # Check if the entry is expired
        timestamp = index[key].get("timestamp", 0)
        if self.is_expired(timestamp):
            # Remove expired entry
            self.invalidate(key)
            return None

        # Read cache file
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            # File doesn't exist, fix the index
            self.invalidate(key)
            return None

        try:
            with open(cache_path, "r") as f:
                result = json.load(f)
                return cast(Dict[str, Any], result)
        except Exception as e:
            logger.error(f"Error reading cache file {cache_path}: {e}")
            return None

    def set(self, key: str, value: Dict[str, Any]) -> None:
        """Store an item in the cache.

        Args:
            key: Cache key
            value: Value to store
        """
        # Write cache file
        cache_path = self._get_cache_path(key)
        try:
            with open(cache_path, "w") as f:
                json.dump(value, f)
        except Exception as e:
            logger.error(f"Error writing cache file {cache_path}: {e}")
            return

        # Update index
        index = self._read_index()
        index[key] = {"timestamp": time.time(), "size": cache_path.stat().st_size}
        self._write_index(index)

        # Enforce size limit
        self._enforce_size_limit()

    def invalidate(self, key: str) -> bool:
        """Invalidate a cache entry.

        Args:
            key: Cache key

        Returns:
            True if the entry was removed, False if it wasn't found
        """
        # Remove cache file
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            cache_path.unlink()

        # Update index
        index = self._read_index()
        if key in index:
            del index[key]
            self._write_index(index)
            return True

        return False

    def clear(self) -> None:
        """Clear all items from the cache."""
        # Remove all cache files
        for file_path in self.cache_dir.glob("*.json"):
            if file_path.name != "index.json":
                file_path.unlink()

        # Reset index
        self._write_index({})


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


# Default cache instance (in-memory)
default_cache = MemoryCache(ttl=3600)  # 1 hour TTL
