"""
Utility classes for LLM service.

This module provides utility classes for request management, flow control,
and optimization in the LLM service.
"""

import threading
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Not allowing requests
    HALF_OPEN = "half_open"  # Testing recovery


class RateLimiter:
    """Rate limiter for API requests."""

    def __init__(self, rate: float, period: float = 1.0, burst: int = 1):
        """Initialize the rate limiter.

        Args:
            rate: Requests per period
            period: Time period in seconds
            burst: Burst capacity (how many requests can be made at once)
        """
        self.rate = rate
        self.period = period
        self.burst = burst
        self.tokens: float = float(burst)
        self.last_update = time.time()
        self._lock = threading.RLock()

    def _update_tokens(self) -> None:
        """Update available tokens based on elapsed time."""
        current_time = time.time()
        elapsed = current_time - self.last_update
        new_tokens = elapsed * self.rate / self.period

        with self._lock:
            self.tokens = float(min(self.burst, self.tokens + new_tokens))
            self.last_update = current_time

    def try_acquire(self) -> bool:
        """Try to acquire a token without blocking.

        Returns:
            True if token acquired, False otherwise
        """
        self._update_tokens()

        with self._lock:
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False

    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        import asyncio  # Import here to avoid circular imports

        while True:
            if self.try_acquire():
                return

            # Calculate wait time for the next token
            with self._lock:
                wait_time = (1 - self.tokens) * self.period / self.rate

            await asyncio.sleep(max(0.01, wait_time))


class CircuitBreaker:
    """Circuit breaker to prevent cascading failures."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
    ):
        """Initialize the circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            half_open_max_calls: Maximum calls in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self.state = CircuitState.CLOSED
        self.failure_count: int = 0
        self.last_failure_time: float = 0.0
        self.half_open_calls = 0
        self._lock = threading.RLock()

    def record_success(self) -> None:
        """Record a successful operation."""
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.half_open_calls += 1
                if self.half_open_calls >= self.half_open_max_calls:
                    # Recovery successful
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.half_open_calls = 0
            elif self.state == CircuitState.CLOSED:
                # Reset any accumulated failures
                self.failure_count = 0

    def record_failure(self) -> None:
        """Record a failed operation."""
        with self._lock:
            current_time = time.time()
            self.last_failure_time = current_time

            if self.state == CircuitState.HALF_OPEN:
                # Failed during recovery attempt
                self.state = CircuitState.OPEN
                self.half_open_calls = 0
            elif self.state == CircuitState.CLOSED:
                self.failure_count += 1
                if self.failure_count >= self.failure_threshold:
                    # Too many failures, open the circuit
                    self.state = CircuitState.OPEN
                    from enterprise_ai.logger import get_logger

                    logger = get_logger("llm.service.utils")
                    logger.warning("Circuit breaker opened due to failures")

    def allow_request(self) -> bool:
        """Check if a request should be allowed.

        Returns:
            True if request is allowed, False otherwise
        """
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:
                current_time = time.time()
                if current_time - self.last_failure_time >= self.recovery_timeout:
                    # Try recovery
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    return True
                return False

            if self.state == CircuitState.HALF_OPEN:
                return self.half_open_calls < self.half_open_max_calls

            return False


class RequestDeduplicator:
    """Deduplicate identical requests within a time window."""

    def __init__(self, ttl: float = 5.0, max_size: int = 1000):
        """Initialize the request deduplicator.

        Args:
            ttl: Time-to-live for deduplication entries in seconds
            max_size: Maximum entries in the deduplication cache
        """
        self.ttl = ttl
        self.max_size = max_size
        self.cache: Dict[str, Tuple[float, Any]] = {}
        self._lock = threading.RLock()

    def _clean_expired(self) -> None:
        """Remove expired entries from the cache."""
        current_time = time.time()
        with self._lock:
            expired_keys = [
                key
                for key, (timestamp, _) in self.cache.items()
                if current_time - timestamp > self.ttl
            ]

            for key in expired_keys:
                del self.cache[key]

    def add(self, key: str, value: Any) -> None:
        """Add a result to the deduplication cache.

        Args:
            key: Deduplication key
            value: Result value
        """
        with self._lock:
            # Enforce maximum size
            if len(self.cache) >= self.max_size and key not in self.cache:
                # Remove oldest item
                oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][0])
                del self.cache[oldest_key]

            # Store with current timestamp
            self.cache[key] = (time.time(), value)

    def get(self, key: str) -> Optional[Any]:
        """Get a result from the deduplication cache if available.

        Args:
            key: Deduplication key

        Returns:
            Cached result or None if not found
        """
        self._clean_expired()

        with self._lock:
            if key in self.cache:
                timestamp, value = self.cache[key]
                return value

        return None


class RequestTracker:
    """Track in-flight requests for load management."""

    def __init__(self, max_concurrent: int = 100):
        """Initialize the request tracker.

        Args:
            max_concurrent: Maximum concurrent requests
        """
        self.max_concurrent = max_concurrent
        self.active_requests = 0
        self.seen_requests: Set[str] = set()
        self._lock = threading.RLock()

    def start_request(self, request_id: str) -> bool:
        """Start tracking a request.

        Args:
            request_id: Unique request identifier

        Returns:
            True if request can proceed, False if overloaded
        """
        with self._lock:
            # Check if already at capacity
            if self.active_requests >= self.max_concurrent and request_id not in self.seen_requests:
                return False

            # Track the request
            self.seen_requests.add(request_id)
            self.active_requests += 1
            return True

    def end_request(self, request_id: str) -> None:
        """End tracking a request.

        Args:
            request_id: Unique request identifier
        """
        with self._lock:
            if request_id in self.seen_requests:
                self.seen_requests.remove(request_id)
                self.active_requests = max(0, self.active_requests - 1)
