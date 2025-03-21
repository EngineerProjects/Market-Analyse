"""
Advanced retry strategies for LLM service requests.

This module provides sophisticated retry mechanisms with various backoff strategies,
designed to handle different types of transient failures in LLM API calls.
"""

import time
import random
import asyncio
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar, Union, cast

from enterprise_ai.logger import get_logger

# Initialize logger
logger = get_logger("llm.retry")

# Type variable for decorated functions
F = TypeVar("F", bound=Callable[..., Any])


class BackoffStrategy(str, Enum):
    """Strategies for retry backoff."""

    CONSTANT = "constant"  # Fixed delay between retries
    LINEAR = "linear"  # Delay increases linearly
    EXPONENTIAL = "exponential"  # Delay increases exponentially
    FIBONACCI = "fibonacci"  # Delay follows fibonacci sequence
    JITTER = "jitter"  # Exponential backoff with random jitter


class RetryableException(Exception):
    """Base class for exceptions that should trigger a retry."""

    pass


class RetryConfig:
    """Configuration for retry behavior."""

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL,
        jitter_factor: float = 0.25,
        retryable_exceptions: Optional[Set[type]] = None,
        retryable_status_codes: Optional[Set[int]] = None,
        retry_on_timeout: bool = True,
        retry_on_connection_error: bool = True,
        retry_on_server_error: bool = True,
        retry_on_rate_limit: bool = True,
    ):
        """Initialize retry configuration.

        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay between retries in seconds
            max_delay: Maximum delay between retries in seconds
            backoff_strategy: Strategy for calculating delay between retries
            jitter_factor: Random factor for jitter strategy (0.0 to 1.0)
            retryable_exceptions: Set of exception types to retry on
            retryable_status_codes: Set of HTTP status codes to retry on
            retry_on_timeout: Whether to retry on timeout errors
            retry_on_connection_error: Whether to retry on connection errors
            retry_on_server_error: Whether to retry on 5xx server errors
            retry_on_rate_limit: Whether to retry on rate limit errors (429)
        """
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_strategy = backoff_strategy
        self.jitter_factor = min(max(jitter_factor, 0.0), 1.0)  # Clamp to [0.0, 1.0]

        # Build default retryable exceptions
        self.retryable_exceptions = retryable_exceptions or set()

        # Add common exception types based on configuration flags
        if retry_on_timeout:
            self.retryable_exceptions.update({TimeoutError})
            # Add httpx.TimeoutException if available
            try:
                import httpx

                self.retryable_exceptions.add(httpx.TimeoutException)
            except (ImportError, AttributeError):
                pass

        if retry_on_connection_error:
            self.retryable_exceptions.update({ConnectionError, ConnectionRefusedError})
            # Add httpx.ConnectError if available
            try:
                import httpx

                self.retryable_exceptions.add(httpx.ConnectError)
            except (ImportError, AttributeError):
                pass

        # Add RetryableException and its subclasses
        self.retryable_exceptions.add(RetryableException)

        # Build list of retryable status codes
        self.retryable_status_codes = retryable_status_codes or set()

        if retry_on_server_error:
            # Add all 5xx status codes
            self.retryable_status_codes.update(range(500, 600))

        if retry_on_rate_limit:
            # Add 429 (Too Many Requests)
            self.retryable_status_codes.add(429)

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a retry attempt.

        Args:
            attempt: Current attempt number (1-based)

        Returns:
            Delay in seconds
        """
        if attempt <= 0:
            return 0.0

        base_delay = 0.0

        # Calculate base delay according to strategy
        if self.backoff_strategy == BackoffStrategy.CONSTANT:
            base_delay = self.initial_delay

        elif self.backoff_strategy == BackoffStrategy.LINEAR:
            base_delay = self.initial_delay * attempt

        elif self.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            base_delay = self.initial_delay * (2 ** (attempt - 1))

        elif self.backoff_strategy == BackoffStrategy.FIBONACCI:
            # Calculate Fibonacci number (simplified algorithm)
            a, b = 1, 1
            for _ in range(attempt - 1):
                a, b = b, a + b
            base_delay = self.initial_delay * a

        elif self.backoff_strategy == BackoffStrategy.JITTER:
            # Exponential backoff with jitter
            exp_delay = self.initial_delay * (2 ** (attempt - 1))
            jitter = random.uniform(-self.jitter_factor, self.jitter_factor) * exp_delay
            base_delay = exp_delay + jitter

        # Clamp to max delay
        return min(base_delay, self.max_delay)

    def should_retry(self, exception: Exception, status_code: Optional[int] = None) -> bool:
        """Determine if a retry should be attempted for a given exception.

        Args:
            exception: The exception that occurred
            status_code: Optional HTTP status code

        Returns:
            True if should retry, False otherwise
        """
        # Check for retryable exception types
        for exc_type in self.retryable_exceptions:
            if isinstance(exception, exc_type):
                return True

        # Check for retryable status codes
        if status_code is not None and status_code in self.retryable_status_codes:
            return True

        return False


def with_retry(config: Optional[RetryConfig] = None) -> Callable[[F], F]:
    """Decorator for functions that should be retried on failure.

    Args:
        config: RetryConfig instance (uses default if None)

    Returns:
        Decorated function
    """
    retry_config = config or RetryConfig()

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrapper for synchronous functions."""
            last_exception = None

            for attempt in range(1, retry_config.max_retries + 2):  # +1 for initial attempt
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    # Extract status code if available
                    status_code = None
                    if hasattr(e, "status_code"):
                        status_code = getattr(e, "status_code")
                    elif hasattr(e, "response") and hasattr(e.response, "status_code"):
                        status_code = e.response.status_code

                    # Check if we should retry
                    if attempt > retry_config.max_retries or not retry_config.should_retry(
                        e, status_code
                    ):
                        logger.debug(f"Not retrying: {e}, status={status_code}, attempt={attempt}")
                        raise

                    # Calculate delay
                    delay = retry_config.calculate_delay(attempt)

                    logger.warning(
                        f"Retry {attempt}/{retry_config.max_retries} after {delay:.2f}s: {e}, status={status_code}"
                    )

                    # Wait and retry
                    time.sleep(delay)

            # This should never be reached, but just in case
            raise last_exception or RuntimeError("Retry logic failed")

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrapper for asynchronous functions."""
            last_exception = None

            for attempt in range(1, retry_config.max_retries + 2):  # +1 for initial attempt
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e

                    # Extract status code if available
                    status_code = None
                    if hasattr(e, "status_code"):
                        status_code = getattr(e, "status_code")
                    elif hasattr(e, "response") and hasattr(e.response, "status_code"):
                        status_code = e.response.status_code

                    # Check if we should retry
                    if attempt > retry_config.max_retries or not retry_config.should_retry(
                        e, status_code
                    ):
                        logger.debug(f"Not retrying: {e}, status={status_code}, attempt={attempt}")
                        raise

                    # Calculate delay
                    delay = retry_config.calculate_delay(attempt)

                    logger.warning(
                        f"Retry {attempt}/{retry_config.max_retries} after {delay:.2f}s: {e}, status={status_code}"
                    )

                    # Wait and retry
                    await asyncio.sleep(delay)

            # This should never be reached, but just in case
            raise last_exception or RuntimeError("Retry logic failed")

        # Choose appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        else:
            return cast(F, wrapper)

    return decorator


# Specialized retry configs
DEFAULT_RETRY_CONFIG = RetryConfig()

# More retries for rate limiting
RATE_LIMIT_RETRY_CONFIG = RetryConfig(
    max_retries=5,
    initial_delay=2.0,
    max_delay=120.0,
    backoff_strategy=BackoffStrategy.EXPONENTIAL,
    retry_on_rate_limit=True,
)

# Fast retries for transient errors
TRANSIENT_RETRY_CONFIG = RetryConfig(
    max_retries=3,
    initial_delay=0.5,
    max_delay=5.0,
    backoff_strategy=BackoffStrategy.JITTER,
    retry_on_connection_error=True,
    retry_on_timeout=True,
)
