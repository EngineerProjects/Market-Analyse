"""
Request orchestration for LLM service.

This module provides advanced parallelization and request management
functionality for optimizing LLM service performance.
"""

import asyncio
import json
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from enterprise_ai.exceptions import LLMError
from enterprise_ai.logger import get_logger
from enterprise_ai.llm.service.utils import (
    RateLimiter,
    CircuitBreaker,
    RequestDeduplicator,
    RequestTracker,
)

# Initialize logger
logger = get_logger("llm.service.orchestration")

# Type variable for future results
T = TypeVar("T")


class RequestPriority(Enum):
    """Priority levels for request scheduling."""

    HIGH = 0
    NORMAL = 1
    LOW = 2
    BACKGROUND = 3


class RequestOrchestrator:
    """Advanced parallelization manager for LLM requests."""

    def __init__(
        self,
        max_concurrent_requests: int = 10,
        max_queue_size: int = 100,
        rate_limits: Optional[Dict[str, float]] = None,
        priority_levels: int = 4,
        adaptive_scaling: bool = True,
        max_retries: int = 3,
    ):
        """Initialize the request orchestrator.

        Args:
            max_concurrent_requests: Maximum concurrent requests
            max_queue_size: Maximum size of the request queue
            rate_limits: Rate limits by provider
            priority_levels: Number of priority levels
            adaptive_scaling: Whether to dynamically adjust concurrency
            max_retries: Maximum number of retries for failed requests
        """
        self.max_concurrent = max_concurrent_requests
        self.rate_limits = rate_limits or {}
        self.adaptive_scaling = adaptive_scaling
        self.max_retries = max_retries

        # Create thread pool with adaptive sizing
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_requests)

        # Create priority queues
        self.queues: List[
            asyncio.Queue[
                Tuple[
                    str,
                    Callable[..., Any],
                    Tuple[Any, ...],
                    Dict[str, Any],
                    asyncio.Future[Any],
                    str,
                    int,
                ]
            ]
        ] = [asyncio.Queue(maxsize=max_queue_size) for _ in range(priority_levels)]

        # Create rate limiters for each provider
        self.limiters = {
            provider: RateLimiter(rate=rate) for provider, rate in self.rate_limits.items()
        }

        # Create circuit breakers for each provider
        self.circuit_breakers = {provider: CircuitBreaker() for provider in self.rate_limits}

        # Request deduplication
        self.deduplicator = RequestDeduplicator()

        # Request tracking
        self.tracker = RequestTracker(max_concurrent=max_concurrent_requests)

        # Statistics
        self.stats = {
            "processed": 0,
            "queued": 0,
            "errors": 0,
            "retries": 0,
            "deduplicated": 0,
            "current_concurrency": 0,
            "avg_latency": 0.0,
            "latency_samples": 0,
        }
        self._stats_lock = threading.RLock()

        # Start worker
        self._shutdown_event = asyncio.Event()
        self._worker_task: Optional[asyncio.Task[None]] = None

    async def start(self) -> None:
        """Start the orchestrator."""
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._process_queue())

    async def shutdown(self) -> None:
        """Shutdown the orchestrator."""
        if self._worker_task is not None:
            self._shutdown_event.set()
            await self._worker_task
            self._worker_task = None
            self.executor.shutdown(wait=True)

    def generate_request_id(self, provider: str, *args: Any, **kwargs: Any) -> str:
        """Generate a unique request ID based on inputs.

        Args:
            provider: Provider name
            *args, **kwargs: Request arguments

        Returns:
            Unique request ID
        """
        # Create a deterministic representation of the request
        request_data = {
            "provider": provider,
            "args": args,
            "kwargs": kwargs,
        }

        # Serialize to JSON with stable ordering
        try:
            request_str = json.dumps(request_data, sort_keys=True)
        except (TypeError, ValueError):
            # If serialization fails, use a less reliable method
            request_str = f"{provider}:{str(args)}:{str(kwargs)}:{time.time()}"

        # Create hash
        import hashlib

        return hashlib.md5(request_str.encode()).hexdigest()

    async def submit(
        self,
        provider: str,
        fn: Callable[..., T],
        *args: Any,
        priority: Union[int, RequestPriority] = RequestPriority.NORMAL,
        deduplicate: bool = True,
        **kwargs: Any,
    ) -> Awaitable[T]:
        """Submit a request to the orchestrator.

        Args:
            provider: Provider name
            fn: Function to execute
            priority: Priority level (RequestPriority or int)
            deduplicate: Whether to deduplicate identical requests
            *args, **kwargs: Arguments for the function

        Returns:
            Future for the result
        """
        # Ensure orchestrator is started
        await self.start()

        # Convert enum to int if needed
        if isinstance(priority, RequestPriority):
            priority_value = priority.value
        else:
            priority_value = min(priority, len(self.queues) - 1)

        # Generate request ID
        request_id = self.generate_request_id(provider, *args, **kwargs)

        # Check deduplication cache
        if deduplicate:
            cached_result = self.deduplicator.get(request_id)
            if cached_result is not None:
                with self._stats_lock:
                    self.stats["deduplicated"] += 1
                future: asyncio.Future[T] = asyncio.Future()
                future.set_result(cached_result)
                return future

        # Create future for result
        future = asyncio.Future()

        # Check if request can be accepted
        if not self.tracker.start_request(request_id):
            future.set_exception(LLMError("Request rejected: system at capacity"))
            return future

        # Check circuit breaker
        if provider in self.circuit_breakers:
            circuit_breaker = self.circuit_breakers[provider]
            if not circuit_breaker.allow_request():
                self.tracker.end_request(request_id)
                future.set_exception(LLMError(f"Circuit breaker open for provider {provider}"))
                return future

        # Queue the request
        await self.queues[priority_value].put(
            (provider, fn, args, kwargs, future, request_id, 0)  # 0 is retry count
        )

        with self._stats_lock:
            self.stats["queued"] += 1

        return future

    async def _process_queue(self) -> None:
        """Process queued requests with priority handling."""
        while not self._shutdown_event.is_set():
            # Check each queue in priority order
            processed = False

            for priority, queue in enumerate(self.queues):
                if not queue.empty():
                    try:
                        (
                            provider,
                            fn,
                            args,
                            kwargs,
                            future,
                            request_id,
                            retry_count,
                        ) = await queue.get()

                        # Apply rate limiting if needed
                        if provider in self.limiters:
                            await self.limiters[provider].acquire()

                        # Submit to thread pool
                        with self._stats_lock:
                            self.stats["current_concurrency"] += 1

                        self._submit_to_pool(
                            provider, fn, args, kwargs, future, request_id, retry_count
                        )
                        processed = True
                        # Process one request per iteration
                        break
                    except Exception as e:
                        logger.error(f"Error processing queue item: {e}")
                        # Continue to next queue item

            if not processed:
                # No items in any queue, wait briefly
                await asyncio.sleep(0.01)

    def _submit_to_pool(
        self,
        provider: str,
        fn: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        future: asyncio.Future[Any],
        request_id: str,
        retry_count: int,
    ) -> None:
        """Submit a task to the thread pool with proper callbacks."""
        start_time = time.time()

        def _done_callback(thread_future: Future[Any]) -> None:
            """Handle completion of thread pool task."""
            try:
                result = thread_future.result()

                # Record successful completion
                if provider in self.circuit_breakers:
                    self.circuit_breakers[provider].record_success()

                # Update deduplication cache if successful
                self.deduplicator.add(request_id, result)

                # Set result on future
                if not future.done():
                    loop = asyncio.get_event_loop()
                    loop.call_soon_threadsafe(lambda: future.set_result(result))

                # Update metrics
                completion_time = time.time() - start_time
                with self._stats_lock:
                    self.stats["current_concurrency"] -= 1
                    self.stats["processed"] += 1

                    # Update average latency
                    total_latency = self.stats["avg_latency"] * self.stats["latency_samples"]
                    self.stats["latency_samples"] += 1
                    self.stats["avg_latency"] = (total_latency + completion_time) / self.stats[
                        "latency_samples"
                    ]

            except Exception as e:
                # Record failure
                if provider in self.circuit_breakers:
                    self.circuit_breakers[provider].record_failure()

                with self._stats_lock:
                    self.stats["current_concurrency"] -= 1
                    self.stats["errors"] += 1

                # Check if we should retry
                if retry_count < self.max_retries:
                    # Retry with backoff
                    retry_delay = 0.5 * (2**retry_count)

                    with self._stats_lock:
                        self.stats["retries"] += 1

                    # Re-queue with higher retry count
                    asyncio.run_coroutine_threadsafe(
                        self._retry_request(
                            provider,
                            fn,
                            args,
                            kwargs,
                            future,
                            request_id,
                            retry_count + 1,
                            retry_delay,
                        ),
                        asyncio.get_event_loop(),
                    )
                else:
                    # Max retries reached, set exception
                    if not future.done():
                        loop = asyncio.get_event_loop()

                        # Define a properly typed function
                        def set_exception_callback(exc: Exception = e) -> None:
                            future.set_exception(exc)

                        loop.call_soon_threadsafe(set_exception_callback)

                    # End request tracking
                    self.tracker.end_request(request_id)

            # End request tracking if no retry
            if retry_count >= self.max_retries:
                self.tracker.end_request(request_id)

            # Adapt concurrency if needed
            if self.adaptive_scaling:
                self._adapt_concurrency()

        # Submit to pool
        thread_future = self.executor.submit(fn, *args, **kwargs)
        thread_future.add_done_callback(_done_callback)

    async def _retry_request(
        self,
        provider: str,
        fn: Callable[..., Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        future: asyncio.Future[Any],
        request_id: str,
        retry_count: int,
        delay: float,
    ) -> None:
        """Retry a failed request with delay.

        Args:
            provider: Provider name
            fn: Function to execute
            args: Function arguments
            kwargs: Function keyword arguments
            future: Future for the result
            request_id: Unique request ID
            retry_count: Current retry count
            delay: Delay before retrying
        """
        # Wait before retry
        await asyncio.sleep(delay)

        # Queue for retry with highest priority
        await self.queues[0].put((provider, fn, args, kwargs, future, request_id, retry_count))

    def _adapt_concurrency(self) -> None:
        """Dynamically adjust concurrency based on performance."""
        if not self.adaptive_scaling:
            return

        with self._stats_lock:
            # Calculate error rate
            error_rate = self.stats["errors"] / max(1, self.stats["processed"])

            # Calculate utilization
            utilization = self.stats["current_concurrency"] / self.max_concurrent

            new_size = self.max_concurrent

            # Adjust based on error rate
            if error_rate > 0.1:  # More than 10% errors
                # Reduce concurrency to improve stability
                new_size = max(1, int(self.max_concurrent * 0.8))
            elif utilization > 0.8 and error_rate < 0.05:
                # High utilization with low error rate = increase capacity
                new_size = min(self.max_concurrent * 2, 100)

            # Apply changes if needed
            if new_size != self.max_concurrent:
                old_size = self.max_concurrent
                self.max_concurrent = new_size

                # Adjust executor if needed
                if new_size > old_size:
                    self.executor._max_workers = new_size

                logger.info(f"Adapted concurrency from {old_size} to {new_size}")
