"""
Metrics collection for the LLM service.

This module provides metrics collection and tracking functionality for
monitoring LLM service performance and usage.
"""

import threading
import time
from typing import Any, Dict


class ServiceMetrics:
    """Metrics collection for the LLM service."""

    def __init__(self) -> None:
        """Initialize metrics collector."""
        self.request_count = 0
        self.error_count = 0
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.token_count = 0
        self.avg_response_time = 0.0
        self.start_time = time.time()
        self.latency_samples = 0
        self.provider_metrics: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def record_request(self, provider: str, cached: bool = False) -> None:
        """Record a request.

        Args:
            provider: Provider name
            cached: Whether the request was served from cache
        """
        with self._lock:
            self.request_count += 1
            if cached:
                self.cache_hit_count += 1
            else:
                self.cache_miss_count += 1

            # Update provider-specific metrics
            if provider not in self.provider_metrics:
                self.provider_metrics[provider] = {
                    "request_count": 0,
                    "error_count": 0,
                    "cache_hit_count": 0,
                    "cache_miss_count": 0,
                }

            self.provider_metrics[provider]["request_count"] += 1
            if cached:
                self.provider_metrics[provider]["cache_hit_count"] += 1
            else:
                self.provider_metrics[provider]["cache_miss_count"] += 1

    def record_error(self, provider: str) -> None:
        """Record an error.

        Args:
            provider: Provider name
        """
        with self._lock:
            self.error_count += 1

            # Update provider-specific metrics
            if provider not in self.provider_metrics:
                self.provider_metrics[provider] = {
                    "request_count": 0,
                    "error_count": 1,
                    "cache_hit_count": 0,
                    "cache_miss_count": 0,
                }
            else:
                self.provider_metrics[provider]["error_count"] += 1

    def record_tokens(self, count: int) -> None:
        """Record token usage.

        Args:
            count: Number of tokens used
        """
        with self._lock:
            self.token_count += count

    def record_response_time(self, time_seconds: float) -> None:
        """Record a response time.

        Args:
            time_seconds: Response time in seconds
        """
        with self._lock:
            # Update average using running average formula
            total_latency = self.avg_response_time * self.latency_samples
            self.latency_samples += 1
            self.avg_response_time = (total_latency + time_seconds) / self.latency_samples

    def get_metrics(self) -> Dict[str, Any]:
        """Get all current metrics.

        Returns:
            Dictionary of metrics
        """
        with self._lock:
            uptime = time.time() - self.start_time
            cache_hit_ratio = 0.0
            if self.request_count > 0:
                cache_hit_ratio = self.cache_hit_count / self.request_count

            error_rate = 0.0
            if self.request_count > 0:
                error_rate = self.error_count / self.request_count

            return {
                "request_count": self.request_count,
                "error_count": self.error_count,
                "cache_hit_count": self.cache_hit_count,
                "cache_miss_count": self.cache_miss_count,
                "token_count": self.token_count,
                "avg_response_time": self.avg_response_time,
                "cache_hit_ratio": cache_hit_ratio,
                "error_rate": error_rate,
                "uptime_seconds": uptime,
                "requests_per_minute": (self.request_count * 60) / max(1, uptime),
                "latency_samples": self.latency_samples,
                "providers": self.provider_metrics,
            }

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.request_count = 0
            self.error_count = 0
            self.cache_hit_count = 0
            self.cache_miss_count = 0
            self.token_count = 0
            self.avg_response_time = 0.0
            self.latency_samples = 0
            self.provider_metrics = {}
            self.start_time = time.time()
