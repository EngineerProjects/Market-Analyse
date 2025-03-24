"""
Provider pooling for LLM service.

This module provides functionality for managing pools of provider instances
to improve resource utilization and performance.
"""

import threading
import time
from typing import Any, Callable, List, Set, Tuple

from enterprise_ai.llm.base import LLMProvider
from enterprise_ai.logger import get_logger

# Initialize logger
logger = get_logger("llm.service.pools")


class ProviderPool:
    """Pool of provider instances for load balancing."""

    def __init__(
        self,
        provider_factory: Callable[[], LLMProvider],
        min_size: int = 1,
        max_size: int = 5,
        idle_timeout: float = 300.0,
    ):
        """Initialize the provider pool.

        Args:
            provider_factory: Factory function to create new providers
            min_size: Minimum number of providers in the pool
            max_size: Maximum number of providers in the pool
            idle_timeout: Seconds to keep idle providers before removing
        """
        self.provider_factory = provider_factory
        self.min_size = min_size
        self.max_size = max_size
        self.idle_timeout = idle_timeout

        self.providers: List[Tuple[LLMProvider, float]] = []  # (provider, last_used_time)
        self.active_providers: Set[LLMProvider] = set()
        self._lock = threading.RLock()

        # Initialize minimum providers
        for _ in range(min_size):
            self._create_provider()

    def _create_provider(self) -> LLMProvider:
        """Create a new provider instance.

        Returns:
            New provider instance
        """
        provider = self.provider_factory()
        with self._lock:
            self.providers.append((provider, time.time()))
        return provider

    def _cleanup_idle(self) -> None:
        """Remove idle providers exceeding the minimum size."""
        current_time = time.time()
        with self._lock:
            # Only clean up if we're over minimum size
            if len(self.providers) > self.min_size:
                # Sort by last used time (oldest first)
                self.providers.sort(key=lambda p: p[1])

                # Remove idle providers beyond minimum size
                for idx, (provider, last_used) in enumerate(self.providers):
                    if idx >= self.min_size and provider not in self.active_providers:
                        if current_time - last_used > self.idle_timeout:
                            # Remove this provider
                            self.providers.pop(idx)

    def acquire(self) -> LLMProvider:
        """Acquire a provider from the pool.

        Returns:
            Provider instance
        """
        self._cleanup_idle()

        with self._lock:
            # Find an available provider
            for idx, (provider, _) in enumerate(self.providers):
                if provider not in self.active_providers:
                    # Update last used time
                    self.providers[idx] = (provider, time.time())
                    self.active_providers.add(provider)
                    return provider

            # No available provider, create new one if under max size
            if len(self.providers) < self.max_size:
                provider = self._create_provider()
                self.active_providers.add(provider)
                return provider

            # At capacity, find least recently used and return it
            self.providers.sort(key=lambda p: p[1])
            provider, _ = self.providers[0]
            # Update last used time
            self.providers[0] = (provider, time.time())
            self.active_providers.add(provider)
            return provider

    def release(self, provider: LLMProvider) -> None:
        """Release a provider back to the pool.

        Args:
            provider: Provider to release
        """
        with self._lock:
            if provider in self.active_providers:
                self.active_providers.remove(provider)

                # Update last used time
                for idx, (p, _) in enumerate(self.providers):
                    if p is provider:
                        self.providers[idx] = (p, time.time())
                        break
