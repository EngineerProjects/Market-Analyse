"""
LLM service components for Enterprise AI.

This module provides modularized components for the LLM service, including
configuration, orchestration, caching, metrics collection, and the core service.
"""

# Configuration classes
from enterprise_ai.llm.service.config import (
    CacheConfig,
    RequestTimeouts,
    ModelSelectionStrategy,
    OrchestratorConfig,
    LLMServiceConfig,
)

# Utility classes
from enterprise_ai.llm.service.utils import (
    RateLimiter,
    CircuitBreaker,
    CircuitState,
    RequestDeduplicator,
    RequestTracker,
)

# Caching
from enterprise_ai.llm.service.caching import HybridCache, generate_cache_key

# Metrics
from enterprise_ai.llm.service.metrics import ServiceMetrics

# Provider handling
from enterprise_ai.llm.service.pools import ProviderPool
from enterprise_ai.llm.service.registry import ProviderRegistration

# Request orchestration
from enterprise_ai.llm.service.orchestration import (
    RequestOrchestrator,
    RequestPriority,
)

# Core service
from enterprise_ai.llm.service.core import LLMService

# Default service and convenience functions
from enterprise_ai.llm.service.defaults import (
    get_default_llm_service,
    default_llm_service,
    complete,
    complete_stream,
    acomplete,
    acomplete_stream,
    get_conversation_manager,
    clear_cache,
    change_model,
    change_provider,
    get_available_models,
    get_similar_models,
    batch_complete,
    get_metrics,
    reset_metrics,
    shutdown,
)

__all__ = [
    # Configuration classes
    "CacheConfig",
    "RequestTimeouts",
    "ModelSelectionStrategy",
    "OrchestratorConfig",
    "LLMServiceConfig",
    # Utility classes
    "RateLimiter",
    "CircuitBreaker",
    "CircuitState",
    "RequestDeduplicator",
    "RequestTracker",
    # Caching
    "HybridCache",
    "generate_cache_key",
    # Metrics
    "ServiceMetrics",
    # Provider handling
    "ProviderPool",
    "ProviderRegistration",
    # Request orchestration
    "RequestOrchestrator",
    "RequestPriority",
    # Core service
    "LLMService",
    # Default service and helper functions
    "get_default_llm_service",
    "default_llm_service",
    "complete",
    "complete_stream",
    "acomplete",
    "acomplete_stream",
    "get_conversation_manager",
    "clear_cache",
    "change_model",
    "change_provider",
    "get_available_models",
    "get_similar_models",
    "batch_complete",
    "get_metrics",
    "reset_metrics",
    "shutdown",
]
