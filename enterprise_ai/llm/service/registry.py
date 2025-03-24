"""
Provider registration for LLM service.

This module provides functionality for registering and managing LLM providers.
"""

from typing import Any, Callable, Dict, Optional, Type, TypeVar, Generic

from enterprise_ai.llm.base import LLMProvider

# Type variable for LLM providers
P = TypeVar("P", bound=LLMProvider)


class ProviderRegistration(Generic[P]):
    """Registration information for an LLM provider."""

    def __init__(
        self,
        name: str,
        provider_class: Type[P],
        default_model: str,
        base_url: Optional[str] = None,
        capabilities: Optional[Dict[str, bool]] = None,
        create_func: Optional[Callable[..., P]] = None,
    ):
        """Initialize the provider registration.

        Args:
            name: Provider name
            provider_class: Provider class type
            default_model: Default model name
            base_url: Default base URL
            capabilities: Default capabilities dictionary
            create_func: Custom factory function for provider creation
        """
        self.name = name
        self.provider_class = provider_class
        self.default_model = default_model
        self.base_url = base_url
        self.capabilities = capabilities or {}
        self.create_func = create_func
