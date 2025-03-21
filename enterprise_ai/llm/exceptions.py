"""
Exception classes for LLM module.

This module extends the Enterprise AI exception hierarchy with LLM-specific exceptions.
"""

from typing import Any, Optional, Dict

from enterprise_ai.exceptions import LLMError, TokenLimitExceeded, ModelNotAvailable, APIError


class TokenCountError(LLMError):
    """Exception raised when there's an error counting tokens."""

    def __init__(self, message: str = "Error counting tokens", model: Optional[str] = None) -> None:
        """Initialize the exception.

        Args:
            message: Error message
            model: Optional model name
        """
        self.model = model
        super().__init__(f"{message} for model {model}" if model else message)


class ProviderNotSupportedError(LLMError):
    """Exception raised when a requested provider is not supported."""

    def __init__(self, provider: str) -> None:
        """Initialize the exception.

        Args:
            provider: Provider name
        """
        self.provider = provider
        super().__init__(f"Provider not supported: {provider}")


class ParameterError(LLMError):
    """Exception raised when there's an error with request parameters."""

    def __init__(self, message: str, parameter: Optional[str] = None) -> None:
        """Initialize the exception.

        Args:
            message: Error message
            parameter: Optional parameter name
        """
        self.parameter = parameter
        msg = f"Parameter error{f' for {parameter}' if parameter else ''}: {message}"
        super().__init__(msg)


class ContextWindowExceededError(TokenLimitExceeded):
    """Exception raised when the context window size is exceeded."""

    def __init__(self, model: str, token_count: int, max_tokens: int) -> None:
        """Initialize the exception.

        Args:
            model: Model name
            token_count: Current token count
            max_tokens: Maximum token limit
        """
        self.model = model
        self.token_count = token_count
        self.max_tokens = max_tokens
        super().__init__(
            f"Context window exceeded for {model}: {token_count} tokens (max: {max_tokens})"
        )


class ImageProcessingError(LLMError):
    """Exception raised when there's an error processing images."""

    def __init__(self, message: str, source: Optional[str] = None) -> None:
        """Initialize the exception.

        Args:
            message: Error message
            source: Optional source of the image
        """
        self.source = source
        msg = f"Image processing error{f' for {source}' if source else ''}: {message}"
        super().__init__(msg)


class ModelCapabilityError(LLMError):
    """Exception raised when a model doesn't support a requested capability."""

    def __init__(self, model: str, capability: str) -> None:
        """Initialize the exception.

        Args:
            model: Model name
            capability: Capability name
        """
        self.model = model
        self.capability = capability
        super().__init__(f"Model {model} does not support capability: {capability}")
