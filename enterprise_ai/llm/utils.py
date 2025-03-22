"""
Utility functions for LLM integration.

This module provides utilities for token counting, image processing,
and retry mechanisms for LLM API calls.
"""

import time
import random
import base64
import asyncio
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, TypeVar, cast

# Third-party imports
import requests
import httpx

from enterprise_ai.schema import Message, Role
from enterprise_ai.logger import get_logger
from enterprise_ai.llm.exceptions import TokenCountError, ImageProcessingError

# Initialize logger
logger = get_logger("llm.utils")

# Type variable for functions
F = TypeVar("F", bound=Callable[..., Any])


def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 20.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
) -> Callable[[F], F]:
    """Decorator for retrying functions with exponential backoff.

    Args:
        max_retries: Maximum number of retries
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Factor to increase delay with each retry
        retryable_exceptions: Tuple of exceptions to retry on (default: connection errors)

    Returns:
        Decorated function
    """
    if retryable_exceptions is None:
        retryable_exceptions = cast(
            Tuple[Type[Exception], ...],
            (
                requests.exceptions.RequestException,
                httpx.HTTPError,
                ConnectionError,
                TimeoutError,
            ),
        )

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            delay = initial_delay
            last_exception: Optional[BaseException] = None

            for retry in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if retry >= max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded: {str(e)}")
                        raise

                    # Calculate backoff with jitter
                    jitter = random.uniform(0.8, 1.2)
                    sleep_time = min(delay * jitter, max_delay)

                    logger.warning(
                        f"Retry {retry + 1}/{max_retries} after {sleep_time:.2f}s due to: {str(e)}"
                    )

                    # Sleep and increase delay
                    time.sleep(sleep_time)
                    delay = min(delay * backoff_factor, max_delay)

            # This should never be reached, but just in case
            if last_exception is not None:
                raise last_exception
            raise RuntimeError("Retry logic failed")

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Similar changes to the async version
            delay = initial_delay
            last_exception: Optional[BaseException] = None

            for retry in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if retry >= max_retries:
                        logger.error(f"Max retries ({max_retries}) exceeded: {str(e)}")
                        raise

                    # Calculate backoff with jitter
                    jitter = random.uniform(0.8, 1.2)
                    sleep_time = min(delay * jitter, max_delay)

                    logger.warning(
                        f"Retry {retry + 1}/{max_retries} after {sleep_time:.2f}s due to: {str(e)}"
                    )

                    # Sleep and increase delay
                    await asyncio.sleep(sleep_time)
                    delay = min(delay * backoff_factor, max_delay)

            # This should never be reached, but just in case
            if last_exception is not None:
                raise last_exception
            raise RuntimeError("Retry logic failed")

        # Check if function is async or sync
        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        return cast(F, wrapper)

    return decorator


class TokenCounter:
    """Utility for counting tokens for different models and providers."""

    # Token counters for different providers and models
    _openai_tokenizers: Dict[str, Any] = {}

    @classmethod
    def count_for_model(cls, model: str, messages: List[Message]) -> int:
        """Count tokens for a specific model.

        Args:
            model: Model name
            messages: List of messages to count tokens for

        Returns:
            Number of tokens

        Raises:
            TokenCountError: If token counting fails or model is not supported
        """
        try:
            if model.startswith(("gpt-", "text-")):
                return cls._count_openai(model, messages)
            elif model.startswith("claude"):
                return cls._count_anthropic(model, messages)
            elif any(m in model.lower() for m in ["llama", "mistral", "gemma", "phi"]):
                return cls._count_ollama(model, messages)
            else:
                # Fallback to a simple character-based estimation
                logger.warning(f"No specific tokenizer for {model}, using simple estimation")
                return cls._count_simple(messages)
        except Exception as e:
            logger.error(f"Error counting tokens: {str(e)}")
            raise TokenCountError(f"Failed to count tokens: {str(e)}", model)

    @classmethod
    def _count_openai(cls, model: str, messages: List[Message]) -> int:
        """Count tokens using OpenAI's tiktoken library.

        Args:
            model: Model name
            messages: List of messages to count tokens for

        Returns:
            Number of tokens

        Raises:
            TokenCountError: If token counting fails
        """
        try:
            # Import here to avoid dependency on tiktoken if not used
            import tiktoken

            if model not in cls._openai_tokenizers:
                encoding = tiktoken.encoding_for_model(model)
                cls._openai_tokenizers[model] = encoding
            else:
                encoding = cls._openai_tokenizers[model]

            # Add tokens for each message according to OpenAI's formula
            num_tokens = 0
            for message in messages:
                # Add message metadata tokens
                num_tokens += 4  # Every message has a 4-token overhead

                # Add role tokens
                num_tokens += len(encoding.encode(message.role.value))

                # Add content tokens
                if message.content:
                    num_tokens += len(encoding.encode(message.content))

                # Add tokens for image attachments (approximate)
                if message.base64_image:
                    # Assume high res image (512x512) at around 1024 tokens
                    num_tokens += 1024

                # Add tokens for tool calls
                if message.tool_calls:
                    for tool_call in message.tool_calls:
                        num_tokens += len(encoding.encode(tool_call.function.name))
                        num_tokens += len(encoding.encode(tool_call.function.arguments))

                # Add tokens for name
                if message.name:
                    num_tokens += len(encoding.encode(message.name))
                    num_tokens += 1  # Token for name field itself

            # Add trailing tokens
            num_tokens += 2  # For the system message indicator

            return num_tokens

        except Exception as e:
            logger.error(f"Error counting tokens for OpenAI model {model}: {e}")
            raise TokenCountError(f"Failed to count tokens: {e}", model)

    @classmethod
    def _count_anthropic(cls, model: str, messages: List[Message]) -> int:
        """Count tokens for Anthropic models.

        This is an approximation as Anthropic doesn't provide an official tokenizer.

        Args:
            model: Model name
            messages: List of messages to count tokens for

        Returns:
            Number of tokens
        """
        try:
            # Import here to avoid dependency on tiktoken if not used
            import tiktoken

            # Anthropic uses a similar tokenizer to OpenAI
            # We'll use cl100k_base for estimation
            if "cl100k_base" not in cls._openai_tokenizers:
                encoding = tiktoken.get_encoding("cl100k_base")
                cls._openai_tokenizers["cl100k_base"] = encoding
            else:
                encoding = cls._openai_tokenizers["cl100k_base"]

            num_tokens = 0

            # Anthropic has a unique format
            for message in messages:
                # Add role-specific tokens
                if message.role == Role.SYSTEM:
                    num_tokens += 15  # Approximate overhead for system message
                else:
                    num_tokens += 5  # Regular message overhead

                # Add content tokens
                if message.content:
                    num_tokens += len(encoding.encode(message.content))

                # Add image tokens (approximation)
                if message.base64_image:
                    num_tokens += 1024  # Assume high res image

            # Add model overhead
            num_tokens += 20

            return num_tokens

        except Exception as e:
            logger.error(f"Error counting tokens for Anthropic model {model}: {e}")
            raise TokenCountError(f"Failed to count tokens: {e}", model)

    @classmethod
    def _count_ollama(cls, model: str, messages: List[Message]) -> int:
        """Count tokens for Ollama models.

        This is a rough approximation since Ollama doesn't expose a tokenizer.

        Args:
            model: Model name
            messages: List of messages to count tokens for

        Returns:
            Number of tokens
        """
        # Ollama's tokenization varies by model, but a rough estimate
        # is 1 token per 4 characters
        total_chars = 0

        for message in messages:
            if message.content:
                total_chars += len(message.content)

            # Add message metadata
            total_chars += len(message.role.value)

            if message.name:
                total_chars += len(message.name)

        # Rough approximation: 4 chars per token
        return total_chars // 4 + 20  # Add overhead

    @classmethod
    def _count_simple(cls, messages: List[Message]) -> int:
        """Simple character-based token estimation.

        Args:
            messages: List of messages to count tokens for

        Returns:
            Estimated number of tokens
        """
        total_chars = 0

        for message in messages:
            if message.content:
                total_chars += len(message.content)

            # Add message metadata
            total_chars += len(message.role.value)

            if message.name:
                total_chars += len(message.name)

        # Very rough approximation: 4 chars per token
        return total_chars // 4 + 10  # Add some overhead


# Utility functions for image handling
def encode_image_file(image_path: Union[str, Path]) -> str:
    """Encode an image file as base64.

    Args:
        image_path: Path to the image file

    Returns:
        Base64-encoded image

    Raises:
        ImageProcessingError: If the image file cannot be encoded
    """
    try:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error encoding image file {image_path}: {str(e)}")
        raise ImageProcessingError(f"Failed to encode image: {str(e)}", str(image_path))


def encode_image_bytes(image_data: bytes) -> str:
    """Encode image bytes as base64.

    Args:
        image_data: Image bytes

    Returns:
        Base64-encoded image

    Raises:
        ImageProcessingError: If the image data cannot be encoded
    """
    try:
        return base64.b64encode(image_data).decode("utf-8")
    except Exception as e:
        logger.error(f"Error encoding image bytes: {str(e)}")
        raise ImageProcessingError(f"Failed to encode image bytes: {str(e)}")


def encode_image_from_url(url: str) -> str:
    """Download and encode an image from a URL.

    Args:
        url: URL of the image

    Returns:
        Base64-encoded image

    Raises:
        ImageProcessingError: If the image cannot be downloaded or encoded
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return encode_image_bytes(response.content)
    except Exception as e:
        logger.error(f"Error downloading image from {url}: {str(e)}")
        raise ImageProcessingError(f"Failed to download image: {str(e)}", url)
