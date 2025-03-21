"""
Anthropic Claude provider implementation for LLM integration.

This module implements the LLMProvider interface for Anthropic's Claude models.
"""

import time
import json
import asyncio
from typing import Any, Dict, Generator, List, Optional, AsyncGenerator, Union, cast

import anthropic
from anthropic import Anthropic, AsyncAnthropic

from enterprise_ai.schema import Message, Role, ToolCall, Function
from enterprise_ai.exceptions import APIError, ModelNotAvailable
from enterprise_ai.llm.base import LLMProvider
from enterprise_ai.llm.exceptions import (
    TokenCountError,
    ModelCapabilityError,
    ParameterError,
    ContextWindowExceededError,
)
from enterprise_ai.llm.utils import TokenCounter, retry_with_exponential_backoff
from enterprise_ai.logger import get_logger, trace_execution

# Initialize logger
logger = get_logger("llm.anthropic")


class AnthropicProvider(LLMProvider):
    """Provider implementation for Anthropic Claude models."""

    # Model capability mappings
    VISION_MODELS = {"claude-3-opus", "claude-3-sonnet", "claude-3-haiku", "claude-3.5-sonnet"}
    TOOL_MODELS = {"claude-3-opus", "claude-3-sonnet", "claude-3-haiku", "claude-3.5-sonnet"}

    # Maximum token contexts for different models
    MODEL_CONTEXT_SIZES = {
        "claude-2": 100000,
        "claude-2.0": 100000,
        "claude-2.1": 200000,
        "claude-3-opus": 200000,
        "claude-3-sonnet": 200000,
        "claude-3-haiku": 200000,
        "claude-3.5-sonnet": 200000,
    }
    DEFAULT_CONTEXT_SIZE = 100000

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        request_timeout: float = 60.0,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Anthropic provider.

        Args:
            model_name: Name of the model to use
            api_key: Anthropic API key (default: from env)
            api_base: API base URL (default: from env)
            request_timeout: Request timeout in seconds
            temperature: Model temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters to pass to Anthropic API
        """
        self.model_name = model_name
        self.request_timeout = request_timeout
        self.temperature = temperature
        self.max_tokens = max_tokens or 4096  # Default to 4096 if not specified
        self.kwargs = kwargs

        # Set base URL
        base_url = api_base if api_base else "https://api.anthropic.com"

        # Initialize clients
        self.client = Anthropic(api_key=api_key, base_url=base_url, timeout=request_timeout)

        self.async_client = AsyncAnthropic(
            api_key=api_key, base_url=base_url, timeout=request_timeout
        )

        # Set default parameters
        self.default_params = {
            "model": model_name,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
            **kwargs,
        }

    def get_model_name(self) -> str:
        """Get the model name.

        Returns:
            Model name
        """
        return self.model_name

    def _prepare_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert our Message objects to Anthropic API format.

        Args:
            messages: List of Message objects

        Returns:
            List of Anthropic API message dictionaries
        """
        anthropic_messages = []
        system_message = None

        for message in messages:
            # Handle system message separately
            if message.role == Role.SYSTEM:
                system_message = message.content
                continue

            # Convert role
            if message.role == Role.USER:
                role = "user"
            elif message.role == Role.ASSISTANT:
                role = "assistant"
            elif message.role == Role.TOOL:
                role = "tool"
            else:
                # Skip unknown roles
                logger.warning(f"Skipping message with unsupported role: {message.role}")
                continue

            msg_dict: Dict[str, Any] = {"role": role}

            # Handle content and images
            if message.base64_image and role == "user":
                if not self.supports_vision():
                    raise ModelCapabilityError(self.model_name, "vision")

                # For vision models, content needs to be a list of objects
                content_list = []

                # Add text content if present
                if message.content:
                    content_list.append({"type": "text", "text": message.content})

                # Add image content
                content_list.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": message.base64_image,
                        },
                    }
                )

                msg_dict["content"] = content_list
            elif message.content is not None:
                msg_dict["content"] = message.content
            else:
                msg_dict["content"] = ""

            # Handle tool calls for assistant messages
            if message.tool_calls and role == "assistant":
                # Convert to Anthropic tool calls format
                tool_calls = []
                for tool_call in message.tool_calls:
                    tool_calls.append(
                        {
                            "id": tool_call.id,
                            "type": "tool_use",
                            "name": tool_call.function.name,
                            "input": json.loads(tool_call.function.arguments),
                        }
                    )

                msg_dict["content"] = []
                if message.content:
                    msg_dict["content"].append({"type": "text", "text": message.content})

                for tool_call in tool_calls:
                    msg_dict["content"].append(
                        {
                            "type": "tool_use",
                            "id": tool_call["id"],
                            "name": tool_call["name"],
                            "input": tool_call["input"],
                        }
                    )

            # Handle tool response
            if message.role == Role.TOOL:
                msg_dict["content"] = []
                msg_dict["content"].append(
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id,
                        "content": message.content or "",
                    }
                )

            anthropic_messages.append(msg_dict)

        return anthropic_messages, system_message

    def _convert_completion_to_message(
        self, completion: Any, full_response: bool = False
    ) -> Message:
        """Convert an Anthropic completion to a Message object.

        Args:
            completion: Anthropic completion object
            full_response: Whether to include the full response

        Returns:
            Message object
        """
        content = None
        tool_calls = []

        # Parse the response content
        if hasattr(completion, "content") and completion.content:
            for block in completion.content:
                if block.type == "text":
                    content = block.text
                elif block.type == "tool_use":
                    # Convert tool use to our format
                    function = Function(name=block.name, arguments=json.dumps(block.input))

                    tool_calls.append(ToolCall(id=block.id, type="function", function=function))

        # Create appropriate message type
        if tool_calls:
            message = Message.from_tool_calls(tool_calls, content or "")
        else:
            message = Message.assistant_message(content)

        # Optionally include full response metadata
        if full_response:
            message.metadata["completion"] = {
                "id": completion.id,
                "model": completion.model,
                "type": completion.type,
                "stop_reason": completion.stop_reason,
                "stop_sequence": completion.stop_sequence,
            }

            if hasattr(completion, "usage"):
                message.metadata["usage"] = {
                    "input_tokens": completion.usage.input_tokens,
                    "output_tokens": completion.usage.output_tokens,
                }

        return message

    @retry_with_exponential_backoff()
    def complete(self, messages: List[Message], **kwargs: Any) -> Message:
        """Generate a completion for the given messages.

        Args:
            messages: List of messages in the conversation
            **kwargs: Additional completion options

        Returns:
            Generated completion message

        Raises:
            APIError: If there's an API error
            ModelNotAvailable: If the model is not available
            TokenLimitExceeded: If the token limit is exceeded
        """
        try:
            # Get API parameters
            params = self.default_params.copy()
            params.update(kwargs)

            # Prepare messages
            anthropic_messages, system_message = self._prepare_messages(messages)

            # Set system message if present
            params["messages"] = anthropic_messages
            if system_message:
                params["system"] = system_message

            # Call the API
            response = self.client.messages.create(**params)

            # Convert to Message object
            return self._convert_completion_to_message(response, full_response=True)

        except anthropic.RateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
            raise APIError(429, f"Anthropic rate limit exceeded: {e}")

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            status_code = getattr(e, "status_code", 500)
            raise APIError(status_code, f"Anthropic API error: {e}")

        except anthropic.BadRequestError as e:
            # Check for token limit errors
            if "maximum context length" in str(e).lower() or "too long" in str(e).lower():
                logger.error(f"Token limit exceeded: {e}")
                raise ContextWindowExceededError(
                    self.model_name, self.count_tokens(messages), self.get_max_tokens()
                )

            logger.error(f"Bad request: {e}")
            raise ParameterError(f"Invalid request: {e}")

        except anthropic.AuthenticationError as e:
            logger.error(f"Authentication error: {e}")
            raise APIError(401, f"Anthropic authentication error: {e}")

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise APIError(500, f"Unexpected error: {e}")

    @retry_with_exponential_backoff()
    def complete_stream(
        self, messages: List[Message], **kwargs: Any
    ) -> Generator[Message, None, None]:
        """Generate a streaming completion for the given messages.

        Args:
            messages: List of messages in the conversation
            **kwargs: Additional completion options

        Yields:
            Partial completion messages

        Raises:
            APIError: If there's an API error
            ModelNotAvailable: If the model is not available
            TokenLimitExceeded: If the token limit is exceeded
        """
        try:
            # Get API parameters
            params = self.default_params.copy()
            params.update(kwargs)
            params["stream"] = True

            # Prepare messages
            anthropic_messages, system_message = self._prepare_messages(messages)

            # Set system message if present
            params["messages"] = anthropic_messages
            if system_message:
                params["system"] = system_message

            # Stream configuration
            current_content = ""
            current_tool_calls = []

            # Call the API
            with self.client.messages.stream(**params) as stream:
                for text in stream.text_stream:
                    current_content += text
                    yield Message.assistant_message(current_content)

                # Handle tool calls if any
                if stream.response.type == "message" and hasattr(stream.response, "content"):
                    for block in stream.response.content:
                        if block.type == "tool_use":
                            # Convert tool use to our format
                            function = Function(name=block.name, arguments=json.dumps(block.input))

                            current_tool_calls.append(
                                ToolCall(id=block.id, type="function", function=function)
                            )

                # Yield final message with tool calls if any
                if current_tool_calls:
                    yield Message.from_tool_calls(current_tool_calls, current_content)
                else:
                    # Final content-only message
                    yield Message.assistant_message(current_content)

        except anthropic.RateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
            raise APIError(429, f"Anthropic rate limit exceeded: {e}")

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            status_code = getattr(e, "status_code", 500)
            raise APIError(status_code, f"Anthropic API error: {e}")

        except anthropic.BadRequestError as e:
            # Check for token limit errors
            if "maximum context length" in str(e).lower() or "too long" in str(e).lower():
                logger.error(f"Token limit exceeded: {e}")
                raise ContextWindowExceededError(
                    self.model_name, self.count_tokens(messages), self.get_max_tokens()
                )

            logger.error(f"Bad request: {e}")
            raise ParameterError(f"Invalid request: {e}")

        except anthropic.AuthenticationError as e:
            logger.error(f"Authentication error: {e}")
            raise APIError(401, f"Anthropic authentication error: {e}")

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise APIError(500, f"Unexpected error: {e}")

    @retry_with_exponential_backoff()
    async def acomplete(self, messages: List[Message], **kwargs: Any) -> Message:
        """Generate a completion for the given messages asynchronously.

        Args:
            messages: List of messages in the conversation
            **kwargs: Additional completion options

        Returns:
            Generated completion message

        Raises:
            APIError: If there's an API error
            ModelNotAvailable: If the model is not available
            TokenLimitExceeded: If the token limit is exceeded
        """
        try:
            # Get API parameters
            params = self.default_params.copy()
            params.update(kwargs)

            # Prepare messages
            anthropic_messages, system_message = self._prepare_messages(messages)

            # Set system message if present
            params["messages"] = anthropic_messages
            if system_message:
                params["system"] = system_message

            # Call the API
            response = await self.async_client.messages.create(**params)

            # Convert to Message object
            return self._convert_completion_to_message(response, full_response=True)

        except anthropic.RateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
            raise APIError(429, f"Anthropic rate limit exceeded: {e}")

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            status_code = getattr(e, "status_code", 500)
            raise APIError(status_code, f"Anthropic API error: {e}")

        except anthropic.BadRequestError as e:
            # Check for token limit errors
            if "maximum context length" in str(e).lower() or "too long" in str(e).lower():
                logger.error(f"Token limit exceeded: {e}")
                raise ContextWindowExceededError(
                    self.model_name, self.count_tokens(messages), self.get_max_tokens()
                )

            logger.error(f"Bad request: {e}")
            raise ParameterError(f"Invalid request: {e}")

        except anthropic.AuthenticationError as e:
            logger.error(f"Authentication error: {e}")
            raise APIError(401, f"Anthropic authentication error: {e}")

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise APIError(500, f"Unexpected error: {e}")

    @retry_with_exponential_backoff()
    async def acomplete_stream(
        self, messages: List[Message], **kwargs: Any
    ) -> AsyncGenerator[Message, None]:
        """Generate a streaming completion for the given messages asynchronously.

        Args:
            messages: List of messages in the conversation
            **kwargs: Additional completion options

        Yields:
            Partial completion messages

        Raises:
            APIError: If there's an API error
            ModelNotAvailable: If the model is not available
            TokenLimitExceeded: If the token limit is exceeded
        """
        try:
            # Get API parameters
            params = self.default_params.copy()
            params.update(kwargs)

            # Prepare messages
            anthropic_messages, system_message = self._prepare_messages(messages)

            # Set system message if present
            params["messages"] = anthropic_messages
            if system_message:
                params["system"] = system_message

            # Stream configuration
            current_content = ""
            current_tool_calls = []

            # Call the API
            async with self.async_client.messages.stream(**params) as stream:
                async for text in stream.text_stream:
                    current_content += text
                    yield Message.assistant_message(current_content)

                # Handle tool calls if any
                if stream.response.type == "message" and hasattr(stream.response, "content"):
                    for block in stream.response.content:
                        if block.type == "tool_use":
                            # Convert tool use to our format
                            function = Function(name=block.name, arguments=json.dumps(block.input))

                            current_tool_calls.append(
                                ToolCall(id=block.id, type="function", function=function)
                            )

                # Yield final message with tool calls if any
                if current_tool_calls:
                    yield Message.from_tool_calls(current_tool_calls, current_content)
                else:
                    # Final content-only message
                    yield Message.assistant_message(current_content)

        except anthropic.RateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
            raise APIError(429, f"Anthropic rate limit exceeded: {e}")

        except anthropic.APIError as e:
            logger.error(f"Anthropic API error: {e}")
            status_code = getattr(e, "status_code", 500)
            raise APIError(status_code, f"Anthropic API error: {e}")

        except anthropic.BadRequestError as e:
            # Check for token limit errors
            if "maximum context length" in str(e).lower() or "too long" in str(e).lower():
                logger.error(f"Token limit exceeded: {e}")
                raise ContextWindowExceededError(
                    self.model_name, self.count_tokens(messages), self.get_max_tokens()
                )

            logger.error(f"Bad request: {e}")
            raise ParameterError(f"Invalid request: {e}")

        except anthropic.AuthenticationError as e:
            logger.error(f"Authentication error: {e}")
            raise APIError(401, f"Anthropic authentication error: {e}")

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise APIError(500, f"Unexpected error: {e}")

    def count_tokens(self, messages: List[Message]) -> int:
        """Count the number of tokens in the given messages.

        Args:
            messages: List of messages to count tokens for

        Returns:
            Number of tokens
        """
        return TokenCounter.count_for_model(self.model_name, messages)

    def get_max_tokens(self) -> int:
        """Get the maximum number of tokens supported by the model.

        Returns:
            Maximum token count
        """
        # Try to get exact model context size, otherwise use default
        for model_prefix, context_size in self.MODEL_CONTEXT_SIZES.items():
            if self.model_name.startswith(model_prefix):
                return context_size

        # Use a safe default
        logger.warning(f"Unknown context size for model {self.model_name}. Using default.")
        return self.DEFAULT_CONTEXT_SIZE

    def supports_vision(self) -> bool:
        """Check if the provider/model supports vision/images.

        Returns:
            True if vision is supported, False otherwise
        """
        return any(model in self.model_name for model in self.VISION_MODELS)

    def supports_tools(self) -> bool:
        """Check if the provider/model supports tool/function calling.

        Returns:
            True if tool calling is supported, False otherwise
        """
        return any(model in self.model_name for model in self.TOOL_MODELS)
