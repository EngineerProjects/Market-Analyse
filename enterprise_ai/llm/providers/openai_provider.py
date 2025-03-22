"""
OpenAI provider implementation for LLM integration.

This module implements the LLMProvider interface for OpenAI models.
"""

import time
import json
import re
import logging
from typing import (
    Any,
    Coroutine,
    Dict,
    Generator,
    List,
    Optional,
    AsyncGenerator,
    Union,
    cast,
    TypedDict,
    Literal,
    Tuple,
    Set,
)

import httpx
import openai
from openai import OpenAI, AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionChunk,
)

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
logger = get_logger("llm.openai")


# Define TypedDict classes for OpenAI message content
class ImageURLDict(TypedDict):
    url: str


class ImageURLContent(TypedDict):
    type: Literal["image_url"]
    image_url: ImageURLDict


class TextContent(TypedDict):
    type: Literal["text"]
    text: str


# Type for tool call data in stream processing
class ToolCallDict(TypedDict):
    id: str
    type: str
    function: Dict[str, str]


ContentItem = Union[TextContent, ImageURLContent]
MessageContent = Union[str, List[ContentItem]]


class OpenAIProvider(LLMProvider):
    """Provider implementation for OpenAI models."""

    # Model capability mappings with comprehensive model support
    VISION_MODELS = {
        # GPT-4 Vision models
        "gpt-4-vision-preview",
        # GPT-4o family with vision
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4o-2024-05-13",
        # GPT-4 Turbo with vision
        "gpt-4-turbo",
        # New models from 2025
        "gpt-4.5",
        "gpt-4o-search-preview",
        "gpt-4o-mini-search-preview",
    }

    TOOL_MODELS = {
        # GPT-3.5 models with function calling
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-1106",
        # GPT-4 models with function calling
        "gpt-4-0125-preview",
        "gpt-4-turbo-preview",
        "gpt-4-1106-preview",
        "gpt-4-vision-preview",
        "gpt-4",
        "gpt-4-turbo",
        # GPT-4o family
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4o-2024-05-13",
        # New models from 2025
        "gpt-4.5",
        "o3-mini",
        "o1-pro",
        "gpt-4o-search-preview",
        "gpt-4o-mini-search-preview",
    }

    # Audio/transcription specific models
    AUDIO_MODELS = {
        "gpt-4o-transcribe",
        "gpt-4o-mini-transcribe",
        "whisper",
    }

    # Maximum token contexts for different models (including 2025 models)
    MODEL_CONTEXT_SIZES = {
        # GPT-3.5 family
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-16k": 16384,
        "gpt-3.5-turbo-0125": 16385,
        # GPT-4 base models
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        # GPT-4 Turbo models
        "gpt-4-turbo": 128000,
        "gpt-4-turbo-preview": 128000,
        # GPT-4o family
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-4o-2024-05-13": 128000,
        # Specialized models
        "gpt-4o-transcribe": 128000,
        "gpt-4o-mini-transcribe": 64000,
        "gpt-4o-search-preview": 128000,
        "gpt-4o-mini-search-preview": 64000,
        # New 2025 models
        "gpt-4.5": 128000,
        "o3-mini": 200000,
        "o1-pro": 200000,
    }
    DEFAULT_CONTEXT_SIZE = 16384  # Increased default from 4096 to be more future-proof

    # Model families for capability detection
    MODEL_FAMILIES = {
        "gpt-3.5": {"vision": False, "tools": True, "audio": False},
        "gpt-4": {"vision": True, "tools": True, "audio": False},
        "gpt-4o": {"vision": True, "tools": True, "audio": False},
        "gpt-4.5": {"vision": True, "tools": True, "audio": False},
        "o1": {"vision": False, "tools": True, "audio": False},
        "o3": {"vision": True, "tools": True, "audio": False},
    }

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        organization: Optional[str] = None,
        request_timeout: float = 60.0,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        validate_model: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the OpenAI provider.

        Args:
            model_name: Name of the model to use
            api_key: OpenAI API key (default: from env)
            api_base: OpenAI API base URL (default: from env)
            organization: OpenAI organization ID (default: from env)
            request_timeout: Request timeout in seconds
            temperature: Model temperature
            max_tokens: Maximum tokens to generate
            validate_model: Whether to validate the model with the API
            **kwargs: Additional parameters to pass to OpenAI API
        """
        self.model_name = model_name
        self.request_timeout = request_timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        self._model_capabilities: Optional[Dict[str, bool]] = None
        self._model_validated: bool = False
        self._available_models: Optional[List[str]] = None

        # Initialize clients
        self.client = OpenAI(
            api_key=api_key, base_url=api_base, organization=organization, timeout=request_timeout
        )

        self.async_client = AsyncOpenAI(
            api_key=api_key, base_url=api_base, organization=organization, timeout=request_timeout
        )

        # Set default parameters
        self.default_params = {"model": model_name, "temperature": temperature, **kwargs}

        if max_tokens is not None:
            self.default_params["max_tokens"] = max_tokens

        # Validate model if requested
        if validate_model:
            self._validate_model()

    def get_model_name(self) -> str:
        """Get the model name.

        Returns:
            Model name
        """
        return self.model_name

    def _validate_model(self) -> bool:
        """Validate if the model exists and is available.

        Returns:
            True if model is valid, False otherwise
        """
        if self._model_validated:
            return True

        try:
            # Try to list available models first
            available_models = self.get_available_models()

            # Check if exact model name exists
            if self.model_name in available_models:
                self._model_validated = True
                return True

            # Try to find a model with the same base name (for cases like -preview suffixes)
            base_name = self.model_name.split("-")[0]
            for model in available_models:
                if model.startswith(base_name):
                    self._model_validated = True
                    return True

            # If model not found in list, try a minimal API call as fallback
            try:
                _ = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": "Test"}],
                    max_tokens=1,
                )
                # If we get here, the model exists
                self._model_validated = True
                return True
            except (openai.BadRequestError, openai.NotFoundError) as e:
                # Check if the error is due to invalid model
                if "model" in str(e).lower() and (
                    "not found" in str(e).lower() or "does not exist" in str(e).lower()
                ):
                    logger.warning(f"Model validation failed: {self.model_name} not found")
                    suggestions = self._suggest_similar_models(self.model_name, available_models)
                    if suggestions:
                        suggestion_str = ", ".join(suggestions[:3])
                        logger.info(f"Similar available models: {suggestion_str}")
                    return False
                else:
                    # Other bad request errors might be due to parameters, not the model itself
                    self._model_validated = True
                    return True
        except Exception as e:
            logger.warning(f"Model validation error: {e}")
            return True  # Assume model is valid if we can't definitively prove otherwise

    def get_available_models(self) -> List[str]:
        """Get a list of available models from the OpenAI API.

        Returns:
            List of available model names
        """
        if self._available_models is not None:
            return self._available_models

        try:
            response = self.client.models.list()
            if hasattr(response, "data"):
                model_list = [model.id for model in response.data]
                self._available_models = model_list
                return model_list
            # Add return for when response.data doesn't exist
            return []  # Return an empty list if no data attribute
        except Exception as e:
            logger.warning(f"Failed to get available models: {e}")
            # Return a list of common models as fallback
            fallback_models = [
                "gpt-4",
                "gpt-4-turbo",
                "gpt-4o",
                "gpt-4o-mini",
                "gpt-3.5-turbo",
                "gpt-3.5-turbo-16k",
            ]
            return fallback_models

    def _suggest_similar_models(self, model_name: str, available_models: List[str]) -> List[str]:
        """Suggest similar models based on name matching.

        Args:
            model_name: The requested model name
            available_models: List of available models

        Returns:
            List of suggested model names
        """
        if not available_models:
            return []

        # Extract model family and version
        parts = model_name.split("-")
        model_family = parts[0]

        # First try to find models in the same family
        family_matches = [m for m in available_models if m.startswith(model_family)]
        if family_matches:
            return family_matches

        # Try partial matching for similar suffixes
        if len(parts) > 1:
            suffix = parts[-1]
            suffix_matches = [m for m in available_models if m.endswith(suffix)]
            if suffix_matches:
                return suffix_matches

        # Default to returning the latest models
        latest_models = []
        for m in available_models:
            if "latest" in m or "turbo" in m or "gpt-4" in m:
                latest_models.append(m)

        return latest_models or available_models[:5]  # Return top 5 models

    def _get_model_family(self) -> str:
        """Extract the model family from the model name.

        Returns:
            Model family (e.g., "gpt-4" from "gpt-4-turbo")
        """
        # Try to match common model families
        for family in ["gpt-4.5", "gpt-4o", "gpt-4", "gpt-3.5", "o3", "o1"]:
            if self.model_name.startswith(family):
                return family

        # Use regex to extract the model family
        match = re.match(r"([a-zA-Z0-9]+-[0-9.]+)", self.model_name)
        if match:
            return match.group(1)

        # Extract the first part before any dash
        parts = self.model_name.split("-")
        return parts[0]

    def _detect_capabilities(self) -> Dict[str, bool]:
        """Detect model capabilities based on model name.

        Returns:
            Dictionary of capabilities
        """
        if self._model_capabilities is not None:
            return self._model_capabilities

        capabilities = {"vision": False, "tools": False, "audio": False}

        # Check exact model matches first
        model_name_lower = self.model_name.lower()

        for vision_model in self.VISION_MODELS:
            if vision_model.lower() in model_name_lower:
                capabilities["vision"] = True
                break

        for tool_model in self.TOOL_MODELS:
            if tool_model.lower() in model_name_lower:
                capabilities["tools"] = True
                break

        for audio_model in self.AUDIO_MODELS:
            if audio_model.lower() in model_name_lower:
                capabilities["audio"] = True
                break

        # Check model family if not all capabilities are determined
        if not all(capabilities.values()):
            model_family = self._get_model_family()

            # Check against known model families
            for family, family_capabilities in self.MODEL_FAMILIES.items():
                if family.lower() in model_family.lower():
                    # Update only unset capabilities
                    for cap_name, cap_value in family_capabilities.items():
                        if not capabilities.get(cap_name, False):
                            capabilities[cap_name] = cap_value
                    break

        # Check for specific suffixes/pattern that indicate capabilities
        if "vision" in model_name_lower or "visual" in model_name_lower:
            capabilities["vision"] = True

        if (
            "transcribe" in model_name_lower
            or "audio" in model_name_lower
            or "whisper" in model_name_lower
        ):
            capabilities["audio"] = True

        # Most modern GPT models (2024+) support tools/function calling
        if any(prefix in model_name_lower for prefix in ["gpt-4", "gpt-3.5-turbo", "o1", "o3"]):
            capabilities["tools"] = True

        self._model_capabilities = capabilities
        return capabilities

    def _prepare_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert our Message objects to OpenAI API format.

        Args:
            messages: List of Message objects

        Returns:
            List of OpenAI API message dictionaries
        """
        openai_messages = []

        for message in messages:
            msg_dict: Dict[str, Any] = {"role": message.role.value}

            # Handle content
            if message.content is not None:
                msg_dict["content"] = message.content

            # Handle image attachments
            if message.base64_image:
                if not self.supports_vision():
                    raise ModelCapabilityError(self.model_name, "vision")

                # For vision models, content needs to be a list of objects
                content_list: List[ContentItem] = []

                # Add text content if present
                if message.content:
                    text_item: TextContent = {"type": "text", "text": message.content}
                    content_list.append(text_item)

                # Add image content
                image_url_data: ImageURLDict = {
                    "url": f"data:image/jpeg;base64,{message.base64_image}"
                }
                image_item: ImageURLContent = {"type": "image_url", "image_url": image_url_data}
                content_list.append(image_item)

                msg_dict["content"] = content_list

            # Handle tool calls
            if message.tool_calls:
                if not self.supports_tools():
                    raise ModelCapabilityError(self.model_name, "tools")

                msg_dict["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in message.tool_calls
                ]

            # Handle name
            if message.name:
                msg_dict["name"] = message.name

            # Handle tool response
            if message.tool_call_id:
                msg_dict["tool_call_id"] = message.tool_call_id

            openai_messages.append(msg_dict)

        return openai_messages

    def _convert_completion_to_message(
        self, completion: Any, full_response: bool = False
    ) -> Message:
        """Convert an OpenAI completion to a Message object.

        Args:
            completion: OpenAI completion object
            full_response: Whether to include the full response

        Returns:
            Message object
        """
        if not hasattr(completion, "choices") or not completion.choices:
            # Handle empty or malformed completion
            return Message.assistant_message("")

        choice = completion.choices[0]
        message_data = choice.message

        # Extract content
        content = message_data.content if hasattr(message_data, "content") else None

        # Create basic message
        if hasattr(message_data, "tool_calls") and message_data.tool_calls:
            # Handle tool calls
            tool_calls_data = message_data.tool_calls
            tool_calls = []

            for tool_call in tool_calls_data:
                function = Function(
                    name=tool_call.function.name, arguments=tool_call.function.arguments
                )

                tool_calls.append(ToolCall(id=tool_call.id, type=tool_call.type, function=function))

            message = Message.from_tool_calls(tool_calls, content or "")
        else:
            # Standard message
            message = Message.assistant_message(content)

        # Optionally include full response metadata
        if full_response:
            if message.metadata is None:
                message.metadata = {}

            message.metadata["completion"] = {
                "id": completion.id if hasattr(completion, "id") else "",
                "model": completion.model if hasattr(completion, "model") else self.model_name,
                "created": completion.created if hasattr(completion, "created") else 0,
                "finish_reason": choice.finish_reason if hasattr(choice, "finish_reason") else "",
            }

            if hasattr(completion, "usage"):
                message.metadata["usage"] = {
                    "prompt_tokens": completion.usage.prompt_tokens
                    if hasattr(completion.usage, "prompt_tokens")
                    else 0,
                    "completion_tokens": completion.usage.completion_tokens
                    if hasattr(completion.usage, "completion_tokens")
                    else 0,
                    "total_tokens": completion.usage.total_tokens
                    if hasattr(completion.usage, "total_tokens")
                    else 0,
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
            openai_messages = self._prepare_messages(messages)

            # Call the API using type cast for compatibility
            response = self.client.chat.completions.create(
                messages=cast(List[ChatCompletionMessageParam], openai_messages), **params
            )

            # Convert to Message object
            return self._convert_completion_to_message(response, full_response=True)

        except openai.RateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
            raise APIError(429, f"OpenAI rate limit exceeded: {e}")

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            status_code = getattr(e, "status_code", 500)
            raise APIError(status_code, f"OpenAI API error: {e}")

        except openai.BadRequestError as e:
            # Check for token limit errors
            if any(
                phrase in str(e).lower()
                for phrase in ["maximum context length", "token limit", "too many tokens"]
            ):
                logger.error(f"Token limit exceeded: {e}")
                raise ContextWindowExceededError(
                    self.model_name, self.count_tokens(messages), self.get_max_tokens()
                )

            # Check for model not found errors
            if any(
                phrase in str(e).lower()
                for phrase in ["model not found", "does not exist", "invalid model"]
            ):
                logger.error(f"Model not found: {self.model_name}")

                # Try to suggest similar models
                suggestions = self._suggest_similar_models(
                    self.model_name, self.get_available_models()
                )
                suggestion_msg = ""
                if suggestions:
                    suggestion_msg = (
                        f" Try one of these models instead: {', '.join(suggestions[:3])}"
                    )

                raise ModelNotAvailable(self.model_name, f"Model not found: {e}{suggestion_msg}")

            logger.error(f"Bad request: {e}")
            raise ParameterError(f"Invalid request: {e}")

        except openai.NotFoundError as e:
            # Handle model not found errors
            logger.error(f"Resource not found: {e}")

            if "model" in str(e).lower():
                suggestions = self._suggest_similar_models(
                    self.model_name, self.get_available_models()
                )
                suggestion_msg = ""
                if suggestions:
                    suggestion_msg = (
                        f" Try one of these models instead: {', '.join(suggestions[:3])}"
                    )

                raise ModelNotAvailable(self.model_name, f"Model not found: {e}{suggestion_msg}")

            raise APIError(404, f"Resource not found: {e}")

        except openai.AuthenticationError as e:
            logger.error(f"Authentication error: {e}")
            raise APIError(401, f"OpenAI authentication error: {e}")

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
            openai_messages = self._prepare_messages(messages)

            # Stream configuration
            current_content = ""
            current_tool_calls: List[ToolCallDict] = []

            # Call the API
            response_stream = self.client.chat.completions.create(
                messages=cast(List[ChatCompletionMessageParam], openai_messages), **params
            )

            for chunk in response_stream:
                if not hasattr(chunk, "choices") or not chunk.choices:
                    continue

                # Extract delta
                delta = chunk.choices[0].delta

                # Update message content
                if hasattr(delta, "content") and delta.content is not None:
                    current_content += delta.content
                    yield Message.assistant_message(current_content)

                # Handle tool calls
                if hasattr(delta, "tool_calls") and delta.tool_calls:
                    for tool_call_delta in delta.tool_calls:
                        if not hasattr(tool_call_delta, "index"):
                            continue

                        tool_call_index = tool_call_delta.index

                        # Extend current_tool_calls list if needed
                        while len(current_tool_calls) <= tool_call_index:
                            current_tool_calls.append(
                                {
                                    "id": "",
                                    "type": "function",
                                    "function": {"name": "", "arguments": ""},
                                }
                            )

                        # Update tool call
                        if hasattr(tool_call_delta, "id") and tool_call_delta.id:
                            current_tool_calls[tool_call_index]["id"] = tool_call_delta.id

                        if hasattr(tool_call_delta, "function"):
                            func_delta = tool_call_delta.function

                            if hasattr(func_delta, "name") and func_delta.name:
                                current_tool_calls[tool_call_index]["function"]["name"] = (
                                    func_delta.name
                                )

                            if hasattr(func_delta, "arguments") and func_delta.arguments:
                                current_args = current_tool_calls[tool_call_index]["function"][
                                    "arguments"
                                ]
                                current_tool_calls[tool_call_index]["function"]["arguments"] = (
                                    current_args + func_delta.arguments
                                )

                    # Convert to Message objects
                    tool_calls = []
                    for tc in current_tool_calls:
                        if tc["id"] and tc["function"]["name"]:
                            function = Function(
                                name=tc["function"]["name"], arguments=tc["function"]["arguments"]
                            )

                            tool_calls.append(
                                ToolCall(id=tc["id"], type=tc["type"], function=function)
                            )

                    if tool_calls:
                        yield Message.from_tool_calls(tool_calls, current_content)

            # Return final message
            if current_tool_calls and any(tc["id"] for tc in current_tool_calls):
                # Final tool call message
                tool_calls = []
                for tc in current_tool_calls:
                    if tc["id"]:
                        function = Function(
                            name=tc["function"]["name"], arguments=tc["function"]["arguments"]
                        )

                        tool_calls.append(ToolCall(id=tc["id"], type=tc["type"], function=function))

                yield Message.from_tool_calls(tool_calls, current_content)
            else:
                # Final content-only message
                yield Message.assistant_message(current_content)

        except openai.RateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
            raise APIError(429, f"OpenAI rate limit exceeded: {e}")

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            status_code = getattr(e, "status_code", 500)
            raise APIError(status_code, f"OpenAI API error: {e}")

        except openai.BadRequestError as e:
            # Check for token limit errors
            if any(
                phrase in str(e).lower()
                for phrase in ["maximum context length", "token limit", "too many tokens"]
            ):
                logger.error(f"Token limit exceeded: {e}")
                raise ContextWindowExceededError(
                    self.model_name, self.count_tokens(messages), self.get_max_tokens()
                )

            # Check for model not found errors
            if any(
                phrase in str(e).lower()
                for phrase in ["model not found", "does not exist", "invalid model"]
            ):
                logger.error(f"Model not found: {self.model_name}")
                raise ModelNotAvailable(self.model_name, f"Model not found: {e}")

            logger.error(f"Bad request: {e}")
            raise ParameterError(f"Invalid request: {e}")

        except openai.NotFoundError as e:
            # Handle model not found errors
            logger.error(f"Resource not found: {e}")

            if "model" in str(e).lower():
                raise ModelNotAvailable(self.model_name, f"Model not found: {e}")

            raise APIError(404, f"Resource not found: {e}")

        except openai.AuthenticationError as e:
            logger.error(f"Authentication error: {e}")
            raise APIError(401, f"OpenAI authentication error: {e}")

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
            openai_messages = self._prepare_messages(messages)

            # Call the API
            response = await self.async_client.chat.completions.create(
                messages=cast(List[ChatCompletionMessageParam], openai_messages), **params
            )

            # Convert to Message object
            return self._convert_completion_to_message(response, full_response=True)

        except openai.RateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
            raise APIError(429, f"OpenAI rate limit exceeded: {e}")

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            status_code = getattr(e, "status_code", 500)
            raise APIError(status_code, f"OpenAI API error: {e}")

        except openai.BadRequestError as e:
            # Check for token limit errors
            if any(
                phrase in str(e).lower()
                for phrase in ["maximum context length", "token limit", "too many tokens"]
            ):
                logger.error(f"Token limit exceeded: {e}")
                raise ContextWindowExceededError(
                    self.model_name, self.count_tokens(messages), self.get_max_tokens()
                )

            # Check for model not found errors
            if any(
                phrase in str(e).lower()
                for phrase in ["model not found", "does not exist", "invalid model"]
            ):
                logger.error(f"Model not found: {self.model_name}")
                raise ModelNotAvailable(self.model_name, f"Model not found: {e}")

            logger.error(f"Bad request: {e}")
            raise ParameterError(f"Invalid request: {e}")

        except openai.NotFoundError as e:
            # Handle model not found errors
            logger.error(f"Resource not found: {e}")

            if "model" in str(e).lower():
                raise ModelNotAvailable(self.model_name, f"Model not found: {e}")

            raise APIError(404, f"Resource not found: {e}")

        except openai.AuthenticationError as e:
            logger.error(f"Authentication error: {e}")
            raise APIError(401, f"OpenAI authentication error: {e}")

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise APIError(500, f"Unexpected error: {e}")

    @retry_with_exponential_backoff()
    async def acomplete_stream(
        self, messages: List[Message], **kwargs: Any
    ) -> Coroutine[Any, Any, AsyncGenerator[Message, None]]:
        """Generate a streaming completion for the given messages asynchronously.

        Args:
            messages: List of messages in the conversation
            **kwargs: Additional completion options

        Returns:
            Async generator yielding partial completion messages

        Raises:
            APIError: If there's an API error
            ModelNotAvailable: If the model is not available
            TokenLimitExceeded: If the token limit is exceeded
        """
        # Get API parameters
        params = self.default_params.copy()
        params.update(kwargs)
        params["stream"] = True

        # Prepare messages
        openai_messages = self._prepare_messages(messages)

        # Create a properly typed async generator function
        async def stream_generator() -> AsyncGenerator[Message, None]:
            try:
                # Stream configuration
                current_content = ""
                current_tool_calls: List[ToolCallDict] = []

                # Call the API - properly await the response
                response_stream = await self.async_client.chat.completions.create(
                    messages=cast(List[ChatCompletionMessageParam], openai_messages), **params
                )

                # Ensure we get a proper stream response
                if not hasattr(response_stream, "__aiter__"):
                    raise APIError(500, "Expected streaming response but got non-iterable")

                # Process the stream
                async for chunk in response_stream:
                    if not hasattr(chunk, "choices") or not chunk.choices:
                        continue

                    # Extract delta
                    delta = chunk.choices[0].delta

                    # Update message content
                    if hasattr(delta, "content") and delta.content is not None:
                        current_content += delta.content
                        yield Message.assistant_message(current_content)

                    # Handle tool calls
                    if hasattr(delta, "tool_calls") and delta.tool_calls:
                        for tool_call_delta in delta.tool_calls:
                            if not hasattr(tool_call_delta, "index"):
                                continue

                            tool_call_index = tool_call_delta.index

                            # Extend current_tool_calls list if needed
                            while len(current_tool_calls) <= tool_call_index:
                                current_tool_calls.append(
                                    {
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    }
                                )

                            # Update tool call
                            if hasattr(tool_call_delta, "id") and tool_call_delta.id:
                                current_tool_calls[tool_call_index]["id"] = tool_call_delta.id

                            if hasattr(tool_call_delta, "function"):
                                func_delta = tool_call_delta.function

                                if hasattr(func_delta, "name") and func_delta.name:
                                    current_tool_calls[tool_call_index]["function"]["name"] = (
                                        func_delta.name
                                    )

                                if hasattr(func_delta, "arguments") and func_delta.arguments:
                                    current_args = current_tool_calls[tool_call_index][
                                        "function"
                                    ].get("arguments", "")
                                    current_tool_calls[tool_call_index]["function"]["arguments"] = (
                                        current_args + func_delta.arguments
                                    )

                        # Convert to Message objects
                        tool_calls = []
                        for tc in current_tool_calls:
                            if tc.get("id") and tc.get("function", {}).get("name"):
                                function = Function(
                                    name=tc["function"]["name"],
                                    arguments=tc["function"]["arguments"],
                                )

                                tool_calls.append(
                                    ToolCall(id=tc["id"], type=tc["type"], function=function)
                                )

                        if tool_calls:
                            yield Message.from_tool_calls(tool_calls, current_content)

                # Return final message
                if current_tool_calls and any(tc.get("id") for tc in current_tool_calls):
                    # Final tool call message
                    tool_calls = []
                    for tc in current_tool_calls:
                        if tc.get("id"):
                            function = Function(
                                name=tc["function"]["name"], arguments=tc["function"]["arguments"]
                            )

                            tool_calls.append(
                                ToolCall(id=tc["id"], type=tc["type"], function=function)
                            )

                    yield Message.from_tool_calls(tool_calls, current_content)
                else:
                    # Final content-only message
                    yield Message.assistant_message(current_content)

            except openai.RateLimitError as e:
                logger.error(f"Rate limit exceeded: {e}")
                raise APIError(429, f"OpenAI rate limit exceeded: {e}")

            except openai.APIError as e:
                logger.error(f"OpenAI API error: {e}")
                status_code = getattr(e, "status_code", 500)
                raise APIError(status_code, f"OpenAI API error: {e}")

            except openai.BadRequestError as e:
                # Check for token limit errors
                if any(
                    phrase in str(e).lower()
                    for phrase in ["maximum context length", "token limit", "too many tokens"]
                ):
                    logger.error(f"Token limit exceeded: {e}")
                    raise ContextWindowExceededError(
                        self.model_name, self.count_tokens(messages), self.get_max_tokens()
                    )

                # Check for model not found errors
                if any(
                    phrase in str(e).lower()
                    for phrase in ["model not found", "does not exist", "invalid model"]
                ):
                    logger.error(f"Model not found: {self.model_name}")
                    raise ModelNotAvailable(self.model_name, f"Model not found: {e}")

                logger.error(f"Bad request: {e}")
                raise ParameterError(f"Invalid request: {e}")

            except openai.NotFoundError as e:
                # Handle model not found errors
                logger.error(f"Resource not found: {e}")

                if "model" in str(e).lower():
                    raise ModelNotAvailable(self.model_name, f"Model not found: {e}")

                raise APIError(404, f"Resource not found: {e}")

            except openai.AuthenticationError as e:
                logger.error(f"Authentication error: {e}")
                raise APIError(401, f"OpenAI authentication error: {e}")

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise APIError(500, f"Unexpected error: {e}")

        async def get_generator() -> AsyncGenerator[Message, None]:
            return stream_generator()

        return get_generator()

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
        # Try to get exact model context size
        for model_prefix, context_size in self.MODEL_CONTEXT_SIZES.items():
            if self.model_name.startswith(model_prefix):
                return context_size

        # Try pattern matching for model families
        model_family = self._get_model_family()
        for family_prefix, context_size in self.MODEL_CONTEXT_SIZES.items():
            if family_prefix.startswith(model_family):
                return context_size

        # Extract from model name if possible
        # Look for patterns like "32k" in model name
        context_match = re.search(r"(\d+)[kK]", self.model_name)
        if context_match:
            return int(context_match.group(1)) * 1024

        # Use safe defaults based on model family
        if "gpt-4" in self.model_name.lower():
            if "turbo" in self.model_name.lower() or "o" in self.model_name.lower():
                return 128000  # Default for GPT-4 Turbo/o models
            return 8192  # Default for base GPT-4

        if "gpt-3.5" in self.model_name.lower():
            if "16k" in self.model_name.lower():
                return 16384
            return 4096  # Default for GPT-3.5

        # Use a safe default
        logger.warning(f"Unknown context size for model {self.model_name}. Using default.")
        return self.DEFAULT_CONTEXT_SIZE

    def supports_vision(self) -> bool:
        """Check if the provider/model supports vision/images.

        Returns:
            True if vision is supported, False otherwise
        """
        capabilities = self._detect_capabilities()
        return capabilities["vision"]

    def supports_tools(self) -> bool:
        """Check if the provider/model supports tool/function calling.

        Returns:
            True if tool calling is supported, False otherwise
        """
        capabilities = self._detect_capabilities()
        return capabilities["tools"]

    def supports_audio(self) -> bool:
        """Check if the provider/model supports audio processing.

        Returns:
            True if audio processing is supported, False otherwise
        """
        capabilities = self._detect_capabilities()
        return capabilities["audio"]
