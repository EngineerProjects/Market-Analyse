"""
OpenAI provider implementation for LLM integration.

This module implements the LLMProvider interface for OpenAI models.
"""

import time
import json
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

    # Model capability mappings
    VISION_MODELS = {
        "gpt-4-vision-preview",
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-4o-mini",
        "gpt-4o-2024-05-13",
    }

    TOOL_MODELS = {
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-1106",
        "gpt-4-0125-preview",
        "gpt-4-turbo-preview",
        "gpt-4-1106-preview",
        "gpt-4-vision-preview",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4",
        "gpt-4-turbo",
        "gpt-4o-2024-05-13",
    }

    # Maximum token contexts for different models
    MODEL_CONTEXT_SIZES = {
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-16k": 16384,
        "gpt-3.5-turbo-0125": 16385,
        "gpt-4": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-turbo": 128000,
        "gpt-4-turbo-preview": 128000,
        "gpt-4o": 128000,
        "gpt-4o-mini": 128000,
        "gpt-4o-2024-05-13": 128000,
    }
    DEFAULT_CONTEXT_SIZE = 4096

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        organization: Optional[str] = None,
        request_timeout: float = 60.0,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
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
            **kwargs: Additional parameters to pass to OpenAI API
        """
        self.model_name = model_name
        self.request_timeout = request_timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs

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

    def get_model_name(self) -> str:
        """Get the model name.

        Returns:
            Model name
        """
        return self.model_name

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
        choice = completion.choices[0]
        message_data = choice.message

        # Extract content
        content = message_data.content

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
                "id": completion.id,
                "model": completion.model,
                "created": completion.created,
                "finish_reason": choice.finish_reason,
            }

            if hasattr(completion, "usage"):
                message.metadata["usage"] = {
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                    "total_tokens": completion.usage.total_tokens,
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
            if "maximum context length" in str(e).lower():
                logger.error(f"Token limit exceeded: {e}")
                raise ContextWindowExceededError(
                    self.model_name, self.count_tokens(messages), self.get_max_tokens()
                )

            logger.error(f"Bad request: {e}")
            raise ParameterError(f"Invalid request: {e}")

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
            if "maximum context length" in str(e).lower():
                logger.error(f"Token limit exceeded: {e}")
                raise ContextWindowExceededError(
                    self.model_name, self.count_tokens(messages), self.get_max_tokens()
                )

            logger.error(f"Bad request: {e}")
            raise ParameterError(f"Invalid request: {e}")

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
            if "maximum context length" in str(e).lower():
                logger.error(f"Token limit exceeded: {e}")
                raise ContextWindowExceededError(
                    self.model_name, self.count_tokens(messages), self.get_max_tokens()
                )

            logger.error(f"Bad request: {e}")
            raise ParameterError(f"Invalid request: {e}")

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
                if "maximum context length" in str(e).lower():
                    logger.error(f"Token limit exceeded: {e}")
                    raise ContextWindowExceededError(
                        self.model_name, self.count_tokens(messages), self.get_max_tokens()
                    )

                logger.error(f"Bad request: {e}")
                raise ParameterError(f"Invalid request: {e}")

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
        return any(self.model_name.startswith(model) for model in self.VISION_MODELS)

    def supports_tools(self) -> bool:
        """Check if the provider/model supports tool/function calling.

        Returns:
            True if tool calling is supported, False otherwise
        """
        return any(self.model_name.startswith(model) for model in self.TOOL_MODELS)
