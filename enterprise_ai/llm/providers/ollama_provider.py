"""
Ollama provider implementation for LLM integration.

This module implements the LLMProvider interface for local Ollama models.
"""

import json
import time
import asyncio
from typing import (
    Any,
    Coroutine,
    Dict,
    Generator,
    List,
    Optional,
    AsyncGenerator,
    Union,
)

import httpx

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
logger = get_logger("llm.ollama")


class OllamaProvider(LLMProvider):
    """Provider implementation for Ollama models."""

    # Model capability mappings - Enhanced vision model support
    VISION_MODELS = {
        "llava",
        "bakllava",
        "moondream",
        "llama3-vision",
        "llama3.1-vision",
        "llama3.2-vision",
        "cogvlm",
    }

    TOOL_MODELS = {"llama3", "mistral", "mixtral", "openhermes", "wizardlm", "deepseek-coder"}

    # Maximum token contexts for different model families
    MODEL_CONTEXT_SIZES = {
        "llama2": 4096,
        "llama3": 8192,
        "llama3.1": 16384,
        "llama3.2": 32768,
        "mistral": 8192,
        "mixtral": 32768,
        "phi": 2048,
        "gemma": 8192,
        "mpt": 8192,
    }
    DEFAULT_CONTEXT_SIZE = 4096

    def __init__(
        self,
        model_name: str,
        api_base: str = "http://localhost:11434",
        request_timeout: float = 60.0,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Ollama provider.

        Args:
            model_name: Name of the model to use
            api_base: Ollama API base URL
            request_timeout: Request timeout in seconds
            temperature: Model temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters to pass to Ollama API
        """
        self.model_name = model_name
        self.api_base = api_base.rstrip("/")
        self.request_timeout = request_timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.kwargs = kwargs

        # Initialize clients
        self.client = httpx.Client(timeout=request_timeout, base_url=api_base)
        self.async_client = httpx.AsyncClient(timeout=request_timeout, base_url=api_base)

        # Set default parameters
        self.default_params = {"model": model_name, "temperature": temperature, **kwargs}

        if max_tokens is not None:
            self.default_params["num_predict"] = max_tokens

    def get_model_name(self) -> str:
        """Get the model name.

        Returns:
            Model name
        """
        return self.model_name

    def _prepare_messages(self, messages: List[Message]) -> Union[str, Dict[str, Any]]:
        """Convert our Message objects to Ollama API format.

        Args:
            messages: List of Message objects

        Returns:
            Formatted string for Ollama prompt or chat messages dict
        """
        # First try to get model info to check if it supports chat format
        try:
            response = self.client.get("/api/show", params={"name": self.model_name})
            response.raise_for_status()
            model_info = response.json()

            # Check if model info indicates chat format support
            if "format" in model_info and model_info["format"] == "chat":
                return self._prepare_chat_messages(messages)

            # Default to legacy prompt format
            return self._prepare_legacy_prompt(messages)

        except Exception as e:
            logger.warning(f"Failed to check model info, using legacy prompt format: {e}")
            return self._prepare_legacy_prompt(messages)

    def _prepare_chat_messages(self, messages: List[Message]) -> Dict[str, Any]:
        """Prepare messages for Ollama chat format.

        Args:
            messages: List of Message objects

        Returns:
            Dict with messages in Ollama chat format
        """
        ollama_messages = []

        for message in messages:
            if message.role == Role.SYSTEM:
                role = "system"
            elif message.role == Role.USER:
                role = "user"
            elif message.role == Role.ASSISTANT:
                role = "assistant"
            elif message.role == Role.TOOL:
                # Ollama doesn't have a tool role, use system as fallback
                role = "system"
                msg_content = f"[Tool Response] {message.name}: {message.content}"
                ollama_messages.append({"role": role, "content": msg_content})
                continue
            else:
                # Skip unknown roles
                logger.warning(f"Skipping message with unsupported role: {message.role}")
                continue

            msg_dict: Dict[str, str] = {"role": role}

            # Handle content
            if message.content is not None:
                msg_dict["content"] = message.content
            else:
                msg_dict["content"] = ""

            # Add image if present (for vision models)
            if message.base64_image and role == "user":
                if not self.supports_vision():
                    raise ModelCapabilityError(self.model_name, "vision")

                # Different models might require different image formats
                if any(
                    vision_model in self.model_name.lower()
                    for vision_model in ["llama3", "llama-3"]
                ):
                    # Format for llama3-vision models
                    msg_dict["content"] += (
                        f"\n<image>\ndata:image/jpeg;base64,{message.base64_image}\n</image>"
                    )
                else:
                    # Format for llava and other models
                    msg_dict["content"] += (
                        f"\n![image](data:image/jpeg;base64,{message.base64_image})"
                    )

            # Add tool calls as formatted text (workaround)
            if message.tool_calls and role == "assistant":
                for tool_call in message.tool_calls:
                    tool_info = f"\n[Function Call] {tool_call.function.name}({tool_call.function.arguments})"
                    msg_dict["content"] += tool_info

            ollama_messages.append(msg_dict)

        return {"messages": ollama_messages}

    def _prepare_legacy_prompt(self, messages: List[Message]) -> str:
        """Convert messages to a legacy string prompt format.

        Args:
            messages: List of Message objects

        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        system_message = None

        # Extract system message if present
        for message in messages:
            if message.role == Role.SYSTEM:
                system_message = message.content
                break

        # Add system message at the beginning if present
        if system_message:
            prompt_parts.append(f"<|system|>\n{system_message}</s>")

        # Process other messages
        for message in messages:
            if message.role == Role.SYSTEM:
                # Already handled
                continue

            if message.role == Role.USER:
                role_tag = "<|user|>"
            elif message.role == Role.ASSISTANT:
                role_tag = "<|assistant|>"
            elif message.role == Role.TOOL:
                role_tag = "<|tool|>"
            else:
                # Skip unknown roles
                continue

            content = message.content or ""

            # Add image if present (for vision models)
            if message.base64_image and message.role == Role.USER:
                if not self.supports_vision():
                    raise ModelCapabilityError(self.model_name, "vision")

                # Different models might require different image formats
                if any(
                    vision_model in self.model_name.lower()
                    for vision_model in ["llama3", "llama-3"]
                ):
                    # Format for llama3-vision models
                    content += f"\n<image>\ndata:image/jpeg;base64,{message.base64_image}\n</image>"
                else:
                    # Format for llava and other models
                    content += f"\n![image](data:image/jpeg;base64,{message.base64_image})"

            # Add tool calls as formatted text
            if message.tool_calls and message.role == Role.ASSISTANT:
                for tool_call in message.tool_calls:
                    content += f"\n[Function Call] {tool_call.function.name}({tool_call.function.arguments})"

            # Add message to prompt
            prompt_parts.append(f"{role_tag}\n{content}</s>")

        # Add assistant prompt at the end
        prompt_parts.append("<|assistant|>")

        return "\n".join(prompt_parts)

    def _convert_completion_to_message(
        self, completion: Dict[str, Any], full_response: bool = False
    ) -> Message:
        """Convert an Ollama completion to a Message object.

        Args:
            completion: Ollama completion response
            full_response: Whether to include the full response

        Returns:
            Message object
        """
        content = completion.get("response", "")

        # Check for tool calls in content (simple regex pattern matching)
        # This is a basic workaround since Ollama doesn't natively support tool calls
        tool_calls: List[ToolCall] = []

        if "[Function Call]" in content:
            # Very simple parsing, would need to be more robust in production
            import re

            pattern = r"\[Function Call\] ([^(]+)\(({.*?})\)"
            matches = re.findall(pattern, content)

            for name, args in matches:
                # Clean up the content by removing the function call text
                content = content.replace(f"[Function Call] {name}({args})", "").strip()

                function = Function(name=name.strip(), arguments=args.strip())

                tool_calls.append(
                    ToolCall(
                        id=f"tc_{hash(name)}_{int(time.time())}",  # Generate a unique ID
                        type="function",
                        function=function,
                    )
                )

        # Create appropriate message type
        if tool_calls:
            message = Message.from_tool_calls(tool_calls, content or "")
        else:
            message = Message.assistant_message(content)

        # Optionally include full response metadata
        if full_response:
            if message.metadata is None:
                message.metadata = {}

            message.metadata["completion"] = {
                "model": completion.get("model", self.model_name),
                "created_at": completion.get("created_at", ""),
                "done": completion.get("done", True),
                "total_duration": completion.get("total_duration", 0),
                "load_duration": completion.get("load_duration", 0),
                "prompt_eval_count": completion.get("prompt_eval_count", 0),
                "eval_count": completion.get("eval_count", 0),
                "eval_duration": completion.get("eval_duration", 0),
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

            # Check if we should use chat or completion endpoint
            endpoint = (
                "/api/chat"
                if isinstance(self._prepare_messages(messages), dict)
                else "/api/generate"
            )

            # Prepare messages or prompt
            message_data = self._prepare_messages(messages)

            # Prepare the request data
            if isinstance(message_data, dict):
                # Chat format
                request_data = {**params, **message_data}
            else:
                # Legacy prompt format
                request_data = {**params, "prompt": message_data}

            # Remove stream parameter if present
            request_data.pop("stream", None)

            # Call the API
            response = self.client.post(endpoint, json=request_data)
            response.raise_for_status()
            completion = response.json()

            # Convert to Message object
            return self._convert_completion_to_message(completion, full_response=True)

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code

            try:
                error_data = e.response.json()
                error_message = error_data.get("error", str(e))
            except Exception:
                error_message = str(e)

            if status_code == 404:
                logger.error(f"Model not found: {self.model_name}")
                raise ModelNotAvailable(self.model_name, f"Model not found: {error_message}")

            if status_code == 429:
                logger.error(f"Rate limit exceeded: {error_message}")
                raise APIError(429, f"Rate limit exceeded: {error_message}")

            if 400 <= status_code < 500:
                # Check for token limit errors in error message
                if any(
                    phrase in error_message.lower()
                    for phrase in ["context length", "token limit", "too long", "too many tokens"]
                ):
                    logger.error(f"Token limit exceeded: {error_message}")
                    raise ContextWindowExceededError(
                        self.model_name, self.count_tokens(messages), self.get_max_tokens()
                    )

                logger.error(f"Bad request: {error_message}")
                raise ParameterError(f"Invalid request: {error_message}")

            logger.error(f"Ollama API error: {error_message}")
            raise APIError(status_code, f"Ollama API error: {error_message}")

        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise APIError(500, f"Request error: {e}")

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

            # Check if we should use chat or completion endpoint
            endpoint = (
                "/api/chat"
                if isinstance(self._prepare_messages(messages), dict)
                else "/api/generate"
            )

            # Prepare messages or prompt
            message_data = self._prepare_messages(messages)

            # Prepare the request data
            if isinstance(message_data, dict):
                # Chat format
                request_data = {**params, **message_data}
            else:
                # Legacy prompt format
                request_data = {**params, "prompt": message_data}

            # Call the API
            with self.client.stream("POST", endpoint, json=request_data) as response:
                response.raise_for_status()

                # Stream configuration
                current_content = ""
                current_raw_chunks: List[Dict[str, Any]] = []

                # Process the stream
                for line in response.iter_lines():
                    if not line.strip():
                        continue

                    try:
                        chunk = json.loads(line)
                        current_raw_chunks.append(chunk)

                        if "response" in chunk:
                            # Add to current content
                            current_content += chunk["response"]
                            yield Message.assistant_message(current_content)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse chunk: {line}")

                # Extract any tool calls at the end
                if "[Function Call]" in current_content:
                    # Use the conversion method to extract tool calls
                    final_message = self._convert_completion_to_message(
                        {"response": current_content}, full_response=False
                    )

                    if hasattr(final_message, "tool_calls") and final_message.tool_calls:
                        yield final_message

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code

            try:
                error_data = e.response.json()
                error_message = error_data.get("error", str(e))
            except Exception:
                error_message = str(e)

            if status_code == 404:
                logger.error(f"Model not found: {self.model_name}")
                raise ModelNotAvailable(self.model_name, f"Model not found: {error_message}")

            if status_code == 429:
                logger.error(f"Rate limit exceeded: {error_message}")
                raise APIError(429, f"Rate limit exceeded: {error_message}")

            if 400 <= status_code < 500:
                # Check for token limit errors in error message
                if any(
                    phrase in error_message.lower()
                    for phrase in ["context length", "token limit", "too long", "too many tokens"]
                ):
                    logger.error(f"Token limit exceeded: {error_message}")
                    raise ContextWindowExceededError(
                        self.model_name, self.count_tokens(messages), self.get_max_tokens()
                    )

                logger.error(f"Bad request: {error_message}")
                raise ParameterError(f"Invalid request: {error_message}")

            logger.error(f"Ollama API error: {error_message}")
            raise APIError(status_code, f"Ollama API error: {error_message}")

        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise APIError(500, f"Request error: {e}")

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

            # Check if we should use chat or completion endpoint
            endpoint = (
                "/api/chat"
                if isinstance(self._prepare_messages(messages), dict)
                else "/api/generate"
            )

            # Prepare messages or prompt
            message_data = self._prepare_messages(messages)

            # Prepare the request data
            if isinstance(message_data, dict):
                # Chat format
                request_data = {**params, **message_data}
            else:
                # Legacy prompt format
                request_data = {**params, "prompt": message_data}

            # Remove stream parameter if present
            request_data.pop("stream", None)

            # Call the API
            response = await self.async_client.post(endpoint, json=request_data)
            response.raise_for_status()
            completion = response.json()

            # Convert to Message object
            return self._convert_completion_to_message(completion, full_response=True)

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code

            try:
                error_data = e.response.json()
                error_message = error_data.get("error", str(e))
            except Exception:
                error_message = str(e)

            if status_code == 404:
                logger.error(f"Model not found: {self.model_name}")
                raise ModelNotAvailable(self.model_name, f"Model not found: {error_message}")

            if status_code == 429:
                logger.error(f"Rate limit exceeded: {error_message}")
                raise APIError(429, f"Rate limit exceeded: {error_message}")

            if 400 <= status_code < 500:
                # Check for token limit errors in error message
                if any(
                    phrase in error_message.lower()
                    for phrase in ["context length", "token limit", "too long", "too many tokens"]
                ):
                    logger.error(f"Token limit exceeded: {error_message}")
                    raise ContextWindowExceededError(
                        self.model_name, self.count_tokens(messages), self.get_max_tokens()
                    )

                logger.error(f"Bad request: {error_message}")
                raise ParameterError(f"Invalid request: {error_message}")

            logger.error(f"Ollama API error: {error_message}")
            raise APIError(status_code, f"Ollama API error: {error_message}")

        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise APIError(500, f"Request error: {e}")

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

        # Check if we should use chat or completion endpoint
        endpoint = (
            "/api/chat" if isinstance(self._prepare_messages(messages), dict) else "/api/generate"
        )

        # Prepare messages or prompt
        message_data = self._prepare_messages(messages)

        # Prepare the request data
        if isinstance(message_data, dict):
            # Chat format
            request_data = {**params, **message_data}
        else:
            # Legacy prompt format
            request_data = {**params, "prompt": message_data}

        # Create a properly typed async generator function
        async def stream_generator() -> AsyncGenerator[Message, None]:
            try:
                # Call the API
                async with self.async_client.stream(
                    "POST", endpoint, json=request_data
                ) as response:
                    response.raise_for_status()

                    # Stream configuration
                    current_content = ""
                    current_raw_chunks: List[Dict[str, Any]] = []

                    # Process the stream
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue

                        try:
                            chunk = json.loads(line)
                            current_raw_chunks.append(chunk)

                            if "response" in chunk:
                                # Add to current content
                                current_content += chunk["response"]
                                yield Message.assistant_message(current_content)
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse chunk: {line}")

                    # Extract any tool calls at the end
                    if "[Function Call]" in current_content:
                        # Use the conversion method to extract tool calls
                        final_message = self._convert_completion_to_message(
                            {"response": current_content}, full_response=False
                        )

                        if hasattr(final_message, "tool_calls") and final_message.tool_calls:
                            yield final_message

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code

                try:
                    error_data = e.response.json()
                    error_message = error_data.get("error", str(e))
                except Exception:
                    error_message = str(e)

                if status_code == 404:
                    logger.error(f"Model not found: {self.model_name}")
                    raise ModelNotAvailable(self.model_name, f"Model not found: {error_message}")

                if status_code == 429:
                    logger.error(f"Rate limit exceeded: {error_message}")
                    raise APIError(429, f"Rate limit exceeded: {error_message}")

                if 400 <= status_code < 500:
                    # Check for token limit errors in error message
                    if any(
                        phrase in error_message.lower()
                        for phrase in [
                            "context length",
                            "token limit",
                            "too long",
                            "too many tokens",
                        ]
                    ):
                        logger.error(f"Token limit exceeded: {error_message}")
                        raise ContextWindowExceededError(
                            self.model_name, self.count_tokens(messages), self.get_max_tokens()
                        )

                    logger.error(f"Bad request: {error_message}")
                    raise ParameterError(f"Invalid request: {error_message}")

                logger.error(f"Ollama API error: {error_message}")
                raise APIError(status_code, f"Ollama API error: {error_message}")

            except httpx.RequestError as e:
                logger.error(f"Request error: {e}")
                raise APIError(500, f"Request error: {e}")

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
        # Try to get model family context size
        for model_family, context_size in self.MODEL_CONTEXT_SIZES.items():
            if model_family in self.model_name.lower():
                return context_size

        # Use a safe default
        logger.warning(f"Unknown context size for model {self.model_name}. Using default.")
        return self.DEFAULT_CONTEXT_SIZE

    def supports_vision(self) -> bool:
        """Check if the provider/model supports vision/images.

        Returns:
            True if vision is supported, False otherwise
        """
        # Check if model name directly matches any vision model names
        if any(model in self.model_name.lower() for model in self.VISION_MODELS):
            return True

        # Check for model families that support vision
        if "vision" in self.model_name.lower():
            return True

        return False

    def supports_tools(self) -> bool:
        """Check if the provider/model supports tool/function calling.

        Returns:
            True if tool calling is supported, False otherwise
        """
        # Ollama models need specific fine-tuning for tool usage
        # Check if model name matches any tool-supporting models
        return any(model in self.model_name.lower() for model in self.TOOL_MODELS)
