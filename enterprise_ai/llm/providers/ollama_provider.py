"""
Ollama provider implementation for LLM integration.

This module implements the LLMProvider interface for local Ollama models.
"""

import json
import time
import re
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
    Tuple,
    cast,
    Set,
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
    """Provider implementation for Ollama models.

    This provider integrates with Ollama to provide access to a wide range of models
    through a consistent interface. It supports user-defined capability detection
    and simplified model validation.
    """

    # Maximum token contexts for different model families (used as fallbacks)
    MODEL_CONTEXT_SIZES = {
        "llama2": 4096,
        "llama3": 8192,
        "llama3.1": 16384,
        "llama3.2": 32768,
        "llama3.3": 131072,
        "mistral": 8192,
        "mixtral": 32768,
        "phi": 2048,
        "phi2": 4096,
        "phi3": 8192,
        "phi4": 131072,
        "gemma": 8192,
        "mpt": 8192,
        "deepseek": 32768,
        "deepseek-coder": 32768,
        "deepseek-v": 32768,
        "deepseek-r": 131072,
    }
    DEFAULT_CONTEXT_SIZE = 8192

    # Basic patterns for identifying model capabilities
    VISION_PATTERNS = [
        "llava",
        "bakllava",
        "vision",
        "multimodal",
        "llama3-vision",
        "llama3.2-vision",
        "llama3.3-vision",
        "deepseek-vl",
        "cogvlm",
        "yi-vl",
        "qwen-vl",
        "clip",
    ]

    def __init__(
        self,
        model_name: str,
        api_base: str = "http://localhost:11434",
        request_timeout: float = 900.0,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        validate_model: bool = False,
        strict_validation: bool = False,
        known_capabilities: Optional[Dict[str, bool]] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Ollama provider.

        Args:
            model_name: Name of the model to use
            api_base: Ollama API base URL
            request_timeout: Request timeout in seconds
            temperature: Model temperature
            max_tokens: Maximum tokens to generate (None uses Ollama's default)
            validate_model: Whether to validate if the model exists
            strict_validation: Whether to raise an exception if model doesn't exist
            known_capabilities: Dictionary of model capabilities (vision, tools, etc.)
            **kwargs: Additional parameters to pass to Ollama API
        """
        self.model_name = model_name
        self.api_base = api_base.rstrip("/")
        self.request_timeout = request_timeout
        self.temperature = temperature
        self.kwargs = kwargs

        # Initialize capability information
        self._model_capabilities = known_capabilities or None
        self._model_info: Optional[Dict[str, Any]] = None
        self._available_models: Optional[List[str]] = None
        self._model_validated: bool = False

        # Initialize clients
        self.client = httpx.Client(timeout=request_timeout, base_url=api_base)
        self.async_client = httpx.AsyncClient(timeout=request_timeout, base_url=api_base)

        # Set default parameters - omit max_tokens if None
        self.default_params = {"model": model_name, "temperature": temperature, **kwargs}

        # Only add num_predict (Ollama's name for max_tokens) if it's specified
        if max_tokens is not None:
            self.default_params["num_predict"] = max_tokens

        # Validate model if requested
        if validate_model:
            self._validate_model(strict_validation)

    def get_model_name(self) -> str:
        """Get the model name.

        Returns:
            Model name
        """
        return self.model_name

    def get_available_models(self) -> List[str]:
        """Fetch the list of available models from Ollama.

        Returns:
            List of available model names
        """
        if self._available_models is not None:
            return self._available_models

        try:
            response = self.client.get("/api/tags")
            response.raise_for_status()
            data = response.json()

            # Extract model names from the response
            model_list: List[str] = []
            models_data = data.get("models", [])

            if isinstance(models_data, list):
                for model_info in models_data:
                    if isinstance(model_info, dict) and "name" in model_info:
                        model_list.append(model_info["name"])

            self._available_models = model_list
            return model_list

        except Exception as e:
            logger.warning(f"Failed to fetch available models: {e}")
            return []

    def _validate_model(self, strict: bool = False) -> bool:
        """Validate if the model exists in available models.

        Args:
            strict: Whether to raise an exception if model isn't found

        Returns:
            True if model is valid, False otherwise
        """
        if self._model_validated:
            return True

        try:
            # Get available models
            available_models = self.get_available_models()

            # Check if exact model name exists
            if self.model_name in available_models:
                self._model_validated = True
                return True

            # Check for model base name without version tag
            base_name = self.model_name.split(":")[0]
            for model in available_models:
                if model.startswith(f"{base_name}:") or model == base_name:
                    self._model_validated = True
                    return True

            if strict:
                suggestions = self._suggest_models(self.model_name, available_models)
                suggestion_msg = ""
                if suggestions:
                    suggestion_msg = f" Did you mean one of these? {', '.join(suggestions)}"

                raise ModelNotAvailable(
                    self.model_name, f"Model not found in Ollama.{suggestion_msg}"
                )

            logger.warning(
                f"Model '{self.model_name}' was not found in available Ollama models. "
                f"Make sure it's pulled or the name is correct."
            )
            return False

        except Exception as e:
            logger.warning(f"Model validation error: {e}")
            return True  # Assume model is valid if we can't check

    def _suggest_models(
        self, model_name: str, available_models: List[str], max_suggestions: int = 3
    ) -> List[str]:
        """Suggest similar models when a model isn't found.

        Args:
            model_name: The requested model name
            available_models: List of available models
            max_suggestions: Maximum number of suggestions to return

        Returns:
            List of suggested model names
        """
        if not available_models:
            return []

        # Simple suggestion based on prefix matching
        base_name = model_name.split(":")[0] if ":" in model_name else model_name

        # First try exact base name matches
        exact_matches = [
            m for m in available_models if m.startswith(f"{base_name}:") or m == base_name
        ]
        if exact_matches:
            return exact_matches[:max_suggestions]

        # Then try partial matches
        partial_matches = [
            m
            for m in available_models
            if base_name.lower() in m.lower()
            or any(part.lower() in m.lower() for part in base_name.split("-"))
        ]

        return partial_matches[:max_suggestions]

    def _detect_capabilities(self) -> Dict[str, bool]:
        """Detect model capabilities based on model name or user-provided capabilities.

        Returns:
            Dictionary of capability flags
        """
        # If capabilities are already set, return them
        if self._model_capabilities is not None:
            return self._model_capabilities

        # Try to get model information if available
        if self._model_info is None:
            try:
                response = self.client.post("/api/show", json={"name": self.model_name})
                if response.status_code == 200:
                    self._model_info = response.json()
            except Exception as e:
                logger.debug(f"Unable to get model info: {e}")

        # Default capabilities
        capabilities = {
            "vision": False,
            "tools": True,  # Most modern models support tools
            "chat_format": True,  # Most modern models support chat format
        }

        # Check for vision capability based on model name
        model_name_lower = self.model_name.lower()
        if any(pattern in model_name_lower for pattern in self.VISION_PATTERNS):
            capabilities["vision"] = True

        # Check model info for more capability hints if available
        if self._model_info:
            details = self._model_info.get("details", {})
            family = details.get("family", "").lower()

            # Vision capability often mentioned in model family
            if any(v in family for v in ["llava", "vision", "clip", "multimodal"]):
                capabilities["vision"] = True

            # Template format can indicate chat capability
            if "template" in self._model_info and self._model_info["template"]:
                if "{{" in self._model_info["template"]:
                    capabilities["chat_format"] = True

        self._model_capabilities = capabilities
        return capabilities

    def _prepare_messages(self, messages: List[Message]) -> Union[str, Dict[str, Any]]:
        """Convert our Message objects to Ollama API format.

        Args:
            messages: List of Message objects

        Returns:
            Formatted string for Ollama prompt or chat messages dict
        """
        # Check capabilities
        capabilities = self._detect_capabilities()
        use_chat_format = capabilities.get("chat_format", True)

        if use_chat_format:
            return self._prepare_chat_messages(messages)
        else:
            return self._prepare_legacy_prompt(messages)

    def _convert_role(self, role: Role) -> str:
        """Convert our role enum to Ollama role string.

        Args:
            role: Role enum value

        Returns:
            Ollama role string
        """
        role_mapping = {
            Role.USER: "user",
            Role.ASSISTANT: "assistant",
            Role.SYSTEM: "system",
            Role.TOOL: "tool",
            Role.AGENT: "user",  # Map agent to user as fallback
        }
        return role_mapping.get(role, "user")  # Default to user if unknown

    def _prepare_chat_messages(self, messages: List[Message]) -> Dict[str, Any]:
        """Prepare messages for Ollama chat format.

        Args:
            messages: List of Message objects

        Returns:
            Dict with messages in Ollama chat format
        """
        ollama_messages = []
        system_message = None

        for message in messages:
            if message.role == Role.SYSTEM:
                # Store system message separately
                system_message = message.content
                continue

            role = self._convert_role(message.role)

            msg_dict: Dict[str, Any] = {"role": role}

            # Handle content and images
            if message.content is not None:
                msg_dict["content"] = message.content
            else:
                msg_dict["content"] = ""

            # Add image if present and model supports vision
            if message.base64_image and role == "user":
                if not self.supports_vision():
                    raise ModelCapabilityError(self.model_name, "vision")

                model_name_lower = self.model_name.lower()

                # Choose image format based on model name pattern
                if (
                    "llama3.2" in model_name_lower
                    or "llama3.3" in model_name_lower
                    or any(
                        v in model_name_lower for v in ["vision-preview", "gemini", "deepseek-vl"]
                    )
                ):
                    # Use images array for newest models
                    if "images" not in msg_dict:
                        msg_dict["images"] = []
                    msg_dict["images"].append(message.base64_image)
                elif any(v in model_name_lower for v in ["llama3", "llama-3"]):
                    # Use embedded format for llama3 models
                    msg_dict["content"] += (
                        f"\n<image>\ndata:image/jpeg;base64,{message.base64_image}\n</image>"
                    )
                else:
                    # Default to markdown format for other models (llava, etc.)
                    msg_dict["content"] += (
                        f"\n![image](data:image/jpeg;base64,{message.base64_image})"
                    )

            # Handle tool calls as formatted text (workaround for models without native tool support)
            if message.tool_calls and role == "assistant":
                for tool_call in message.tool_calls:
                    tool_info = (
                        f"\n[Function Call] {tool_call.function.name}"
                        f"({tool_call.function.arguments})"
                    )
                    msg_dict["content"] += tool_info

            # Add name for tool responses
            if message.role == Role.TOOL and message.name:
                msg_dict["name"] = message.name

            # Add tool_call_id if present
            if message.role == Role.TOOL and message.tool_call_id:
                msg_dict["tool_call_id"] = message.tool_call_id

            ollama_messages.append(msg_dict)

        # Build final request structure
        request: Dict[str, Any] = {"messages": ollama_messages}

        # Add system message if present
        if system_message:
            request["system"] = system_message

        return request

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

            # Map roles to tags
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

                model_name_lower = self.model_name.lower()

                # Choose image format based on model name pattern
                if "llama3.2" in model_name_lower or "llama3.3" in model_name_lower:
                    # Format for llama3.2-vision models
                    content += f"\n<image>\ndata:image/jpeg;base64,{message.base64_image}\n</image>"
                else:
                    # Format for llava and other models
                    content += f"\n![image](data:image/jpeg;base64,{message.base64_image})"

            # Add tool calls as formatted text
            if message.tool_calls and message.role == Role.ASSISTANT:
                for tool_call in message.tool_calls:
                    content += (
                        f"\n[Function Call] {tool_call.function.name}"
                        f"({tool_call.function.arguments})"
                    )

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
        # Check both formats - generate endpoint and chat endpoint
        content = ""
        tool_calls: List[ToolCall] = []

        # Try generate endpoint format
        if "response" in completion:
            content = completion.get("response", "")

            # Check for tool calls in content
            if "[Function Call]" in content:
                # Simple parsing for function calls
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

        # Try chat endpoint format
        elif "message" in completion and isinstance(completion["message"], dict):
            message_data = completion["message"]
            content = message_data.get("content", "")

            # Check for tool_calls in the message
            if "tool_calls" in message_data and message_data["tool_calls"]:
                for tc in message_data["tool_calls"]:
                    if isinstance(tc, dict) and "function" in tc:
                        function_data = tc["function"]

                        # Handle different formats of function arguments
                        arguments = function_data.get("arguments", "")
                        if isinstance(arguments, dict):
                            arguments = json.dumps(arguments)

                        function = Function(name=function_data.get("name", ""), arguments=arguments)

                        tool_calls.append(
                            ToolCall(
                                id=tc.get(
                                    "id",
                                    f"tc_{hash(function_data.get('name', ''))}_{int(time.time())}",
                                ),
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

            metadata_dict = {
                "model": completion.get("model", self.model_name),
                "created_at": completion.get("created_at", ""),
                "done": completion.get("done", True),
            }

            # Add additional fields if they exist
            for field in [
                "total_duration",
                "load_duration",
                "prompt_eval_count",
                "eval_count",
                "eval_duration",
                "done_reason",
            ]:
                if field in completion:
                    metadata_dict[field] = completion.get(field, 0)

            message.metadata["completion"] = metadata_dict

        return message

    def _prepare_request_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare request parameters for Ollama API.

        Args:
            params: Original request parameters

        Returns:
            Modified parameters for Ollama
        """
        request_params = params.copy()

        # Handle max_tokens/num_predict conversion
        if "max_tokens" in request_params:
            # Convert from OpenAI naming convention to Ollama's
            if request_params["max_tokens"] is not None:
                request_params["num_predict"] = request_params.pop("max_tokens")
            else:
                # Remove the parameter if None to use Ollama's default
                request_params.pop("max_tokens")

        return request_params

    def _make_request(
        self, endpoint: str, messages: List[Message], params: Dict[str, Any], use_chat: bool
    ) -> Dict[str, Any]:
        """Make a request to the Ollama API.

        Args:
            endpoint: API endpoint
            messages: Message list to send
            params: Request parameters
            use_chat: Whether to use chat format

        Returns:
            API response
        """
        request_data = params.copy()

        if use_chat:
            # Chat format expects a messages array
            chat_data = self._prepare_chat_messages(messages)
            request_data.update(chat_data)
            endpoint = "/api/chat"
        else:
            # Legacy format expects a prompt string
            prompt = self._prepare_legacy_prompt(messages)
            request_data["prompt"] = prompt
            endpoint = "/api/generate"

        # Explicitly set stream to false for regular requests
        request_data["stream"] = False

        # Make the request
        response = self.client.post(endpoint, json=request_data)
        response.raise_for_status()

        # Handle the response more robustly
        try:
            # First try standard JSON parsing
            return cast(Dict[str, Any], response.json())
        except json.JSONDecodeError:
            # If that fails, try parsing the first line only
            response_text = response.text.strip()
            logger.debug("Received non-standard JSON response, attempting line-by-line parsing")

            # Find the first complete JSON object
            try:
                first_json_end = response_text.find("\n")
                if first_json_end > 0:
                    first_line = response_text[:first_json_end]
                    return cast(Dict[str, Any], json.loads(first_line))
                else:
                    # If no newline, try parsing the whole text again
                    return cast(Dict[str, Any], json.loads(response_text))
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Ollama response: {response_text[:200]}...")
                raise APIError(500, f"Failed to parse Ollama response: {e}")

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

            # Prepare request parameters
            request_params = self._prepare_request_parameters(params)

            # Remove stream parameter if present (we'll handle it separately)
            request_params.pop("stream", None)

            # Check capabilities for API format
            capabilities = self._detect_capabilities()
            use_chat = capabilities.get("chat_format", True)

            # Make the request
            if use_chat:
                endpoint = "/api/chat"
            else:
                endpoint = "/api/generate"

            response = self._make_request(endpoint, messages, request_params, use_chat)

            # Convert to Message object
            return self._convert_completion_to_message(response, full_response=True)

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

            # Always enable streaming
            params["stream"] = True

            # Prepare request parameters
            request_params = self._prepare_request_parameters(params)

            # Determine if we should use chat or generate endpoint
            capabilities = self._detect_capabilities()
            use_chat = capabilities.get("chat_format", True)

            endpoint = "/api/chat" if use_chat else "/api/generate"

            # Prepare message data
            if use_chat:
                message_data = self._prepare_chat_messages(messages)
                request_data = {**request_params, **message_data}
            else:
                prompt = self._prepare_legacy_prompt(messages)
                request_data = {**request_params, "prompt": prompt}

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
                        # Parse the JSON line
                        parsed_data = json.loads(line)

                        # Only add dictionary values to current_raw_chunks
                        if isinstance(parsed_data, dict):
                            dict_chunk: Dict[str, Any] = parsed_data
                            current_raw_chunks.append(dict_chunk)

                            # Handle response based on endpoint
                            if use_chat and "message" in dict_chunk:
                                # Chat endpoint format
                                message_chunk = dict_chunk["message"]
                                if isinstance(message_chunk, dict) and "content" in message_chunk:
                                    current_content += message_chunk["content"]
                                    yield Message.assistant_message(current_content)
                            elif "response" in dict_chunk:
                                # Generate endpoint format
                                current_content += dict_chunk["response"]
                                yield Message.assistant_message(current_content)
                        else:
                            logger.warning(f"Received non-dict JSON: {parsed_data}")

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

            # Prepare request parameters
            request_params = self._prepare_request_parameters(params)

            # Remove stream parameter if present (we'll handle it separately)
            request_params.pop("stream", None)

            # Determine if we should use chat or generate endpoint
            capabilities = self._detect_capabilities()
            use_chat = capabilities.get("chat_format", True)

            endpoint = "/api/chat" if use_chat else "/api/generate"

            # Prepare message data
            if use_chat:
                message_data = self._prepare_chat_messages(messages)
                request_data = {**request_params, **message_data}
            else:
                prompt = self._prepare_legacy_prompt(messages)
                request_data = {**request_params, "prompt": prompt}

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

        # Always enable streaming
        params["stream"] = True

        # Prepare request parameters
        request_params = self._prepare_request_parameters(params)

        # Create a properly typed async generator function
        async def stream_generator() -> AsyncGenerator[Message, None]:
            try:
                # Determine if we should use chat or generate endpoint
                capabilities = self._detect_capabilities()
                use_chat = capabilities.get("chat_format", True)

                endpoint = "/api/chat" if use_chat else "/api/generate"

                # Prepare message data
                if use_chat:
                    message_data = self._prepare_chat_messages(messages)
                    request_data = {**request_params, **message_data}
                else:
                    prompt = self._prepare_legacy_prompt(messages)
                    request_data = {**request_params, "prompt": prompt}

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
                            # Parse the JSON line
                            parsed_data = json.loads(line)

                            # Only add dictionary values to current_raw_chunks
                            if isinstance(parsed_data, dict):
                                dict_chunk: Dict[str, Any] = parsed_data
                                current_raw_chunks.append(dict_chunk)

                                # Handle response based on endpoint
                                if use_chat and "message" in dict_chunk:
                                    # Chat endpoint format
                                    message_chunk = dict_chunk["message"]
                                    if (
                                        isinstance(message_chunk, dict)
                                        and "content" in message_chunk
                                    ):
                                        current_content += message_chunk["content"]
                                        yield Message.assistant_message(current_content)
                                elif "response" in dict_chunk:
                                    # Generate endpoint format
                                    current_content += dict_chunk["response"]
                                    yield Message.assistant_message(current_content)
                            else:
                                logger.warning(f"Received non-dict JSON: {parsed_data}")

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
        # Try to get context size from model info
        if self._model_info and "details" in self._model_info:
            _ = self._model_info["details"]

            # First check if there's a context window explicitly specified
            if "model_info" in self._model_info:
                model_info = self._model_info["model_info"]
                for key in [
                    "llama.context_length",
                    "general.context_length",
                    "context_length",
                    "max_context_length",
                ]:
                    if key in model_info and isinstance(model_info[key], (int, str)):
                        try:
                            return int(model_info[key])
                        except (ValueError, TypeError):
                            pass

        # Extract from model name if possible
        model_name_lower = self.model_name.lower()
        context_match = re.search(r"[-_](\d+)[kK](?:ctx)?", model_name_lower)
        if context_match:
            return int(context_match.group(1)) * 1024

        # Try to get model family context size as fallback
        for model_family, context_size in self.MODEL_CONTEXT_SIZES.items():
            if model_family in model_name_lower:
                return context_size

        # Use a safe default
        logger.warning(f"Unknown context size for model {self.model_name}. Using default.")
        return self.DEFAULT_CONTEXT_SIZE

    def supports_vision(self) -> bool:
        """Check if the provider/model supports vision/images.

        Returns:
            True if vision is supported, False otherwise
        """
        capabilities = self._detect_capabilities()
        return capabilities.get("vision", False)

    def supports_tools(self) -> bool:
        """Check if the provider/model supports tool/function calling.

        Returns:
            True if tool calling is supported, False otherwise
        """
        capabilities = self._detect_capabilities()
        return capabilities.get("tools", True)  # Most modern models support tools
