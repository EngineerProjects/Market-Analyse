"""
Ollama provider implementation for LLM integration.

This module implements the LLMProvider interface for local Ollama models with
enhanced resilience, performance optimization, and advanced capabilities.
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
    OllamaError,
    OllamaModelUnavailableError,
    OllamaConnectionError,
)
from enterprise_ai.llm.utils import TokenCounter, retry_with_exponential_backoff
from enterprise_ai.logger import get_logger, trace_execution

# Initialize logger
logger = get_logger("llm.ollama")


class OllamaProvider(LLMProvider):
    """Provider implementation for Ollama models.

    This provider integrates with Ollama to provide access to a wide range of models
    through a consistent interface. It supports robust error handling, performance
    optimization, and advanced model management.
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
        "qwen": 32768,
        "qwen2": 32768,
        "yi": 4096,
        "yi-1.5": 8192,
        "yi-2": 128000,
        "wizard": 8192,
        "falcon": 4096,
        "openchat": 8192,
        "vicuna": 4096,
    }
    DEFAULT_CONTEXT_SIZE = 8192

    # Basic patterns for identifying model capabilities
    VISION_PATTERNS = [
        "llava",
        "bakllava",
        "vision",
        "multimodal",
        "-vision",
        "vl-",
        "-vl",
        "cogvlm",
        "qwen-vl",
        "clip",
    ]

    # Models known to support function/tool calling
    TOOL_CALLING_PATTERNS = [
        "llama3",
        "llama3.1",
        "llama3.2",
        "llama3.3",
        "gpt-4",
        "claude-3",
        "qwen2",
        "yi-1.5",
        "yi-2",
        "mistral-nemo",
        "gemini",
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
        connection_pool_size: int = 10,
        model_cache_size: int = 3,
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
            connection_pool_size: Size of the connection pool for HTTP requests
            model_cache_size: Maximum number of models to keep loaded in memory
            **kwargs: Additional parameters to pass to Ollama API
        """
        self.model_name = model_name
        self.api_base = api_base.rstrip("/")
        self.request_timeout = request_timeout
        self.temperature = temperature
        self.model_cache_size = model_cache_size
        self.kwargs = kwargs

        # Initialize capability information
        self._model_capabilities = known_capabilities or None
        self._model_info: Optional[Dict[str, Any]] = None
        self._available_models: Optional[List[str]] = None
        # Explicitly initialize as boolean to avoid type error
        self._model_validated: bool = False
        self._ollama_version: Optional[str] = None

        # Create connection pools for better performance
        limits = httpx.Limits(max_connections=connection_pool_size)

        # Initialize clients
        self.client = httpx.Client(
            timeout=request_timeout,
            base_url=self.api_base,
            limits=limits,
            http2=True,  # Enable HTTP/2 for better performance
        )

        self.async_client = httpx.AsyncClient(
            timeout=request_timeout, base_url=self.api_base, limits=limits, http2=True
        )

        # Set default parameters - omit num_predict if None
        self.default_params = {"model": model_name, "temperature": temperature, **kwargs}

        # Only add num_predict (Ollama's name for max_tokens) if it's specified
        if max_tokens is not None:
            self.default_params["num_predict"] = max_tokens

        # Detect Ollama version for API behavior adjustments
        self._detect_ollama_version()

        # Validate model if requested
        if validate_model:
            self._validate_model(strict_validation)

        # Manage model cache if enabled
        if model_cache_size > 0:
            self._manage_model_cache(model_cache_size)

    def _detect_ollama_version(self) -> str:
        """Detect the Ollama API version.

        Returns:
            String representing the API version or "unknown"
        """
        if self._ollama_version is not None:
            return self._ollama_version

        try:
            response = self.client.get("/api/version")
            response.raise_for_status()
            data = response.json()
            version = str(data.get("version", "unknown"))
            self._ollama_version = version
            logger.debug(f"Detected Ollama version: {version}")
            return version
        except Exception as e:
            logger.warning(f"Failed to detect Ollama version: {e}")
            self._ollama_version = "unknown"
            return "unknown"

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

        except httpx.ConnectError as e:
            logger.warning(f"Connection error when fetching models: {e}")
            raise OllamaConnectionError(f"Failed to connect to Ollama server: {e}")
        except httpx.HTTPStatusError as e:
            logger.warning(f"HTTP error when fetching models: {e}")
            raise APIError(e.response.status_code, f"Failed to fetch models: {e}")
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
                self._model_validated = True  # Explicitly use boolean True
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

                raise OllamaModelUnavailableError(
                    self.model_name, f"Model not found in Ollama.{suggestion_msg}"
                )

            logger.warning(
                f"Model '{self.model_name}' was not found in available Ollama models. "
                f"Make sure it's pulled or the name is correct."
            )
            return False

        except OllamaModelUnavailableError:
            # Re-raise specific exception
            raise
        except Exception as e:
            logger.warning(f"Model validation error: {e}")
            if strict:
                raise OllamaError(f"Error validating model: {e}")
            return False

    def validate_model(self, strict: bool = False) -> bool:
        """Public alias for _validate_model.

        Args:
            strict: Whether to raise an exception if model isn't found

        Returns:
            True if model is valid, False otherwise
        """
        return self._validate_model(strict)

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

        # Try different matching strategies
        matches = []

        # 1. Exact base name matches
        exact_matches = [
            m for m in available_models if m.startswith(f"{base_name}:") or m == base_name
        ]
        matches.extend(exact_matches)

        # 2. Substring matches
        if not matches:
            substring_matches = [m for m in available_models if base_name.lower() in m.lower()]
            matches.extend(substring_matches)

        # 3. Model family matches (e.g., suggest llama3 if llama2 was requested)
        if not matches:
            # Extract model family
            family_match = re.match(r"([a-zA-Z]+)(\d*)", base_name)
            if family_match:
                family = family_match.group(1).lower()
                family_matches = [m for m in available_models if m.lower().startswith(family)]
                matches.extend(family_matches)

        # Return unique suggestions up to max_suggestions
        unique_matches = []
        for m in matches:
            if m not in unique_matches:
                unique_matches.append(m)
                if len(unique_matches) >= max_suggestions:
                    break

        return unique_matches

    def _ensure_model_loaded(self, model_name: Optional[str] = None) -> bool:
        """Ensure the specified model is loaded with retries.

        Args:
            model_name: Name of model to load (defaults to self.model_name)

        Returns:
            True if model was loaded successfully, False otherwise
        """
        target_model = model_name or self.model_name
        max_attempts = 3
        backoff_time = 2  # seconds

        for attempt in range(1, max_attempts + 1):
            try:
                # Check if model exists
                available_models = self.get_available_models()
                if target_model in available_models:
                    return True

                # Try to pull the model
                logger.info(
                    f"Model {target_model} not found. Attempting to pull (attempt {attempt}/{max_attempts})..."
                )
                response = self.client.post(
                    "/api/pull",
                    json={"name": target_model},
                    timeout=600.0,  # Longer timeout for pulling
                )
                response.raise_for_status()

                # Verify model was pulled successfully
                for _ in range(10):  # Poll for model availability
                    time.sleep(3)
                    if target_model in self.get_available_models():
                        logger.info(f"Successfully pulled model {target_model}")
                        return True

                logger.warning(f"Model {target_model} not available after pull")
            except Exception as e:
                logger.warning(
                    f"Error pulling model {target_model} (attempt {attempt}/{max_attempts}): {e}"
                )

            # Backoff before retry
            if attempt < max_attempts:
                sleep_time = backoff_time * (2 ** (attempt - 1))
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)

        # All attempts failed
        logger.error(f"Failed to load model {target_model} after {max_attempts} attempts")
        return False

    def _check_model_health(self, model_name: str) -> Dict[str, Any]:
        """Check health status of a model.

        Args:
            model_name: Name of model to check

        Returns:
            Dictionary with health information
        """
        # Add explicit type annotation for the dictionary
        health_info: Dict[str, Union[bool, None, int]] = {
            "available": False,
            "loaded": False,
            "response_time_ms": None,
            "memory_usage": None,
        }

        try:
            # Check availability
            available_models = self.get_available_models()
            health_info["available"] = model_name in available_models

            if health_info["available"]:
                # Check if loaded by sending a minimal prompt
                start_time = time.time()
                response = self.client.post(
                    "/api/generate",
                    json={"model": model_name, "prompt": "test", "max_tokens": 1},
                    timeout=50.0,
                )
                response_time = time.time() - start_time
                # This line will now work with the explicit type annotation
                health_info["response_time_ms"] = int(response_time * 1000)
                health_info["loaded"] = response.status_code == 200

                # Try to get memory usage if available
                try:
                    usage_response = self.client.get(f"/api/show?name={model_name}")
                    if usage_response.status_code == 200:
                        model_info = usage_response.json()
                        health_info["memory_usage"] = model_info.get("parameters", 0)
                except Exception:
                    pass  # Ignore errors getting memory usage
        except Exception as e:
            logger.warning(f"Error checking model health: {e}")

        return health_info

    def _manage_model_cache(self, max_models: int = 3) -> None:
        """Manage the model cache to prevent too many models loaded.

        Args:
            max_models: Maximum number of models to keep loaded
        """
        try:
            available_models = self.get_available_models()

            # Get health info for each model
            models_health = []
            for model in available_models:
                health = self._check_model_health(model)
                if health["loaded"]:
                    models_health.append((model, health["response_time_ms"] or 9999))

            # If we have too many models loaded, unload the slowest ones
            if len(models_health) > max_models:
                # Sort by response time (descending)
                models_health.sort(key=lambda x: x[1], reverse=True)

                # Unload excess models, but keep current model loaded
                models_to_unload = [
                    m
                    for m, _ in models_health[: len(models_health) - max_models]
                    if m != self.model_name
                ]

                for model in models_to_unload:
                    logger.info(f"Unloading model from cache: {model}")
                    try:
                        self.client.delete(f"/api/delete?name={model}")
                    except Exception as e:
                        logger.warning(f"Error unloading model {model}: {e}")
        except Exception as e:
            logger.warning(f"Error managing model cache: {e}")

    def _detect_model_capabilities(self) -> Dict[str, bool]:
        """Detect model capabilities based on model name and known patterns.

        Returns:
            Dictionary with capability flags
        """
        if self._model_capabilities is not None:
            return self._model_capabilities

        # Default capabilities
        capabilities = {"vision": False, "tools": False}

        # Check model name against known patterns
        model_name_lower = self.model_name.lower()

        # Check vision capability
        for pattern in self.VISION_PATTERNS:
            if pattern.lower() in model_name_lower:
                capabilities["vision"] = True
                break

        # Check tool calling capability
        for pattern in self.TOOL_CALLING_PATTERNS:
            if pattern.lower() in model_name_lower:
                capabilities["tools"] = True
                break

        # Store and return capabilities
        self._model_capabilities = capabilities
        return capabilities

    def _prepare_chat_messages(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert our Message objects to Ollama chat format.

        Args:
            messages: List of Message objects

        Returns:
            List of Ollama-formatted message dictionaries
        """
        ollama_messages = []

        for message in messages:
            role = message.role.value

            # Map message roles to Ollama format
            if role == "system":
                ollama_role = "system"
            elif role == "user":
                ollama_role = "user"
            elif role == "assistant":
                ollama_role = "assistant"
            elif role == "tool":
                # Ollama doesn't directly support tool messages, convert to user
                ollama_role = "user"
                content = f"Tool result from {message.name}: {message.content}"
                ollama_messages.append({"role": ollama_role, "content": content})
                continue
            else:
                # Skip unknown roles
                continue

            # Add content
            msg_dict: Dict[str, Any] = {"role": ollama_role}

            # Handle images for vision models
            if message.base64_image and role == "user":
                if not self.supports_vision():
                    raise ModelCapabilityError(self.model_name, "vision")

                # Format depends on Ollama version and model type
                # Try content list format
                content_list = []

                # Add text content if present
                if message.content:
                    content_list.append({"type": "text", "text": message.content})

                # Add image content
                content_list.append(
                    {
                        "type": "image",
                        "data": message.base64_image,
                    }
                )

                msg_dict["content"] = content_list
            else:
                # Regular content
                msg_dict["content"] = message.content if message.content is not None else ""

            ollama_messages.append(msg_dict)

        return ollama_messages

    def _prepare_chat_payload(self, messages: List[Message], **kwargs: Any) -> Dict[str, Any]:
        """Prepare the payload for a chat API request.

        Args:
            messages: List of Message objects
            **kwargs: Additional parameters

        Returns:
            Chat API request payload
        """
        # Start with default parameters
        payload = self.default_params.copy()

        # Update with any additional parameters
        payload.update(kwargs)

        # Ensure model is set
        payload["model"] = payload.get("model", self.model_name)

        # Convert our messages to Ollama format
        payload["messages"] = self._prepare_chat_messages(messages)

        # Handle tool calls if requested and supported
        if "tools" in kwargs and self.supports_tools():
            payload["options"] = payload.get("options", {})
            payload["options"]["tools"] = kwargs["tools"]

        return payload

    def _prepare_legacy_prompt(self, messages: List[Message]) -> str:
        """Convert messages to a legacy text prompt format.

        Args:
            messages: List of Message objects

        Returns:
            Formatted prompt string
        """
        prompt = ""

        # Extract system message if present
        system_content = None
        for message in messages:
            if message.role == Role.SYSTEM:
                system_content = message.content
                break

        # Add system message if present
        if system_content:
            prompt += f"<s>\n{system_content}\n</s>\n\n"

        # Add conversation messages
        for message in messages:
            if message.role == Role.SYSTEM:
                continue  # Already handled

            if message.role == Role.USER:
                prompt += f"User: {message.content}\n"
            elif message.role == Role.ASSISTANT:
                prompt += f"Assistant: {message.content}\n"
            elif message.role == Role.TOOL:
                prompt += f"Tool ({message.name}): {message.content}\n"

        # Add final assistant prompt
        prompt += "Assistant: "

        return prompt

    def _parse_completion(self, response: Dict[str, Any]) -> str:
        """Parse completion response with fallbacks for API variations.

        Args:
            response: Raw API response

        Returns:
            Extracted completion text
        """
        # Try the standard response format first
        if "response" in response:
            return str(response["response"])

        # Fallbacks for other formats
        if "completion" in response:
            return str(response["completion"])
        if "content" in response and isinstance(response["content"], str):
            return str(response["content"])
        if "message" in response and "content" in response["message"]:
            return str(response["message"]["content"])

        # Last resort, convert the whole response to a string
        logger.warning(f"Unknown response format: {response}")
        return str(response)

    def _parse_streaming_delta(self, data: Dict[str, Any]) -> str:
        """Extract content from a streaming response chunk with fallbacks.

        Args:
            data: Response chunk data

        Returns:
            Extracted text delta or empty string
        """
        # Try standard format
        if "response" in data:
            return str(data["response"])

        # Try other known formats
        if "delta" in data:
            return str(data["delta"])
        if "chunk" in data:
            return str(data["chunk"])
        if "content" in data and isinstance(data["content"], str):
            return str(data["content"])

        # Try to detect if this is a message completion format
        if "message" in data and isinstance(data["message"], dict):
            if "content" in data["message"]:
                return str(data["message"]["content"])
            if "delta" in data["message"]:
                return str(data["message"]["delta"])

        # No content found
        return ""

    @retry_with_exponential_backoff(max_retries=2)
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
            # If fallback_models specified, try with fallbacks
            fallback_models = kwargs.pop("fallback_models", [])
            if fallback_models:
                return self._complete_with_fallbacks(messages, fallback_models, **kwargs)

            # Otherwise, standard completion
            return self._complete_internal(messages, **kwargs)
        except OllamaModelUnavailableError:
            # Try auto-suggest fallbacks if model not available
            available_models = self.get_available_models()
            auto_fallbacks = self._suggest_models(
                self.model_name, available_models, max_suggestions=2
            )

            if auto_fallbacks:
                logger.info(
                    f"Model {self.model_name} not available, trying suggested alternatives: {auto_fallbacks}"
                )
                return self._complete_with_fallbacks(messages, auto_fallbacks, **kwargs)

            # No fallbacks available, re-raise
            raise

    def _complete_with_fallbacks(
        self, messages: List[Message], fallback_models: List[str], **kwargs: Any
    ) -> Message:
        """Try completion with multiple fallback models.

        Args:
            messages: List of messages
            fallback_models: List of fallback model names
            **kwargs: Additional options

        Returns:
            Completion message

        Raises:
            ModelNotAvailable: If all models fail
        """
        # Save original model
        original_model = self.model_name
        last_error: Optional[OllamaModelUnavailableError] = None

        # Try with each fallback model
        for fallback in fallback_models:
            try:
                logger.info(f"Trying fallback model: {fallback}")
                self.model_name = fallback

                # Update default params
                self.default_params["model"] = fallback

                # Attempt completion
                result = self._complete_internal(messages, **kwargs)

                # Add fallback info to metadata
                if result.metadata is None:
                    result.metadata = {}

                result.metadata["fallback_info"] = {
                    "original_model": original_model,
                    "fallback_model": fallback,
                }

                return result
            except Exception as e:
                logger.warning(f"Fallback model {fallback} failed: {e}")
                # Create proper OllamaModelUnavailableError for type safety
                if isinstance(e, OllamaModelUnavailableError):
                    last_error = e
                else:
                    last_error = OllamaModelUnavailableError(
                        fallback, f"Fallback failed with error: {str(e)}"
                    )

        # Restore original model
        self.model_name = original_model
        self.default_params["model"] = original_model

        # All fallbacks failed
        if last_error:
            raise last_error
        else:
            raise OllamaModelUnavailableError(
                original_model, f"All fallback models failed: {fallback_models}"
            )

    def _complete_internal(self, messages: List[Message], **kwargs: Any) -> Message:
        """Internal completion implementation.

        Args:
            messages: List of messages
            **kwargs: Additional options

        Returns:
            Completion message
        """
        use_chat_api = True

        # Check if legacy /api/generate should be used (for compatibility)
        if kwargs.pop("use_legacy_api", False):
            use_chat_api = False

        # Chat API is preferred and has better support for complex interactions
        if use_chat_api:
            return self._complete_chat(messages, **kwargs)
        else:
            return self._complete_generate(messages, **kwargs)

    def _complete_chat(self, messages: List[Message], **kwargs: Any) -> Message:
        """Generate a completion using the chat API.

        Args:
            messages: List of messages
            **kwargs: Additional options

        Returns:
            Completion message
        """
        try:
            # Prepare payload
            payload = self._prepare_chat_payload(messages, **kwargs)

            # Call the API
            response = self.client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

            # Extract content
            content = self._parse_completion(data)

            # Create response message
            message = Message.assistant_message(content)

            # Add metadata
            message.metadata = {
                "response_format": "chat",
                "model": self.model_name,
                "raw_response": {
                    "done": data.get("done", True),
                    "total_duration": data.get("total_duration", 0),
                    "load_duration": data.get("load_duration", 0),
                    "eval_count": data.get("eval_count", 0),
                },
            }

            # Check if there are tool calls
            if self.supports_tools() and kwargs.get("tools"):
                # Try to parse tool calls from the content
                try:
                    self._extract_tool_calls(content, message)
                except Exception as e:
                    logger.warning(f"Error extracting tool calls: {e}")

            return message

        except httpx.ConnectError as e:
            logger.error(f"Connection error: {e}")
            raise OllamaConnectionError(f"Failed to connect to Ollama: {e}")
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code

            # Check for specific error types
            if status_code == 404:
                raise OllamaModelUnavailableError(self.model_name, f"Model not found: {e}")
            elif status_code == 400:
                error_text = e.response.text
                if "context window" in error_text.lower() or "token limit" in error_text.lower():
                    raise ContextWindowExceededError(
                        self.model_name, self.count_tokens(messages), self.get_max_tokens()
                    )
                else:
                    raise ParameterError(f"Invalid request parameters: {e}")
            else:
                raise APIError(status_code, f"Ollama API error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise OllamaError(f"Error generating completion: {e}")

    def _complete_generate(self, messages: List[Message], **kwargs: Any) -> Message:
        """Generate a completion using the legacy generate API.

        Args:
            messages: List of messages
            **kwargs: Additional options

        Returns:
            Completion message
        """
        try:
            # Convert messages to a prompt
            prompt = self._prepare_legacy_prompt(messages)

            # Prepare payload
            payload = self.default_params.copy()
            payload.update(kwargs)
            payload["prompt"] = prompt

            # Remove chat-specific parameters
            payload.pop("messages", None)
            payload.pop("tools", None)
            payload.pop("tool_choice", None)

            # Call the API
            response = self.client.post("/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()

            # Extract content
            content = self._parse_completion(data)

            # Create response message
            message = Message.assistant_message(content)

            # Add metadata
            message.metadata = {
                "response_format": "generate",
                "model": self.model_name,
                "raw_response": {
                    "done": data.get("done", True),
                    "total_duration": data.get("total_duration", 0),
                    "load_duration": data.get("load_duration", 0),
                    "eval_count": data.get("eval_count", 0),
                },
            }

            return message

        except httpx.ConnectError as e:
            logger.error(f"Connection error: {e}")
            raise OllamaConnectionError(f"Failed to connect to Ollama: {e}")
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code

            # Check for specific error types
            if status_code == 404:
                raise OllamaModelUnavailableError(self.model_name, f"Model not found: {e}")
            elif status_code == 400:
                error_text = e.response.text
                if "context window" in error_text.lower() or "token limit" in error_text.lower():
                    raise ContextWindowExceededError(
                        self.model_name, self.count_tokens(messages), self.get_max_tokens()
                    )
                else:
                    raise ParameterError(f"Invalid request parameters: {e}")
            else:
                raise APIError(status_code, f"Ollama API error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise OllamaError(f"Error generating completion: {e}")

    def _extract_tool_calls(self, content: str, message: Message) -> None:
        """Extract tool calls from response content.

        Args:
            content: Response content text
            message: Message to update with tool calls
        """
        # Look for function call patterns in the content
        function_blocks = []

        # Try to find JSON blocks
        json_pattern = r"```(?:json)?\s*({[\s\S]*?})\s*```"
        json_matches = re.finditer(json_pattern, content, re.MULTILINE)

        for match in json_matches:
            try:
                # Extract the JSON block
                json_str = match.group(1)
                data = json.loads(json_str)

                # Check if it looks like a function call
                if "name" in data and ("arguments" in data or "params" in data or "args" in data):
                    function_blocks.append(data)
            except json.JSONDecodeError:
                continue

        # If no JSON blocks found, try plain text parsing
        if not function_blocks:
            # Look for function call syntax: function_name(args)
            fn_pattern = r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(([^)]*)\)"
            fn_matches = re.finditer(fn_pattern, content)

            for match in fn_matches:
                try:
                    name = match.group(1)
                    args_str = match.group(2)

                    # Try to parse arguments as JSON or as a simple string
                    try:
                        # Add braces to make it valid JSON
                        args_json = f"{{{args_str}}}"
                        arguments = json.loads(args_json)
                    except json.JSONDecodeError:
                        # Fall back to simple string
                        arguments = {"text": args_str}

                    function_blocks.append({"name": name, "arguments": arguments})
                except Exception:
                    continue

        # Convert to ToolCall objects
        tool_calls = []

        for idx, fn_data in enumerate(function_blocks):
            try:
                # Get function name
                name = fn_data.get("name", "")

                # Get arguments, handling different formats
                args: Union[Dict[str, Any], str] = {}
                if "arguments" in fn_data:
                    args = fn_data["arguments"]
                elif "params" in fn_data:
                    args = fn_data["params"]
                elif "args" in fn_data:
                    args = fn_data["args"]

                # Convert arguments to JSON string if not already
                args_str_json: str  # Use a different variable name here
                if not isinstance(args, str):
                    args_str_json = json.dumps(args)
                else:
                    args_str_json = args

                # Create function object
                function = Function(name=name, arguments=args_str_json)

                # Create tool call
                tool_calls.append(
                    ToolCall(
                        id=f"call_{idx}_{int(time.time())}", type="function", function=function
                    )
                )
            except Exception as e:
                logger.warning(f"Error creating tool call: {e}")

        # Update message with tool calls if any found
        if tool_calls:
            message.tool_calls = tool_calls

            # Update content to remove function call blocks
            clean_content = content
            for match in re.finditer(json_pattern, content, re.MULTILINE):
                clean_content = clean_content.replace(match.group(0), "")

            # Trim any extra whitespace
            clean_content = clean_content.strip()
            if clean_content:
                message.content = clean_content
            else:
                message.content = None

    @retry_with_exponential_backoff(max_retries=2)
    def complete_stream(
        self, messages: List[Message], **kwargs: Any
    ) -> Generator[Message, None, None]:
        """Generate a streaming completion with improved error handling.

        Args:
            messages: List of messages in the conversation
            **kwargs: Additional completion options

        Yields:
            Partial completion messages

        Raises:
            APIError: If there's an API error
            ModelNotAvailable: If the model is not available
        """
        use_chat_api = True

        # Check if legacy /api/generate should be used (for compatibility)
        if kwargs.pop("use_legacy_api", False):
            use_chat_api = False

        # Chat API is preferred and has better support for complex interactions
        if use_chat_api:
            yield from self._complete_chat_stream(messages, **kwargs)
        else:
            yield from self._complete_generate_stream(messages, **kwargs)

    def _complete_chat_stream(
        self, messages: List[Message], **kwargs: Any
    ) -> Generator[Message, None, None]:
        """Generate a streaming completion using the chat API.

        Args:
            messages: List of messages
            **kwargs: Additional options

        Yields:
            Partial completion messages
        """
        # Prepare payload
        payload = self._prepare_chat_payload(messages, **kwargs)
        payload["stream"] = True

        content_buffer = ""
        done = False
        retries = 0
        max_retries = 3

        while not done and retries <= max_retries:
            try:
                # Stream from API
                with self.client.stream(
                    "POST", "/api/chat", json=payload, timeout=None
                ) as response:
                    response.raise_for_status()

                    for line in response.iter_lines():
                        if not line:
                            continue

                        try:
                            data = json.loads(line)

                            # Handle done flag
                            if data.get("done", False):
                                done = True
                                break

                            # Extract content with fallbacks
                            new_content = self._parse_streaming_delta(data)
                            if new_content:
                                content_buffer += new_content
                                yield Message.assistant_message(content_buffer)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in stream: {line}")
                            continue

                    # If we made it through the loop, we're done
                    done = True
            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                # Only retry on connection issues
                retries += 1
                if retries <= max_retries:
                    logger.warning(
                        f"Stream connection error, retrying ({retries}/{max_retries}): {e}"
                    )
                    time.sleep(1 * retries)  # Incremental backoff
                else:
                    # Final attempt failed
                    logger.error(f"Stream connection failed after {max_retries} retries: {e}")
                    raise OllamaConnectionError(f"Streaming connection failed: {e}")
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                if status_code == 404:
                    raise OllamaModelUnavailableError(self.model_name, f"Model not found: {e}")
                else:
                    raise APIError(status_code, f"Ollama API error during streaming: {e}")
            except Exception as e:
                # Don't retry on other errors
                logger.error(f"Stream error: {e}")
                raise OllamaError(f"Error in streaming completion: {e}")

        # Yield the final complete message
        if content_buffer:
            final_message = Message.assistant_message(content_buffer)

            # Try to extract tool calls if requested
            if self.supports_tools() and kwargs.get("tools"):
                try:
                    self._extract_tool_calls(content_buffer, final_message)
                except Exception as e:
                    logger.warning(f"Error extracting tool calls from stream: {e}")

            yield final_message

    def _complete_generate_stream(
        self, messages: List[Message], **kwargs: Any
    ) -> Generator[Message, None, None]:
        """Generate a streaming completion using the legacy generate API.

        Args:
            messages: List of messages
            **kwargs: Additional options

        Yields:
            Partial completion messages
        """
        # Convert messages to a prompt
        prompt = self._prepare_legacy_prompt(messages)

        # Prepare payload
        payload = self.default_params.copy()
        payload.update(kwargs)
        payload["prompt"] = prompt
        payload["stream"] = True

        # Remove chat-specific parameters
        payload.pop("messages", None)
        payload.pop("tools", None)
        payload.pop("tool_choice", None)

        content_buffer = ""
        done = False
        retries = 0
        max_retries = 3

        while not done and retries <= max_retries:
            try:
                # Stream from API
                with self.client.stream(
                    "POST", "/api/generate", json=payload, timeout=None
                ) as response:
                    response.raise_for_status()

                    for line in response.iter_lines():
                        if not line:
                            continue

                        try:
                            data = json.loads(line)

                            # Handle done flag
                            if data.get("done", False):
                                done = True
                                break

                            # Extract content with fallbacks
                            new_content = self._parse_streaming_delta(data)
                            if new_content:
                                content_buffer += new_content
                                yield Message.assistant_message(content_buffer)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in stream: {line}")
                            continue

                    # If we made it through the loop, we're done
                    done = True
            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                # Only retry on connection issues
                retries += 1
                if retries <= max_retries:
                    logger.warning(
                        f"Stream connection error, retrying ({retries}/{max_retries}): {e}"
                    )
                    time.sleep(1 * retries)  # Incremental backoff
                else:
                    # Final attempt failed
                    logger.error(f"Stream connection failed after {max_retries} retries: {e}")
                    raise OllamaConnectionError(f"Streaming connection failed: {e}")
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                if status_code == 404:
                    raise OllamaModelUnavailableError(self.model_name, f"Model not found: {e}")
                else:
                    raise APIError(status_code, f"Ollama API error during streaming: {e}")
            except Exception as e:
                # Don't retry on other errors
                logger.error(f"Stream error: {e}")
                raise OllamaError(f"Error in streaming completion: {e}")

        # Yield the final complete message
        if content_buffer:
            yield Message.assistant_message(content_buffer)

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
        """
        try:
            # Check if legacy API should be used
            use_chat_api = not kwargs.pop("use_legacy_api", False)

            if use_chat_api:
                return await self._acomplete_chat(messages, **kwargs)
            else:
                return await self._acomplete_generate(messages, **kwargs)
        except OllamaModelUnavailableError as e:
            # Try auto-suggest fallbacks if model not available
            available_models = self.get_available_models()
            auto_fallbacks = self._suggest_models(
                self.model_name, available_models, max_suggestions=2
            )

            if auto_fallbacks:
                logger.info(
                    f"Model {self.model_name} not available, trying suggested alternatives: {auto_fallbacks}"
                )

                # Try each fallback
                last_error: OllamaModelUnavailableError = e
                original_model = self.model_name

                for fallback in auto_fallbacks:
                    try:
                        self.model_name = fallback
                        self.default_params["model"] = fallback

                        # Try completion with fallback
                        if use_chat_api:
                            result = await self._acomplete_chat(messages, **kwargs)
                        else:
                            result = await self._acomplete_generate(messages, **kwargs)

                        # Add fallback info
                        if result.metadata is None:
                            result.metadata = {}

                        result.metadata["fallback_info"] = {
                            "original_model": original_model,
                            "fallback_model": fallback,
                        }

                        return result
                    except Exception as fallback_error:
                        logger.warning(f"Fallback model {fallback} failed: {fallback_error}")
                        if isinstance(fallback_error, OllamaModelUnavailableError):
                            last_error = fallback_error
                        else:
                            # Create proper error type
                            last_error = OllamaModelUnavailableError(
                                fallback, f"Failed with error: {str(fallback_error)}"
                            )

                # Restore original model
                self.model_name = original_model
                self.default_params["model"] = original_model

                # All fallbacks failed
                raise last_error

            # No fallbacks available
            raise

    async def _acomplete_chat(self, messages: List[Message], **kwargs: Any) -> Message:
        """Generate a completion using the chat API asynchronously.

        Args:
            messages: List of messages
            **kwargs: Additional options

        Returns:
            Completion message
        """
        try:
            # Prepare payload
            payload = self._prepare_chat_payload(messages, **kwargs)

            # Call the API
            response = await self.async_client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()

            # Extract content
            content = self._parse_completion(data)

            # Create response message
            message = Message.assistant_message(content)

            # Add metadata
            message.metadata = {
                "response_format": "chat",
                "model": self.model_name,
                "raw_response": {
                    "done": data.get("done", True),
                    "total_duration": data.get("total_duration", 0),
                    "load_duration": data.get("load_duration", 0),
                    "eval_count": data.get("eval_count", 0),
                },
            }

            # Check if there are tool calls
            if self.supports_tools() and kwargs.get("tools"):
                # Try to parse tool calls from the content
                try:
                    self._extract_tool_calls(content, message)
                except Exception as e:
                    logger.warning(f"Error extracting tool calls: {e}")

            return message

        except httpx.ConnectError as e:
            logger.error(f"Connection error: {e}")
            raise OllamaConnectionError(f"Failed to connect to Ollama: {e}")
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code

            # Check for specific error types
            if status_code == 404:
                raise OllamaModelUnavailableError(self.model_name, f"Model not found: {e}")
            elif status_code == 400:
                error_text = e.response.text
                if "context window" in error_text.lower() or "token limit" in error_text.lower():
                    raise ContextWindowExceededError(
                        self.model_name, self.count_tokens(messages), self.get_max_tokens()
                    )
                else:
                    raise ParameterError(f"Invalid request parameters: {e}")
            else:
                raise APIError(status_code, f"Ollama API error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise OllamaError(f"Error generating completion: {e}")

    async def _acomplete_generate(self, messages: List[Message], **kwargs: Any) -> Message:
        """Generate a completion using the legacy generate API asynchronously.

        Args:
            messages: List of messages
            **kwargs: Additional options

        Returns:
            Completion message
        """
        try:
            # Convert messages to a prompt
            prompt = self._prepare_legacy_prompt(messages)

            # Prepare payload
            payload = self.default_params.copy()
            payload.update(kwargs)
            payload["prompt"] = prompt

            # Remove chat-specific parameters
            payload.pop("messages", None)
            payload.pop("tools", None)
            payload.pop("tool_choice", None)

            # Call the API
            response = await self.async_client.post("/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()

            # Extract content
            content = self._parse_completion(data)

            # Create response message
            message = Message.assistant_message(content)

            # Add metadata
            message.metadata = {
                "response_format": "generate",
                "model": self.model_name,
                "raw_response": {
                    "done": data.get("done", True),
                    "total_duration": data.get("total_duration", 0),
                    "load_duration": data.get("load_duration", 0),
                    "eval_count": data.get("eval_count", 0),
                },
            }

            return message

        except httpx.ConnectError as e:
            logger.error(f"Connection error: {e}")
            raise OllamaConnectionError(f"Failed to connect to Ollama: {e}")
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code

            # Check for specific error types
            if status_code == 404:
                raise OllamaModelUnavailableError(self.model_name, f"Model not found: {e}")
            elif status_code == 400:
                error_text = e.response.text
                if "context window" in error_text.lower() or "token limit" in error_text.lower():
                    raise ContextWindowExceededError(
                        self.model_name, self.count_tokens(messages), self.get_max_tokens()
                    )
                else:
                    raise ParameterError(f"Invalid request parameters: {e}")
            else:
                raise APIError(status_code, f"Ollama API error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise OllamaError(f"Error generating completion: {e}")

    @retry_with_exponential_backoff()
    async def acomplete_stream(
        self, messages: List[Message], **kwargs: Any
    ) -> Coroutine[Any, Any, AsyncGenerator[Message, None]]:
        """Generate a streaming completion for the given messages asynchronously.

        Args:
            messages: List of messages in the conversation
            **kwargs: Additional completion options

        Returns:
            Coroutine that when awaited returns an AsyncGenerator yielding partial completion messages

        Raises:
            APIError: If there's an API error
            ModelNotAvailable: If the model is not available
        """
        use_chat_api = not kwargs.pop("use_legacy_api", False)

        # Create a properly typed async coroutine that returns a generator
        async def get_stream_generator() -> AsyncGenerator[Message, None]:
            async def stream_generator() -> AsyncGenerator[Message, None]:
                if use_chat_api:
                    async for message in self._acomplete_chat_stream(messages, **kwargs):
                        yield message
                else:
                    async for message in self._acomplete_generate_stream(messages, **kwargs):
                        yield message

            return stream_generator()

        # Return the coroutine, not the generator itself
        return get_stream_generator()

    async def _acomplete_chat_stream(
        self, messages: List[Message], **kwargs: Any
    ) -> AsyncGenerator[Message, None]:
        """Generate a streaming completion using the chat API asynchronously.

        Args:
            messages: List of messages
            **kwargs: Additional options

        Yields:
            Partial completion messages
        """
        # Prepare payload
        payload = self._prepare_chat_payload(messages, **kwargs)
        payload["stream"] = True

        content_buffer = ""
        done = False
        retries = 0
        max_retries = 3

        while not done and retries <= max_retries:
            try:
                # Stream from API
                async with self.async_client.stream(
                    "POST", "/api/chat", json=payload, timeout=None
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line:
                            continue

                        try:
                            data = json.loads(line)

                            # Handle done flag
                            if data.get("done", False):
                                done = True
                                break

                            # Extract content with fallbacks
                            new_content = self._parse_streaming_delta(data)
                            if new_content:
                                content_buffer += new_content
                                yield Message.assistant_message(content_buffer)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in stream: {line}")
                            continue

                    # If we made it through the loop, we're done
                    done = True
            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                # Only retry on connection issues
                retries += 1
                if retries <= max_retries:
                    logger.warning(
                        f"Stream connection error, retrying ({retries}/{max_retries}): {e}"
                    )
                    await asyncio.sleep(1 * retries)  # Incremental backoff
                else:
                    # Final attempt failed
                    logger.error(f"Stream connection failed after {max_retries} retries: {e}")
                    raise OllamaConnectionError(f"Streaming connection failed: {e}")
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                if status_code == 404:
                    raise OllamaModelUnavailableError(self.model_name, f"Model not found: {e}")
                else:
                    raise APIError(status_code, f"Ollama API error during streaming: {e}")
            except Exception as e:
                # Don't retry on other errors
                logger.error(f"Stream error: {e}")
                raise OllamaError(f"Error in streaming completion: {e}")

        # Yield the final complete message
        if content_buffer:
            final_message = Message.assistant_message(content_buffer)

            # Try to extract tool calls if requested
            if self.supports_tools() and kwargs.get("tools"):
                try:
                    self._extract_tool_calls(content_buffer, final_message)
                except Exception as e:
                    logger.warning(f"Error extracting tool calls from stream: {e}")

            yield final_message

    async def _acomplete_generate_stream(
        self, messages: List[Message], **kwargs: Any
    ) -> AsyncGenerator[Message, None]:
        """Generate a streaming completion using the legacy generate API asynchronously.

        Args:
            messages: List of messages
            **kwargs: Additional options

        Yields:
            Partial completion messages
        """
        # Convert messages to a prompt
        prompt = self._prepare_legacy_prompt(messages)

        # Prepare payload
        payload = self.default_params.copy()
        payload.update(kwargs)
        payload["prompt"] = prompt
        payload["stream"] = True

        # Remove chat-specific parameters
        payload.pop("messages", None)
        payload.pop("tools", None)
        payload.pop("tool_choice", None)

        content_buffer = ""
        done = False
        retries = 0
        max_retries = 3

        while not done and retries <= max_retries:
            try:
                # Stream from API
                async with self.async_client.stream(
                    "POST", "/api/generate", json=payload, timeout=None
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line:
                            continue

                        try:
                            data = json.loads(line)

                            # Handle done flag
                            if data.get("done", False):
                                done = True
                                break

                            # Extract content with fallbacks
                            new_content = self._parse_streaming_delta(data)
                            if new_content:
                                content_buffer += new_content
                                yield Message.assistant_message(content_buffer)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in stream: {line}")
                            continue

                    # If we made it through the loop, we're done
                    done = True
            except (httpx.ConnectError, httpx.ReadTimeout) as e:
                # Only retry on connection issues
                retries += 1
                if retries <= max_retries:
                    logger.warning(
                        f"Stream connection error, retrying ({retries}/{max_retries}): {e}"
                    )
                    await asyncio.sleep(1 * retries)  # Incremental backoff
                else:
                    # Final attempt failed
                    logger.error(f"Stream connection failed after {max_retries} retries: {e}")
                    raise OllamaConnectionError(f"Streaming connection failed: {e}")
            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                if status_code == 404:
                    raise OllamaModelUnavailableError(self.model_name, f"Model not found: {e}")
                else:
                    raise APIError(status_code, f"Ollama API error during streaming: {e}")
            except Exception as e:
                # Don't retry on other errors
                logger.error(f"Stream error: {e}")
                raise OllamaError(f"Error in streaming completion: {e}")

        # Yield the final complete message
        if content_buffer:
            yield Message.assistant_message(content_buffer)

    def count_tokens(self, messages: List[Message]) -> int:
        """Count the number of tokens in the given messages.

        Args:
            messages: List of messages to count tokens for

        Returns:
            Number of tokens
        """
        # Use the optimized token counter for Ollama models
        return TokenCounter._count_ollama_tokens_optimized(self.model_name, messages)

    def get_max_tokens(self) -> int:
        """Get the maximum number of tokens supported by the model.

        Returns:
            Maximum token count
        """
        # Extract context size from model info if available
        if self._model_info and "context_size" in self._model_info:
            return int(self._model_info["context_size"])

        # Check the model name against known context sizes
        for model_family, context_size in self.MODEL_CONTEXT_SIZES.items():
            if model_family in self.model_name.lower():
                return context_size

        # Look for patterns like "8k", "16k", "32k" in model name
        ctx_match = re.search(r"(\d+)[kK](?:ctx)?", self.model_name)
        if ctx_match:
            return int(ctx_match.group(1)) * 1024

        # Use a safe default
        logger.info(f"Using default context size for model {self.model_name}")
        return self.DEFAULT_CONTEXT_SIZE

    def supports_vision(self) -> bool:
        """Check if the provider/model supports vision/images.

        Returns:
            True if vision is supported, False otherwise
        """
        # Use pre-configured capabilities if available
        capabilities = self._detect_model_capabilities()
        return capabilities["vision"]

    def supports_tools(self) -> bool:
        """Check if the provider/model supports tool/function calling.

        Returns:
            True if tool calling is supported, False otherwise
        """
        # Use pre-configured capabilities if available
        capabilities = self._detect_model_capabilities()
        return capabilities["tools"]

    def register_custom_model(
        self, base_model: str, custom_name: str, capabilities: Optional[Dict[str, bool]] = None
    ) -> bool:
        """Register a custom fine-tuned model.

        Args:
            base_model: Name of the base model
            custom_name: Name to use for the custom model
            capabilities: Optional dict of model capabilities

        Returns:
            True if registration successful
        """
        # Check if model already exists
        available_models = self.get_available_models()
        if custom_name in available_models:
            logger.info(f"Custom model {custom_name} already registered")
            return True

        try:
            # Register the custom model
            logger.info(f"Registering custom model {custom_name} based on {base_model}")

            # Update capabilities registry
            if capabilities:
                # Initialize model capabilities dict if needed
                if self._model_capabilities is None:
                    self._model_capabilities = {}

                # Create a new dictionary to avoid modifying the input
                self._model_capabilities = dict(capabilities)

            return True
        except Exception as e:
            logger.error(f"Error registering custom model: {e}")
            return False

    def use_versioned_model(self, model_name: str, version: str) -> bool:
        """Use a specific version of a model.

        Args:
            model_name: Base model name
            version: Version tag

        Returns:
            True if successful
        """
        versioned_name = f"{model_name}:{version}"

        # Try to switch to the versioned model
        return self.change_model(versioned_name)

    def change_model(self, model_name: str) -> bool:
        """Change the current model.

        Args:
            model_name: New model name

        Returns:
            True if successful
        """
        # Don't do anything if it's the same model
        if model_name == self.model_name:
            return True

        try:
            # Update model name
            self.model_name = model_name

            # Update default parameters
            self.default_params["model"] = model_name

            # Reset validation status
            self._model_validated = False

            # Validate model availability
            is_valid = self._validate_model(False)

            # Reset capability cache
            self._model_capabilities = None

            return is_valid
        except Exception as e:
            logger.error(f"Error changing model: {e}")
            return False

    def run_diagnostics(self) -> Dict[str, Any]:
        """Run diagnostics on the Ollama provider.

        Returns:
            Dictionary with diagnostic information
        """
        # Create a fresh dictionary for diagnostics
        diagnostics: Dict[str, Any] = {
            "api_status": "unknown",
            "version": "unknown",
            "models_available": 0,
            "current_model": {"name": self.model_name, "status": "unknown", "capabilities": None},
            "connection_info": {"base_url": self.api_base, "timeout": self.request_timeout},
            "tests": {},
        }

        # Check API status
        try:
            response = self.client.get("/")
            diagnostics["api_status"] = "online" if response.status_code == 200 else "error"
        except Exception as e:
            diagnostics["api_status"] = f"offline: {str(e)}"

        # Get version
        diagnostics["version"] = self._detect_ollama_version()

        # Check available models
        try:
            models = self.get_available_models()
            diagnostics["models_available"] = len(models)
        except Exception:
            pass

        # Check current model
        try:
            diagnostics["current_model"]["status"] = (
                "available" if self._validate_model() else "unavailable"
            )
            capabilities_dict = {
                "vision": self.supports_vision(),
                "tools": self.supports_tools(),
            }
            diagnostics["current_model"]["capabilities"] = capabilities_dict
        except Exception:
            pass

        # Run simple test
        try:
            start_time = time.time()
            self.complete([Message.user_message("hello")])
            completion_time = time.time() - start_time
            diagnostics["tests"]["simple_completion"] = {
                "success": True,
                "time_seconds": round(completion_time, 2),
            }
        except Exception as e:
            diagnostics["tests"]["simple_completion"] = {"success": False, "error": str(e)}

        return diagnostics
