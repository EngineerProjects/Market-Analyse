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
    through a consistent interface. It dynamically detects model capabilities and adapts
    to different model requirements and API formats.
    """

    # Maximum token contexts for different model families (used as fallbacks)
    MODEL_CONTEXT_SIZES = {
        "llama2": 4096,
        "llama3": 8192,
        "llama3.1": 16384,
        "llama3.2": 32768,
        "llama3.3": 131072,  # These are estimated - adjust based on actual values
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
    DEFAULT_CONTEXT_SIZE = 8192  # Increased default

    # Patterns for identifying model capabilities based on name
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

    TOOL_PATTERNS = [
        "llama",
        "mistral",
        "mixtral",
        "openhermes",
        "wizard",
        "deepseek",
        "phi",
        "neural-chat",
        "qwen",
        "yi",
        "vicuna",
        "nous",
        "orca",
    ]

    def __init__(
        self,
        model_name: str,
        api_base: str = "http://localhost:11434",
        request_timeout: float = 60.0,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        validate_model: bool = True,
        strict_validation: bool = False,
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
            **kwargs: Additional parameters to pass to Ollama API
        """
        self.model_name = model_name
        self.api_base = api_base.rstrip("/")
        self.request_timeout = request_timeout
        self.temperature = temperature
        self.kwargs = kwargs

        # Dynamic capability and model info cache
        self._capabilities: Optional[Dict[str, bool]] = None
        self._model_info: Optional[Dict[str, Any]] = None
        self._api_version: Optional[str] = None
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
            model_exists = self.validate_model()
            if not model_exists and strict_validation:
                available = self.get_available_models()
                suggestions = self._suggest_models(self.model_name, available)

                suggestion_msg = ""
                if suggestions:
                    suggestion_msg = f" Did you mean one of these? {', '.join(suggestions)}"

                raise ModelNotAvailable(
                    self.model_name, f"Model not found in Ollama.{suggestion_msg}"
                )
            elif not model_exists:
                logger.warning(
                    f"Model '{self.model_name}' was not found in available Ollama models. "
                    f"Make sure it's pulled or the name is correct."
                )

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

    def validate_model(self) -> bool:
        """Validate if the current model exists in Ollama.

        Returns:
            True if the model exists, False otherwise
        """
        if self._model_validated:
            return True

        # First try to validate via the show endpoint (more reliable)
        try:
            response = self.client.get("/api/show", params={"name": self.model_name})
            if response.status_code == 200:
                self._model_validated = True
                # Also update model info since we have it
                self._model_info = response.json()
                return True
        except Exception:
            pass

        # Fall back to checking available models list
        available_models = self.get_available_models()
        if self.model_name in available_models:
            self._model_validated = True
            return True

        # Check if the model exists with a tag
        model_base = self.model_name.split(":")[0] if ":" in self.model_name else self.model_name
        if any(m.startswith(f"{model_base}:") for m in available_models):
            self._model_validated = True
            return True

        return False

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

        # Simple suggestion based on prefix matching (more sophisticated methods could be used)
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

    def _detect_api_version(self) -> str:
        """Detect Ollama API version for compatibility handling.

        Returns:
            Semantic version string (or "latest" if detection fails)
        """
        if self._api_version is not None:
            return self._api_version

        try:
            response = self.client.get("/api/version")
            response.raise_for_status()
            version_info = response.json()
            version = str(version_info.get("version", "0.0.0"))
            logger.debug(f"Detected Ollama version: {version}")
            self._api_version = version
            return version
        except Exception as e:
            logger.warning(f"Failed to detect Ollama version: {e}, assuming latest")
            self._api_version = "latest"
            return "latest"

    def _discover_model_capabilities(self) -> Dict[str, bool]:
        """Dynamically discover model capabilities by querying Ollama API.

        Returns:
            Dictionary of capability flags
        """
        if self._capabilities is not None:
            return self._capabilities

        try:
            # First, try to check if model info has already been fetched
            if self._model_info is None:
                # Query the model using the /api/show endpoint
                response = self.client.get("/api/show", params={"name": self.model_name})
                response.raise_for_status()
                self._model_info = response.json()

            # Extract model details
            model_info = self._model_info
            details = model_info.get("details", {})
            family = details.get("family", "").lower()
            families = details.get("families", [])
            if isinstance(families, list):
                families = [f.lower() for f in families]
            else:
                families = []

            # Get parameter size
            parameter_size_str = details.get("parameter_size", "")
            parameter_size = self._extract_parameter_size(parameter_size_str)

            # Check for capability hints in parameter description
            model_name_lower = self.model_name.lower()

            # Template info can indicate chat capability
            chat_format = False
            if "template" in model_info and model_info["template"]:
                chat_format = "{{" in model_info["template"]

            # Determine capabilities based on model info and name
            vision_capability = (
                # Check for vision in family or families
                any(v in family for v in ["llava", "vision", "clip", "multimodal"])
                or any(
                    any(v in f for f in families) for v in ["llava", "vision", "clip", "multimodal"]
                )
                or
                # Check for vision in model name
                any(v in model_name_lower for v in self.VISION_PATTERNS)
                or
                # Parameter size description sometimes contains vision info
                any(
                    v in parameter_size_str.lower()
                    for v in ["vision", "multimodal", "vl", "visual"]
                )
            )

            # Tool capability is more common in larger models and specific families
            tools_capability = (
                # Check for tool-capable families
                any(t in family for t in ["llama", "mistral", "mixtral", "wizard"])
                or any(
                    any(t in f for f in families) for t in ["llama", "mistral", "mixtral", "wizard"]
                )
                or
                # Check model name patterns
                any(t in model_name_lower for t in self.TOOL_PATTERNS)
                or
                # Larger models (>7B) are more likely to support tools
                parameter_size >= 7000000000
            )

            # Store and return capabilities
            capabilities = {
                "vision": vision_capability,
                "tools": tools_capability,
                "chat_format": chat_format,
            }

            self._capabilities = capabilities
            return capabilities

        except Exception as e:
            logger.warning(f"Failed to discover model capabilities: {e}")
            # Default fallback capabilities based on model name
            model_name_lower = self.model_name.lower()

            # More aggressive fallback detection
            vision_capable = any(v in model_name_lower for v in self.VISION_PATTERNS)

            # Look for model size indicators that suggest advanced capabilities
            parameter_size = self._extract_parameter_size_from_name(model_name_lower)

            # Models 7B and larger are more likely to support tools
            tools_capable = (
                any(t in model_name_lower for t in self.TOOL_PATTERNS)
                or parameter_size >= 7000000000
            )

            fallback = {
                "vision": vision_capable,
                "tools": tools_capable,
                "chat_format": True,  # Most modern models support chat format
            }

            self._capabilities = fallback
            return fallback

    def _extract_parameter_size(self, size_str: str) -> int:
        """Extract parameter size in number of parameters from a string description.

        Args:
            size_str: String describing parameter size (e.g. "7B", "13B")

        Returns:
            Number of parameters as integer
        """
        try:
            # Look for patterns like "7B", "13B", "70b", etc.
            match = re.search(r"(\d+(?:\.\d+)?)([bmtq])", size_str.lower())
            if match:
                value = float(match.group(1))
                unit = match.group(2)

                if unit == "b":
                    return int(value * 1_000_000_000)
                elif unit == "m":
                    return int(value * 1_000_000)
                elif unit == "t":
                    return int(value * 1_000_000_000_000)
                elif unit == "q":
                    return int(value * 1_000_000_000_000_000)

            # Try to extract just numbers if they're large enough
            number_match = re.search(r"(\d+)", size_str)
            if number_match:
                value = int(number_match.group(1))
                if value > 1000:  # Likely millions or billions
                    return value * 1_000_000  # Assume millions if it's a large number
        except Exception:
            pass

        return 0  # Default if parsing fails

    def _extract_parameter_size_from_name(self, model_name: str) -> int:
        """Extract parameter size from model name.

        Args:
            model_name: Name of the model (e.g. "llama2:7b", "llama3:70b")

        Returns:
            Estimated number of parameters
        """
        try:
            # Extract size indicators like 7b, 13b, 70b
            size_match = re.search(r"[:\-_](\d+(?:\.\d+)?)([bmtq])", model_name.lower())
            if size_match:
                value = float(size_match.group(1))
                unit = size_match.group(2)

                if unit == "b":
                    return int(value * 1_000_000_000)
                elif unit == "m":
                    return int(value * 1_000_000)
                elif unit == "t":
                    return int(value * 1_000_000_000_000)
                elif unit == "q":
                    return int(value * 1_000_000_000_000_000)

            # Check for specific model family sizes (these are approximations)
            if "phi4-mini" in model_name:
                return 3_400_000_000  # 3.4B
            elif "phi3-mini" in model_name:
                return 3_800_000_000  # 3.8B
            elif "phi3" in model_name:
                return 14_000_000_000  # 14B
            elif "phi2" in model_name:
                return 2_700_000_000  # 2.7B
            elif "mixtral-8x7b" in model_name or "mixtral:8x7b" in model_name:
                return 47_000_000_000  # 8x7B = 56B effective parameters
            elif "mixtral-8x22b" in model_name or "mixtral:8x22b" in model_name:
                return 176_000_000_000  # 8x22B
            elif "deepseek-coder" in model_name and "33b" not in model_name:
                return 16_000_000_000  # 16B is common size
        except Exception:
            pass

        return 0  # Default if extraction fails

    def _prepare_messages(self, messages: List[Message]) -> Union[str, Dict[str, Any]]:
        """Convert our Message objects to Ollama API format.

        Args:
            messages: List of Message objects

        Returns:
            Formatted string for Ollama prompt or chat messages dict
        """
        # First try to get model info to check if it supports chat format
        try:
            if self._capabilities is None:
                self._discover_model_capabilities()

            # Default to chat format if capabilities haven't been determined
            use_chat_format = True
            if self._capabilities is not None:
                use_chat_format = self._capabilities.get("chat_format", True)

            if use_chat_format:
                return self._prepare_chat_messages(messages)

            # Default to legacy prompt format
            return self._prepare_legacy_prompt(messages)

        except Exception as e:
            logger.warning(f"Failed to check model capabilities, using legacy format: {e}")
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
        content = completion.get("response", "")

        # Check for tool calls in content
        tool_calls: List[ToolCall] = []

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

    def _try_with_formats(
        self, messages: List[Message], endpoint: str, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Try multiple formats and return the first successful response.

        Args:
            messages: Message list to send
            endpoint: API endpoint
            params: Request parameters

        Returns:
            API response

        Raises:
            Exception from the last attempt if all attempts fail
        """
        exceptions = []

        # Try chat format first if we think it's supported
        if self._capabilities is None:
            self._discover_model_capabilities()

        # Default to True if capabilities not determined
        use_chat_first = True
        if self._capabilities is not None:
            use_chat_first = self._capabilities.get("chat_format", True)

        attempts = [
            # First attempt based on detected capabilities
            lambda: self._make_request(endpoint, messages, params, use_chat=use_chat_first),
            # Second attempt with opposite format
            lambda: self._make_request(endpoint, messages, params, use_chat=not use_chat_first),
        ]

        for attempt_func in attempts:
            try:
                return attempt_func()
            except Exception as e:
                exceptions.append(e)
                logger.warning(f"Request format attempt failed: {e}")

        # All attempts failed, raise the last exception
        raise exceptions[-1]

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

        # Make the request
        response = self.client.post(endpoint, json=request_data)
        response.raise_for_status()
        return cast(Dict[str, Any], response.json())

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

            # Try multiple formats if needed
            response = self._try_with_formats(messages, "/api/chat", request_params)

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
            use_chat = True
            if self._capabilities is None:
                self._discover_model_capabilities()

            if self._capabilities is not None:
                use_chat = self._capabilities.get("chat_format", True)

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
            use_chat = True
            if self._capabilities is None:
                self._discover_model_capabilities()

            if self._capabilities is not None:
                use_chat = self._capabilities.get("chat_format", True)

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
                use_chat = True
                if self._capabilities is None:
                    self._discover_model_capabilities()

                if self._capabilities is not None:
                    use_chat = self._capabilities.get("chat_format", True)

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
            details = self._model_info["details"]

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

            # Try to extract context size from parameter_size
            if "parameter_size" in details:
                param_size = details["parameter_size"]
                if isinstance(param_size, str):
                    # Look for context window in parameter size (e.g. "7B-32K")
                    try:
                        context_match = re.search(r"(\d+)[kK]", param_size)
                        if context_match:
                            return int(context_match.group(1)) * 1024
                    except Exception:
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
        # First check model name for known vision models (fast path)
        model_name_lower = self.model_name.lower()
        if any(pattern in model_name_lower for pattern in self.VISION_PATTERNS):
            return True

        # Then do API-based capability discovery
        if self._capabilities is None:
            self._discover_model_capabilities()

        if self._capabilities is not None:
            return self._capabilities.get("vision", False)

        # Default fallback if capabilities couldn't be determined
        return False

    def supports_tools(self) -> bool:
        """Check if the provider/model supports tool/function calling.

        Returns:
            True if tool calling is supported, False otherwise
        """
        # For larger models (typically 7B+), assume tool support for most modern architectures
        parameter_size = self._extract_parameter_size_from_name(self.model_name.lower())
        if parameter_size >= 7_000_000_000:  # 7B or larger
            # First check model name for known tool-supporting models
            model_name_lower = self.model_name.lower()
            if any(pattern in model_name_lower for pattern in self.TOOL_PATTERNS):
                return True

        # Then do API-based capability discovery
        if self._capabilities is None:
            self._discover_model_capabilities()

        if self._capabilities is not None:
            return self._capabilities.get("tools", False)

        # Default fallback if capabilities couldn't be determined
        return False
