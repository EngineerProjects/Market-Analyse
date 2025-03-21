"""
Tests for the Enterprise AI LLM module.

This module contains unit tests for the LLM integration components,
including providers, conversation management, and utility functions.
"""

import base64
import json
import os
import pytest
from unittest import mock
from pathlib import Path
from typing import List, Dict, Any, Generator

from enterprise_ai.schema import Message, Role
from enterprise_ai.exceptions import LLMError, ModelNotAvailable, APIError
from enterprise_ai.llm.base import LLMProvider, ConversationManager
from enterprise_ai.llm.exceptions import (
    TokenCountError,
    ProviderNotSupportedError,
    ParameterError,
    ContextWindowExceededError,
    ImageProcessingError,
    ModelCapabilityError,
)
from enterprise_ai.llm.utils import TokenCounter, encode_image_file
from enterprise_ai.llm.providers import OpenAIProvider, AnthropicProvider, OllamaProvider
from enterprise_ai.llm import LLMService


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name
        self.kwargs = kwargs
        self.supports_vision_flag = kwargs.get("supports_vision", False)
        self.supports_tools_flag = kwargs.get("supports_tools", False)
        self.max_tokens_value = kwargs.get("max_tokens", 2048)
        self.calls = []

    def get_model_name(self) -> str:
        return self.model_name

    def complete(self, messages: List[Message], **kwargs: Any) -> Message:
        self.calls.append(("complete", messages, kwargs))
        return Message.assistant_message("This is a mock response")

    def complete_stream(
        self, messages: List[Message], **kwargs: Any
    ) -> Generator[Message, None, None]:
        self.calls.append(("complete_stream", messages, kwargs))
        yield Message.assistant_message("This is a mock ")
        yield Message.assistant_message("This is a mock streaming ")
        yield Message.assistant_message("This is a mock streaming response")

    async def acomplete(self, messages: List[Message], **kwargs: Any) -> Message:
        self.calls.append(("acomplete", messages, kwargs))
        return Message.assistant_message("This is a mock async response")

    async def acomplete_stream(
        self, messages: List[Message], **kwargs: Any
    ) -> Generator[Message, None, None]:
        self.calls.append(("acomplete_stream", messages, kwargs))
        yield Message.assistant_message("This is a mock ")
        yield Message.assistant_message("This is a mock streaming ")
        yield Message.assistant_message("This is a mock streaming async response")

    def count_tokens(self, messages: List[Message]) -> int:
        return sum(len(m.content or "") for m in messages) // 4  # Simple approximation

    def get_max_tokens(self) -> int:
        return self.max_tokens_value

    def supports_vision(self) -> bool:
        return self.supports_vision_flag

    def supports_tools(self) -> bool:
        return self.supports_tools_flag


@pytest.fixture
def mock_provider():
    """Fixture for a mock LLM provider."""
    return MockLLMProvider("mock-model")


@pytest.fixture
def vision_provider():
    """Fixture for a mock LLM provider with vision support."""
    return MockLLMProvider("vision-model", supports_vision=True)


@pytest.fixture
def tool_provider():
    """Fixture for a mock LLM provider with tool support."""
    return MockLLMProvider("tool-model", supports_tools=True)


class TestTokenCounter:
    """Tests for the TokenCounter utility."""

    def test_count_for_model_openai(self):
        """Test token counting for OpenAI models."""
        messages = [
            Message.system_message("You are a helpful assistant."),
            Message.user_message("Hello, how are you?"),
            Message.assistant_message("I'm doing well, thank you!"),
        ]

        # Mock the OpenAI tokenizer
        with mock.patch("enterprise_ai.llm.utils.TokenCounter._count_openai") as mock_count:
            mock_count.return_value = 42
            count = TokenCounter.count_for_model("gpt-4", messages)
            assert count == 42
            mock_count.assert_called_once_with("gpt-4", messages)

    def test_count_for_model_anthropic(self):
        """Test token counting for Anthropic models."""
        messages = [
            Message.system_message("You are a helpful assistant."),
            Message.user_message("Hello, how are you?"),
            Message.assistant_message("I'm doing well, thank you!"),
        ]

        # Mock the Anthropic tokenizer
        with mock.patch("enterprise_ai.llm.utils.TokenCounter._count_anthropic") as mock_count:
            mock_count.return_value = 42
            count = TokenCounter.count_for_model("claude-3-opus", messages)
            assert count == 42
            mock_count.assert_called_once_with("claude-3-opus", messages)

    def test_count_for_model_ollama(self):
        """Test token counting for Ollama models."""
        messages = [
            Message.system_message("You are a helpful assistant."),
            Message.user_message("Hello, how are you?"),
            Message.assistant_message("I'm doing well, thank you!"),
        ]

        # Mock the Ollama tokenizer
        with mock.patch("enterprise_ai.llm.utils.TokenCounter._count_ollama") as mock_count:
            mock_count.return_value = 42
            count = TokenCounter.count_for_model("llama3", messages)
            assert count == 42
            mock_count.assert_called_once_with("llama3", messages)

    def test_count_for_model_fallback(self):
        """Test token counting fallback for unknown models."""
        messages = [
            Message.system_message("You are a helpful assistant."),
            Message.user_message("Hello, how are you?"),
            Message.assistant_message("I'm doing well, thank you!"),
        ]

        # Mock the simple tokenizer
        with mock.patch("enterprise_ai.llm.utils.TokenCounter._count_simple") as mock_count:
            mock_count.return_value = 42
            count = TokenCounter.count_for_model("unknown-model", messages)
            assert count == 42
            mock_count.assert_called_once_with(messages)

    def test_count_simple(self):
        """Test the simple token counting method."""
        messages = [
            Message.system_message("You are a helpful assistant."),
            Message.user_message("Hello, how are you?"),
            Message.assistant_message("I'm doing well, thank you!"),
        ]

        count = TokenCounter._count_simple(messages)
        # Simple estimation: total chars / 4 + overhead
        expected = (
            len("You are a helpful assistant.")
            + len("system")
            + len("Hello, how are you?")
            + len("user")
            + len("I'm doing well, thank you!")
            + len("assistant")
        ) // 4 + 10

        assert count == expected


class TestConversationManager:
    """Tests for the ConversationManager class."""

    def test_initialization(self, mock_provider):
        """Test ConversationManager initialization."""
        # Without system message
        conv = ConversationManager(mock_provider)
        assert conv.provider == mock_provider
        assert conv.messages == []

        # With system message
        conv = ConversationManager(mock_provider, system_message="You are a helpful assistant.")
        assert len(conv.messages) == 1
        assert conv.messages[0].role == Role.SYSTEM
        assert conv.messages[0].content == "You are a helpful assistant."

    def test_add_messages(self, mock_provider):
        """Test adding messages to conversation."""
        conv = ConversationManager(mock_provider)

        # Add system message
        conv.add_system_message("You are a helpful assistant.")
        assert len(conv.messages) == 1
        assert conv.messages[0].role == Role.SYSTEM

        # Add user message
        conv.add_user_message("Hello!")
        assert len(conv.messages) == 2
        assert conv.messages[1].role == Role.USER
        assert conv.messages[1].content == "Hello!"

        # Add assistant message
        conv.add_assistant_message("How can I help you?")
        assert len(conv.messages) == 3
        assert conv.messages[2].role == Role.ASSISTANT
        assert conv.messages[2].content == "How can I help you?"

        # Replace system message
        conv.add_system_message("You are an AI assistant.")
        assert len(conv.messages) == 3  # Count stays the same
        assert conv.messages[0].content == "You are an AI assistant."

        # Add tool message
        conv.add_tool_message("42", "calculator", "calc_123")
        assert len(conv.messages) == 4
        assert conv.messages[3].role == Role.TOOL
        assert conv.messages[3].content == "42"
        assert conv.messages[3].name == "calculator"
        assert conv.messages[3].tool_call_id == "calc_123"

    def test_get_messages(self, mock_provider):
        """Test getting messages from conversation."""
        conv = ConversationManager(mock_provider)

        # Add messages
        conv.add_system_message("You are a helpful assistant.")
        conv.add_user_message("Hello!")
        conv.add_assistant_message("How can I help you?")

        # Get messages
        messages = conv.get_messages()
        assert len(messages) == 3
        assert messages[0].role == Role.SYSTEM
        assert messages[1].role == Role.USER
        assert messages[2].role == Role.ASSISTANT

        # Ensure copy is returned (not reference)
        messages.append(Message.user_message("This shouldn't be added"))
        assert len(conv.messages) == 3

    def test_count_tokens(self, mock_provider):
        """Test counting tokens in conversation."""
        conv = ConversationManager(mock_provider)

        # Add messages
        conv.add_system_message("You are a helpful assistant.")
        conv.add_user_message("Hello!")

        # Count tokens
        with mock.patch.object(mock_provider, "count_tokens", return_value=42):
            count = conv.count_tokens()
            assert count == 42
            mock_provider.count_tokens.assert_called_once_with(conv.messages)

    def test_prune_to_fit_context(self, mock_provider):
        """Test pruning conversation to fit context window."""
        # Set up mock provider with token counting
        mock_provider.count_tokens = lambda msgs: sum(len(m.content or "") for m in msgs)
        mock_provider.get_max_tokens = lambda: 50

        conv = ConversationManager(mock_provider, max_tokens=50, token_limit_pct=0.8)

        # Add messages that will exceed the token budget
        conv.add_system_message("S" * 10)  # 10 tokens
        conv.add_user_message("U1" * 5)  # 10 tokens
        conv.add_assistant_message("A1" * 5)  # 10 tokens
        conv.add_user_message("U2" * 5)  # 10 tokens
        conv.add_assistant_message("A2" * 5)  # 10 tokens

        # Total: 50 tokens, with 80% limit (40 tokens) and 10 reserve = 30 token budget
        # Will need to remove some messages

        pruned = conv.prune_to_fit_context(reserve_tokens=10)

        # Verify that some messages were removed
        assert len(pruned) < 5

        # Verify system message is preserved
        assert pruned[0].content == "S" * 10

        # Verify token count is now within budget
        assert mock_provider.count_tokens(pruned) <= 30

    def test_clear(self, mock_provider):
        """Test clearing conversation history."""
        conv = ConversationManager(mock_provider)

        # Add messages
        conv.add_system_message("You are a helpful assistant.")
        conv.add_user_message("Hello!")
        conv.add_assistant_message("How can I help you?")

        # Clear with keep_system=True (default)
        conv.clear()
        assert len(conv.messages) == 1
        assert conv.messages[0].role == Role.SYSTEM

        # Add messages again
        conv.add_user_message("Hello again!")

        # Clear with keep_system=False
        conv.clear(keep_system=False)
        assert len(conv.messages) == 0


class TestLLMService:
    """Tests for the LLMService class."""

    def test_initialization(self):
        """Test LLMService initialization."""
        # Mock the config and provider creation
        with (
            mock.patch("enterprise_ai.llm.config") as mock_config,
            mock.patch("enterprise_ai.llm.LLMService._create_provider") as mock_create,
        ):
            # Set up mock config
            mock_config.llm = {"default": mock.MagicMock(api_type="openai", model="gpt-4")}

            # Set up mock provider
            mock_provider = MockLLMProvider("gpt-4")
            mock_create.return_value = mock_provider

            # Initialize with default values
            service = LLMService()
            assert service.provider_name == "openai"
            assert service.model_name == "gpt-4"
            assert service.provider == mock_provider

            # Initialize with explicit values
            service = LLMService(provider_name="anthropic", model_name="claude-3")
            assert service.provider_name == "anthropic"
            assert service.model_name == "claude-3"

    def test_create_provider(self):
        """Test provider creation logic."""
        # Mock the providers and config
        with (
            mock.patch("enterprise_ai.llm.OpenAIProvider") as mock_openai,
            mock.patch("enterprise_ai.llm.AnthropicProvider") as mock_anthropic,
            mock.patch("enterprise_ai.llm.OllamaProvider") as mock_ollama,
            mock.patch("enterprise_ai.llm.config") as mock_config,
        ):
            # Set up mock config
            mock_config.llm = {
                "default": mock.MagicMock(
                    api_key="default-key", base_url="default-url", temperature=0.7, max_tokens=1000
                ),
                "openai": mock.MagicMock(
                    api_key="openai-key", base_url="openai-url", temperature=0.5, max_tokens=2000
                ),
                "anthropic": mock.MagicMock(
                    api_key="anthropic-key",
                    base_url="anthropic-url",
                    temperature=0.6,
                    max_tokens=3000,
                ),
                "ollama": mock.MagicMock(base_url="ollama-url", temperature=0.8, max_tokens=4000),
            }

            # Test OpenAI provider creation
            service = LLMService(provider_name="openai", model_name="gpt-4")
            service._create_provider("openai", "gpt-4")
            mock_openai.assert_called_once_with(
                model_name="gpt-4",
                api_key="openai-key",
                api_base="openai-url",
                temperature=0.5,
                max_tokens=2000,
            )

            # Test Anthropic provider creation
            mock_openai.reset_mock()
            service._create_provider("anthropic", "claude-3")
            mock_anthropic.assert_called_once_with(
                model_name="claude-3",
                api_key="anthropic-key",
                api_base="anthropic-url",
                temperature=0.6,
                max_tokens=3000,
            )

            # Test Ollama provider creation
            mock_anthropic.reset_mock()
            service._create_provider("ollama", "llama3")
            mock_ollama.assert_called_once_with(
                model_name="llama3", api_base="ollama-url", temperature=0.8, max_tokens=4000
            )

            # Test unsupported provider
            with pytest.raises(ProviderNotSupportedError):
                service._create_provider("unsupported", "model")

    def test_get_conversation_manager(self, mock_provider):
        """Test getting a conversation manager."""
        # Mock the provider creation
        with mock.patch("enterprise_ai.llm.LLMService._create_provider") as mock_create:
            mock_create.return_value = mock_provider

            service = LLMService(provider_name="openai", model_name="gpt-4")

            # Get conversation manager
            conv = service.get_conversation_manager(
                system_message="You are a helpful assistant.", max_tokens=1000
            )

            assert isinstance(conv, ConversationManager)
            assert conv.provider == mock_provider
            assert conv.messages[0].content == "You are a helpful assistant."
            assert conv.max_tokens == 1000

    def test_completion_methods(self, mock_provider):
        """Test completion methods."""
        # Mock the provider creation
        with mock.patch("enterprise_ai.llm.LLMService._create_provider") as mock_create:
            mock_create.return_value = mock_provider

            service = LLMService(provider_name="openai", model_name="gpt-4")

            # Test complete
            messages = [Message.user_message("Hello!")]
            response = service.complete(messages, temperature=0.7)
            assert response.content == "This is a mock response"
            assert ("complete", messages, {"temperature": 0.7}) in mock_provider.calls

            # Test complete_stream
            mock_provider.calls.clear()
            responses = list(service.complete_stream(messages, temperature=0.8))
            assert responses[-1].content == "This is a mock streaming response"
            assert ("complete_stream", messages, {"temperature": 0.8}) in mock_provider.calls

    def test_property_methods(self, mock_provider):
        """Test property accessor methods."""
        # Configure mock provider
        mock_provider.max_tokens_value = 4000
        mock_provider.supports_vision_flag = True
        mock_provider.supports_tools_flag = False

        # Mock the provider creation
        with mock.patch("enterprise_ai.llm.LLMService._create_provider") as mock_create:
            mock_create.return_value = mock_provider

            service = LLMService(provider_name="openai", model_name="gpt-4")

            # Test properties
            assert service.max_tokens == 4000
            assert service.supports_vision is True
            assert service.supports_tools is False


class TestProviders:
    """Tests for the provider implementations."""

    def test_openai_provider(self):
        """Test OpenAI provider initialization."""
        with (
            mock.patch("enterprise_ai.llm.providers.openai_provider.OpenAI") as mock_client,
            mock.patch("enterprise_ai.llm.providers.openai_provider.AsyncOpenAI") as mock_async,
        ):
            provider = OpenAIProvider(
                model_name="gpt-4",
                api_key="test-key",
                api_base="https://api.test.com",
                temperature=0.7,
                max_tokens=1000,
            )

            assert provider.model_name == "gpt-4"
            assert provider.temperature == 0.7
            assert provider.max_tokens == 1000

            # Check client initialization
            mock_client.assert_called_once()
            mock_async.assert_called_once()

    def test_anthropic_provider(self):
        """Test Anthropic provider initialization."""
        with (
            mock.patch("enterprise_ai.llm.providers.anthropic_provider.Anthropic") as mock_client,
            mock.patch(
                "enterprise_ai.llm.providers.anthropic_provider.AsyncAnthropic"
            ) as mock_async,
        ):
            provider = AnthropicProvider(
                model_name="claude-3",
                api_key="test-key",
                api_base="https://api.test.com",
                temperature=0.7,
                max_tokens=1000,
            )

            assert provider.model_name == "claude-3"
            assert provider.temperature == 0.7
            assert provider.max_tokens == 1000

            # Check client initialization
            mock_client.assert_called_once()
            mock_async.assert_called_once()

    def test_ollama_provider(self):
        """Test Ollama provider initialization."""
        with (
            mock.patch("enterprise_ai.llm.providers.ollama_provider.httpx.Client") as mock_client,
            mock.patch(
                "enterprise_ai.llm.providers.ollama_provider.httpx.AsyncClient"
            ) as mock_async,
        ):
            provider = OllamaProvider(
                model_name="llama3",
                api_base="http://localhost:11434",
                temperature=0.7,
                max_tokens=1000,
            )

            assert provider.model_name == "llama3"
            assert provider.temperature == 0.7
            assert provider.max_tokens == 1000

            # Check client initialization
            mock_client.assert_called_once()
            mock_async.assert_called_once()
