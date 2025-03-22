"""
Tests for the enhanced Enterprise AI LLM service.

This file adds tests for the new functionality in the LLM service implementation,
including model validation, fallback models, model/provider switching,
and capability detection.
"""

import pytest
from unittest import mock
from typing import List, Dict, Any, Generator, AsyncGenerator, Optional

from enterprise_ai.schema import Message, Role
from enterprise_ai.exceptions import ModelNotAvailable, LLMError
from enterprise_ai.llm.base import LLMProvider
from enterprise_ai.llm.exceptions import ProviderNotSupportedError
from enterprise_ai.llm.service import LLMService
from enterprise_ai.llm.service import (
    change_model,
    change_provider,
    get_available_models,
    get_similar_models,
)


# Define a mock provider class for testing
class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        """Initialize the mock provider.

        Args:
            model_name: Name of the model
            **kwargs: Additional options, including:
                - supports_vision (bool): Whether the model supports images
                - supports_tools (bool): Whether the model supports tools
                - supports_audio (bool): Whether the model supports audio
                - max_tokens (int): Maximum tokens supported
        """
        self.model_name = model_name
        self.kwargs = kwargs
        self.supports_vision_flag = kwargs.get("supports_vision", False)
        self.supports_tools_flag = kwargs.get("supports_tools", False)
        self.supports_audio_flag = kwargs.get("supports_audio", False)
        self.max_tokens_value = kwargs.get("max_tokens", 2048)
        self.calls = []

    def get_model_name(self) -> str:
        """Get the model name."""
        return self.model_name

    def complete(self, messages: List[Message], **kwargs: Any) -> Message:
        """Generate a completion for the given messages."""
        self.calls.append(("complete", messages, kwargs))
        return Message.assistant_message("This is a mock response")

    def complete_stream(
        self, messages: List[Message], **kwargs: Any
    ) -> Generator[Message, None, None]:
        """Generate a streaming completion for the given messages."""
        self.calls.append(("complete_stream", messages, kwargs))
        yield Message.assistant_message("This is a mock ")
        yield Message.assistant_message("This is a mock streaming ")
        yield Message.assistant_message("This is a mock streaming response")

    async def acomplete(self, messages: List[Message], **kwargs: Any) -> Message:
        """Generate a completion asynchronously."""
        self.calls.append(("acomplete", messages, kwargs))
        return Message.assistant_message("This is a mock async response")

    async def acomplete_stream(
        self, messages: List[Message], **kwargs: Any
    ) -> AsyncGenerator[Message, None]:
        """Generate a streaming completion asynchronously."""
        self.calls.append(("acomplete_stream", messages, kwargs))
        yield Message.assistant_message("This is a mock ")
        yield Message.assistant_message("This is a mock streaming ")
        yield Message.assistant_message("This is a mock streaming async response")

    def count_tokens(self, messages: List[Message]) -> int:
        """Count tokens in messages using a simple approximation."""
        return sum(len(m.content or "") for m in messages) // 4

    def get_max_tokens(self) -> int:
        """Get the maximum number of tokens supported."""
        return self.max_tokens_value

    def supports_vision(self) -> bool:
        """Check if the provider supports vision/images."""
        return self.supports_vision_flag

    def supports_tools(self) -> bool:
        """Check if the provider supports tool/function calling."""
        return self.supports_tools_flag

    def supports_audio(self) -> bool:
        """Check if the provider supports audio processing."""
        return self.supports_audio_flag

    def get_available_models(self) -> List[str]:
        """Get a list of available models."""
        if hasattr(self, "available_models"):
            return self.available_models
        return ["model1", "model2", "model3"]

    def validate_model(self) -> bool:
        """Validate if the current model exists."""
        if hasattr(self, "model_valid"):
            return self.model_valid
        return True

    def _suggest_models(
        self, model_name: str, available_models: List[str], max_suggestions: int = 3
    ) -> List[str]:
        """Suggest similar models based on name."""
        if hasattr(self, "suggested_models"):
            return self.suggested_models[:max_suggestions]
        return [m for m in available_models if model_name[0:3] in m][:max_suggestions]


@pytest.fixture
def mock_provider():
    """Fixture for a standard mock LLM provider."""
    return MockLLMProvider("mock-model")


@pytest.fixture
def vision_provider():
    """Fixture for a mock LLM provider with vision support."""
    return MockLLMProvider("vision-model", supports_vision=True)


@pytest.fixture
def tool_provider():
    """Fixture for a mock LLM provider with tool support."""
    return MockLLMProvider("tool-model", supports_tools=True)


@pytest.fixture
def audio_provider():
    """Fixture for a mock LLM provider with audio support."""
    return MockLLMProvider("audio-model", supports_audio=True)


class TestLLMServiceEnhancements:
    """Tests for the enhanced LLMService functionality."""

    # This test was removed at user's request
    # def test_get_available_models(self):
    #    """Test retrieving available models from provider."""
    #    # Test removed as requested

    def test_validate_current_model_no_method(self):
        """Test validating a model when the provider doesn't have a validate_model method."""

        # Create a simple object without any validation method
        class SimpleProvider:
            def get_model_name(self):
                return "test-model"

        provider = SimpleProvider()

        # Mock the provider creation
        with mock.patch("enterprise_ai.llm.service.LLMService._create_provider") as mock_create:
            mock_create.return_value = provider

            service = LLMService(provider_name="openai", model_name="gpt-4")
            service._model_validated = False  # Ensure we're not using cached result

            # Test validation with a provider that doesn't have validate_model
            is_valid = service.validate_current_model()
            assert is_valid is True

    def test_get_similar_models(self):
        """Test getting similar models to a given model."""
        # Create custom mock provider
        mock_provider = mock.MagicMock()
        available_models = ["gpt-3", "gpt-3.5", "gpt-4", "gpt-4-turbo"]
        mock_provider.get_available_models = mock.MagicMock(return_value=available_models)
        mock_provider._suggest_models = mock.MagicMock(return_value=["gpt-4", "gpt-4-turbo"])

        # Mock the provider creation
        with mock.patch("enterprise_ai.llm.service.LLMService._create_provider") as mock_create:
            mock_create.return_value = mock_provider

            service = LLMService(provider_name="openai", model_name="gpt-4")

            # Test get_similar_models with provider method
            models = service.get_similar_models("gpt-4-vision", max_suggestions=2)
            assert models == ["gpt-4", "gpt-4-turbo"]
            mock_provider._suggest_models.assert_called_with("gpt-4-vision", available_models, 2)

            # Test get_similar_models without provider method
            new_mock = mock.MagicMock()
            new_mock.get_available_models = mock.MagicMock(return_value=available_models)
            # No _suggest_models method on this mock
            mock_create.return_value = new_mock

            # Reset service to use new provider
            service = LLMService(provider_name="openai", model_name="gpt-4")

            # Service should use fallback model matching logic
            models = service.get_similar_models("gpt-4", max_suggestions=2)
            assert isinstance(models, list)
            assert all(isinstance(model, str) for model in models)

            # Test with empty available models
            new_mock.get_available_models.return_value = []
            models = service.get_similar_models("gpt-4", max_suggestions=2)
            assert models == []

    def test_supports_audio(self):
        """Test audio capability detection with a provider that supports it."""

        # Create a custom provider class that has supports_audio
        class AudioProvider:
            def supports_audio(self):
                return True

        provider = AudioProvider()

        # Mock the provider creation
        with mock.patch("enterprise_ai.llm.service.LLMService._create_provider") as mock_create:
            mock_create.return_value = provider

            service = LLMService(provider_name="openai", model_name="gpt-4")

            # Test with supports_audio method
            assert service.supports_audio is True

    def test_supports_audio_with_exception(self):
        """Test audio capability detection when the method raises an exception."""

        # Create a custom provider class with a method that raises an exception
        class BrokenAudioProvider:
            def supports_audio(self):
                raise Exception("Not implemented")

        provider = BrokenAudioProvider()

        # Mock the provider creation
        with mock.patch("enterprise_ai.llm.service.LLMService._create_provider") as mock_create:
            mock_create.return_value = provider

            service = LLMService(provider_name="openai", model_name="gpt-4")

            # Test with supports_audio method that raises exception
            assert service.supports_audio is False

    def test_supports_audio_no_method(self):
        """Test audio capability detection with a provider that doesn't support it."""

        # Create a provider without the supports_audio method
        class BasicProvider:
            def get_model_name(self):
                return "test-model"

        provider = BasicProvider()

        # Mock the provider creation
        with mock.patch("enterprise_ai.llm.service.LLMService._create_provider") as mock_create:
            mock_create.return_value = provider

            service = LLMService(provider_name="openai", model_name="gpt-4")

            # Test without supports_audio method
            assert service.supports_audio is False

    def test_change_model(self):
        """Test changing model while keeping the same provider."""
        # Mock the provider creation
        with mock.patch("enterprise_ai.llm.service.LLMService._create_provider") as mock_create:
            # Set up mock provider for initial service
            initial_provider = mock.MagicMock()
            initial_provider.validate_model = mock.MagicMock(return_value=True)
            mock_create.return_value = initial_provider

            service = LLMService(provider_name="openai", model_name="gpt-3.5-turbo")

            # Set up second mock provider for new model
            new_provider = mock.MagicMock()
            new_provider.validate_model = mock.MagicMock(return_value=True)
            mock_create.return_value = new_provider

            # Test changing to valid model
            result = service.change_model("gpt-4")
            assert result is True
            assert service.model_name == "gpt-4"
            assert service.provider == new_provider

            # Test changing to same model (should return True without creating new provider)
            mock_create.reset_mock()
            result = service.change_model("gpt-4")
            assert result is True
            mock_create.assert_not_called()

            # Test changing to invalid model
            invalid_provider = mock.MagicMock()
            invalid_provider.validate_model = mock.MagicMock(return_value=False)
            mock_create.return_value = invalid_provider

            result = service.change_model("nonexistent-model")
            assert result is False
            assert service.model_name == "gpt-4"  # Should not change

            # Test with strict validation
            service.strict_validation = True
            mock_create.side_effect = ModelNotAvailable("nonexistent-model", "Model not found")
            with pytest.raises(ModelNotAvailable):
                service.change_model("nonexistent-model")

    def test_change_provider(self):
        """Test changing provider and model."""
        # Mock the provider creation
        with mock.patch("enterprise_ai.llm.service.LLMService._create_provider") as mock_create:
            # Set up mock provider for initial service
            initial_provider = mock.MagicMock()
            mock_create.return_value = initial_provider

            service = LLMService(provider_name="openai", model_name="gpt-4")

            # Set up second mock provider for new provider
            new_provider = mock.MagicMock()
            new_provider.validate_model = mock.MagicMock(return_value=True)
            mock_create.return_value = new_provider

            # Test changing to valid provider
            result = service.change_provider("anthropic", "claude-3")
            assert result is True
            assert service.provider_name == "anthropic"
            assert service.model_name == "claude-3"
            assert service.provider == new_provider

            # Test changing to invalid provider
            with pytest.raises(ProviderNotSupportedError):
                service.change_provider("invalid-provider")

            # Test changing with invalid model
            invalid_provider = mock.MagicMock()
            invalid_provider.validate_model = mock.MagicMock(return_value=False)
            mock_create.return_value = invalid_provider

            result = service.change_provider("ollama", "nonexistent-model")
            assert result is False
            assert service.provider_name == "anthropic"  # Should not change

            # Test with model validation exception
            error_provider = mock.MagicMock()
            error_provider.validate_model = mock.MagicMock(
                side_effect=Exception("Validation error")
            )
            mock_create.return_value = error_provider

            result = service.change_provider("openai", "gpt-4")
            assert result is False

    def test_fallback_models(self):
        """Test fallback model behavior when a model is not available."""
        # Set up mock providers
        main_provider = mock.MagicMock()
        main_provider.complete = mock.MagicMock(side_effect=ModelNotAvailable("gpt-4", "Not found"))

        fallback_provider = mock.MagicMock()
        fallback_response = Message.assistant_message("This is from the fallback model")
        fallback_provider.complete = mock.MagicMock(return_value=fallback_response)

        # Mock the provider creation
        with mock.patch("enterprise_ai.llm.service.LLMService._create_provider") as mock_create:
            # First return the failing provider, then the fallback provider
            mock_create.side_effect = [main_provider, fallback_provider]

            # Create service with fallback models
            service = LLMService(
                provider_name="openai",
                model_name="gpt-4",
                fallback_models={"openai": "gpt-3.5-turbo"},
            )

            # Call complete and test fallback behavior
            messages = [Message.user_message("Hello!")]
            response = service.complete(messages)

            # Check that original model was tried
            main_provider.complete.assert_called_once_with(messages)

            # Check that fallback model was created and used
            mock_create.assert_any_call("openai", "gpt-3.5-turbo")
            fallback_provider.complete.assert_called_once_with(messages)

            # Check that we got the fallback response
            assert response == fallback_response

            # Test that service now uses the fallback provider
            assert service.provider == fallback_provider

    # This test was removed at user's request
    # def test_without_fallback_models(self):
    #    """Test behavior when a model is not available and no fallback is configured."""
    #    # Test removed as requested


class TestHelperFunctions:
    """Tests for the global helper functions."""

    def test_change_model_helper(self):
        """Test the change_model helper function."""
        with mock.patch("enterprise_ai.llm.service.default_llm_service") as mock_service:
            mock_service.change_model = mock.MagicMock(return_value=True)

            # Test helper function
            result = change_model("gpt-4")
            assert result is True
            mock_service.change_model.assert_called_once_with("gpt-4", True)

            # Test with validate=False
            mock_service.change_model.reset_mock()
            result = change_model("gpt-4", validate=False)
            assert result is True
            mock_service.change_model.assert_called_once_with("gpt-4", False)

    def test_change_provider_helper(self):
        """Test the change_provider helper function."""
        with mock.patch("enterprise_ai.llm.service.default_llm_service") as mock_service:
            mock_service.change_provider = mock.MagicMock(return_value=True)

            # Test helper function
            result = change_provider("anthropic", "claude-3")
            assert result is True
            mock_service.change_provider.assert_called_once_with("anthropic", "claude-3", True)

            # Test with default model
            mock_service.change_provider.reset_mock()
            result = change_provider("anthropic")
            assert result is True
            mock_service.change_provider.assert_called_once_with("anthropic", None, True)

            # Test with validate=False
            mock_service.change_provider.reset_mock()
            result = change_provider("anthropic", "claude-3", validate=False)
            assert result is True
            mock_service.change_provider.assert_called_once_with("anthropic", "claude-3", False)

    def test_get_available_models_helper(self):
        """Test the get_available_models helper function."""
        with mock.patch("enterprise_ai.llm.service.default_llm_service") as mock_service:
            mock_service.get_available_models = mock.MagicMock(
                return_value=["model1", "model2", "model3"]
            )

            # Test helper function
            models = get_available_models()
            assert models == ["model1", "model2", "model3"]
            mock_service.get_available_models.assert_called_once()

    def test_get_similar_models_helper(self):
        """Test the get_similar_models helper function."""
        with mock.patch("enterprise_ai.llm.service.default_llm_service") as mock_service:
            mock_service.get_similar_models = mock.MagicMock(return_value=["gpt-4", "gpt-4-turbo"])

            # Test helper function
            models = get_similar_models("gpt-4-vision")
            assert models == ["gpt-4", "gpt-4-turbo"]
            mock_service.get_similar_models.assert_called_once_with("gpt-4-vision", 3)

            # Test with custom max_suggestions
            mock_service.get_similar_models.reset_mock()
            models = get_similar_models("gpt-4-vision", max_suggestions=5)
            assert models == ["gpt-4", "gpt-4-turbo"]
            mock_service.get_similar_models.assert_called_once_with("gpt-4-vision", 5)
