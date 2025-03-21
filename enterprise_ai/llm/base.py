"""
Base classes and interfaces for LLM integration.

This module defines the core abstractions for working with Language Models,
including the provider interface, token management, and conversation utilities.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, AsyncGenerator, TypeVar, cast

from enterprise_ai.schema import Message, Role
from enterprise_ai.exceptions import LLMError, TokenLimitExceeded
from enterprise_ai.logger import get_logger

# Initialize logger
logger = get_logger("llm.base")

# Type variable for LLM providers
T = TypeVar("T", bound="LLMProvider")


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def __init__(self, model_name: str, **kwargs: Any) -> None:
        """Initialize the provider with the model name and additional options.

        Args:
            model_name: Name of the model to use
            **kwargs: Additional provider-specific options
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the model name.

        Returns:
            Model name
        """
        pass

    @abstractmethod
    def complete(self, messages: List[Message], **kwargs: Any) -> Message:
        """Generate a completion for the given messages.

        Args:
            messages: List of messages in the conversation
            **kwargs: Additional completion options

        Returns:
            Generated completion message
        """
        pass

    @abstractmethod
    def complete_stream(
        self, messages: List[Message], **kwargs: Any
    ) -> Generator[Message, None, None]:
        """Generate a streaming completion for the given messages.

        Args:
            messages: List of messages in the conversation
            **kwargs: Additional completion options

        Returns:
            Generator yielding partial completion messages
        """
        pass

    @abstractmethod
    async def acomplete(self, messages: List[Message], **kwargs: Any) -> Message:
        """Generate a completion for the given messages asynchronously.

        Args:
            messages: List of messages in the conversation
            **kwargs: Additional completion options

        Returns:
            Generated completion message
        """
        pass

    @abstractmethod
    async def acomplete_stream(
        self, messages: List[Message], **kwargs: Any
    ) -> AsyncGenerator[Message, None]:
        """Generate a streaming completion for the given messages asynchronously.

        Args:
            messages: List of messages in the conversation
            **kwargs: Additional completion options

        Returns:
            Async generator yielding partial completion messages
        """
        pass

    @abstractmethod
    def count_tokens(self, messages: List[Message]) -> int:
        """Count the number of tokens in the given messages.

        Args:
            messages: List of messages to count tokens for

        Returns:
            Number of tokens
        """
        pass

    @abstractmethod
    def get_max_tokens(self) -> int:
        """Get the maximum number of tokens supported by the model.

        Returns:
            Maximum token count
        """
        pass

    @abstractmethod
    def supports_vision(self) -> bool:
        """Check if the provider/model supports vision/images.

        Returns:
            True if vision is supported, False otherwise
        """
        pass

    @abstractmethod
    def supports_tools(self) -> bool:
        """Check if the provider/model supports tool/function calling.

        Returns:
            True if tool calling is supported, False otherwise
        """
        pass


class ConversationManager:
    """Manages conversation context for LLM interactions."""

    def __init__(
        self,
        provider: LLMProvider,
        system_message: Optional[str] = None,
        max_tokens: Optional[int] = None,
        token_limit_pct: float = 0.9,
    ) -> None:
        """Initialize the conversation manager.

        Args:
            provider: LLM provider to use
            system_message: Optional system message to start the conversation
            max_tokens: Optional maximum token limit (defaults to provider's limit)
            token_limit_pct: Percentage of maximum tokens to use (default: 90%)
        """
        self.provider = provider
        self.messages: List[Message] = []
        self.token_limit_pct = token_limit_pct

        # Set maximum tokens
        if max_tokens is not None:
            self.max_tokens = max_tokens
        else:
            self.max_tokens = provider.get_max_tokens()

        # Add system message if provided
        if system_message:
            self.add_system_message(system_message)

    def add_system_message(self, content: str) -> None:
        """Add a system message to the conversation.

        Args:
            content: System message content
        """
        message = Message.system_message(content)

        # If there's already a system message, replace it
        if self.messages and self.messages[0].role == Role.SYSTEM:
            self.messages[0] = message
        else:
            # Insert at the beginning
            self.messages.insert(0, message)

    def add_user_message(self, content: str, base64_image: Optional[str] = None) -> None:
        """Add a user message to the conversation.

        Args:
            content: User message content
            base64_image: Optional base64-encoded image
        """
        self.messages.append(Message.user_message(content, base64_image=base64_image))

    def add_assistant_message(
        self, content: Optional[str] = None, tool_calls: Optional[List[Any]] = None
    ) -> None:
        """Add an assistant message to the conversation.

        Args:
            content: Assistant message content
            tool_calls: Optional tool calls made by the assistant
        """
        if tool_calls:
            self.messages.append(Message.from_tool_calls(tool_calls, content or ""))
        else:
            self.messages.append(Message.assistant_message(content))

    def add_tool_message(self, content: str, name: str, tool_call_id: str) -> None:
        """Add a tool message to the conversation.

        Args:
            content: Tool message content
            name: Tool name
            tool_call_id: Tool call ID
        """
        self.messages.append(Message.tool_message(content, name, tool_call_id))

    def get_messages(self) -> List[Message]:
        """Get all messages in the conversation.

        Returns:
            List of messages
        """
        return self.messages.copy()

    def count_tokens(self) -> int:
        """Count the number of tokens in the conversation.

        Returns:
            Number of tokens
        """
        return self.provider.count_tokens(self.messages)

    def prune_to_fit_context(self, reserve_tokens: int = 0) -> List[Message]:
        """Prune messages to fit within context window.

        The method preserves system message and recent messages, removing
        older messages from the middle of the conversation if needed.

        Args:
            reserve_tokens: Number of tokens to reserve for future responses

        Returns:
            List of pruned messages

        Raises:
            ContextWindowExceededError: If messages can't be pruned to fit
        """
        from enterprise_ai.llm.exceptions import ContextWindowExceededError

        # Calculate token budget
        token_budget = int(self.max_tokens * self.token_limit_pct) - reserve_tokens
        current_tokens = self.count_tokens()

        if current_tokens <= token_budget:
            return self.messages.copy()

        # Need to prune
        pruned_messages = self.messages.copy()
        system_message = None

        # Extract system message if present
        if pruned_messages and pruned_messages[0].role == Role.SYSTEM:
            system_message = pruned_messages.pop(0)

        # Keep removing messages from the middle until we fit the budget
        while (
            len(pruned_messages) > 2
            and self.provider.count_tokens(
                [system_message] + pruned_messages if system_message else pruned_messages
            )
            > token_budget
        ):
            # Remove a message from the middle (preserve most recent context)
            middle_idx = len(pruned_messages) // 3
            pruned_messages.pop(middle_idx)

        # Check if we still exceed the token budget
        final_messages = [system_message] + pruned_messages if system_message else pruned_messages
        final_token_count = self.provider.count_tokens(final_messages)

        if final_token_count > token_budget:
            # If we still don't fit, raise an error
            raise ContextWindowExceededError(
                self.provider.get_model_name(), final_token_count, token_budget
            )

        # Update internal messages and return
        self.messages = final_messages
        return final_messages

    def clear(self, keep_system: bool = True) -> None:
        """Clear the conversation history.

        Args:
            keep_system: Whether to keep the system message
        """
        if keep_system and self.messages and self.messages[0].role == Role.SYSTEM:
            system_message = self.messages[0]
            self.messages = [system_message]
        else:
            self.messages = []
