"""
Schema definitions for Enterprise AI.

This module provides the core data models and type definitions used throughout
the Enterprise AI framework, including agent roles, message types, states, and
team structures. These schemas provide a foundation for type-safe operations
and data consistency across the system.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union, TypeVar, cast

from pydantic import BaseModel, Field, model_validator

# Type variable for generic typing
T = TypeVar('T')


# Role and Agent Type Definitions
class Role(str, Enum):
    """Message role options for conversation interactions."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    AGENT = "agent"  # Added for inter-agent communication


class AgentRole(str, Enum):
    """Agent role types for team hierarchies."""
    MANAGER = "manager"
    DEVELOPER = "developer"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    CUSTOM = "custom"  # For custom defined roles


# Message type literal definitions - keeping for backward compatibility
ROLE_VALUES = tuple(role.value for role in Role)
# We're not using ROLE_TYPE anymore, but keeping it for backward compatibility
# in case there's existing code that relies on it
ROLE_TYPE = Role  # Using the Enum directly instead of Literal


# Tool and Function Call Schemas
class Function(BaseModel):
    """Represents a function definition in a tool call."""
    name: str
    arguments: str


class ToolCall(BaseModel):
    """Represents a tool/function call in a message."""
    id: str
    type: str = "function"
    function: Function


# Choice options for tool usage
class ToolChoice(str, Enum):
    """Tool choice options for LLM interactions."""
    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"


TOOL_CHOICE_VALUES = tuple(choice.value for choice in ToolChoice)
TOOL_CHOICE_TYPE = Literal[TOOL_CHOICE_VALUES]  # type: ignore


# Execution States
class AgentState(str, Enum):
    """Agent execution states."""
    IDLE = "IDLE"
    RUNNING = "RUNNING"
    THINKING = "THINKING"
    ACTING = "ACTING"
    FINISHED = "FINISHED"
    ERROR = "ERROR"
    BLOCKED = "BLOCKED"
    WAITING = "WAITING"


# Team states
class TeamState(str, Enum):
    """Team execution states."""
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    REVIEWING = "REVIEWING"
    COMPLETED = "COMPLETED"
    ERROR = "ERROR"


# Task states
class TaskState(str, Enum):
    """Task execution states."""
    PENDING = "PENDING"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    REVIEW = "REVIEW"
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


# Messages
class Message(BaseModel):
    """Represents a chat message in the conversation."""
    role: Role
    content: Optional[str] = Field(default=None)
    tool_calls: Optional[List[ToolCall]] = Field(default=None)
    name: Optional[str] = Field(default=None)
    tool_call_id: Optional[str] = Field(default=None)
    base64_image: Optional[str] = Field(default=None)
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=lambda: {})
    
    def __add__(self, other: Union[List["Message"], "Message"]) -> List["Message"]:
        """Support Message + list or Message + Message operations."""
        if isinstance(other, list):
            return [self] + other
        elif isinstance(other, Message):
            return [self, other]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(self).__name__}' and '{type(other).__name__}'"
            )

    def __radd__(self, other: List["Message"]) -> List["Message"]:
        """Support list + Message operations."""
        if isinstance(other, list):
            return other + [self]
        else:
            raise TypeError(
                f"unsupported operand type(s) for +: '{type(other).__name__}' and '{type(self).__name__}'"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        message: Dict[str, Any] = {"role": self.role}
        if self.content is not None:
            message["content"] = self.content
        if self.tool_calls is not None:
            message["tool_calls"] = [tool_call.model_dump() for tool_call in self.tool_calls]
        if self.name is not None:
            message["name"] = self.name
        if self.tool_call_id is not None:
            message["tool_call_id"] = self.tool_call_id
        if self.base64_image is not None:
            message["base64_image"] = self.base64_image
        if self.metadata:
            message["metadata"] = self.metadata
        return message

    @classmethod
    def user_message(
        cls, content: str, base64_image: Optional[str] = None, **kwargs: Any
    ) -> "Message":
        """Create a user message."""
        return cls(role=Role.USER, content=content, base64_image=base64_image, **kwargs)

    @classmethod
    def system_message(cls, content: str, **kwargs: Any) -> "Message":
        """Create a system message."""
        return cls(role=Role.SYSTEM, content=content, **kwargs)

    @classmethod
    def assistant_message(
        cls, content: Optional[str] = None, base64_image: Optional[str] = None, **kwargs: Any
    ) -> "Message":
        """Create an assistant message."""
        return cls(role=Role.ASSISTANT, content=content, base64_image=base64_image, **kwargs)

    @classmethod
    def tool_message(
        cls, content: str, name: str, tool_call_id: str, base64_image: Optional[str] = None, **kwargs: Any
    ) -> "Message":
        """Create a tool message."""
        return cls(
            role=Role.TOOL,
            content=content,
            name=name,
            tool_call_id=tool_call_id,
            base64_image=base64_image,
            **kwargs,
        )
    
    @classmethod
    def agent_message(
        cls, content: str, name: str, base64_image: Optional[str] = None, **kwargs: Any
    ) -> "Message":
        """Create an agent message for inter-agent communication."""
        return cls(
            role=Role.AGENT,
            content=content,
            name=name,
            base64_image=base64_image,
            **kwargs,
        )

    @classmethod
    def from_tool_calls(
        cls,
        tool_calls: List[Any],
        content: Union[str, List[str]] = "",
        base64_image: Optional[str] = None,
        **kwargs: Any,
    ) -> "Message":
        """Create ToolCallsMessage from raw tool calls."""
        formatted_calls = [
            ToolCall(
                id=call.id,
                function=Function(
                    name=call.function.name,
                    arguments=call.function.arguments
                ),
                type="function"
            )
            for call in tool_calls
        ]
        
        # Convert content to string if it's a list
        final_content: Optional[str] = None
        if isinstance(content, list):
            final_content = "\n".join(content)
        else:
            final_content = content
            
        return cls(
            role=Role.ASSISTANT,
            content=final_content,
            tool_calls=formatted_calls,
            base64_image=base64_image,
            **kwargs,
        )


class Memory(BaseModel):
    """Memory store for agent messages."""
    messages: List[Message] = Field(default_factory=list)
    max_messages: int = Field(default=100)
    metadata: Dict[str, Any] = Field(default_factory=lambda: {})

    def add_message(self, message: Message) -> None:
        """Add a message to memory."""
        self.messages.append(message)
        # Optional: Implement message limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def add_messages(self, messages: List[Message]) -> None:
        """Add multiple messages to memory."""
        self.messages.extend(messages)
        # Check for message limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def clear(self) -> None:
        """Clear all messages."""
        self.messages.clear()

    def get_recent_messages(self, n: int) -> List[Message]:
        """Get n most recent messages."""
        return self.messages[-n:]

    def to_dict_list(self) -> List[dict]:
        """Convert messages to list of dicts."""
        return [msg.to_dict() for msg in self.messages]


# Agent and Team Schemas
class AgentProfile(BaseModel):
    """Profile information for an agent."""
    name: str = Field(..., description="Unique name of the agent")
    role: str = Field(..., description="Role of the agent (from AgentRole enum)")
    description: Optional[str] = Field(None, description="Description of the agent")
    capabilities: List[str] = Field(default_factory=list, description="List of agent capabilities")
    specialties: List[str] = Field(default_factory=list, description="Areas of specialty for the agent")
    system_prompt: Optional[str] = Field(None, description="System-level instruction prompt")


class AgentConfig(BaseModel):
    """Configuration for an agent instance."""
    profile: AgentProfile
    model_name: str = Field("default", description="Name of LLM model to use")
    max_steps: int = Field(10, description="Maximum steps before termination")
    tools: List[str] = Field(default_factory=list, description="List of tool names available to agent")
    allowed_tools: List[str] = Field(default_factory=list, description="List of tool names the agent is allowed to use")
    
    @model_validator(mode="after")
    def validate_tools(self) -> "AgentConfig":
        """Validate tool configurations."""
        if not self.allowed_tools:
            self.allowed_tools = self.tools.copy()
        return self


class TaskDefinition(BaseModel):
    """Definition of a task to be performed by an agent or team."""
    id: str = Field(..., description="Unique task identifier")
    title: str = Field(..., description="Short task title")
    description: str = Field(..., description="Detailed task description")
    assigned_to: Optional[str] = Field(None, description="Agent or team assigned to the task")
    state: TaskState = Field(default=TaskState.PENDING, description="Current task state")
    priority: int = Field(default=1, description="Priority level (higher is more important)")
    deadline: Optional[datetime] = Field(None, description="Deadline for task completion")
    parent_task: Optional[str] = Field(None, description="Parent task ID if this is a subtask")
    subtasks: List[str] = Field(default_factory=list, description="List of subtask IDs")
    dependencies: List[str] = Field(default_factory=list, description="List of task IDs this task depends on")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional task metadata")


class TeamConfig(BaseModel):
    """Configuration for a team of agents."""
    name: str = Field(..., description="Unique team name")
    description: Optional[str] = Field(None, description="Team description")
    manager: str = Field(..., description="Name of the manager agent")
    members: List[str] = Field(default_factory=list, description="List of team member agent names")
    max_size: int = Field(10, description="Maximum team size")
    
    @model_validator(mode="after")
    def validate_team_size(self) -> "TeamConfig":
        """Validate team size constraints."""
        if len(self.members) + 1 > self.max_size:  # +1 for manager
            raise ValueError(f"Team size exceeds maximum allowed ({self.max_size})")
        if self.manager not in self.members:
            self.members.append(self.manager)
        return self