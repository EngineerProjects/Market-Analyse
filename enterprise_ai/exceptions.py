"""
Custom exception classes for Enterprise AI.

This module provides a hierarchy of exception types for different error scenarios
in the Enterprise AI framework, enabling precise error handling and reporting
throughout the system.
"""

from typing import Any, Optional

class EnterpriseAIError(Exception):
    """Base exception for all Enterprise AI errors."""
    def __init__(self, message: str = "An error occurred in Enterprise AI") -> None:
        self.message = message
        super().__init__(self.message)


# Configuration Errors
class ConfigError(EnterpriseAIError):
    """Base class for configuration-related errors."""
    def __init__(self, message: str = "Error in Enterprise AI configuration") -> None:
        super().__init__(message)


class ConfigFileError(ConfigError):
    """Error loading or parsing configuration files."""
    def __init__(self, file_path: Optional[str] = None, message: Optional[str] = None) -> None:
        self.file_path = file_path
        msg = message or f"Error loading configuration file: {file_path}"
        super().__init__(msg)


class ConfigValueError(ConfigError):
    """Error with configuration values."""
    def __init__(self, key: Optional[str] = None, value: Optional[Any] = None, message: Optional[str] = None) -> None:
        self.key = key
        self.value = value
        msg = message or f"Invalid configuration value for {key}: {value}"
        super().__init__(msg)


# LLM Errors
class LLMError(EnterpriseAIError):
    """Base class for LLM-related errors."""
    def __init__(self, message: str = "Error in LLM operation") -> None:
        super().__init__(message)


class TokenLimitExceeded(LLMError):
    """Exception raised when the token limit is exceeded."""
    def __init__(self, message: str = "Token limit exceeded") -> None:
        super().__init__(message)


class ModelNotAvailable(LLMError):
    """Exception raised when a requested LLM model is not available."""
    def __init__(self, model_name: Optional[str] = None, message: Optional[str] = None) -> None:
        self.model_name = model_name
        msg = message or f"Model not available: {model_name}"
        super().__init__(msg)


class APIError(LLMError):
    """Exception raised when an API error occurs."""
    def __init__(self, status_code: Optional[int] = None, message: Optional[str] = None) -> None:
        self.status_code = status_code
        msg = message or f"API error occurred: {status_code}"
        super().__init__(msg)


# Tool Errors
class ToolError(EnterpriseAIError):
    """Base class for tool-related errors."""
    def __init__(self, message: str = "Error in tool execution") -> None:
        super().__init__(message)


class ToolNotFound(ToolError):
    """Exception raised when a requested tool is not found."""
    def __init__(self, tool_name: Optional[str] = None, message: Optional[str] = None) -> None:
        self.tool_name = tool_name
        msg = message or f"Tool not found: {tool_name}"
        super().__init__(msg)


class ToolExecutionError(ToolError):
    """Exception raised when a tool execution fails."""
    def __init__(self, tool_name: Optional[str] = None, error: Optional[Any] = None, 
                 message: Optional[str] = None) -> None:
        self.tool_name = tool_name
        self.error = error
        msg = message or f"Error executing tool {tool_name}: {error}"
        super().__init__(msg)


class ToolPermissionError(ToolError):
    """Exception raised when a tool permission error occurs."""
    def __init__(self, tool_name: Optional[str] = None, agent_name: Optional[str] = None, 
                 message: Optional[str] = None) -> None:
        self.tool_name = tool_name
        self.agent_name = agent_name
        msg = message or f"Permission denied for {agent_name} to use tool {tool_name}"
        super().__init__(msg)


# Agent Errors
class AgentError(EnterpriseAIError):
    """Base class for agent-related errors."""
    def __init__(self, message: str = "Error in agent operation") -> None:
        super().__init__(message)


class AgentNotFound(AgentError):
    """Exception raised when a requested agent is not found."""
    def __init__(self, agent_name: Optional[str] = None, message: Optional[str] = None) -> None:
        self.agent_name = agent_name
        msg = message or f"Agent not found: {agent_name}"
        super().__init__(msg)


class AgentExecutionError(AgentError):
    """Exception raised when an agent execution fails."""
    def __init__(self, agent_name: Optional[str] = None, error: Optional[Any] = None, 
                 message: Optional[str] = None) -> None:
        self.agent_name = agent_name
        self.error = error
        msg = message or f"Error executing agent {agent_name}: {error}"
        super().__init__(msg)


class AgentStateError(AgentError):
    """Exception raised when an agent is in an invalid state for an operation."""
    def __init__(self, agent_name: Optional[str] = None, current_state: Optional[str] = None, 
                 required_state: Optional[str] = None, message: Optional[str] = None) -> None:
        self.agent_name = agent_name
        self.current_state = current_state
        self.required_state = required_state
        msg = message or f"Agent {agent_name} in state {current_state}, but requires {required_state}"
        super().__init__(msg)


# Team Errors
class TeamError(EnterpriseAIError):
    """Base class for team-related errors."""
    def __init__(self, message: str = "Error in team operation") -> None:
        super().__init__(message)


class TeamNotFound(TeamError):
    """Exception raised when a requested team is not found."""
    def __init__(self, team_name: Optional[str] = None, message: Optional[str] = None) -> None:
        self.team_name = team_name
        msg = message or f"Team not found: {team_name}"
        super().__init__(msg)


class TeamConfigError(TeamError):
    """Exception raised when a team configuration error occurs."""
    def __init__(self, team_name: Optional[str] = None, error: Optional[Any] = None, 
                 message: Optional[str] = None) -> None:
        self.team_name = team_name
        self.error = error
        msg = message or f"Team configuration error for {team_name}: {error}"
        super().__init__(msg)


class TeamSizeError(TeamError):
    """Exception raised when a team size constraint is violated."""
    def __init__(self, team_name: Optional[str] = None, current_size: Optional[int] = None, 
                 max_size: Optional[int] = None, message: Optional[str] = None) -> None:
        self.team_name = team_name
        self.current_size = current_size
        self.max_size = max_size
        msg = message or f"Team {team_name} size ({current_size}) exceeds maximum ({max_size})"
        super().__init__(msg)


# Task Errors
class TaskError(EnterpriseAIError):
    """Base class for task-related errors."""
    def __init__(self, message: str = "Error in task operation") -> None:
        super().__init__(message)


class TaskNotFound(TaskError):
    """Exception raised when a requested task is not found."""
    def __init__(self, task_id: Optional[str] = None, message: Optional[str] = None) -> None:
        self.task_id = task_id
        msg = message or f"Task not found: {task_id}"
        super().__init__(msg)


class TaskStateError(TaskError):
    """Exception raised when a task is in an invalid state for an operation."""
    def __init__(self, task_id: Optional[str] = None, current_state: Optional[str] = None, 
                 required_state: Optional[str] = None, message: Optional[str] = None) -> None:
        self.task_id = task_id
        self.current_state = current_state
        self.required_state = required_state
        msg = message or f"Task {task_id} in state {current_state}, but requires {required_state}"
        super().__init__(msg)


class TaskDependencyError(TaskError):
    """Exception raised when a task dependency constraint is violated."""
    def __init__(self, task_id: Optional[str] = None, dependency_id: Optional[str] = None, 
                 message: Optional[str] = None) -> None:
        self.task_id = task_id
        self.dependency_id = dependency_id
        msg = message or f"Task {task_id} depends on incomplete task {dependency_id}"
        super().__init__(msg)


# Workflow Errors
class WorkflowError(EnterpriseAIError):
    """Base class for workflow-related errors."""
    def __init__(self, message: str = "Error in workflow operation") -> None:
        super().__init__(message)


class WorkflowNotFound(WorkflowError):
    """Exception raised when a requested workflow is not found."""
    def __init__(self, workflow_id: Optional[str] = None, message: Optional[str] = None) -> None:
        self.workflow_id = workflow_id
        msg = message or f"Workflow not found: {workflow_id}"
        super().__init__(msg)


class WorkflowExecutionError(WorkflowError):
    """Exception raised when a workflow execution fails."""
    def __init__(self, workflow_id: Optional[str] = None, error: Optional[Any] = None, 
                 message: Optional[str] = None) -> None:
        self.workflow_id = workflow_id
        self.error = error
        msg = message or f"Error executing workflow {workflow_id}: {error}"
        super().__init__(msg)


# Sandbox Errors
class SandboxError(EnterpriseAIError):
    """Base class for sandbox-related errors."""
    def __init__(self, message: str = "Error in sandbox operation") -> None:
        super().__init__(message)


class SandboxTimeoutError(SandboxError):
    """Exception raised when a sandbox operation times out."""
    def __init__(self, timeout: Optional[int] = None, message: Optional[str] = None) -> None:
        self.timeout = timeout
        msg = message or f"Sandbox operation timed out after {timeout} seconds"
        super().__init__(msg)


class SandboxResourceError(SandboxError):
    """Exception raised when sandbox resource limits are exceeded."""
    def __init__(self, resource: Optional[str] = None, limit: Optional[Any] = None, 
                 message: Optional[str] = None) -> None:
        self.resource = resource
        self.limit = limit
        msg = message or f"Sandbox {resource} limit exceeded: {limit}"
        super().__init__(msg)


# Security Errors
class SecurityError(EnterpriseAIError):
    """Base class for security-related errors."""
    def __init__(self, message: str = "Security error") -> None:
        super().__init__(message)


class AuthorizationError(SecurityError):
    """Exception raised when an authorization error occurs."""
    def __init__(self, entity: Optional[str] = None, action: Optional[str] = None, 
                 message: Optional[str] = None) -> None:
        self.entity = entity
        self.action = action
        msg = message or f"Authorization error: {entity} not authorized for {action}"
        super().__init__(msg)


class UnsafeOperationError(SecurityError):
    """Exception raised when an unsafe operation is attempted."""
    def __init__(self, operation: Optional[str] = None, reason: Optional[str] = None, 
                 message: Optional[str] = None) -> None:
        self.operation = operation
        self.reason = reason
        msg = message or f"Unsafe operation {operation}: {reason}"
        super().__init__(msg)


# File and I/O Errors
class FileOperationError(EnterpriseAIError):
    """Base class for file operation errors."""
    def __init__(self, message: str = "Error in file operation") -> None:
        super().__init__(message)


class FileReadError(FileOperationError):
    """Exception raised when a file read operation fails."""
    def __init__(self, path: Optional[str] = None, error: Optional[Any] = None, 
                 message: Optional[str] = None) -> None:
        self.path = path
        self.error = error
        msg = message or f"Error reading file {path}: {error}"
        super().__init__(msg)


class FileWriteError(FileOperationError):
    """Exception raised when a file write operation fails."""
    def __init__(self, path: Optional[str] = None, error: Optional[Any] = None, 
                 message: Optional[str] = None) -> None:
        self.path = path
        self.error = error
        msg = message or f"Error writing to file {path}: {error}"
        super().__init__(msg)