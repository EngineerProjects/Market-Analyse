"""
Tool framework for Enterprise AI.

This module provides a collection of tools that agents can use to interact with
their environment, including file operations, Python execution, terminal commands,
and more. It also includes the core abstractions for tool execution and security.
"""

from enterprise_ai.tool.authorization import (
    AuthorizationManager,
    ToolPermission,
    auth_manager,
)

from enterprise_ai.tool.base import BaseTool, ToolInput, ToolResult

from enterprise_ai.tool.file_operators import FileOperator

from enterprise_ai.tool.python_execute import PythonExecute, SecurityVisitor

from enterprise_ai.tool.terminal import Terminal

from enterprise_ai.tool.tool_collection import ToolCollection

# Export commonly used classes and instances
__all__ = [
    # Authorization
    "AuthorizationManager",
    "ToolPermission",
    "auth_manager",
    # Base classes
    "BaseTool",
    "ToolInput",
    "ToolResult",
    # Tool implementations
    "FileOperator",
    "PythonExecute",
    "Terminal",
    # Tool management
    "ToolCollection",
    "SecurityVisitor",
]
