"""
Tests for the exceptions module.

This module contains tests for the custom exception classes defined in the
exceptions module, ensuring they function as expected and provide
accurate error information.
"""

import unittest

from enterprise_ai.exceptions import (
    AgentExecutionError,
    AgentNotFound,
    AgentStateError,
    AuthorizationError,
    ConfigFileError,
    ConfigValueError,
    EnterpriseAIError,
    FileReadError,
    LLMError,
    SandboxTimeoutError,
    TaskDependencyError,
    TaskNotFound,
    TeamSizeError,
    ToolExecutionError,
    ToolNotFound,
    TokenLimitExceeded,
    UnsafeOperationError,
    WorkflowExecutionError,
)


class TestExceptions(unittest.TestCase):
    """Test case for the exceptions module."""

    def test_base_exceptions(self):
        """Test base exception classes."""
        base_ex = EnterpriseAIError("Base error")
        self.assertEqual(str(base_ex), "Base error")

        # Test LLM error
        llm_ex = LLMError("LLM failed")
        self.assertEqual(str(llm_ex), "LLM failed")
        self.assertTrue(isinstance(llm_ex, EnterpriseAIError))

    def test_config_exceptions(self):
        """Test configuration exception classes."""
        # Test ConfigFileError
        file_ex = ConfigFileError("/path/to/config.toml")
        self.assertEqual(file_ex.file_path, "/path/to/config.toml")
        self.assertIn("/path/to/config.toml", str(file_ex))

        # Test ConfigValueError
        value_ex = ConfigValueError("api_key", "invalid_key")
        self.assertEqual(value_ex.key, "api_key")
        self.assertEqual(value_ex.value, "invalid_key")
        self.assertIn("api_key", str(value_ex))

    def test_llm_exceptions(self):
        """Test LLM exception classes."""
        # Test TokenLimitExceeded
        token_ex = TokenLimitExceeded("Too many tokens")
        self.assertEqual(str(token_ex), "Too many tokens")

        # Test custom message
        token_ex2 = TokenLimitExceeded()
        self.assertEqual(str(token_ex2), "Token limit exceeded")

    def test_tool_exceptions(self):
        """Test tool exception classes."""
        # Test ToolNotFound
        not_found_ex = ToolNotFound("nonexistent_tool")
        self.assertEqual(not_found_ex.tool_name, "nonexistent_tool")
        self.assertIn("nonexistent_tool", str(not_found_ex))

        # Test ToolExecutionError
        exec_ex = ToolExecutionError("calculator", "division by zero")
        self.assertEqual(exec_ex.tool_name, "calculator")
        self.assertEqual(exec_ex.error, "division by zero")
        self.assertIn("calculator", str(exec_ex))
        self.assertIn("division by zero", str(exec_ex))

    def test_agent_exceptions(self):
        """Test agent exception classes."""
        # Test AgentNotFound
        not_found_ex = AgentNotFound("agent1")
        self.assertEqual(not_found_ex.agent_name, "agent1")

        # Test AgentStateError
        state_ex = AgentStateError("agent1", "RUNNING", "IDLE")
        self.assertEqual(state_ex.agent_name, "agent1")
        self.assertEqual(state_ex.current_state, "RUNNING")
        self.assertEqual(state_ex.required_state, "IDLE")
        self.assertIn("RUNNING", str(state_ex))
        self.assertIn("IDLE", str(state_ex))

        # Test AgentExecutionError
        exec_ex = AgentExecutionError("agent1", "timeout")
        self.assertEqual(exec_ex.agent_name, "agent1")
        self.assertEqual(exec_ex.error, "timeout")

    def test_team_exceptions(self):
        """Test team exception classes."""
        # Test TeamSizeError
        size_ex = TeamSizeError("team1", 11, 10)
        self.assertEqual(size_ex.team_name, "team1")
        self.assertEqual(size_ex.current_size, 11)
        self.assertEqual(size_ex.max_size, 10)
        self.assertIn("exceeds", str(size_ex))

    def test_task_exceptions(self):
        """Test task exception classes."""
        # Test TaskNotFound
        not_found_ex = TaskNotFound("task-123")
        self.assertEqual(not_found_ex.task_id, "task-123")

        # Test TaskDependencyError
        dep_ex = TaskDependencyError("task-2", "task-1")
        self.assertEqual(dep_ex.task_id, "task-2")
        self.assertEqual(dep_ex.dependency_id, "task-1")
        self.assertIn("depends", str(dep_ex))

    def test_sandbox_exceptions(self):
        """Test sandbox exception classes."""
        # Test SandboxTimeoutError
        timeout_ex = SandboxTimeoutError(60)
        self.assertEqual(timeout_ex.timeout, 60)
        self.assertIn("60 seconds", str(timeout_ex))

    def test_security_exceptions(self):
        """Test security exception classes."""
        # Test AuthorizationError
        auth_ex = AuthorizationError("agent1", "write_file")
        self.assertEqual(auth_ex.entity, "agent1")
        self.assertEqual(auth_ex.action, "write_file")

        # Test UnsafeOperationError
        unsafe_ex = UnsafeOperationError("rm -rf /", "dangerous command")
        self.assertEqual(unsafe_ex.operation, "rm -rf /")
        self.assertEqual(unsafe_ex.reason, "dangerous command")

    def test_file_exceptions(self):
        """Test file operation exception classes."""
        # Test FileReadError
        read_ex = FileReadError("/tmp/file.txt", "permission denied")
        self.assertEqual(read_ex.path, "/tmp/file.txt")
        self.assertEqual(read_ex.error, "permission denied")
        self.assertIn("permission denied", str(read_ex))

    def test_workflow_exceptions(self):
        """Test workflow exception classes."""
        # Test WorkflowExecutionError
        exec_ex = WorkflowExecutionError("workflow-1", "agent failed")
        self.assertEqual(exec_ex.workflow_id, "workflow-1")
        self.assertEqual(exec_ex.error, "agent failed")
        self.assertIn("workflow-1", str(exec_ex))


if __name__ == "__main__":
    unittest.main()
