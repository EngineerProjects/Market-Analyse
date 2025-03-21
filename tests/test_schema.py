"""
Tests for the schema module.

This module contains tests for the data models and type definitions
in the schema module, ensuring they function as expected and maintain
type safety.
"""

import unittest

from enterprise_ai.schema import (
    AgentConfig,
    AgentProfile,
    AgentRole,
    AgentState,
    Function,
    Memory,
    Message,
    Role,
    TaskDefinition,
    TaskState,
    TeamConfig,
    ToolCall,
)


class TestSchema(unittest.TestCase):
    """Test case for the schema module."""

    def test_message_creation(self):
        """Test creating and manipulating message objects."""
        # Test basic message creation
        user_msg = Message.user_message("Hello")
        self.assertEqual(user_msg.role, "user")
        self.assertEqual(user_msg.content, "Hello")

        # Test system message
        sys_msg = Message.system_message("You are an assistant")
        self.assertEqual(sys_msg.role, "system")

        # Test assistant message
        asst_msg = Message.assistant_message("I can help with that")
        self.assertEqual(asst_msg.role, "assistant")

        # Test tool message
        tool_msg = Message.tool_message(content="Result", name="calculator", tool_call_id="12345")
        self.assertEqual(tool_msg.role, "tool")
        self.assertEqual(tool_msg.name, "calculator")

        # Test agent message
        agent_msg = Message.agent_message(content="Task completed", name="developer_agent")
        self.assertEqual(agent_msg.role, "agent")
        self.assertEqual(agent_msg.name, "developer_agent")

        # Test message combination
        msg_list = user_msg + asst_msg
        self.assertIsInstance(msg_list, list)
        self.assertEqual(len(msg_list), 2)

        # Test message to dict conversion
        msg_dict = user_msg.to_dict()
        self.assertIsInstance(msg_dict, dict)
        self.assertEqual(msg_dict["role"], "user")

    def test_memory(self):
        """Test memory functionality."""
        memory = Memory()
        self.assertEqual(len(memory.messages), 0)

        # Add a message
        msg = Message.user_message("Test")
        memory.add_message(msg)
        self.assertEqual(len(memory.messages), 1)

        # Add multiple messages
        memory.add_messages(
            [Message.system_message("System"), Message.assistant_message("Assistant")]
        )
        self.assertEqual(len(memory.messages), 3)

        # Test get recent
        recent = memory.get_recent_messages(2)
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[0].role, "system")
        self.assertEqual(recent[1].role, "assistant")

        # Test clear
        memory.clear()
        self.assertEqual(len(memory.messages), 0)

        # Test message limit
        small_memory = Memory(max_messages=2)
        small_memory.add_messages(
            [Message.user_message("1"), Message.user_message("2"), Message.user_message("3")]
        )
        self.assertEqual(len(small_memory.messages), 2)
        self.assertEqual(small_memory.messages[0].content, "2")
        self.assertEqual(small_memory.messages[1].content, "3")

    def test_tool_call(self):
        """Test tool call functionality."""
        function = Function(name="calculator", arguments='{"a": 1, "b": 2}')
        tool_call = ToolCall(id="123", function=function)

        self.assertEqual(tool_call.type, "function")
        self.assertEqual(tool_call.function.name, "calculator")

        # Test tool call in message
        msg = Message(role="assistant", content="Calculating", tool_calls=[tool_call])
        self.assertIsNotNone(msg.tool_calls)
        self.assertEqual(len(msg.tool_calls), 1)

    def test_agent_profile(self):
        """Test agent profile functionality."""
        profile = AgentProfile(
            name="dev_agent",
            role=AgentRole.DEVELOPER.value,
            description="Development specialist",
            capabilities=["coding", "debugging"],
        )

        self.assertEqual(profile.name, "dev_agent")
        self.assertEqual(profile.role, "developer")
        self.assertEqual(len(profile.capabilities), 2)

    def test_agent_config(self):
        """Test agent configuration functionality."""
        profile = AgentProfile(name="dev_agent", role=AgentRole.DEVELOPER.value)

        config = AgentConfig(profile=profile, model_name="gpt-4", tools=["bash", "python_execute"])

        self.assertEqual(config.profile.name, "dev_agent")
        self.assertEqual(config.model_name, "gpt-4")
        self.assertEqual(len(config.tools), 2)
        self.assertEqual(len(config.allowed_tools), 2)

    def test_task_definition(self):
        """Test task definition functionality."""
        task = TaskDefinition(
            id="task-123",
            title="Implement Feature",
            description="Implement the new feature X",
            priority=2,
        )

        self.assertEqual(task.id, "task-123")
        self.assertEqual(task.state, TaskState.PENDING)

        # Create subtask
        subtask = TaskDefinition(
            id="task-124",
            title="Write Tests",
            description="Write tests for feature X",
            parent_task="task-123",
        )

        self.assertEqual(subtask.parent_task, "task-123")

        # Link tasks
        task.subtasks.append(subtask.id)
        self.assertEqual(len(task.subtasks), 1)

    def test_team_config(self):
        """Test team configuration functionality."""
        team = TeamConfig(
            name="dev_team",
            description="Development team",
            manager="manager_agent",
            members=["dev_agent_1", "dev_agent_2"],
        )

        self.assertEqual(team.name, "dev_team")
        self.assertEqual(len(team.members), 3)  # Manager is added to members

        # Test team size validation
        with self.assertRaises(ValueError):
            TeamConfig(
                name="large_team",
                manager="manager",
                members=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
            )


if __name__ == "__main__":
    unittest.main()
