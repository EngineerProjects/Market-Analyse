## Current Project Analysis

Your Enterprise AI platform aims to create a multi-agent AI system with:

- **Agent Hierarchy** for structured teams with manager-worker relationships
- **Role-Based Specialization** for domain-specific expertise
- **Multi-Tool Integration** with role-appropriate tooling
- **Workflow Orchestration** for complex, multi-agent task management
- **Execution Environments** for secure code execution
- **Team Communication** for agent knowledge sharing

The reference project (OpenManus) already implements many foundational components like agent types, tool systems, and execution environments, but lacks the team-based hierarchy and specialization you're planning to build.

## Development Strategy

I recommend starting with the core infrastructure first, then building agent and tool systems before implementing the team structure that differentiates your project.

### Phase 1: Core Foundation (Where to start)

Begin by implementing these essential components:

1. **Project Structure Setup**

   ```bash
   mkdir -p enterprise_ai/{agent,team/templates,flow,prompt,sandbox/core,tool/search}
   touch enterprise_ai/__init__.py
   ```

1. **Schema Definition**

   - Start with `schema.py` to define your core data models, especially agent roles, message types, and team structures.
   - This will establish type definitions that everything else will build upon.

1. **Configuration System**

   - Implement `config.py` to manage settings for LLMs, sandboxes, and other configurable elements.
   - Adapt from the reference but simplify for your initial needs.

1. **Exception Handling**

   - Create `exceptions.py` with custom exceptions for your system.
   - Include specific exceptions for team operations and agent coordination.

1. **Logging**

   - Implement `logger.py` for structured logging across the application.

### Phase 2: Base Agent & Tool Systems

1. **Base Agent**

   - Implement `agent/base.py` as the foundation for all agent types.
   - Define the agent interface with execution state management.

1. **Tool System**

   - Create `tool/base.py` with the tool interface.
   - Implement `tool/tool_collection.py` for tool management.

### Phase 3: Team Structure (Your Differentiator)

Once the core is in place, focus on your key innovation:

1. **Team Registry**

   - Implement `team/registry.py` for role and agent registration.
   - Define role templates in the `team/templates/` directory.

1. **Team Coordination**

   - Create `team/coordinator.py` for managing agent interactions.
   - Implement `team/hierarchy.py` for team structure management.
