# Enterprise AI Logging System

This document provides guidance on how to use the Enterprise AI logging system effectively in your development and deployment.

## Overview

The Enterprise AI logging system is designed to provide:

- Consistent logging across all components
- Context-aware logging for agents and teams
- Multiple output destinations (console, files)
- Flexible configuration
- Performance-optimized logging
- Tracing capabilities for debugging

## Basic Usage

### Importing the Logger

```python
# Import the logger module
from enterprise_ai.logger import debug, info, warning, error, critical, exception
```

### Log Levels

The logger supports the following log levels:

- `debug`: Detailed information, typically useful only for diagnosing problems
- `info`: Confirmation that things are working as expected
- `success`: Successful operations (loguru extension)
- `warning`: Indication of something unexpected that isn't a problem
- `error`: Due to a more serious problem, some functionality couldn't be performed
- `critical`: Serious error, the program may be unable to continue
- `exception`: Used in exception handlers, includes traceback automatically

### Example Usage

```python
# Basic logging
info("Starting execution")
debug("Debug information: {}", additional_info)

# Logging with formatting
warning("Unexpected value: {value}", value=result)

# Error logging
try:
    # Some operation
    raise ValueError("Invalid input")
except Exception as e:
    exception("An error occurred: {}", e)
    # or
    error("An error occurred: {}", e)
```

## Component-Specific Logging

### Getting a Component Logger

```python
from enterprise_ai.logger import get_logger

# Get a logger for a specific component
logger = get_logger("component_name")

# Use the component logger
logger.info("Component-specific message")
```

### Agent and Team Loggers

```python
from enterprise_ai.logger import get_agent_logger, get_team_logger

# Get a logger for a specific agent
agent_logger = get_agent_logger("agent-123", "developer")
agent_logger.info("Agent is processing task")

# Get a logger for a specific team
team_logger = get_team_logger("team-456")
team_logger.info("Team is coordinating")
```

## Contextual Logging

### Using the Context Decorator

```python
from enterprise_ai.logger import with_context, info

@with_context(operation="process_data", task_id="task-123")
def process_data():
    # All log messages inside this function will include the context
    info("Processing data")
```

### Nested Context

```python
@with_context(operation="outer_operation")
def outer_function():
    info("Outer function log")  # Includes outer_operation

    @with_context(sub_operation="inner_operation")
    def inner_function():
        info("Inner function log")  # Includes outer_operation and inner_operation

    inner_function()
```

## Execution Tracing

The `trace_execution` decorator automatically logs entry and exit points of functions, which is useful for debugging and performance analysis.

```python
from enterprise_ai.logger import trace_execution

@trace_execution("custom_name")  # Custom name is optional
def my_function():
    # Function implementation
    pass

# Also works with async functions
@trace_execution()
async def my_async_function():
    # Async implementation
    pass
```

## Configuration

### Default Configuration

The logging system starts with default configuration but can be customized:

```python
from enterprise_ai.logger import LoggerConfig, configure

# Create a custom configuration
config = LoggerConfig(
    console_level="INFO",    # Minimum level for console output
    file_level="DEBUG",      # Minimum level for file output
    log_dir="/path/to/logs",  # Where to store log files
    format="custom_format",  # Custom log format
    retention="10 days",     # How long to keep log files
    rotation="100 MB",       # When to rotate log files
)

# Apply the configuration
configure(config)
```

### Integration with Enterprise AI Configuration

The logging system is designed to integrate with the Enterprise AI configuration system:

```python
# Configuration is automatically loaded from enterprise_ai.config
# You don't need to explicitly configure the logger in most cases
```

## Best Practices

1. **Use Appropriate Log Levels**: Reserve `debug` for detailed information and `info` for standard operational messages.

1. **Include Context**: Always include relevant context in log messages, such as agent IDs, task IDs, etc.

1. **Structured Messages**: Use named placeholders for better readability and structure.

   ```python
   logger.info("Processing {task} with {params}", task=task_id, params=parameters)
   ```

1. **Log Exceptions**: Always log exceptions with the `exception` method to include traceback information.

1. **Use Component Loggers**: Create specific loggers for different components to make log filtering easier.

1. **Sensitive Information**: Never log sensitive information such as API keys or user credentials.

1. **Performance Consideration**: Use lazy evaluation for expensive operations:

   ```python
   # This will only be evaluated if the DEBUG level is enabled
   logger.debug("Expensive calculation: {result}", result=lambda: expensive_calculation())
   ```

## Advanced Usage

### Adding Custom Log Handlers

For advanced use cases, you may need to add custom log handlers (e.g., to send logs to a remote service):

```python
from loguru import logger as _logger

# Add a custom handler
_logger.add(
    "my_custom_sink",
    level="INFO",
    format="custom_format",
    # Additional parameters as needed
)
```

### Custom Log Filters

You can create custom filters to process log records:

```python
def my_filter(record):
    # Process or filter the record
    record["extra"]["custom_field"] = "custom_value"
    return True  # Keep the record, return False to drop it

# Add a handler with the filter
_logger.add(sys.stderr, filter=my_filter)
```
