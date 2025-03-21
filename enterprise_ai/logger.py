"""
Logging system for Enterprise AI.

This module provides a centralized, configurable logging system for all components
of the Enterprise AI platform. It supports contextual logging for agents and teams,
multiple output destinations, and custom log formats.
"""

import os
import sys
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, cast

from loguru import logger as _logger

from enterprise_ai.config import PROJECT_ROOT, config

# Define a type variable for the loguru logger
LoguruLogger = TypeVar('LoguruLogger')


class LoggerConfig:
    """Configuration for the logging system."""
    
    # Constants for log levels
    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    
    def __init__(
        self,
        console_level: str = INFO,
        file_level: str = DEBUG,
        log_dir: Optional[Path] = None,
        format: str = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        retention: str = "10 days",
        rotation: str = "100 MB",
        enable_context: bool = True,
    ):
        """Initialize logger configuration.
        
        Args:
            console_level: Minimum level for console output
            file_level: Minimum level for file output
            log_dir: Directory to store log files (defaults to PROJECT_ROOT/logs)
            format: Log message format
            retention: How long to keep log files
            rotation: When to rotate log files
            enable_context: Whether to enable contextual logging
        """
        self.console_level = console_level
        self.file_level = file_level
        self.log_dir = log_dir or PROJECT_ROOT / "logs"
        self.format = format
        self.retention = retention
        self.rotation = rotation
        self.enable_context = enable_context
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)


class EnterpriseLogger:
    """Centralized logging facility for Enterprise AI."""
    
    _instance: Optional["EnterpriseLogger"] = None
    _initialized: bool = False
    _context_var: Dict[str, Any] = {}
    
    def __new__(cls, *args: Any, **kwargs: Any) -> "EnterpriseLogger":
        """Singleton pattern implementation."""
        if cls._instance is None:
            cls._instance = super(EnterpriseLogger, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, logger_config: Optional[LoggerConfig] = None):
        """Initialize the logger.
        
        Args:
            logger_config: Configuration for the logger
        """
        # Only initialize once (singleton pattern)
        if self._initialized:
            return
            
        self._logger_config = logger_config or self._load_config_from_system()
        self._configure_logger()
        self._initialized = True
    
    def _load_config_from_system(self) -> LoggerConfig:
        """Load logger configuration from the system configuration."""
        # In a real implementation, you would load from your config system
        # For now, we'll use default values
        return LoggerConfig()
    
    def _configure_logger(self) -> None:
        """Configure the loguru logger with our settings."""
        # Remove default handlers
        _logger.remove()
        
        # Add console handler
        _logger.add(
            sys.stderr,
            level=self._logger_config.console_level,
            format=self._logger_config.format,
            colorize=True,
        )
        
        # Add file handler
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self._logger_config.log_dir / f"enterprise_ai_{current_time}.log"
        
        _logger.add(
            log_file,
            level=self._logger_config.file_level,
            format=self._logger_config.format,
            rotation=self._logger_config.rotation,
            retention=self._logger_config.retention,
            compression="zip",
        )
    
    def get_logger(self, name: str) -> Any:
        """Get a logger for a specific component.
        
        Args:
            name: Name of the component
            
        Returns:
            Logger instance
        """
        return _logger.bind(name=name)
    
    def get_agent_logger(self, agent_id: str, agent_type: str) -> Any:
        """Get a logger for a specific agent.
        
        Args:
            agent_id: Unique ID of the agent
            agent_type: Type of the agent
            
        Returns:
            Logger instance with agent context
        """
        return _logger.bind(
            name="agent",
            agent_id=agent_id,
            agent_type=agent_type,
        )
    
    def get_team_logger(self, team_id: str) -> Any:
        """Get a logger for a specific team.
        
        Args:
            team_id: Unique ID of the team
            
        Returns:
            Logger instance with team context
        """
        return _logger.bind(
            name="team",
            team_id=team_id,
        )
    
    def with_context(self, **context: Any) -> Callable:
        """Decorator to add context to log messages.
        
        Args:
            **context: Context key-value pairs
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                # Save original context
                original_context = self._context_var.copy()
                
                try:
                    # Update context
                    self._context_var.update(context)
                    
                    # Set context in logger
                    contextual_logger = _logger.bind(**self._context_var)
                    
                    # Execute function with contextual logger
                    with _logger.contextualize(**self._context_var):
                        return func(*args, **kwargs)
                finally:
                    # Restore original context
                    self._context_var = original_context
            
            return wrapper
        
        return decorator
    
    def trace_execution(self, name: Optional[str] = None) -> Callable:
        """Decorator to trace function execution with entry/exit logs.
        
        Args:
            name: Optional name for the trace (defaults to function name)
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable) -> Callable:
            func_name = name or func.__name__
            
            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                _logger.debug(f"Entering {func_name}")
                try:
                    result = await func(*args, **kwargs)
                    _logger.debug(f"Exiting {func_name}")
                    return result
                except Exception as e:
                    _logger.exception(f"Error in {func_name}: {e}")
                    raise
            
            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                _logger.debug(f"Entering {func_name}")
                try:
                    result = func(*args, **kwargs)
                    _logger.debug(f"Exiting {func_name}")
                    return result
                except Exception as e:
                    _logger.exception(f"Error in {func_name}: {e}")
                    raise
            
            return async_wrapper if asyncio_iscoroutinefunction(func) else sync_wrapper
        
        return decorator
    
    def configure(self, new_config: LoggerConfig) -> None:
        """Reconfigure the logger with new settings.
        
        Args:
            new_config: New configuration for the logger
        """
        self._logger_config = new_config
        _logger.remove()  # Remove all handlers
        self._configure_logger()  # Reconfigure


# Determine if a function is a coroutine function (for trace_execution)
def asyncio_iscoroutinefunction(func: Callable) -> bool:
    """Check if a function is a coroutine function.
    
    This function handles the case where asyncio is not available.
    
    Args:
        func: Function to check
        
    Returns:
        True if the function is a coroutine function, False otherwise
    """
    try:
        import asyncio
        return asyncio.iscoroutinefunction(func)
    except ImportError:
        return False


# Create global logger instance
logger_instance = EnterpriseLogger()

# Export common log functions directly
debug = _logger.debug
info = _logger.info
success = _logger.success
warning = _logger.warning
error = _logger.error
critical = _logger.critical
exception = _logger.exception

# Export logger configuration function
configure = logger_instance.configure

# Export context creation function
with_context = logger_instance.with_context

# Export logger getters
get_logger = logger_instance.get_logger
get_agent_logger = logger_instance.get_agent_logger
get_team_logger = logger_instance.get_team_logger

# Export execution tracing
trace_execution = logger_instance.trace_execution


if __name__ == "__main__":
    # Simple test if module is run directly
    logger = get_logger("test")
    logger.info("This is a test message")
    logger.debug("This is a debug message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    agent_logger = get_agent_logger("agent-1", "developer")
    agent_logger.info("Agent is starting")
    
    team_logger = get_team_logger("team-1")
    team_logger.info("Team is collaborating")
    
    @with_context(operation="test_operation")
    def test_context():
        _logger.info("This message has context")
    
    test_context()