"""
Tests for the Enterprise AI logging system.

This module contains tests for the logging system, including configuration,
log levels, contextual logging, and output destinations.
"""

import asyncio
import os
import re
import shutil
import sys
import tempfile
from io import StringIO
from pathlib import Path
from unittest import mock

import pytest

from enterprise_ai.logger import (
    EnterpriseLogger,
    LoggerConfig,
    configure,
    debug,
    error,
    get_agent_logger,
    get_logger,
    get_team_logger,
    info,
    trace_execution,
    warning,
    with_context,
    _logger,  # Import the internal logger for testing
)


class TestLoggerConfig:
    """Tests for the LoggerConfig class."""
    
    def test_default_config(self):
        """Test that default configuration works."""
        config = LoggerConfig()
        assert config.console_level == LoggerConfig.INFO
        assert config.file_level == LoggerConfig.DEBUG
        assert config.log_dir.exists()
    
    def test_custom_config(self):
        """Test that custom configuration works."""
        temp_dir = Path(tempfile.mkdtemp())
        try:
            config = LoggerConfig(
                console_level=LoggerConfig.WARNING,
                file_level=LoggerConfig.ERROR,
                log_dir=temp_dir,
                format="custom_format",
                retention="5 days",
                rotation="50 MB",
                enable_context=False,
            )
            
            assert config.console_level == LoggerConfig.WARNING
            assert config.file_level == LoggerConfig.ERROR
            assert config.log_dir == temp_dir
            assert config.format == "custom_format"
            assert config.retention == "5 days"
            assert config.rotation == "50 MB"
            assert config.enable_context is False
        finally:
            # Clean up
            shutil.rmtree(temp_dir)


class TestEnterpriseLogger:
    """Tests for the EnterpriseLogger class."""
    
    def setup_method(self):
        """Set up test environment."""
        # Create a temporary directory for log files
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Reset the EnterpriseLogger singleton
        EnterpriseLogger._instance = None
        EnterpriseLogger._initialized = False
    
    def teardown_method(self):
        """Clean up test environment."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_singleton_pattern(self):
        """Test that EnterpriseLogger follows the singleton pattern."""
        logger1 = EnterpriseLogger()
        logger2 = EnterpriseLogger()
        
        assert logger1 is logger2
    
    def test_configure_logger(self):
        """Test that logger configuration works."""
        # Create a logger with custom configuration
        config = LoggerConfig(
            console_level=LoggerConfig.WARNING,
            file_level=LoggerConfig.ERROR,
            log_dir=self.temp_dir,
        )
        
        logger = EnterpriseLogger(config)
        
        # Check that configuration was applied
        assert logger._logger_config.console_level == LoggerConfig.WARNING
        assert logger._logger_config.file_level == LoggerConfig.ERROR
        assert logger._logger_config.log_dir == self.temp_dir
    
    def test_get_logger(self):
        """Test that get_logger returns a logger with the correct name."""
        logger = EnterpriseLogger(
            LoggerConfig(log_dir=self.temp_dir)
        )
        
        # Mock the bind method to check the parameters
        with mock.patch("enterprise_ai.logger._logger.bind") as mock_bind:
            mock_bind.return_value = mock.MagicMock()
            component_logger = logger.get_logger("test_component")
            
            # Check that bind was called with the correct name
            mock_bind.assert_called_once_with(name="test_component")
    
    def test_get_agent_logger(self):
        """Test that get_agent_logger returns a logger with agent context."""
        logger = EnterpriseLogger(
            LoggerConfig(log_dir=self.temp_dir)
        )
        
        # Mock the bind method to check the parameters
        with mock.patch("enterprise_ai.logger._logger.bind") as mock_bind:
            mock_bind.return_value = mock.MagicMock()
            agent_logger = logger.get_agent_logger("agent-1", "developer")
            
            # Check that bind was called with the correct agent context
            mock_bind.assert_called_once_with(
                name="agent",
                agent_id="agent-1",
                agent_type="developer",
            )
    
    def test_get_team_logger(self):
        """Test that get_team_logger returns a logger with team context."""
        logger = EnterpriseLogger(
            LoggerConfig(log_dir=self.temp_dir)
        )
        
        # Mock the bind method to check the parameters
        with mock.patch("enterprise_ai.logger._logger.bind") as mock_bind:
            mock_bind.return_value = mock.MagicMock()
            team_logger = logger.get_team_logger("team-1")
            
            # Check that bind was called with the correct team context
            mock_bind.assert_called_once_with(
                name="team",
                team_id="team-1",
            )


class TestLoggingFunctions:
    """Tests for the logging functions."""
    
    def setup_method(self):
        """Set up test environment."""
        # Create a temporary directory for log files
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Configure logger
        self.config = LoggerConfig(log_dir=self.temp_dir)
        configure(self.config)
        
        # Create a temporary file to capture log output
        self.log_capture = tempfile.NamedTemporaryFile(delete=False)
        self.log_path = Path(self.log_capture.name)
    
    def teardown_method(self):
        """Clean up test environment."""
        # Close and remove log capture file
        self.log_capture.close()
        os.unlink(self.log_path)
        
        # Remove temporary directory
        shutil.rmtree(self.temp_dir)
    
    def test_log_levels(self):
        """Test that log levels work as expected."""
        # Create a StringIO to capture log output
        string_io = StringIO()
        handler_id = _logger.add(string_io, level=LoggerConfig.DEBUG, format="{level} - {message}")
        
        # Create loggers
        logger1 = get_logger("test1")
        
        # Log messages at different levels
        logger1.debug("Debug message")
        logger1.info("Info message")
        logger1.warning("Warning message")
        logger1.error("Error message")
        
        # Check the log output
        log_output = string_io.getvalue()
        assert "DEBUG - Debug message" in log_output
        assert "INFO - Info message" in log_output
        assert "WARNING - Warning message" in log_output
        assert "ERROR - Error message" in log_output
        
        # Clean up
        _logger.remove(handler_id)
    
    def test_with_context(self):
        """Test that with_context adds context to log messages."""
        # Create a mock for the logger
        with mock.patch("enterprise_ai.logger._logger.bind") as mock_bind:
            # Define a function with context
            @with_context(operation="test_operation", user_id="user-1")
            def test_function():
                info("Test message with context")
            
            # Call the function
            test_function()
            
            # Check that context was added
            mock_bind.assert_called_with(
                operation="test_operation",
                user_id="user-1",
            )
    
    @pytest.mark.asyncio
    async def test_trace_execution_async(self):
        """Test that trace_execution works with async functions.
        
        Note: This test requires the pytest-asyncio plugin. Install it with:
        pip install pytest-asyncio
        """
        # Create a mock for the logger
        with mock.patch("enterprise_ai.logger._logger.debug") as mock_debug:
            # Define an async function with trace
            @trace_execution("async_function")
            async def async_test():
                await asyncio.sleep(0.01)
                return "result"
            
            # Call the function
            result = await async_test()
            
            # Check that trace logs were generated
            mock_debug.assert_any_call("Entering async_function")
            mock_debug.assert_any_call("Exiting async_function")
            
            # Check that function result is correct
            assert result == "result"
    
    def test_trace_execution_sync(self):
        """Test that trace_execution works with sync functions."""
        # Create a mock for the logger
        with mock.patch("enterprise_ai.logger._logger.debug") as mock_debug:
            # Define a sync function with trace
            @trace_execution()  # Use default name
            def sync_test():
                return "result"
            
            # Call the function
            result = sync_test()
            
            # Check that trace logs were generated
            mock_debug.assert_any_call("Entering sync_test")
            mock_debug.assert_any_call("Exiting sync_test")
            
            # Check that function result is correct
            assert result == "result"
    
    def test_trace_execution_exception(self):
        """Test that trace_execution handles exceptions correctly."""
        # Create a mock for the logger
        with mock.patch("enterprise_ai.logger._logger.exception") as mock_exception:
            # Define a function that raises an exception
            @trace_execution("exception_function")
            def exception_test():
                raise ValueError("Test exception")
            
            # Call the function and check that it re-raises
            with pytest.raises(ValueError, match="Test exception"):
                exception_test()
            
            # Check that the exception was logged
            mock_exception.assert_called_with(
                "Error in exception_function: Test exception"
            )


if __name__ == "__main__":
    pytest.main(["-v", __file__])