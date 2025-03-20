# Individual Test Commands

Here are the commands to run tests for each file individually. This approach will help you isolate and fix issues one at a time:

## Foundation Components Tests

```bash
# Test config implementation
python -m pytest tests/test_config.py -v

# Test schema implementation
python -m pytest tests/test_schema.py -v

# Test exceptions (once you add tests)
python -m pytest tests/test_exceptions.py -v

# Test logger (once you add tests)
python -m pytest tests/test_logger.py -v
```

## Tool Framework Tests

```bash
# Test base tool classes
python -m pytest tests/tool/test_base.py -v

# Test tool collection
python -m pytest tests/tool/test_tool_collection.py -v

# Test file operators
python -m pytest tests/tool/test_file_operators.py -v

# Test authorization
python -m pytest tests/tool/test_authorization.py -v

# Test terminal (once you add tests)
python -m pytest tests/tool/test_terminal.py -v

# Test Python execution (once you add tests)
python -m pytest tests/tool/test_python_execute.py -v
```

## Run Tests with Coverage

To see test coverage for specific modules:

```bash
# Coverage for config
python -m pytest --cov=enterprise_ai.config tests/test_config.py

# Coverage for schema
python -m pytest --cov=enterprise_ai.schema tests/test_schema.py

# Coverage for tool base
python -m pytest --cov=enterprise_ai.tool.base tests/tool/test_base.py

# Coverage for tool collection
python -m pytest --cov=enterprise_ai.tool.tool_collection tests/tool/test_tool_collection.py

# Coverage for file operators
python -m pytest --cov=enterprise_ai.tool.file_operators tests/tool/test_file_operators.py

# Coverage for authorization
python -m pytest --cov=enterprise_ai.tool.authorization tests/tool/test_authorization.py
```

## Run Tests for Specific Test Methods

If you want to run a specific test method:

```bash
# Run specific test method
python -m pytest tests/test_schema.py::TestMessageSchema::test_from_tool_calls -v

# Run specific test method in config
python -m pytest tests/test_config.py::TestConfig::test_env_variable_handling -v
```

## Useful Flags

- `-v`: Verbose output
- `--tb=native`: Native Python traceback formatting
- `-s`: Show print statements during test execution
- `--pdb`: Drop into debugger on failure

For example, to debug a specific failing test:

```bash
python -m pytest tests/test_schema.py::TestMessageSchema::test_from_tool_calls -v --pdb
```

These commands should help you systematically address the test failures one by one.
