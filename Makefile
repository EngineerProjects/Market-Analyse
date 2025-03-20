.PHONY: setup install test lint format clean pre-commit help all

# Default Python interpreter
PYTHON := python3
# Default command to run pytest
PYTEST := pytest -xvs
# Default format for documentation
DOCFORMAT := html

help:
	@echo "Enterprise-AI Development Makefile"
	@echo "=================================="
	@echo "setup        - Install development dependencies and set up pre-commit"
	@echo "install      - Install package in development mode"
	@echo "test         - Run tests"
	@echo "lint         - Run linting checks"
	@echo "format       - Format code with Ruff"
	@echo "clean        - Clean up build artifacts"
	@echo "pre-commit   - Run pre-commit hooks on all files"
	@echo "all          - Run all checks (lint, test)"

setup:
	@echo "Installing UV if not already installed..."
	@if ! command -v uv > /dev/null; then \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	@echo "Adding UV to PATH if needed..."
	@if ! command -v uv > /dev/null; then \
		export PATH="$$HOME/.cargo/bin:$$PATH"; \
	fi
	@echo "Installing development dependencies..."
	@uv pip install -e ".[dev]"
	@echo "Setting up pre-commit hooks..."
	@pre-commit install

install:
	@uv pip install -e .

test:
	@$(PYTEST) tests/

lint:
	@ruff check .
	@mypy --show-error-codes enterprise_ai/

format:
	@ruff format .
	@ruff check --fix .

clean:
	@rm -rf build/
	@rm -rf dist/
	@rm -rf *.egg-info
	@rm -rf __pycache__/
	@rm -rf .pytest_cache/
	@rm -rf .ruff_cache/
	@rm -rf .mypy_cache/
	@rm -rf logs/
	@rm -rf .coverage
	@rm -rf .tox/
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type d -name "*.egg-info" -exec rm -rf {} +
	@find . -type f -name "*.pyc" -delete

pre-commit:
	@pre-commit run --all-files

all: lint test