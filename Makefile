VENV := .venv
PYTHON := $(VENV)/bin/python

.PHONY: setup test lint format pre-commit docs coverage serve-coverage clean help all

setup:
	@echo "Setting up development environment..."
	@if ! command -v uv > /dev/null; then \
		echo "Error: UV package manager not found. Please install it first."; \
		echo "curl -LsSf https://astral.sh/uv/install.sh | sh"; \
		exit 1; \
	fi
	@echo "Creating virtual environment with UV..."
	@uv venv $(VENV)
	@echo "Installing project dependencies from pyproject.toml..."
	@uv pip install --python $(PYTHON) -e ".[dev]"
	@echo "Setup completed successfully!"

install:
	@uv pip install --python $(PYTHON) -e .

test:
	@echo "Running tests..."
	@uv run --python $(PYTHON) pytest -xvs tests/

lint:
	@echo "Running linters..."
	@uv run --python $(PYTHON) ruff check .
	@uv run --python $(PYTHON) mypy --show-error-codes enterprise_ai/

format:
	@echo "Formatting code..."
	@uv run --python $(PYTHON) ruff format .
	@uv run --python $(PYTHON) ruff check --fix .
	@echo "Formatting Markdown files..."
	@which mdformat >/dev/null 2>&1 || uv pip install mdformat
	@mdformat .

pre-commit:
	@echo "Running pre-commit hooks..."
	@uv run --python $(PYTHON) pre-commit run --all-files

docs:
	@echo "Generating documentation..."
	@uv run --python $(PYTHON) pdoc -o docs --html --force enterprise_ai

coverage:
	@echo "Generating coverage report..."
	@$(VENV)/bin/pytest --cov=enterprise_ai --cov-report=html

serve-coverage:
	@echo "Serving coverage report on http://localhost:8000"
	@python3 -m http.server --directory htmlcov 8000

clean:
	@echo "Cleaning build artifacts..."
	@rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .ruff_cache/ .mypy_cache/ htmlcov/ .coverage logs/
	@find . -type d -name __pycache__ -exec rm -rf {} +

notebook:
	@echo "Checking for Jupyter Notebook..."
	@if ! $(VENV)/bin/pip show notebook > /dev/null 2>&1; then \
		echo "Installing Jupyter Notebook..."; \
		uv pip install --python $(PYTHON) notebook; \
	fi
	@echo "Launching Jupyter Notebook..."
	@$(VENV)/bin/jupyter notebook

all: lint test pre-commit coverage
help:
	@echo "Enterprise-AI Development Makefile"
	@echo "=================================="
	@echo "setup          - Create virtual env and install deps"
	@echo "install        - Install package in dev mode"
	@echo "test           - Run tests with verbose output"
	@echo "lint           - Run static analysis checks"
	@echo "format         - Format and fix code"
	@echo "docs           - Generate API documentation"
	@echo "coverage       - Generate test coverage report"
	@echo "serve-coverage - Serve coverage report on port 8000"
	@echo "clean          - Remove build artifacts"
	@echo "pre-commit     - Run all pre-commit checks"
	@echo "notebook       - Launch Jupyter Notebook"
	@echo "all            - Run full quality checks (lint + test + coverage)"
