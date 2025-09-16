.PHONY: install lint format type-check test clean help

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install development dependencies
	pip install -e ".[dev]"

install-pre-commit: ## Install pre-commit hooks
	pre-commit install

format: ## Format code with black and isort
	poetry run black agentic_rag tests
	poetry run isort agentic_rag tests

lint: ## Run linting with flake8
	poetry run flake8 agentic_rag tests

type-check: ## Run type checking with mypy
	poetry run mypy agentic_rag

test: ## Run tests with pytest
	poetry run pytest

check-all: format lint type-check test ## Run all checks (format, lint, type-check, test)

clean: ## Clean up cache files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
