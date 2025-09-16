"""Test configuration and fixtures."""

import os

import pytest


@pytest.fixture(autouse=True)
def setup_test_env():
    """Set up test environment variables."""
    # Set test API keys to avoid authentication errors
    os.environ["OPENAI_API_KEY"] = "sk-test_key_for_testing_only_not_real"
    os.environ["HUGGINGFACE_API_TOKEN"] = "test_token_for_testing"

    yield

    # Clean up is not necessary since these are test-only values
