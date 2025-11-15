"""Tests for OpenRouter generator component."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from agentic_rag.components.generators.openrouter_generator import OpenRouterGenerator


@pytest.fixture
def mock_response():
    """Create a mock response object."""
    response = Mock()
    response.json.return_value = {
        "choices": [
            {
                "message": {"content": "This is a test response"},
                "finish_reason": "stop",
                "index": 0,
            }
        ],
        "model": "openai/gpt-3.5-turbo",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    response.raise_for_status = Mock()
    return response


@pytest.fixture
def generator():
    """Create a test generator instance."""
    return OpenRouterGenerator(
        api_key="test-api-key",
        model="openai/gpt-3.5-turbo",
    )


def test_init_with_api_key():
    """Test initialization with API key."""
    gen = OpenRouterGenerator(api_key="test-key", model="test-model")
    assert gen.api_key == "test-key"
    assert gen.model == "test-model"
    assert gen.client is not None
    assert gen.async_client is not None


def test_init_without_api_key():
    """Test initialization without API key raises error."""
    with pytest.raises(ValueError, match="OpenRouter API key required"):
        OpenRouterGenerator(model="test-model")


def test_to_dict(generator):
    """Test serialization to dictionary."""
    data = generator.to_dict()
    assert data["init_parameters"]["api_key"] == "test-api-key"
    assert data["init_parameters"]["model"] == "openai/gpt-3.5-turbo"
    assert "timeout" in data["init_parameters"]


def test_from_dict():
    """Test deserialization from dictionary."""
    data = {
        "type": "agentic_rag.components.generators.openrouter_generator.OpenRouterGenerator",
        "init_parameters": {
            "api_key": "test-key",
            "model": "test-model",
            "timeout": 30.0,
        },
    }
    gen = OpenRouterGenerator.from_dict(data)
    assert gen.api_key == "test-key"
    assert gen.model == "test-model"


@patch("httpx.Client.post")
def test_run_success(mock_post, generator, mock_response):
    """Test successful synchronous run."""
    mock_post.return_value = mock_response

    result = generator.run(prompt="Test prompt")

    assert "replies" in result
    assert "meta" in result
    assert len(result["replies"]) == 1
    assert result["replies"][0] == "This is a test response"
    assert result["meta"][0]["model"] == "openai/gpt-3.5-turbo"
    assert result["meta"][0]["usage"]["total_tokens"] == 15

    # Verify the API was called correctly
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args[1]
    assert "headers" in call_kwargs
    assert call_kwargs["headers"]["Authorization"] == "Bearer test-api-key"
    assert call_kwargs["json"]["model"] == "openai/gpt-3.5-turbo"
    assert call_kwargs["json"]["messages"][0]["content"] == "Test prompt"


@patch("httpx.Client.post")
def test_run_with_generation_kwargs(mock_post, generator, mock_response):
    """Test run with custom generation kwargs."""
    mock_post.return_value = mock_response

    result = generator.run(
        prompt="Test prompt",
        generation_kwargs={"temperature": 0.7, "max_tokens": 100},
    )

    assert "replies" in result

    # Verify generation kwargs were passed
    call_kwargs = mock_post.call_args[1]
    assert call_kwargs["json"]["temperature"] == 0.7
    assert call_kwargs["json"]["max_tokens"] == 100


@pytest.mark.asyncio
@patch("httpx.AsyncClient.post", new_callable=AsyncMock)
async def test_run_async_success(mock_post, generator, mock_response):
    """Test successful asynchronous run."""
    mock_post.return_value = mock_response

    result = await generator.run_async(prompt="Test prompt")

    assert "replies" in result
    assert "meta" in result
    assert len(result["replies"]) == 1
    assert result["replies"][0] == "This is a test response"
    assert result["meta"][0]["model"] == "openai/gpt-3.5-turbo"

    # Verify the API was called correctly
    mock_post.assert_called_once()
    call_kwargs = mock_post.call_args[1]
    assert "headers" in call_kwargs
    assert call_kwargs["headers"]["Authorization"] == "Bearer test-api-key"
    assert call_kwargs["json"]["model"] == "openai/gpt-3.5-turbo"
    assert call_kwargs["json"]["messages"][0]["content"] == "Test prompt"


@pytest.mark.asyncio
@patch("httpx.AsyncClient.post", new_callable=AsyncMock)
async def test_run_async_with_generation_kwargs(mock_post, generator, mock_response):
    """Test async run with custom generation kwargs."""
    mock_post.return_value = mock_response

    result = await generator.run_async(
        prompt="Test prompt",
        generation_kwargs={"temperature": 0.8, "max_tokens": 200},
    )

    assert "replies" in result

    # Verify generation kwargs were passed
    call_kwargs = mock_post.call_args[1]
    assert call_kwargs["json"]["temperature"] == 0.8
    assert call_kwargs["json"]["max_tokens"] == 200


@patch("httpx.Client.post")
def test_run_http_error(mock_post, generator):
    """Test handling of HTTP errors in sync run."""
    import httpx

    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_response.json.return_value = {"error": "Internal error"}

    error = httpx.HTTPStatusError(
        "500 Internal Server Error",
        request=Mock(),
        response=mock_response,
    )
    mock_post.return_value.raise_for_status.side_effect = error

    with pytest.raises(RuntimeError, match="OpenRouter API request failed"):
        generator.run(prompt="Test prompt")


@pytest.mark.asyncio
@patch("httpx.AsyncClient.post", new_callable=AsyncMock)
async def test_run_async_http_error(mock_post, generator):
    """Test handling of HTTP errors in async run."""
    import httpx

    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    mock_response.json.return_value = {"error": "Internal error"}
    mock_response.raise_for_status = Mock()

    error = httpx.HTTPStatusError(
        "500 Internal Server Error",
        request=Mock(),
        response=mock_response,
    )
    mock_response.raise_for_status.side_effect = error
    mock_post.return_value = mock_response

    with pytest.raises(RuntimeError, match="OpenRouter API request failed"):
        await generator.run_async(prompt="Test prompt")
