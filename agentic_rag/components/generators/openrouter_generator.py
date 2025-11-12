"""OpenRouter LLM generator component for Haystack."""

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import requests  # type: ignore[import-untyped]
from haystack import component, default_from_dict, default_to_dict

if TYPE_CHECKING:
    from ...config import Config


@component
class OpenRouterGenerator:
    """
    Generator component that uses OpenRouter API for LLM completions.

    OpenRouter provides access to multiple LLM providers through a single API.

    Usage:
        ```python
        # With explicit API key
        generator = OpenRouterGenerator(
            api_key="your-api-key",
            model="anthropic/claude-3.5-sonnet",
        )

        # With Config object
        from agentic_rag import Config
        config = Config(openrouter_api_key="your-api-key")
        generator = OpenRouterGenerator(config=config, model="anthropic/claude-3.5-sonnet")

        response = generator.run(prompt="What is RAG?")
        print(response["replies"][0])
        ```

    Supported models include:
    - anthropic/claude-3.5-sonnet
    - openai/gpt-4-turbo
    - google/gemini-pro
    - meta-llama/llama-3.1-70b-instruct
    - mistralai/mistral-large
    - And many more! See https://openrouter.ai/models
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-3.5-turbo",
        api_base: str = "https://openrouter.ai/api/v1",
        generation_kwargs: Optional[Dict[str, Any]] = None,
        streaming_callback: Optional[Any] = None,
        config: Optional["Config"] = None,
    ):
        """
        Initialize OpenRouter generator.

        Args:
            api_key: OpenRouter API key (overrides config)
            model: Model to use (e.g., "anthropic/claude-3.5-sonnet").
            api_base: Base URL for OpenRouter API.
            generation_kwargs: Additional parameters for generation (temperature, max_tokens, etc.).
            streaming_callback: Callback for streaming responses (not yet implemented).
            config: Config object with API key (required if api_key not provided)
        """
        # Priority: explicit api_key > config object
        if config is not None:
            self.api_key = api_key or config.openrouter_api_key
        else:
            self.api_key = api_key

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Provide via config parameter:\n"
                "  config = Config(openrouter_api_key='your-key')\n"
                "  OpenRouterGenerator(config=config, model='...')"
            )

        self.model = model
        self.api_base = api_base
        self.generation_kwargs = generation_kwargs or {}
        self.streaming_callback = streaming_callback

    def to_dict(self) -> Dict[str, Any]:
        """Serialize component to dictionary."""
        serialized: Dict[str, Any] = default_to_dict(
            self,
            api_key=self.api_key,
            model=self.model,
            api_base=self.api_base,
            generation_kwargs=self.generation_kwargs,
            streaming_callback=self.streaming_callback,
        )
        return serialized

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpenRouterGenerator":
        """Deserialize component from dictionary."""
        instance: "OpenRouterGenerator" = default_from_dict(cls, data)
        return instance

    @component.output_types(replies=List[str], meta=List[Dict[str, Any]])  # type: ignore[misc]
    def run(
        self,
        prompt: str,
        generation_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate text using OpenRouter API.

        Args:
            prompt: Input prompt for generation.
            generation_kwargs: Optional generation parameters that override defaults.

        Returns:
            Dictionary with:
                - replies: List of generated text responses
                - meta: List of metadata dictionaries with usage info
        """
        # Merge generation kwargs
        params = {**self.generation_kwargs, **(generation_kwargs or {})}

        # Prepare request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/arkhai-io/agentic-rag",
            "X-Title": "Agentic RAG",
        }

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            **params,
        }

        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()

            data = response.json()

            # Extract response
            replies = [choice["message"]["content"] for choice in data["choices"]]

            # Extract metadata
            meta = [
                {
                    "model": data.get("model", self.model),
                    "usage": data.get("usage", {}),
                    "finish_reason": choice.get("finish_reason"),
                    "index": choice.get("index"),
                }
                for choice in data["choices"]
            ]

            return {
                "replies": replies,
                "meta": meta,
            }

        except requests.exceptions.RequestException as e:
            error_msg = f"OpenRouter API request failed: {str(e)}"
            if hasattr(e, "response") and e.response is not None:
                try:
                    error_details = e.response.json()
                    error_msg = f"{error_msg}\nDetails: {error_details}"
                except Exception:
                    error_msg = f"{error_msg}\nResponse: {e.response.text}"

            raise RuntimeError(error_msg) from e
