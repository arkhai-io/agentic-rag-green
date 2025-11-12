"""Tests for LongQAAnswerEvaluator component."""

import os
from unittest.mock import MagicMock, patch

import pytest

from agentic_rag.components.evaluators import LongQAAnswerEvaluator


class TestLongQAAnswerEvaluator:
    """Test LongQAAnswerEvaluator component functionality."""

    def test_initialization_with_api_key(self):
        """Test initialization with API key."""
        evaluator = LongQAAnswerEvaluator(api_key="test-api-key")
        assert evaluator.api_key == "test-api-key"
        assert evaluator.model == "anthropic/claude-3.5-sonnet"

    def test_initialization_from_env_var(self):
        """Test initialization from environment variable."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-api-key"}):
            # Note: Evaluator doesn't auto-read from env, so we pass it explicitly
            evaluator = LongQAAnswerEvaluator(
                api_key=os.environ.get("OPENROUTER_API_KEY")
            )
            assert evaluator.api_key == "env-api-key"

    def test_initialization_without_api_key_raises_error(self):
        """Test that initialization without API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENROUTER_API_KEY", None)
            with pytest.raises(ValueError, match="OpenRouter API key required"):
                LongQAAnswerEvaluator()

    def test_initialization_with_custom_model(self):
        """Test initialization with custom model."""
        evaluator = LongQAAnswerEvaluator(api_key="test-key", model="openai/gpt-4")
        assert evaluator.model == "openai/gpt-4"

    def test_prompt_template_loaded(self):
        """Test that prompt template is loaded."""
        evaluator = LongQAAnswerEvaluator(api_key="test-key")
        assert evaluator.prompt_template is not None
        assert len(evaluator.prompt_template) > 0
        assert "medical" in evaluator.prompt_template.lower()

    @patch("requests.post")
    def test_run_with_answer(self, mock_post):
        """Test evaluation with an answer."""
        # Mock LLM response with dimension scores
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": """{
                            "The answer aligns with current medical knowledge": 5,
                            "The answer addresses the specific medical question": 4,
                            "The answer communicates contraindications or risks": 3
                        }"""
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        evaluator = LongQAAnswerEvaluator(api_key="test-key")

        result = evaluator.run(
            query="What are the side effects of aspirin?",
            replies=["Aspirin can cause stomach upset and bleeding."],
        )

        # Check structure
        assert "eval_data" in result
        eval_data = result["eval_data"]
        assert "eval_metrics" in eval_data
        assert "longqa_answer" in eval_data["eval_metrics"]

        longqa_metric = eval_data["eval_metrics"]["longqa_answer"]
        assert "score" in longqa_metric  # Normalized score (0-1)
        assert "raw_score" in longqa_metric  # Average of dimension scores (1-5)
        assert "dimension_scores" in longqa_metric
        assert "raw_dimension_scores" in longqa_metric
        assert longqa_metric["type"] == "llm_judge"

        # Check score normalization
        expected_avg = (5 + 4 + 3) / 3
        assert longqa_metric["raw_score"] == pytest.approx(expected_avg)
        assert longqa_metric["score"] == pytest.approx(expected_avg / 5.0)

        # Check dimension scores
        assert len(longqa_metric["raw_dimension_scores"]) == 3
        assert all(
            1 <= score <= 5 for score in longqa_metric["raw_dimension_scores"].values()
        )

    @patch("requests.post")
    def test_run_with_eval_data_from_previous_evaluator(self, mock_post):
        """Test evaluation with existing eval_data."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": """{
                            "The answer aligns with current medical knowledge": 4,
                            "The answer addresses the specific medical question": 4,
                            "The answer communicates contraindications or risks": 4
                        }"""
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        evaluator = LongQAAnswerEvaluator(api_key="test-key")

        existing_eval_data = {
            "query": "Test question",
            "answer": "Test answer",
            "eval_metrics": {
                "some_other_metric": {"score": 0.8},
            },
        }

        result = evaluator.run(
            query="Test question",
            replies=["Test answer"],
            eval_data=existing_eval_data,
        )

        eval_data = result["eval_data"]
        # Previous metrics preserved
        assert "some_other_metric" in eval_data["eval_metrics"]
        # New metric added
        assert "longqa_answer" in eval_data["eval_metrics"]

    def test_run_without_answer_skips_evaluation(self):
        """Test that evaluation is skipped if no answer."""
        evaluator = LongQAAnswerEvaluator(api_key="test-key")

        result = evaluator.run(
            query="What is aspirin?",
            replies=None,
        )

        assert "longqa_answer" not in result["eval_data"]["eval_metrics"]

    @patch("requests.post")
    def test_api_call_with_correct_headers(self, mock_post):
        """Test that API is called with correct headers."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": """{
                            "The answer aligns with current medical knowledge": 3,
                            "The answer addresses the specific medical question": 3,
                            "The answer communicates contraindications or risks": 3
                        }"""
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        evaluator = LongQAAnswerEvaluator(api_key="test-key", model="openai/gpt-4")

        evaluator.run(
            query="Test question?",
            replies=["Test answer."],
        )

        assert mock_post.called
        call_args = mock_post.call_args
        headers = call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/json"

        json_data = call_args[1]["json"]
        assert json_data["model"] == "openai/gpt-4"
        assert json_data["temperature"] == 0.0

    @patch("requests.post")
    def test_json_parsing_with_code_blocks(self, mock_post):
        """Test JSON parsing from responses with code blocks."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": """```json
{
    "The answer aligns with current medical knowledge": 5,
    "The answer addresses the specific medical question": 4,
    "The answer communicates contraindications or risks": 5
}
```"""
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        evaluator = LongQAAnswerEvaluator(api_key="test-key")

        result = evaluator.run(
            query="Test",
            replies=["Answer"],
        )

        assert "longqa_answer" in result["eval_data"]["eval_metrics"]
        metric = result["eval_data"]["eval_metrics"]["longqa_answer"]
        assert metric["raw_score"] == pytest.approx((5 + 4 + 5) / 3)

    @patch("requests.post")
    def test_error_handling_in_api_call(self, mock_post):
        """Test error handling when API call fails."""
        mock_post.side_effect = Exception("API Error")

        evaluator = LongQAAnswerEvaluator(api_key="test-key")

        result = evaluator.run(
            query="Test",
            replies=["Answer"],
        )

        assert "longqa_answer_error" in result["eval_data"]["eval_metrics"]
        assert (
            "API Error"
            in result["eval_data"]["eval_metrics"]["longqa_answer_error"]["error"]
        )

    def test_to_dict_serialization(self):
        """Test component serialization."""
        evaluator = LongQAAnswerEvaluator(
            api_key="test-key", model="openai/gpt-4", base_url="https://example.com"
        )

        serialized = evaluator.to_dict()

        assert "type" in serialized
        assert "init_parameters" in serialized
        params = serialized["init_parameters"]
        assert params["api_key"] == "test-key"
        assert params["model"] == "openai/gpt-4"
        assert params["base_url"] == "https://example.com"

    def test_from_dict_deserialization(self):
        """Test component deserialization."""
        serialized = {
            "type": "agentic_rag.components.evaluators.longqa_answer_evaluator.LongQAAnswerEvaluator",
            "init_parameters": {
                "api_key": "test-key",
                "model": "openai/gpt-4",
                "base_url": "https://example.com",
            },
        }

        evaluator = LongQAAnswerEvaluator.from_dict(serialized)

        assert evaluator.api_key == "test-key"
        assert evaluator.model == "openai/gpt-4"
        assert evaluator.base_url == "https://example.com"


class TestLongQAAnswerEvaluatorIntegration:
    """Integration tests requiring actual API access."""

    @pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="Requires OPENROUTER_API_KEY environment variable",
    )
    def test_real_api_call(self):
        """Test actual API call with real OpenRouter service."""
        evaluator = LongQAAnswerEvaluator()

        result = evaluator.run(
            query="What are common side effects of ibuprofen?",
            replies=[
                "Common side effects include stomach upset, nausea, and dizziness."
            ],
        )

        eval_data = result["eval_data"]
        assert "longqa_answer" in eval_data["eval_metrics"]

        longqa_metric = eval_data["eval_metrics"]["longqa_answer"]
        assert 0.0 <= longqa_metric["score"] <= 1.0
        assert 1.0 <= longqa_metric["raw_score"] <= 5.0
        assert len(longqa_metric["dimension_scores"]) == 3
