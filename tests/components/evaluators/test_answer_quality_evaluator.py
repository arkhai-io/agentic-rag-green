"""Tests for AnswerQualityEvaluator component."""

import os
from unittest.mock import MagicMock, patch

import pytest

from agentic_rag.components.evaluators import AnswerQualityEvaluator


class TestAnswerQualityEvaluator:
    """Test AnswerQualityEvaluator component functionality."""

    def test_initialization_with_api_key(self):
        """Test that AnswerQualityEvaluator can be initialized with an API key."""
        evaluator = AnswerQualityEvaluator(api_key="test-api-key")
        assert evaluator.api_key == "test-api-key"
        assert evaluator.model == "anthropic/claude-3.5-sonnet"
        assert evaluator.base_url == "https://openrouter.ai/api/v1"

    def test_initialization_from_env_var(self):
        """Test that AnswerQualityEvaluator reads API key from environment."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-api-key"}):
            evaluator = AnswerQualityEvaluator()
            assert evaluator.api_key == "env-api-key"

    def test_initialization_without_api_key_raises_error(self):
        """Test that initialization without API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            # Clear any existing OPENROUTER_API_KEY
            os.environ.pop("OPENROUTER_API_KEY", None)
            with pytest.raises(ValueError, match="OpenRouter API key is required"):
                AnswerQualityEvaluator()

    def test_initialization_with_custom_model(self):
        """Test that custom model can be specified."""
        evaluator = AnswerQualityEvaluator(
            api_key="test-key", model="openai/gpt-4-turbo"
        )
        assert evaluator.model == "openai/gpt-4-turbo"

    def test_prompt_template_loaded(self):
        """Test that prompt template is loaded on initialization."""
        evaluator = AnswerQualityEvaluator(api_key="test-key")
        assert evaluator.prompt_template is not None
        assert len(evaluator.prompt_template) > 0
        assert "COMPLETENESS" in evaluator.prompt_template
        assert "CORRECTNESS" in evaluator.prompt_template

    @patch("requests.post")
    def test_run_with_ground_truth(self, mock_post):
        """Test evaluation with ground truth answer."""
        # Mock API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": """{
                            "completeness_score": 4,
                            "completeness_reasoning": "Covers most key points but missing some details",
                            "key_missing_info": ["Detail 1", "Detail 2"],
                            "correctness_score": 5,
                            "correctness_reasoning": "All information is factually accurate",
                            "factual_errors": []
                        }"""
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        evaluator = AnswerQualityEvaluator(api_key="test-key")
        result = evaluator.run(
            query="What is Python?",
            replies=["Python is a programming language."],
            ground_truth_answer="Python is a high-level programming language...",
        )

        # Check that eval_data is returned
        assert "eval_data" in result
        eval_data = result["eval_data"]

        # Check structure
        assert "query" in eval_data
        assert "answer" in eval_data
        assert "ground_truth_answer" in eval_data
        assert "eval_metrics" in eval_data

        # Check metrics were added
        metrics = eval_data["eval_metrics"]
        assert "answer_quality_completeness" in metrics
        assert "answer_quality_correctness" in metrics
        assert "answer_quality_overall" in metrics

        # Check normalized scores (1-5 scale normalized to 0-1)
        assert metrics["answer_quality_completeness"]["score"] == 4 / 5.0
        assert metrics["answer_quality_correctness"]["score"] == 5 / 5.0
        assert metrics["answer_quality_overall"]["score"] == 4.5 / 5.0

        # Check raw scores preserved
        assert metrics["answer_quality_completeness"]["raw_score"] == 4
        assert metrics["answer_quality_correctness"]["raw_score"] == 5

        # Check reasoning is included
        assert "reasoning" in metrics["answer_quality_completeness"]
        assert "reasoning" in metrics["answer_quality_correctness"]

    @patch("requests.post")
    def test_run_with_eval_data_from_previous_evaluator(self, mock_post):
        """Test that evaluator can receive eval_data from previous evaluator."""
        # Mock API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": """{
                            "completeness_score": 3,
                            "completeness_reasoning": "Partially complete",
                            "key_missing_info": ["Info 1"],
                            "correctness_score": 4,
                            "correctness_reasoning": "Mostly correct",
                            "factual_errors": ["Minor error"]
                        }"""
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        evaluator = AnswerQualityEvaluator(api_key="test-key")

        # Simulate eval_data from a previous evaluator
        existing_eval_data = {
            "query": "What is Python?",
            "answer": "Python is a language.",
            "ground_truth_answer": "Python is a high-level language.",
            "eval_metrics": {
                "some_other_metric": {"score": 0.8, "type": "other"},
            },
        }

        result = evaluator.run(
            query="What is Python?",
            replies=["Python is a language."],
            eval_data=existing_eval_data,
        )

        eval_data = result["eval_data"]

        # Check that previous metrics are preserved
        assert "some_other_metric" in eval_data["eval_metrics"]

        # Check that new metrics are added
        assert "answer_quality_completeness" in eval_data["eval_metrics"]
        assert "answer_quality_correctness" in eval_data["eval_metrics"]

    def test_run_without_ground_truth_skips_evaluation(self):
        """Test that evaluation is skipped if no ground truth is provided."""
        evaluator = AnswerQualityEvaluator(api_key="test-key")

        result = evaluator.run(
            query="What is Python?",
            replies=["Python is a programming language."],
            ground_truth_answer=None,
        )

        eval_data = result["eval_data"]

        # Should not have answer quality metrics
        assert "answer_quality_completeness" not in eval_data["eval_metrics"]
        assert "answer_quality_correctness" not in eval_data["eval_metrics"]

    def test_run_without_answer_skips_evaluation(self):
        """Test that evaluation is skipped if no answer is provided."""
        evaluator = AnswerQualityEvaluator(api_key="test-key")

        result = evaluator.run(
            query="What is Python?",
            replies=None,
            ground_truth_answer="Python is a high-level programming language.",
        )

        eval_data = result["eval_data"]

        # Should not have answer quality metrics
        assert "answer_quality_completeness" not in eval_data["eval_metrics"]
        assert "answer_quality_correctness" not in eval_data["eval_metrics"]

    @patch("requests.post")
    def test_api_call_with_correct_headers(self, mock_post):
        """Test that API is called with correct headers and parameters."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": """{
                            "completeness_score": 3,
                            "completeness_reasoning": "Test",
                            "key_missing_info": [],
                            "correctness_score": 3,
                            "correctness_reasoning": "Test",
                            "factual_errors": []
                        }"""
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        evaluator = AnswerQualityEvaluator(
            api_key="test-key", model="openai/gpt-4-turbo"
        )

        evaluator.run(
            query="What is Python?",
            replies=["Python is a language."],
            ground_truth_answer="Python is a high-level language.",
        )

        # Check API was called
        assert mock_post.called

        # Check call arguments
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://openrouter.ai/api/v1/chat/completions"

        # Check headers
        headers = call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Content-Type"] == "application/json"
        assert "HTTP-Referer" in headers
        assert "X-Title" in headers

        # Check request body
        json_data = call_args[1]["json"]
        assert json_data["model"] == "openai/gpt-4-turbo"
        assert json_data["temperature"] == 0.0
        assert "messages" in json_data

    @patch("requests.post")
    def test_json_parsing_with_code_blocks(self, mock_post):
        """Test that JSON is correctly parsed from responses with code blocks."""
        # Mock response with JSON in code blocks
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": """```json
{
    "completeness_score": 4,
    "completeness_reasoning": "Good coverage",
    "key_missing_info": [],
    "correctness_score": 5,
    "correctness_reasoning": "Accurate",
    "factual_errors": []
}
```"""
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        evaluator = AnswerQualityEvaluator(api_key="test-key")

        result = evaluator.run(
            query="What is Python?",
            replies=["Python is a language."],
            ground_truth_answer="Python is a high-level language.",
        )

        eval_data = result["eval_data"]

        # Should successfully parse and add metrics
        assert "answer_quality_completeness" in eval_data["eval_metrics"]
        assert (
            eval_data["eval_metrics"]["answer_quality_completeness"]["score"] == 4 / 5.0
        )

    @patch("requests.post")
    def test_error_handling_in_api_call(self, mock_post):
        """Test that API errors are handled gracefully."""
        # Mock API error
        mock_post.side_effect = Exception("API Error")

        evaluator = AnswerQualityEvaluator(api_key="test-key")

        result = evaluator.run(
            query="What is Python?",
            replies=["Python is a language."],
            ground_truth_answer="Python is a high-level language.",
        )

        eval_data = result["eval_data"]

        # Should have metrics with error information and score 0
        assert "answer_quality_completeness" in eval_data["eval_metrics"]
        assert "answer_quality_correctness" in eval_data["eval_metrics"]
        assert "answer_quality_overall" in eval_data["eval_metrics"]

        # Scores should be 0 when there's an error
        assert eval_data["eval_metrics"]["answer_quality_completeness"]["score"] == 0.0
        assert eval_data["eval_metrics"]["answer_quality_correctness"]["score"] == 0.0
        assert eval_data["eval_metrics"]["answer_quality_overall"]["score"] == 0.0

        # Error message should be in reasoning
        assert (
            "API Error"
            in eval_data["eval_metrics"]["answer_quality_completeness"]["reasoning"]
        )

    @patch("requests.post")
    def test_score_validation(self, mock_post):
        """Test that scores are validated to be in 1-5 range."""
        # Mock response with out-of-range scores
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": """{
                            "completeness_score": 10,
                            "completeness_reasoning": "Test",
                            "key_missing_info": [],
                            "correctness_score": 0,
                            "correctness_reasoning": "Test",
                            "factual_errors": []
                        }"""
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        evaluator = AnswerQualityEvaluator(api_key="test-key")

        result = evaluator.run(
            query="What is Python?",
            replies=["Python is a language."],
            ground_truth_answer="Python is a high-level language.",
        )

        eval_data = result["eval_data"]
        metrics = eval_data["eval_metrics"]

        # Scores should be clamped to valid range
        assert (
            metrics["answer_quality_completeness"]["raw_score"] == 5
        )  # Clamped from 10
        assert metrics["answer_quality_correctness"]["raw_score"] == 1  # Clamped from 0

    def test_to_dict_serialization(self):
        """Test that component can be serialized to dict."""
        evaluator = AnswerQualityEvaluator(
            api_key="test-key", model="openai/gpt-4", base_url="https://example.com"
        )

        serialized = evaluator.to_dict()

        assert "type" in serialized
        assert "init_parameters" in serialized
        assert serialized["init_parameters"]["api_key"] == "test-key"
        assert serialized["init_parameters"]["model"] == "openai/gpt-4"
        assert serialized["init_parameters"]["base_url"] == "https://example.com"

    def test_from_dict_deserialization(self):
        """Test that component can be deserialized from dict."""
        serialized = {
            "type": "agentic_rag.components.evaluators.answer_quality_evaluator.AnswerQualityEvaluator",
            "init_parameters": {
                "api_key": "test-key",
                "model": "openai/gpt-4",
                "base_url": "https://example.com",
            },
        }

        evaluator = AnswerQualityEvaluator.from_dict(serialized)

        assert evaluator.api_key == "test-key"
        assert evaluator.model == "openai/gpt-4"
        assert evaluator.base_url == "https://example.com"

    @patch("requests.post")
    def test_metadata_preservation(self, mock_post):
        """Test that metadata like key_missing_info and factual_errors are preserved."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": """{
                            "completeness_score": 3,
                            "completeness_reasoning": "Missing some points",
                            "key_missing_info": ["Detail A", "Detail B", "Detail C"],
                            "correctness_score": 4,
                            "correctness_reasoning": "One minor error",
                            "factual_errors": ["Error about dates"]
                        }"""
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        evaluator = AnswerQualityEvaluator(api_key="test-key")

        result = evaluator.run(
            query="What is Python?",
            replies=["Python is a language."],
            ground_truth_answer="Python is a high-level language.",
        )

        eval_data = result["eval_data"]
        metrics = eval_data["eval_metrics"]

        # Check that detailed metadata is preserved
        assert metrics["answer_quality_completeness"]["key_missing_info"] == [
            "Detail A",
            "Detail B",
            "Detail C",
        ]
        assert metrics["answer_quality_correctness"]["factual_errors"] == [
            "Error about dates"
        ]


class TestAnswerQualityEvaluatorIntegration:
    """Integration tests that require actual API access."""

    @pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="Requires OPENROUTER_API_KEY environment variable",
    )
    def test_real_api_call(self):
        """Test actual API call with real OpenRouter service."""
        evaluator = AnswerQualityEvaluator()

        result = evaluator.run(
            query="What is the capital of France?",
            replies=["Paris is the capital of France."],
            ground_truth_answer="The capital of France is Paris.",
        )

        eval_data = result["eval_data"]
        metrics = eval_data["eval_metrics"]

        # Should have all metrics
        assert "answer_quality_completeness" in metrics
        assert "answer_quality_correctness" in metrics
        assert "answer_quality_overall" in metrics

        # Scores should be high for this correct answer
        assert metrics["answer_quality_completeness"]["score"] >= 0.8
        assert metrics["answer_quality_correctness"]["score"] >= 0.8

    @pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="Requires OPENROUTER_API_KEY environment variable",
    )
    def test_real_api_call_with_incomplete_answer(self):
        """Test actual API call with an incomplete answer."""
        evaluator = AnswerQualityEvaluator()

        result = evaluator.run(
            query="What is Python?",
            replies=["Python is a language."],
            ground_truth_answer="Python is a high-level, interpreted programming language known for its simplicity and readability.",
        )

        eval_data = result["eval_data"]
        metrics = eval_data["eval_metrics"]

        # Should detect incompleteness
        assert "answer_quality_completeness" in metrics
        assert metrics["answer_quality_completeness"]["score"] < 0.9

        # But should still be correct
        assert metrics["answer_quality_correctness"]["score"] >= 0.6
