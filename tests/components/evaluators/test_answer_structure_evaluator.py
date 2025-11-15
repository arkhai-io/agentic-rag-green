"""Tests for AnswerStructureEvaluator component."""

import os
from unittest.mock import MagicMock, patch

import pytest

from agentic_rag.components.evaluators import AnswerStructureEvaluator


class TestAnswerStructureEvaluator:
    """Test suite for AnswerStructureEvaluator."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        evaluator = AnswerStructureEvaluator(api_key="test-key")
        assert evaluator.api_key == "test-key"
        assert evaluator.model == "anthropic/claude-3.5-sonnet"
        assert evaluator.base_url == "https://openrouter.ai/api/v1"

    def test_init_without_api_key(self):
        """Test initialization without API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenRouter API key required"):
                AnswerStructureEvaluator()

    def test_init_with_env_var(self):
        """Test initialization with environment variable."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-key"}):
            # Note: Evaluator doesn't auto-read from env, so we pass it explicitly
            evaluator = AnswerStructureEvaluator(
                api_key=os.environ.get("OPENROUTER_API_KEY")
            )
            assert evaluator.api_key == "env-key"

    @patch("httpx.Client.post")
    def test_successful_evaluation(self, mock_post):
        """Test successful answer structure evaluation."""
        # Mock API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"organization": 5, "formatting": 4, "hierarchy": 5, "clarity": 4, "summary": "Well structured answer"}'
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        evaluator = AnswerStructureEvaluator(api_key="test-key")

        result = evaluator.run(
            query="What is Python?",
            replies=[
                "Python is a high-level programming language. It emphasizes code readability."
            ],
        )

        # Verify eval_data structure
        assert "eval_data" in result
        eval_data = result["eval_data"]
        assert "eval_metrics" in eval_data

        # Check individual dimension scores
        assert "answer_structure_organization" in eval_data["eval_metrics"]
        assert "answer_structure_formatting" in eval_data["eval_metrics"]
        assert "answer_structure_hierarchy" in eval_data["eval_metrics"]
        assert "answer_structure_clarity" in eval_data["eval_metrics"]
        assert "answer_structure_overall" in eval_data["eval_metrics"]

        # Verify scores are normalized to 0-1
        org_score = eval_data["eval_metrics"]["answer_structure_organization"]["score"]
        assert 0 <= org_score <= 1
        assert (
            eval_data["eval_metrics"]["answer_structure_organization"]["raw_score"] == 5
        )

    @patch("httpx.Client.post")
    def test_evaluation_with_existing_eval_data(self, mock_post):
        """Test that evaluation extends existing eval_data."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"organization": 3, "formatting": 3, "hierarchy": 3, "clarity": 3, "summary": "Average structure"}'
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        evaluator = AnswerStructureEvaluator(api_key="test-key")

        # Provide existing eval_data
        existing_eval_data = {
            "query": "Test query",
            "answer": "Test answer",
            "eval_metrics": {"some_other_metric": {"score": 0.5}},
        }

        result = evaluator.run(
            query="What is Python?",
            replies=["Python is a programming language."],
            eval_data=existing_eval_data,
        )

        # Check that existing metric is preserved
        assert "some_other_metric" in result["eval_data"]["eval_metrics"]
        # Check that new metrics are added
        assert "answer_structure_overall" in result["eval_data"]["eval_metrics"]

    @patch("httpx.Client.post")
    def test_empty_reply(self, mock_post):
        """Test handling of empty reply."""
        evaluator = AnswerStructureEvaluator(api_key="test-key")

        result = evaluator.run(query="Test query", replies=[""])

        # Should return eval_data without structure metrics
        assert "eval_data" in result
        assert "answer_structure_overall" not in result["eval_data"].get(
            "eval_metrics", {}
        )

    @patch("httpx.Client.post")
    def test_error_handling(self, mock_post):
        """Test error handling when API call fails."""
        mock_post.side_effect = Exception("API Error")

        evaluator = AnswerStructureEvaluator(api_key="test-key")

        result = evaluator.run(
            query="What is Python?",
            replies=["Python is a programming language."],
        )

        # Should have metrics with error information and score 0
        assert "answer_structure_overall" in result["eval_data"]["eval_metrics"]
        assert (
            result["eval_data"]["eval_metrics"]["answer_structure_overall"]["score"]
            == 0.0
        )
        assert (
            "error" in result["eval_data"]["eval_metrics"]["answer_structure_overall"]
        )

    @patch("httpx.Client.post")
    def test_json_parsing_with_markdown(self, mock_post):
        """Test JSON parsing when response contains markdown code blocks."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '```json\n{"organization": 4, "formatting": 4, "hierarchy": 4, "clarity": 4, "summary": "Good structure"}\n```'
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        evaluator = AnswerStructureEvaluator(api_key="test-key")

        result = evaluator.run(
            query="Test", replies=["Test answer with good structure."]
        )

        # Should successfully parse JSON from markdown
        assert "answer_structure_overall" in result["eval_data"]["eval_metrics"]
        assert (
            result["eval_data"]["eval_metrics"]["answer_structure_overall"]["score"] > 0
        )

    def test_serialization(self):
        """Test component serialization."""
        evaluator = AnswerStructureEvaluator(
            api_key="test-key",
            model="custom-model",
            base_url="https://custom.api",
        )

        # Serialize
        config = evaluator.to_dict()
        assert config["init_parameters"]["api_key"] == "test-key"
        assert config["init_parameters"]["model"] == "custom-model"
        assert config["init_parameters"]["base_url"] == "https://custom.api"

        # Deserialize
        new_evaluator = AnswerStructureEvaluator.from_dict(config)
        assert new_evaluator.api_key == "test-key"
        assert new_evaluator.model == "custom-model"
        assert new_evaluator.base_url == "https://custom.api"


@pytest.mark.integration
class TestAnswerStructureEvaluatorIntegration:
    """Integration tests for AnswerStructureEvaluator."""

    def test_real_api_call(self):
        """Test with real API call (requires OPENROUTER_API_KEY)."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            pytest.skip("OPENROUTER_API_KEY not set")

        evaluator = AnswerStructureEvaluator(api_key=api_key)

        result = evaluator.run(
            query="What is machine learning?",
            replies=[
                "Machine learning is a subset of artificial intelligence. "
                "It involves training algorithms on data. "
                "The algorithms learn patterns and make predictions."
            ],
        )

        # Verify response structure
        assert "eval_data" in result
        assert "eval_metrics" in result["eval_data"]
        assert "answer_structure_overall" in result["eval_data"]["eval_metrics"]

        # Verify scores are in valid range
        overall = result["eval_data"]["eval_metrics"]["answer_structure_overall"]
        assert 0 <= overall["score"] <= 1
        assert 1 <= overall["raw_score"] <= 5
