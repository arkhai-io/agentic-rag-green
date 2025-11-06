"""Tests for CommunicationQualityEvaluator component."""

import os
from unittest.mock import MagicMock, patch

import pytest

from agentic_rag.components.evaluators import CommunicationQualityEvaluator


class TestCommunicationQualityEvaluator:
    """Test suite for CommunicationQualityEvaluator."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        evaluator = CommunicationQualityEvaluator(api_key="test-key")
        assert evaluator.api_key == "test-key"
        assert evaluator.model == "anthropic/claude-3.5-sonnet"
        assert evaluator.base_url == "https://openrouter.ai/api/v1"

    def test_init_without_api_key(self):
        """Test initialization without API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenRouter API key is required"):
                CommunicationQualityEvaluator()

    def test_init_with_env_var(self):
        """Test initialization with environment variable."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-key"}):
            evaluator = CommunicationQualityEvaluator()
            assert evaluator.api_key == "env-key"

    @patch("requests.post")
    def test_successful_evaluation(self, mock_post):
        """Test successful communication quality evaluation."""
        # Mock API response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"tone_appropriateness": 5, "professionalism": 4, "bias_and_fairness": 5}'
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        evaluator = CommunicationQualityEvaluator(api_key="test-key")

        result = evaluator.run(
            query="What is Python?",
            replies=[
                "Python is a high-level programming language known for its readability."
            ],
        )

        # Verify eval_data structure
        assert "eval_data" in result
        eval_data = result["eval_data"]
        assert "eval_metrics" in eval_data

        # Check individual dimension scores
        assert "communication_quality_tone" in eval_data["eval_metrics"]
        assert "communication_quality_professionalism" in eval_data["eval_metrics"]
        assert "communication_quality_bias" in eval_data["eval_metrics"]
        assert "communication_quality_overall" in eval_data["eval_metrics"]

        # Verify scores are normalized to 0-1
        tone_score = eval_data["eval_metrics"]["communication_quality_tone"]["score"]
        assert 0 <= tone_score <= 1
        assert eval_data["eval_metrics"]["communication_quality_tone"]["raw_score"] == 5

    @patch("requests.post")
    def test_evaluation_with_existing_eval_data(self, mock_post):
        """Test that evaluation extends existing eval_data."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"tone_appropriateness": 3, "professionalism": 3, "bias_and_fairness": 3}'
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        evaluator = CommunicationQualityEvaluator(api_key="test-key")

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
        assert "communication_quality_overall" in result["eval_data"]["eval_metrics"]

    @patch("requests.post")
    def test_empty_reply(self, mock_post):
        """Test handling of empty reply."""
        evaluator = CommunicationQualityEvaluator(api_key="test-key")

        result = evaluator.run(query="Test query", replies=[""])

        # Should return eval_data without communication quality metrics
        assert "eval_data" in result
        assert "communication_quality_overall" not in result["eval_data"].get(
            "eval_metrics", {}
        )

    @patch("requests.post")
    def test_error_handling(self, mock_post):
        """Test error handling when API call fails."""
        mock_post.side_effect = Exception("API Error")

        evaluator = CommunicationQualityEvaluator(api_key="test-key")

        result = evaluator.run(
            query="What is Python?",
            replies=["Python is a programming language."],
        )

        # Should have metrics with error information and score 0
        assert "communication_quality_overall" in result["eval_data"]["eval_metrics"]
        assert (
            result["eval_data"]["eval_metrics"]["communication_quality_overall"][
                "score"
            ]
            == 0.0
        )
        assert (
            "error"
            in result["eval_data"]["eval_metrics"]["communication_quality_overall"]
        )

    @patch("requests.post")
    def test_json_parsing_with_markdown(self, mock_post):
        """Test JSON parsing when response contains markdown code blocks."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '```json\n{"tone_appropriateness": 4, "professionalism": 4, "bias_and_fairness": 4}\n```'
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        evaluator = CommunicationQualityEvaluator(api_key="test-key")

        result = evaluator.run(
            query="Test", replies=["Test answer with good communication."]
        )

        # Should successfully parse JSON from markdown
        assert "communication_quality_overall" in result["eval_data"]["eval_metrics"]
        assert (
            result["eval_data"]["eval_metrics"]["communication_quality_overall"][
                "score"
            ]
            > 0
        )

    def test_serialization(self):
        """Test component serialization."""
        evaluator = CommunicationQualityEvaluator(
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
        new_evaluator = CommunicationQualityEvaluator.from_dict(config)
        assert new_evaluator.api_key == "test-key"
        assert new_evaluator.model == "custom-model"
        assert new_evaluator.base_url == "https://custom.api"


@pytest.mark.integration
class TestCommunicationQualityEvaluatorIntegration:
    """Integration tests for CommunicationQualityEvaluator."""

    def test_real_api_call(self):
        """Test with real API call (requires OPENROUTER_API_KEY)."""
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            pytest.skip("OPENROUTER_API_KEY not set")

        evaluator = CommunicationQualityEvaluator(api_key=api_key)

        result = evaluator.run(
            query="What is machine learning?",
            replies=[
                "Machine learning is a subset of artificial intelligence that enables "
                "systems to learn and improve from experience without being explicitly "
                "programmed. It focuses on developing algorithms that can access data "
                "and use it to learn for themselves."
            ],
        )

        # Verify response structure
        assert "eval_data" in result
        assert "eval_metrics" in result["eval_data"]
        assert "communication_quality_overall" in result["eval_data"]["eval_metrics"]

        # Verify scores are in valid range
        overall = result["eval_data"]["eval_metrics"]["communication_quality_overall"]
        assert 0 <= overall["score"] <= 1
        assert 1 <= overall["raw_score"] <= 5
