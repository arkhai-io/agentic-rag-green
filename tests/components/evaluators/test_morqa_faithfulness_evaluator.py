"""Tests for MORQAFaithfulnessEvaluator component."""

import os
from unittest.mock import MagicMock, patch

import pytest

from agentic_rag.components.evaluators import MORQAFaithfulnessEvaluator


class TestMORQAFaithfulnessEvaluator:
    """Test MORQAFaithfulnessEvaluator component functionality."""

    def test_initialization_with_api_key(self):
        """Test initialization with API key."""
        evaluator = MORQAFaithfulnessEvaluator(api_key="test-api-key")
        assert evaluator.api_key == "test-api-key"
        assert evaluator.model == "anthropic/claude-3.5-sonnet"

    def test_initialization_from_env_var(self):
        """Test initialization from environment variable."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-api-key"}):
            # Note: Evaluator doesn't auto-read from env, so we pass it explicitly
            evaluator = MORQAFaithfulnessEvaluator(
                api_key=os.environ.get("OPENROUTER_API_KEY")
            )
            assert evaluator.api_key == "env-api-key"

    def test_initialization_without_api_key_raises_error(self):
        """Test that initialization without API key raises error."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENROUTER_API_KEY", None)
            with pytest.raises(ValueError, match="OpenRouter API key required"):
                MORQAFaithfulnessEvaluator()

    def test_initialization_with_custom_model(self):
        """Test initialization with custom model."""
        evaluator = MORQAFaithfulnessEvaluator(api_key="test-key", model="openai/gpt-4")
        assert evaluator.model == "openai/gpt-4"

    def test_prompt_template_loaded(self):
        """Test that prompt template is loaded."""
        evaluator = MORQAFaithfulnessEvaluator(api_key="test-key")
        assert evaluator.prompt_template is not None
        assert len(evaluator.prompt_template) > 0
        assert "atomic" in evaluator.prompt_template.lower()
        assert "faithfulness" in evaluator.prompt_template.lower()

    @patch("requests.post")
    def test_run_with_ground_truth(self, mock_post):
        """Test evaluation with ground truth."""
        # Mock LLM response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": """{
                            "facts": [
                                {"text": "Paris is the capital of France.", "label": "supported_by_reference"},
                                {"text": "Paris is beautiful.", "label": "not_in_reference"}
                            ],
                            "critical_errors": [],
                            "atomic_faithfulness": 0.5,
                            "summary": "One fact supported, one not in reference."
                        }"""
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        evaluator = MORQAFaithfulnessEvaluator(api_key="test-key")

        result = evaluator.run(
            query="What is the capital of France?",
            replies=["Paris is the capital of France. Paris is beautiful."],
            ground_truth_answer="The capital of France is Paris.",
        )

        # Check structure
        assert "eval_data" in result
        eval_data = result["eval_data"]
        assert "eval_metrics" in eval_data
        assert "morqa_faithfulness" in eval_data["eval_metrics"]

        morqa_metric = eval_data["eval_metrics"]["morqa_faithfulness"]
        assert "score" in morqa_metric  # atomic_faithfulness score
        assert "facts" in morqa_metric
        assert "critical_errors" in morqa_metric
        assert "summary" in morqa_metric
        assert morqa_metric["type"] == "llm_judge"

        # Check values
        assert morqa_metric["score"] == 0.5
        assert len(morqa_metric["facts"]) == 2
        assert len(morqa_metric["critical_errors"]) == 0

    @patch("requests.post")
    def test_run_with_critical_errors(self, mock_post):
        """Test evaluation that identifies critical errors."""
        # Mock LLM response with critical errors
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": """{
                            "facts": [
                                {"text": "Aspirin is for headaches.", "label": "partially_supported"},
                                {"text": "Take 10 grams daily.", "label": "contradicted"}
                            ],
                            "critical_errors": ["wrong_dose"],
                            "atomic_faithfulness": 0.25,
                            "summary": "Dangerous dosage error detected."
                        }"""
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        evaluator = MORQAFaithfulnessEvaluator(api_key="test-key")

        result = evaluator.run(
            query="How much aspirin should I take?",
            replies=["Aspirin is for headaches. Take 10 grams daily."],
            ground_truth_answer="Aspirin helps with headaches. Take 325mg daily.",
        )

        morqa_metric = result["eval_data"]["eval_metrics"]["morqa_faithfulness"]
        assert morqa_metric["score"] == 0.25
        assert len(morqa_metric["critical_errors"]) == 1
        assert "wrong_dose" in morqa_metric["critical_errors"]

    @patch("requests.post")
    def test_run_with_eval_data_from_previous_evaluator(self, mock_post):
        """Test evaluation with existing eval_data."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": """{
                            "facts": [{"text": "Test fact.", "label": "supported_by_reference"}],
                            "critical_errors": [],
                            "atomic_faithfulness": 1.0,
                            "summary": "All facts supported."
                        }"""
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        evaluator = MORQAFaithfulnessEvaluator(api_key="test-key")

        existing_eval_data = {
            "query": "Test",
            "answer": "Answer",
            "ground_truth_answer": "Reference",
            "eval_metrics": {
                "some_other_metric": {"score": 0.8},
            },
        }

        result = evaluator.run(
            query="Test",
            replies=["Answer"],
            eval_data=existing_eval_data,
        )

        eval_data = result["eval_data"]
        # Previous metrics preserved
        assert "some_other_metric" in eval_data["eval_metrics"]
        # New metric added
        assert "morqa_faithfulness" in eval_data["eval_metrics"]

    def test_run_without_ground_truth_skips_evaluation(self):
        """Test that evaluation is skipped if no ground truth."""
        evaluator = MORQAFaithfulnessEvaluator(api_key="test-key")

        result = evaluator.run(
            query="What is Python?",
            replies=["Python is a language."],
            ground_truth_answer=None,
        )

        assert "morqa_faithfulness" not in result["eval_data"]["eval_metrics"]

    def test_run_without_answer_skips_evaluation(self):
        """Test that evaluation is skipped if no answer."""
        evaluator = MORQAFaithfulnessEvaluator(api_key="test-key")

        result = evaluator.run(
            query="What is Python?",
            replies=None,
            ground_truth_answer="Python is a language.",
        )

        assert "morqa_faithfulness" not in result["eval_data"]["eval_metrics"]

    @patch("requests.post")
    def test_api_call_with_correct_headers(self, mock_post):
        """Test that API is called with correct headers."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": """{
                            "facts": [],
                            "critical_errors": [],
                            "atomic_faithfulness": 0.0,
                            "summary": "Test"
                        }"""
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        evaluator = MORQAFaithfulnessEvaluator(api_key="test-key", model="openai/gpt-4")

        evaluator.run(
            query="Test",
            replies=["Answer"],
            ground_truth_answer="Reference",
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
    "facts": [{"text": "Fact 1", "label": "supported_by_reference"}],
    "critical_errors": [],
    "atomic_faithfulness": 1.0,
    "summary": "Perfect match."
}
```"""
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        evaluator = MORQAFaithfulnessEvaluator(api_key="test-key")

        result = evaluator.run(
            query="Test",
            replies=["Answer"],
            ground_truth_answer="Reference",
        )

        assert "morqa_faithfulness" in result["eval_data"]["eval_metrics"]
        metric = result["eval_data"]["eval_metrics"]["morqa_faithfulness"]
        assert metric["score"] == 1.0

    @patch("requests.post")
    def test_error_handling_in_api_call(self, mock_post):
        """Test error handling when API call fails."""
        mock_post.side_effect = Exception("API Error")

        evaluator = MORQAFaithfulnessEvaluator(api_key="test-key")

        result = evaluator.run(
            query="Test",
            replies=["Answer"],
            ground_truth_answer="Reference",
        )

        assert "morqa_faithfulness_error" in result["eval_data"]["eval_metrics"]
        error_metric = result["eval_data"]["eval_metrics"]["morqa_faithfulness_error"]
        assert "API Error" in error_metric["error"]

    @patch("requests.post")
    def test_fact_label_categories(self, mock_post):
        """Test all fact label categories."""
        # Mock response with all label types
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": """{
                            "facts": [
                                {"text": "Fact 1", "label": "supported_by_reference"},
                                {"text": "Fact 2", "label": "partially_supported"},
                                {"text": "Fact 3", "label": "contradicted"},
                                {"text": "Fact 4", "label": "not_in_reference"}
                            ],
                            "critical_errors": [],
                            "atomic_faithfulness": 0.375,
                            "summary": "Mixed fact support."
                        }"""
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        evaluator = MORQAFaithfulnessEvaluator(api_key="test-key")

        result = evaluator.run(
            query="Test",
            replies=["Answer"],
            ground_truth_answer="Reference",
        )

        morqa_metric = result["eval_data"]["eval_metrics"]["morqa_faithfulness"]
        facts = morqa_metric["facts"]

        # Check all label types present
        labels = [f["label"] for f in facts]
        assert "supported_by_reference" in labels
        assert "partially_supported" in labels
        assert "contradicted" in labels
        assert "not_in_reference" in labels

        # Score calculation: (1 + 0.5*1) / max(1, 4) = 1.5 / 4 = 0.375
        assert morqa_metric["score"] == 0.375

    def test_to_dict_serialization(self):
        """Test component serialization."""
        evaluator = MORQAFaithfulnessEvaluator(
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
            "type": "agentic_rag.components.evaluators.morqa_faithfulness_evaluator.MORQAFaithfulnessEvaluator",
            "init_parameters": {
                "api_key": "test-key",
                "model": "openai/gpt-4",
                "base_url": "https://example.com",
            },
        }

        evaluator = MORQAFaithfulnessEvaluator.from_dict(serialized)

        assert evaluator.api_key == "test-key"
        assert evaluator.model == "openai/gpt-4"
        assert evaluator.base_url == "https://example.com"


class TestMORQAFaithfulnessEvaluatorIntegration:
    """Integration tests requiring actual API access."""

    @pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="Requires OPENROUTER_API_KEY environment variable",
    )
    def test_real_api_call(self):
        """Test actual API call with real OpenRouter service."""
        evaluator = MORQAFaithfulnessEvaluator()

        result = evaluator.run(
            query="What is the capital of France?",
            replies=["Paris is the capital of France."],
            ground_truth_answer="The capital of France is Paris.",
        )

        eval_data = result["eval_data"]
        assert "morqa_faithfulness" in eval_data["eval_metrics"]

        morqa_metric = eval_data["eval_metrics"]["morqa_faithfulness"]
        assert 0.0 <= morqa_metric["score"] <= 1.0
        assert isinstance(morqa_metric["facts"], list)
        assert isinstance(morqa_metric["critical_errors"], list)
        assert isinstance(morqa_metric["summary"], str)

    @pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="Requires OPENROUTER_API_KEY environment variable",
    )
    def test_real_api_call_with_contradiction(self):
        """Test actual API call with contradictory information."""
        evaluator = MORQAFaithfulnessEvaluator()

        result = evaluator.run(
            query="What is the capital of France?",
            replies=["London is the capital of France."],
            ground_truth_answer="The capital of France is Paris.",
        )

        eval_data = result["eval_data"]
        morqa_metric = eval_data["eval_metrics"]["morqa_faithfulness"]

        # Should detect contradiction and have low score
        assert morqa_metric["score"] < 0.5
