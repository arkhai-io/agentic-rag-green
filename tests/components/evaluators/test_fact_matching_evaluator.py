"""Tests for FactMatchingEvaluator component."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agentic_rag.components.evaluators import FactMatchingEvaluator


class TestFactMatchingEvaluator:
    """Test FactMatchingEvaluator component functionality."""

    def test_initialization_with_api_key(self):
        """Test that FactMatchingEvaluator can be initialized with an API key."""
        evaluator = FactMatchingEvaluator(api_key="test-api-key")
        assert evaluator.api_key == "test-api-key"
        assert evaluator.llm_model == "anthropic/claude-3.5-sonnet"
        assert evaluator.similarity_threshold == 0.75
        assert evaluator.matching_strategy == "greedy"

    def test_initialization_from_env_var(self):
        """Test that FactMatchingEvaluator reads API key from environment."""
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "env-api-key"}):
            evaluator = FactMatchingEvaluator()
            assert evaluator.api_key == "env-api-key"

    def test_initialization_without_api_key_raises_error(self):
        """Test that initialization without API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENROUTER_API_KEY", None)
            with pytest.raises(ValueError, match="OpenRouter API key is required"):
                FactMatchingEvaluator()

    def test_initialization_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        evaluator = FactMatchingEvaluator(
            api_key="test-key",
            llm_model="openai/gpt-4",
            embedding_model="all-mpnet-base-v2",
            similarity_threshold=0.8,
            matching_strategy="optimal",
        )
        assert evaluator.llm_model == "openai/gpt-4"
        assert evaluator.embedding_model_name == "all-mpnet-base-v2"
        assert evaluator.similarity_threshold == 0.8
        assert evaluator.matching_strategy == "optimal"

    def test_prompt_template_loaded(self):
        """Test that prompt template is loaded on initialization."""
        evaluator = FactMatchingEvaluator(api_key="test-key")
        assert evaluator.prompt_template is not None
        assert len(evaluator.prompt_template) > 0
        assert "atomic facts" in evaluator.prompt_template

    @patch("requests.post")
    @patch("sentence_transformers.SentenceTransformer")
    def test_run_with_ground_truth(self, mock_transformer, mock_post):
        """Test fact matching evaluation with ground truth."""
        # Mock LLM responses for fact extraction
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": '{"facts": ["Python is a programming language.", "Python is high-level."]}'
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        # Mock sentence embeddings
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]])
        mock_transformer.return_value = mock_encoder

        evaluator = FactMatchingEvaluator(api_key="test-key")
        evaluator.encoder = mock_encoder

        result = evaluator.run(
            query="What is Python?",
            replies=["Python is a high-level programming language."],
            ground_truth_answer="Python is a high-level programming language.",
        )

        # Check structure
        assert "eval_data" in result
        eval_data = result["eval_data"]
        assert "eval_metrics" in eval_data
        assert "fact_matching" in eval_data["eval_metrics"]

        fact_matching = eval_data["eval_metrics"]["fact_matching"]
        assert "score" in fact_matching  # F1 score
        assert "precision" in fact_matching
        assert "recall" in fact_matching
        assert "model_facts" in fact_matching
        assert "gold_facts" in fact_matching
        assert fact_matching["type"] == "llm_judge"

    @patch("requests.post")
    @patch("sentence_transformers.SentenceTransformer")
    def test_greedy_matching_strategy(self, mock_transformer, mock_post):
        """Test greedy matching strategy."""
        # Mock LLM responses
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"facts": ["Fact 1.", "Fact 2."]}'}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        # Mock embeddings with high similarity
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.array([[1.0, 0.0], [0.8, 0.2]])
        mock_transformer.return_value = mock_encoder

        evaluator = FactMatchingEvaluator(
            api_key="test-key", matching_strategy="greedy"
        )
        evaluator.encoder = mock_encoder

        result = evaluator.run(
            query="Test",
            replies=["Answer"],
            ground_truth_answer="Reference",
        )

        assert "fact_matching" in result["eval_data"]["eval_metrics"]

    @patch("requests.post")
    @patch("sentence_transformers.SentenceTransformer")
    def test_optimal_matching_strategy(self, mock_transformer, mock_post):
        """Test optimal (Hungarian) matching strategy."""
        # Mock LLM responses
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"facts": ["Fact 1.", "Fact 2."]}'}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        # Mock embeddings
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.array([[1.0, 0.0], [0.8, 0.2]])
        mock_transformer.return_value = mock_encoder

        evaluator = FactMatchingEvaluator(
            api_key="test-key", matching_strategy="optimal"
        )
        evaluator.encoder = mock_encoder

        result = evaluator.run(
            query="Test",
            replies=["Answer"],
            ground_truth_answer="Reference",
        )

        assert "fact_matching" in result["eval_data"]["eval_metrics"]

    def test_run_without_ground_truth_skips_evaluation(self):
        """Test that evaluation is skipped if no ground truth."""
        evaluator = FactMatchingEvaluator(api_key="test-key")

        result = evaluator.run(
            query="What is Python?",
            replies=["Python is a programming language."],
            ground_truth_answer=None,
        )

        assert "fact_matching" not in result["eval_data"]["eval_metrics"]

    def test_run_without_answer_skips_evaluation(self):
        """Test that evaluation is skipped if no answer."""
        evaluator = FactMatchingEvaluator(api_key="test-key")

        result = evaluator.run(
            query="What is Python?",
            replies=None,
            ground_truth_answer="Python is a programming language.",
        )

        assert "fact_matching" not in result["eval_data"]["eval_metrics"]

    @patch("requests.post")
    def test_api_call_with_correct_headers(self, mock_post):
        """Test that API is called with correct headers."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"facts": []}'}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        evaluator = FactMatchingEvaluator(api_key="test-key", llm_model="openai/gpt-4")
        evaluator._extract_facts("What is Python?", "Python is a language.", "model")

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
                        "content": '```json\n{"facts": ["Fact 1", "Fact 2"]}\n```'
                    }
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        evaluator = FactMatchingEvaluator(api_key="test-key")
        facts = evaluator._extract_facts("Question", "Answer", "model")

        assert facts == ["Fact 1", "Fact 2"]

    @patch("requests.post")
    @patch("sentence_transformers.SentenceTransformer")
    def test_error_handling_in_api_call(self, mock_transformer, mock_post):
        """Test error handling when API call fails."""
        mock_post.side_effect = Exception("API Error")
        mock_transformer.return_value = MagicMock()

        evaluator = FactMatchingEvaluator(api_key="test-key")

        result = evaluator.run(
            query="What is Python?",
            replies=["Python is a language."],
            ground_truth_answer="Python is a programming language.",
        )

        # When fact extraction fails, it returns empty lists, which results in 0 scores
        assert "fact_matching" in result["eval_data"]["eval_metrics"]
        fact_matching = result["eval_data"]["eval_metrics"]["fact_matching"]
        assert fact_matching["score"] == 0.0
        assert len(fact_matching["model_facts"]) == 0
        assert len(fact_matching["gold_facts"]) == 0

    def test_compute_similarity_matrix(self):
        """Test similarity matrix computation."""
        evaluator = FactMatchingEvaluator(api_key="test-key")

        facts1 = ["Python is great.", "Java is okay."]
        facts2 = ["Python is excellent.", "C++ is fast."]

        matrix = evaluator._compute_similarity_matrix(facts1, facts2)

        assert matrix.shape == (2, 2)
        assert np.all(matrix >= -1) and np.all(matrix <= 1)  # Valid cosine similarities

    def test_match_facts_with_empty_lists(self):
        """Test fact matching with empty lists."""
        evaluator = FactMatchingEvaluator(api_key="test-key")

        result = evaluator._match_facts([], ["Some fact"])
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0
        assert len(result["matches"]) == 0

    def test_to_dict_serialization(self):
        """Test component serialization."""
        evaluator = FactMatchingEvaluator(
            api_key="test-key",
            llm_model="openai/gpt-4",
            embedding_model="all-mpnet-base-v2",
            similarity_threshold=0.8,
            matching_strategy="optimal",
        )

        serialized = evaluator.to_dict()

        assert "type" in serialized
        assert "init_parameters" in serialized
        params = serialized["init_parameters"]
        assert params["api_key"] == "test-key"
        assert params["llm_model"] == "openai/gpt-4"
        assert params["embedding_model"] == "all-mpnet-base-v2"
        assert params["similarity_threshold"] == 0.8
        assert params["matching_strategy"] == "optimal"

    def test_from_dict_deserialization(self):
        """Test component deserialization."""
        serialized = {
            "type": "agentic_rag.components.evaluators.fact_matching_evaluator.FactMatchingEvaluator",
            "init_parameters": {
                "api_key": "test-key",
                "llm_model": "openai/gpt-4",
                "embedding_model": "all-mpnet-base-v2",
                "similarity_threshold": 0.8,
                "matching_strategy": "optimal",
            },
        }

        evaluator = FactMatchingEvaluator.from_dict(serialized)

        assert evaluator.api_key == "test-key"
        assert evaluator.llm_model == "openai/gpt-4"
        assert evaluator.similarity_threshold == 0.8
        assert evaluator.matching_strategy == "optimal"


class TestFactMatchingEvaluatorIntegration:
    """Integration tests requiring actual API access."""

    @pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="Requires OPENROUTER_API_KEY environment variable",
    )
    def test_real_api_call(self):
        """Test actual API call with real OpenRouter service."""
        evaluator = FactMatchingEvaluator(matching_strategy="greedy")

        result = evaluator.run(
            query="What is the capital of France?",
            replies=["Paris is the capital of France. It's a beautiful city."],
            ground_truth_answer="The capital of France is Paris.",
        )

        eval_data = result["eval_data"]
        assert "fact_matching" in eval_data["eval_metrics"]

        fact_matching = eval_data["eval_metrics"]["fact_matching"]
        assert 0.0 <= fact_matching["score"] <= 1.0
        assert 0.0 <= fact_matching["precision"] <= 1.0
        assert 0.0 <= fact_matching["recall"] <= 1.0
        assert len(fact_matching["model_facts"]) > 0
        assert len(fact_matching["gold_facts"]) > 0
