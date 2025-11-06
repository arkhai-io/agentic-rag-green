"""Tests for CoherenceEvaluator component."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agentic_rag.components.evaluators import CoherenceEvaluator


class TestCoherenceEvaluator:
    """Test suite for CoherenceEvaluator."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        with patch("sentence_transformers.SentenceTransformer"):
            evaluator = CoherenceEvaluator()
            assert evaluator.embedding_model_name == "all-MiniLM-L6-v2"

    @patch("agentic_rag.components.evaluators.coherence_evaluator.SentenceTransformer")
    def test_init_custom_model(self, mock_transformer):
        """Test initialization with custom embedding model."""
        mock_transformer.return_value = MagicMock()
        evaluator = CoherenceEvaluator(embedding_model="custom-model")
        assert evaluator.embedding_model_name == "custom-model"

    def test_split_sentences(self):
        """Test sentence splitting."""
        with patch("sentence_transformers.SentenceTransformer"):
            evaluator = CoherenceEvaluator()

            text = "First sentence. Second sentence! Third sentence?"
            sentences = evaluator._split_sentences(text)

            assert len(sentences) == 3
            assert "First sentence" in sentences
            assert "Second sentence" in sentences
            assert "Third sentence" in sentences

    def test_split_sentences_empty(self):
        """Test sentence splitting with empty text."""
        with patch("sentence_transformers.SentenceTransformer"):
            evaluator = CoherenceEvaluator()
            sentences = evaluator._split_sentences("")
            assert len(sentences) == 0

    def test_compute_cosine_similarity(self):
        """Test cosine similarity computation."""
        with patch("sentence_transformers.SentenceTransformer"):
            evaluator = CoherenceEvaluator()

            emb1 = np.array([1.0, 0.0, 0.0])
            emb2 = np.array([1.0, 0.0, 0.0])
            sim = evaluator._compute_cosine_similarity(emb1, emb2)

            assert sim == pytest.approx(1.0)

    def test_compute_cosine_similarity_orthogonal(self):
        """Test cosine similarity with orthogonal vectors."""
        with patch("sentence_transformers.SentenceTransformer"):
            evaluator = CoherenceEvaluator()

            emb1 = np.array([1.0, 0.0, 0.0])
            emb2 = np.array([0.0, 1.0, 0.0])
            sim = evaluator._compute_cosine_similarity(emb1, emb2)

            assert sim == pytest.approx(0.0)

    def test_analyze_coherence_empty_text(self):
        """Test coherence analysis with empty text."""
        with patch("sentence_transformers.SentenceTransformer"):
            evaluator = CoherenceEvaluator()

            result = evaluator._analyze_coherence("")

            assert result["coherence_score"] == 0.0
            assert result["num_sentences"] == 0

    @patch("sentence_transformers.SentenceTransformer")
    def test_analyze_coherence_single_sentence(self, mock_transformer):
        """Test coherence analysis with single sentence."""
        mock_encoder = MagicMock()
        mock_transformer.return_value = mock_encoder

        evaluator = CoherenceEvaluator()

        result = evaluator._analyze_coherence("This is a single sentence.")

        # Single sentence should be perfectly coherent
        assert result["coherence_score"] == 1.0
        assert result["num_sentences"] == 1

    @patch("sentence_transformers.SentenceTransformer")
    def test_analyze_coherence_multiple_sentences(self, mock_transformer):
        """Test coherence analysis with multiple sentences."""
        mock_encoder = MagicMock()
        # Mock embeddings for 3 sentences + full text
        mock_encoder.encode.return_value = np.array(
            [
                [1.0, 0.0, 0.0],  # Sentence 1
                [0.9, 0.1, 0.0],  # Sentence 2 (similar to 1)
                [0.8, 0.2, 0.0],  # Sentence 3 (similar to 2)
                [0.9, 0.1, 0.0],  # Full text
            ]
        )
        mock_transformer.return_value = mock_encoder

        evaluator = CoherenceEvaluator()

        text = "First sentence. Second sentence. Third sentence."
        result = evaluator._analyze_coherence(text)

        assert result["num_sentences"] == 3
        assert 0 <= result["coherence_score"] <= 1
        assert 0 <= result["min_coherence"] <= 1
        assert result["coherence_variance"] >= 0
        assert 0 <= result["topic_consistency"] <= 1

    @patch("sentence_transformers.SentenceTransformer")
    def test_run_successful(self, mock_transformer):
        """Test successful coherence evaluation."""
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.9, 0.1],
            ]
        )
        mock_transformer.return_value = mock_encoder

        evaluator = CoherenceEvaluator()

        result = evaluator.run(
            query="What is Python?",
            replies=["Python is a language. It is used for programming."],
        )

        # Verify structure
        assert "eval_data" in result
        assert "eval_metrics" in result["eval_data"]
        assert "coherence" in result["eval_data"]["eval_metrics"]

        coherence = result["eval_data"]["eval_metrics"]["coherence"]
        assert "score" in coherence
        assert "min_coherence" in coherence
        assert "coherence_variance" in coherence
        assert "topic_consistency" in coherence
        assert "num_sentences" in coherence
        assert coherence["type"] == "reference_free"

    @patch("sentence_transformers.SentenceTransformer")
    def test_run_with_existing_eval_data(self, mock_transformer):
        """Test that evaluation extends existing eval_data."""
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.array(
            [[1.0, 0.0], [0.9, 0.1], [0.9, 0.1]]
        )
        mock_transformer.return_value = mock_encoder

        evaluator = CoherenceEvaluator()

        existing_eval_data = {
            "query": "Test",
            "answer": "Test answer",
            "eval_metrics": {"other_metric": {"score": 0.5}},
        }

        result = evaluator.run(
            query="Test",
            replies=["First sentence. Second sentence."],
            eval_data=existing_eval_data,
        )

        # Check that existing metric is preserved
        assert "other_metric" in result["eval_data"]["eval_metrics"]
        # Check that coherence is added
        assert "coherence" in result["eval_data"]["eval_metrics"]

    @patch("sentence_transformers.SentenceTransformer")
    def test_run_empty_reply(self, mock_transformer):
        """Test handling of empty reply."""
        mock_transformer.return_value = MagicMock()
        evaluator = CoherenceEvaluator()

        result = evaluator.run(query="Test", replies=[""])

        # Should return eval_data without coherence metric
        assert "eval_data" in result
        assert "coherence" not in result["eval_data"].get("eval_metrics", {})

    def test_run_error_handling(self):
        """Test error handling when encoding fails."""
        with patch(
            "agentic_rag.components.evaluators.coherence_evaluator.SentenceTransformer"
        ) as mock_transformer:
            mock_encoder = MagicMock()
            mock_encoder.encode.side_effect = Exception("Encoding error")
            mock_transformer.return_value = mock_encoder

            evaluator = CoherenceEvaluator()

            result = evaluator.run(
                query="Test", replies=["First sentence. Second sentence."]
            )

            # Should have coherence metric with error
            assert "coherence" in result["eval_data"]["eval_metrics"]
            assert result["eval_data"]["eval_metrics"]["coherence"]["score"] == 0.0
            assert "error" in result["eval_data"]["eval_metrics"]["coherence"]

    @patch("agentic_rag.components.evaluators.coherence_evaluator.SentenceTransformer")
    def test_serialization(self, mock_transformer):
        """Test component serialization."""
        mock_transformer.return_value = MagicMock()

        evaluator = CoherenceEvaluator(embedding_model="custom-model")

        # Serialize
        config = evaluator.to_dict()
        assert config["init_parameters"]["embedding_model"] == "custom-model"

        # Deserialize
        new_evaluator = CoherenceEvaluator.from_dict(config)
        assert new_evaluator.embedding_model_name == "custom-model"


@pytest.mark.integration
class TestCoherenceEvaluatorIntegration:
    """Integration tests for CoherenceEvaluator."""

    def test_real_encoding(self):
        """Test with real sentence transformer model."""
        evaluator = CoherenceEvaluator(embedding_model="all-MiniLM-L6-v2")

        result = evaluator.run(
            query="What is machine learning?",
            replies=[
                "Machine learning is a type of AI. "
                "It uses algorithms to learn from data. "
                "The models improve with experience."
            ],
        )

        # Verify response structure
        assert "eval_data" in result
        assert "coherence" in result["eval_data"]["eval_metrics"]

        coherence = result["eval_data"]["eval_metrics"]["coherence"]
        assert 0 <= coherence["score"] <= 1
        assert coherence["num_sentences"] == 3
