"""Tests for ROUGEEvaluator component."""

import pytest

from agentic_rag.components.evaluators import ROUGEEvaluator


class TestROUGEEvaluator:
    """Test ROUGEEvaluator component functionality."""

    def test_initialization_default(self):
        """Test that ROUGEEvaluator can be initialized with defaults."""
        evaluator = ROUGEEvaluator()
        assert evaluator.rouge_type == "rougeL"
        assert evaluator.use_stemmer is True

    def test_initialization_custom(self):
        """Test that ROUGEEvaluator can be initialized with custom parameters."""
        evaluator = ROUGEEvaluator(rouge_type="rouge1", use_stemmer=False)
        assert evaluator.rouge_type == "rouge1"
        assert evaluator.use_stemmer is False

    def test_run_with_ground_truth(self):
        """Test evaluation with ground truth answer."""
        evaluator = ROUGEEvaluator(rouge_type="rougeL")

        result = evaluator.run(
            query="What is the capital of France?",
            replies=["The capital of France is Paris."],
            ground_truth_answer="The capital of France is Paris.",
        )

        # Check that eval_data is returned
        assert "eval_data" in result
        eval_data = result["eval_data"]

        # Check structure
        assert "query" in eval_data
        assert "answer" in eval_data
        assert "ground_truth_answer" in eval_data
        assert "eval_metrics" in eval_data

        # Check metric was added
        assert "rougel" in eval_data["eval_metrics"]
        rouge_metric = eval_data["eval_metrics"]["rougel"]

        # Perfect match should have high score
        assert rouge_metric["score"] == pytest.approx(1.0, abs=0.01)
        assert rouge_metric["precision"] == pytest.approx(1.0, abs=0.01)
        assert rouge_metric["recall"] == pytest.approx(1.0, abs=0.01)
        assert rouge_metric["rouge_type"] == "rougeL"
        assert rouge_metric["type"] == "lexical_overlap"

    def test_run_with_partial_match(self):
        """Test evaluation with partial matching answer."""
        evaluator = ROUGEEvaluator(rouge_type="rouge1")

        result = evaluator.run(
            query="What is Python?",
            replies=["Python is a language."],
            ground_truth_answer="Python is a high-level programming language.",
        )

        eval_data = result["eval_data"]
        assert "rouge1" in eval_data["eval_metrics"]
        rouge_metric = eval_data["eval_metrics"]["rouge1"]

        # Partial match should have moderate score
        assert 0.0 < rouge_metric["score"] < 1.0
        # Should have both precision and recall
        assert 0.0 <= rouge_metric["precision"] <= 1.0
        assert 0.0 <= rouge_metric["recall"] <= 1.0

    def test_run_with_no_match(self):
        """Test evaluation with completely different answers."""
        evaluator = ROUGEEvaluator()

        result = evaluator.run(
            query="What is the capital of France?",
            replies=["Tokyo is a city in Japan."],
            ground_truth_answer="The capital of France is Paris.",
        )

        eval_data = result["eval_data"]
        assert "rougel" in eval_data["eval_metrics"]
        rouge_metric = eval_data["eval_metrics"]["rougel"]

        # No match should have very low score
        assert rouge_metric["score"] < 0.3

    def test_run_with_eval_data_from_previous_evaluator(self):
        """Test that evaluator can receive eval_data from previous evaluator."""
        evaluator = ROUGEEvaluator()

        # Simulate eval_data from a previous evaluator
        existing_eval_data = {
            "query": "What is Python?",
            "answer": "Python is a language.",
            "ground_truth_answer": "Python is a programming language.",
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

        # Check that new metric is added
        assert "rougel" in eval_data["eval_metrics"]

    def test_run_without_ground_truth_skips_evaluation(self):
        """Test that evaluation is skipped if no ground truth is provided."""
        evaluator = ROUGEEvaluator()

        result = evaluator.run(
            query="What is Python?",
            replies=["Python is a programming language."],
            ground_truth_answer=None,
        )

        eval_data = result["eval_data"]

        # Should not have ROUGE metric
        assert "rougel" not in eval_data["eval_metrics"]

    def test_run_without_answer_skips_evaluation(self):
        """Test that evaluation is skipped if no answer is provided."""
        evaluator = ROUGEEvaluator()

        result = evaluator.run(
            query="What is Python?",
            replies=None,
            ground_truth_answer="Python is a programming language.",
        )

        eval_data = result["eval_data"]

        # Should not have ROUGE metric
        assert "rougel" not in eval_data["eval_metrics"]

    def test_different_rouge_types(self):
        """Test ROUGE with different types."""
        # Test ROUGE-1 (unigrams)
        evaluator1 = ROUGEEvaluator(rouge_type="rouge1")
        result1 = evaluator1.run(
            query="Test",
            replies=["Python is great."],
            ground_truth_answer="Python is awesome.",
        )
        assert "rouge1" in result1["eval_data"]["eval_metrics"]

        # Test ROUGE-2 (bigrams)
        evaluator2 = ROUGEEvaluator(rouge_type="rouge2")
        result2 = evaluator2.run(
            query="Test",
            replies=["Python is great."],
            ground_truth_answer="Python is awesome.",
        )
        assert "rouge2" in result2["eval_data"]["eval_metrics"]

        # Test ROUGE-L (longest common subsequence)
        evaluatorL = ROUGEEvaluator(rouge_type="rougeL")
        resultL = evaluatorL.run(
            query="Test",
            replies=["Python is great."],
            ground_truth_answer="Python is awesome.",
        )
        assert "rougel" in resultL["eval_data"]["eval_metrics"]

    def test_stemmer_effect(self):
        """Test that stemmer affects scores."""
        # With stemmer
        evaluator_stem = ROUGEEvaluator(rouge_type="rouge1", use_stemmer=True)
        result_stem = evaluator_stem.run(
            query="Test",
            replies=["Running quickly."],
            ground_truth_answer="Run quick.",
        )

        # Without stemmer
        evaluator_no_stem = ROUGEEvaluator(rouge_type="rouge1", use_stemmer=False)
        result_no_stem = evaluator_no_stem.run(
            query="Test",
            replies=["Running quickly."],
            ground_truth_answer="Run quick.",
        )

        # Both should have metrics
        assert "rouge1" in result_stem["eval_data"]["eval_metrics"]
        assert "rouge1" in result_no_stem["eval_data"]["eval_metrics"]

        # Stemmer should give higher score (matches stems)
        score_stem = result_stem["eval_data"]["eval_metrics"]["rouge1"]["score"]
        score_no_stem = result_no_stem["eval_data"]["eval_metrics"]["rouge1"]["score"]
        assert score_stem >= score_no_stem

    def test_to_dict_serialization(self):
        """Test that component can be serialized to dict."""
        evaluator = ROUGEEvaluator(rouge_type="rouge2", use_stemmer=False)

        serialized = evaluator.to_dict()

        assert "type" in serialized
        assert "init_parameters" in serialized
        assert serialized["init_parameters"]["rouge_type"] == "rouge2"
        assert serialized["init_parameters"]["use_stemmer"] is False

    def test_from_dict_deserialization(self):
        """Test that component can be deserialized from dict."""
        serialized = {
            "type": "agentic_rag.components.evaluators.rouge_evaluator.ROUGEEvaluator",
            "init_parameters": {
                "rouge_type": "rouge2",
                "use_stemmer": False,
            },
        }

        evaluator = ROUGEEvaluator.from_dict(serialized)

        assert evaluator.rouge_type == "rouge2"
        assert evaluator.use_stemmer is False

    def test_recall_oriented_behavior(self):
        """Test that ROUGE is recall-oriented (penalizes missing information)."""
        evaluator = ROUGEEvaluator(rouge_type="rouge1")

        # Model answer missing important information
        result = evaluator.run(
            query="What is Python?",
            replies=["Python is a language."],
            ground_truth_answer="Python is a high-level interpreted object-oriented programming language.",
        )

        rouge_metric = result["eval_data"]["eval_metrics"]["rouge1"]

        # Recall should be low (missing many words from reference)
        assert rouge_metric["recall"] < 0.5
        # Precision might be decent (words it has are correct)
        assert rouge_metric["precision"] > rouge_metric["recall"]

    def test_empty_strings(self):
        """Test handling of empty strings."""
        evaluator = ROUGEEvaluator()

        result = evaluator.run(
            query="Test",
            replies=[""],
            ground_truth_answer="Some answer.",
        )

        eval_data = result["eval_data"]
        # Empty string should skip evaluation (treated as no answer)
        assert "rougel" not in eval_data["eval_metrics"]
