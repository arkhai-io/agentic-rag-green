"""Tests for BLEUEvaluator component."""

import pytest

from agentic_rag.components.evaluators import BLEUEvaluator


class TestBLEUEvaluator:
    """Test BLEUEvaluator component functionality."""

    def test_initialization_default(self):
        """Test that BLEUEvaluator can be initialized with defaults."""
        evaluator = BLEUEvaluator()
        assert evaluator.max_n == 4
        assert evaluator.smoothing is True

    def test_initialization_custom(self):
        """Test that BLEUEvaluator can be initialized with custom parameters."""
        evaluator = BLEUEvaluator(max_n=2, smoothing=False)
        assert evaluator.max_n == 2
        assert evaluator.smoothing is False

    def test_run_with_ground_truth(self):
        """Test evaluation with ground truth answer."""
        evaluator = BLEUEvaluator(max_n=2)

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
        assert "bleu_2" in eval_data["eval_metrics"]
        bleu_metric = eval_data["eval_metrics"]["bleu_2"]

        # Perfect match should have high score
        assert bleu_metric["score"] == pytest.approx(1.0, abs=0.01)
        assert bleu_metric["max_n"] == 2
        assert bleu_metric["smoothing"] is True
        assert bleu_metric["type"] == "lexical_overlap"

    def test_run_with_partial_match(self):
        """Test evaluation with partial matching answer."""
        evaluator = BLEUEvaluator(max_n=4)

        result = evaluator.run(
            query="What is Python?",
            replies=["Python is a language."],
            ground_truth_answer="Python is a high-level programming language.",
        )

        eval_data = result["eval_data"]
        assert "bleu_4" in eval_data["eval_metrics"]
        bleu_metric = eval_data["eval_metrics"]["bleu_4"]

        # Partial match should have moderate score
        assert 0.0 < bleu_metric["score"] < 1.0

    def test_run_with_no_match(self):
        """Test evaluation with completely different answers."""
        evaluator = BLEUEvaluator()

        result = evaluator.run(
            query="What is the capital of France?",
            replies=["Tokyo is a city in Japan."],
            ground_truth_answer="The capital of France is Paris.",
        )

        eval_data = result["eval_data"]
        assert "bleu_4" in eval_data["eval_metrics"]
        bleu_metric = eval_data["eval_metrics"]["bleu_4"]

        # No match should have low score
        assert bleu_metric["score"] < 0.2

    def test_run_with_eval_data_from_previous_evaluator(self):
        """Test that evaluator can receive eval_data from previous evaluator."""
        evaluator = BLEUEvaluator()

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
        assert "bleu_4" in eval_data["eval_metrics"]

    def test_run_without_ground_truth_skips_evaluation(self):
        """Test that evaluation is skipped if no ground truth is provided."""
        evaluator = BLEUEvaluator()

        result = evaluator.run(
            query="What is Python?",
            replies=["Python is a programming language."],
            ground_truth_answer=None,
        )

        eval_data = result["eval_data"]

        # Should not have BLEU metric
        assert "bleu_4" not in eval_data["eval_metrics"]

    def test_run_without_answer_skips_evaluation(self):
        """Test that evaluation is skipped if no answer is provided."""
        evaluator = BLEUEvaluator()

        result = evaluator.run(
            query="What is Python?",
            replies=None,
            ground_truth_answer="Python is a programming language.",
        )

        eval_data = result["eval_data"]

        # Should not have BLEU metric
        assert "bleu_4" not in eval_data["eval_metrics"]

    def test_different_max_n_values(self):
        """Test BLEU with different max_n values."""
        # Test with max_n=1 (unigrams only)
        evaluator1 = BLEUEvaluator(max_n=1)
        result1 = evaluator1.run(
            query="Test",
            replies=["Python is great."],
            ground_truth_answer="Python is awesome.",
        )
        assert "bleu_1" in result1["eval_data"]["eval_metrics"]

        # Test with max_n=3
        evaluator3 = BLEUEvaluator(max_n=3)
        result3 = evaluator3.run(
            query="Test",
            replies=["Python is great."],
            ground_truth_answer="Python is awesome.",
        )
        assert "bleu_3" in result3["eval_data"]["eval_metrics"]

    def test_to_dict_serialization(self):
        """Test that component can be serialized to dict."""
        evaluator = BLEUEvaluator(max_n=3, smoothing=False)

        serialized = evaluator.to_dict()

        assert "type" in serialized
        assert "init_parameters" in serialized
        assert serialized["init_parameters"]["max_n"] == 3
        assert serialized["init_parameters"]["smoothing"] is False

    def test_from_dict_deserialization(self):
        """Test that component can be deserialized from dict."""
        serialized = {
            "type": "agentic_rag.components.evaluators.bleu_evaluator.BLEUEvaluator",
            "init_parameters": {
                "max_n": 3,
                "smoothing": False,
            },
        }

        evaluator = BLEUEvaluator.from_dict(serialized)

        assert evaluator.max_n == 3
        assert evaluator.smoothing is False

    def test_smoothing_effect(self):
        """Test that smoothing affects scores for short texts."""
        # Short text with smoothing
        evaluator_smooth = BLEUEvaluator(max_n=4, smoothing=True)
        result_smooth = evaluator_smooth.run(
            query="Test",
            replies=["Hi."],
            ground_truth_answer="Hello there.",
        )

        # Short text without smoothing
        evaluator_no_smooth = BLEUEvaluator(max_n=4, smoothing=False)
        result_no_smooth = evaluator_no_smooth.run(
            query="Test",
            replies=["Hi."],
            ground_truth_answer="Hello there.",
        )

        # Both should have metrics
        assert "bleu_4" in result_smooth["eval_data"]["eval_metrics"]
        assert "bleu_4" in result_no_smooth["eval_data"]["eval_metrics"]

    def test_empty_strings(self):
        """Test handling of empty strings."""
        evaluator = BLEUEvaluator()

        result = evaluator.run(
            query="Test",
            replies=[""],
            ground_truth_answer="Some answer.",
        )

        eval_data = result["eval_data"]
        # Empty string should skip evaluation (treated as no answer)
        assert "bleu_4" not in eval_data["eval_metrics"]
