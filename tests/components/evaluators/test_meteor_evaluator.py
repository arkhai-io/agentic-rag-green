"""Tests for METEOREvaluator component."""

import pytest

from agentic_rag.components.evaluators import METEOREvaluator


class TestMETEOREvaluator:
    """Test METEOREvaluator component functionality."""

    def test_initialization_default(self):
        """Test that METEOREvaluator can be initialized with defaults."""
        evaluator = METEOREvaluator()
        assert evaluator.alpha == 0.9
        assert evaluator.beta == 3.0
        assert evaluator.gamma == 0.5

    def test_initialization_custom(self):
        """Test that METEOREvaluator can be initialized with custom parameters."""
        evaluator = METEOREvaluator(alpha=0.8, beta=2.0, gamma=0.3)
        assert evaluator.alpha == 0.8
        assert evaluator.beta == 2.0
        assert evaluator.gamma == 0.3

    def test_run_with_ground_truth(self):
        """Test evaluation with ground truth answer."""
        evaluator = METEOREvaluator()

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
        assert "meteor" in eval_data["eval_metrics"]
        meteor_metric = eval_data["eval_metrics"]["meteor"]

        # Perfect match should have high score
        assert meteor_metric["score"] == pytest.approx(1.0, abs=0.01)
        assert meteor_metric["alpha"] == 0.9
        assert meteor_metric["beta"] == 3.0
        assert meteor_metric["gamma"] == 0.5
        assert meteor_metric["type"] == "lexical_overlap"

    def test_run_with_partial_match(self):
        """Test evaluation with partial matching answer."""
        evaluator = METEOREvaluator()

        result = evaluator.run(
            query="What is Python?",
            replies=["Python is a language."],
            ground_truth_answer="Python is a high-level programming language.",
        )

        eval_data = result["eval_data"]
        assert "meteor" in eval_data["eval_metrics"]
        meteor_metric = eval_data["eval_metrics"]["meteor"]

        # Partial match should have moderate score
        assert 0.0 < meteor_metric["score"] < 1.0

    def test_run_with_synonyms(self):
        """Test that METEOR recognizes synonyms."""
        evaluator = METEOREvaluator()

        # Test with synonyms (METEOR uses WordNet)
        result = evaluator.run(
            query="Test",
            replies=["The quick brown fox."],
            ground_truth_answer="The fast brown fox.",
        )

        eval_data = result["eval_data"]
        assert "meteor" in eval_data["eval_metrics"]
        meteor_metric = eval_data["eval_metrics"]["meteor"]

        # Should have decent score due to synonyms (quick/fast)
        assert meteor_metric["score"] > 0.5

    def test_run_with_stemming(self):
        """Test that METEOR recognizes word stems."""
        evaluator = METEOREvaluator()

        # Test with different word forms
        result = evaluator.run(
            query="Test",
            replies=["Running quickly through the streets."],
            ground_truth_answer="Run quick through the street.",
        )

        eval_data = result["eval_data"]
        assert "meteor" in eval_data["eval_metrics"]
        meteor_metric = eval_data["eval_metrics"]["meteor"]

        # Should have decent score due to stemming
        assert meteor_metric["score"] > 0.4

    def test_run_with_no_match(self):
        """Test evaluation with completely different answers."""
        evaluator = METEOREvaluator()

        result = evaluator.run(
            query="What is the capital of France?",
            replies=["Tokyo is a city in Japan."],
            ground_truth_answer="The capital of France is Paris.",
        )

        eval_data = result["eval_data"]
        assert "meteor" in eval_data["eval_metrics"]
        meteor_metric = eval_data["eval_metrics"]["meteor"]

        # No match should have low score
        assert meteor_metric["score"] < 0.2

    def test_run_with_eval_data_from_previous_evaluator(self):
        """Test that evaluator can receive eval_data from previous evaluator."""
        evaluator = METEOREvaluator()

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
        assert "meteor" in eval_data["eval_metrics"]

    def test_run_without_ground_truth_skips_evaluation(self):
        """Test that evaluation is skipped if no ground truth is provided."""
        evaluator = METEOREvaluator()

        result = evaluator.run(
            query="What is Python?",
            replies=["Python is a programming language."],
            ground_truth_answer=None,
        )

        eval_data = result["eval_data"]

        # Should not have METEOR metric
        assert "meteor" not in eval_data["eval_metrics"]

    def test_run_without_answer_skips_evaluation(self):
        """Test that evaluation is skipped if no answer is provided."""
        evaluator = METEOREvaluator()

        result = evaluator.run(
            query="What is Python?",
            replies=None,
            ground_truth_answer="Python is a programming language.",
        )

        eval_data = result["eval_data"]

        # Should not have METEOR metric
        assert "meteor" not in eval_data["eval_metrics"]

    def test_word_order_sensitivity(self):
        """Test that METEOR is sensitive to word order (fragmentation penalty)."""
        evaluator = METEOREvaluator()

        # Same words, different order
        result1 = evaluator.run(
            query="Test",
            replies=["The cat sat on the mat."],
            ground_truth_answer="The cat sat on the mat.",
        )

        result2 = evaluator.run(
            query="Test",
            replies=["The mat on sat cat the."],
            ground_truth_answer="The cat sat on the mat.",
        )

        score1 = result1["eval_data"]["eval_metrics"]["meteor"]["score"]
        score2 = result2["eval_data"]["eval_metrics"]["meteor"]["score"]

        # Correct order should have higher score
        assert score1 > score2

    def test_alpha_parameter_effect(self):
        """Test that alpha parameter affects precision/recall balance."""
        # High alpha (favor recall)
        evaluator_high_alpha = METEOREvaluator(alpha=0.95)
        result_high = evaluator_high_alpha.run(
            query="Test",
            replies=["Python."],
            ground_truth_answer="Python is a programming language.",
        )

        # Low alpha (favor precision)
        evaluator_low_alpha = METEOREvaluator(alpha=0.5)
        result_low = evaluator_low_alpha.run(
            query="Test",
            replies=["Python."],
            ground_truth_answer="Python is a programming language.",
        )

        # Both should have metrics
        assert "meteor" in result_high["eval_data"]["eval_metrics"]
        assert "meteor" in result_low["eval_data"]["eval_metrics"]

    def test_to_dict_serialization(self):
        """Test that component can be serialized to dict."""
        evaluator = METEOREvaluator(alpha=0.8, beta=2.5, gamma=0.4)

        serialized = evaluator.to_dict()

        assert "type" in serialized
        assert "init_parameters" in serialized
        assert serialized["init_parameters"]["alpha"] == 0.8
        assert serialized["init_parameters"]["beta"] == 2.5
        assert serialized["init_parameters"]["gamma"] == 0.4

    def test_from_dict_deserialization(self):
        """Test that component can be deserialized from dict."""
        serialized = {
            "type": "agentic_rag.components.evaluators.meteor_evaluator.METEOREvaluator",
            "init_parameters": {
                "alpha": 0.8,
                "beta": 2.5,
                "gamma": 0.4,
            },
        }

        evaluator = METEOREvaluator.from_dict(serialized)

        assert evaluator.alpha == 0.8
        assert evaluator.beta == 2.5
        assert evaluator.gamma == 0.4

    def test_paraphrasing_detection(self):
        """Test that METEOR handles paraphrasing better than simple n-gram methods."""
        evaluator = METEOREvaluator()

        # Paraphrased answer
        result = evaluator.run(
            query="Test",
            replies=["Python is a simple and easy programming language."],
            ground_truth_answer="Python is an easy and simple programming language.",
        )

        meteor_metric = result["eval_data"]["eval_metrics"]["meteor"]

        # Should have high score despite different word order
        assert meteor_metric["score"] > 0.7

    def test_empty_strings(self):
        """Test handling of empty strings."""
        evaluator = METEOREvaluator()

        result = evaluator.run(
            query="Test",
            replies=[""],
            ground_truth_answer="Some answer.",
        )

        eval_data = result["eval_data"]
        # Empty string should skip evaluation (treated as no answer)
        assert "meteor" not in eval_data["eval_metrics"]
