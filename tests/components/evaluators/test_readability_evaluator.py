"""Tests for ReadabilityEvaluator component."""

import pytest

from agentic_rag.components.evaluators import ReadabilityEvaluator


class TestReadabilityEvaluator:
    """Test suite for ReadabilityEvaluator."""

    def test_init(self):
        """Test initialization."""
        evaluator = ReadabilityEvaluator()
        assert evaluator is not None

    def test_count_syllables_simple(self):
        """Test syllable counting with simple words."""
        evaluator = ReadabilityEvaluator()

        assert evaluator._count_syllables("cat") == 1
        assert evaluator._count_syllables("hello") == 2
        assert evaluator._count_syllables("computer") == 3
        assert evaluator._count_syllables("education") == 4

    def test_count_syllables_silent_e(self):
        """Test syllable counting with silent 'e'."""
        evaluator = ReadabilityEvaluator()

        # Words ending in 'e' should have the 'e' not counted
        assert evaluator._count_syllables("make") == 1
        assert evaluator._count_syllables("time") == 1

    def test_split_sentences(self):
        """Test sentence splitting."""
        evaluator = ReadabilityEvaluator()

        text = "First sentence. Second sentence! Third sentence?"
        sentences = evaluator._split_sentences(text)

        assert len(sentences) == 3
        assert "First sentence" in sentences
        assert "Second sentence" in sentences
        assert "Third sentence" in sentences

    def test_split_words(self):
        """Test word splitting."""
        evaluator = ReadabilityEvaluator()

        text = "Hello, world! This is a test."
        words = evaluator._split_words(text)

        assert len(words) == 6
        assert "Hello" in words
        assert "world" in words
        assert "This" in words

    def test_flesch_reading_ease(self):
        """Test Flesch Reading Ease calculation."""
        evaluator = ReadabilityEvaluator()

        # Simple text should have high reading ease
        score = evaluator._compute_flesch_reading_ease(
            total_words=10, total_sentences=2, total_syllables=15
        )

        assert 0 <= score <= 100

    def test_flesch_kincaid_grade(self):
        """Test Flesch-Kincaid Grade Level calculation."""
        evaluator = ReadabilityEvaluator()

        grade = evaluator._compute_flesch_kincaid_grade(
            total_words=10, total_sentences=2, total_syllables=15
        )

        assert grade >= 0

    def test_gunning_fog(self):
        """Test Gunning Fog Index calculation."""
        evaluator = ReadabilityEvaluator()

        fog = evaluator._compute_gunning_fog(
            total_words=10, total_sentences=2, complex_words=3
        )

        assert fog >= 0

    def test_smog_index(self):
        """Test SMOG Index calculation."""
        evaluator = ReadabilityEvaluator()

        smog = evaluator._compute_smog_index(total_sentences=2, complex_words=3)

        assert smog >= 0

    def test_coleman_liau(self):
        """Test Coleman-Liau Index calculation."""
        evaluator = ReadabilityEvaluator()

        cli = evaluator._compute_coleman_liau(
            total_words=10, total_sentences=2, total_chars=50
        )

        assert cli >= 0

    def test_ari(self):
        """Test Automated Readability Index calculation."""
        evaluator = ReadabilityEvaluator()

        ari = evaluator._compute_ari(total_words=10, total_sentences=2, total_chars=50)

        assert ari >= 0

    def test_analyze_text_empty(self):
        """Test text analysis with empty text."""
        evaluator = ReadabilityEvaluator()

        result = evaluator._analyze_text("")

        assert result["flesch_reading_ease"] == 0.0
        assert result["flesch_kincaid_grade"] == 0.0
        assert result.get("num_sentences", 0) == 0

    def test_analyze_text_simple(self):
        """Test text analysis with simple text."""
        evaluator = ReadabilityEvaluator()

        text = "The cat sat on the mat. The dog ran fast."
        result = evaluator._analyze_text(text)

        # Verify all metrics are present
        assert "flesch_reading_ease" in result
        assert "flesch_kincaid_grade" in result
        assert "gunning_fog" in result
        assert "smog_index" in result
        assert "coleman_liau" in result
        assert "ari" in result
        assert "avg_sentence_length" in result
        assert "avg_word_length" in result
        assert "avg_syllables_per_word" in result

        # Simple text should have high reading ease
        assert result["flesch_reading_ease"] > 60

    def test_analyze_text_complex(self):
        """Test text analysis with complex text."""
        evaluator = ReadabilityEvaluator()

        text = (
            "The implementation of sophisticated algorithms necessitates "
            "comprehensive understanding of computational complexity theory."
        )
        result = evaluator._analyze_text(text)

        # Complex text should have lower reading ease (below average threshold of 60)
        assert result["flesch_reading_ease"] < 60
        assert result["flesch_kincaid_grade"] > 10

    def test_run_successful(self):
        """Test successful readability evaluation."""
        evaluator = ReadabilityEvaluator()

        result = evaluator.run(
            query="What is Python?",
            replies=["Python is a programming language. It is easy to learn."],
        )

        # Verify structure
        assert "eval_data" in result
        assert "eval_metrics" in result["eval_data"]
        assert "readability" in result["eval_data"]["eval_metrics"]

        readability = result["eval_data"]["eval_metrics"]["readability"]
        assert "score" in readability
        assert "flesch_reading_ease" in readability
        assert "flesch_kincaid_grade" in readability
        assert "gunning_fog" in readability
        assert "smog_index" in readability
        assert "coleman_liau" in readability
        assert "ari" in readability
        assert readability["type"] == "reference_free"

        # Score should be normalized to 0-1
        assert 0 <= readability["score"] <= 1

    def test_run_with_existing_eval_data(self):
        """Test that evaluation extends existing eval_data."""
        evaluator = ReadabilityEvaluator()

        existing_eval_data = {
            "query": "Test",
            "answer": "Test answer",
            "eval_metrics": {"other_metric": {"score": 0.5}},
        }

        result = evaluator.run(
            query="Test",
            replies=["This is a simple sentence. It is easy to read."],
            eval_data=existing_eval_data,
        )

        # Check that existing metric is preserved
        assert "other_metric" in result["eval_data"]["eval_metrics"]
        # Check that readability is added
        assert "readability" in result["eval_data"]["eval_metrics"]

    def test_run_empty_reply(self):
        """Test handling of empty reply."""
        evaluator = ReadabilityEvaluator()

        result = evaluator.run(query="Test", replies=[""])

        # Should return eval_data without readability metric
        assert "eval_data" in result
        assert "readability" not in result["eval_data"].get("eval_metrics", {})

    def test_run_error_handling(self):
        """Test error handling when analysis fails."""
        evaluator = ReadabilityEvaluator()

        # Mock the _analyze_text method to raise an exception
        def mock_analyze(*args, **kwargs):
            raise Exception("Analysis error")

        evaluator._analyze_text = mock_analyze

        result = evaluator.run(query="Test", replies=["Test answer."])

        # Should have readability metric with error
        assert "readability" in result["eval_data"]["eval_metrics"]
        assert result["eval_data"]["eval_metrics"]["readability"]["score"] == 0.0
        assert "error" in result["eval_data"]["eval_metrics"]["readability"]

    def test_serialization(self):
        """Test component serialization."""
        evaluator = ReadabilityEvaluator()

        # Serialize
        config = evaluator.to_dict()
        assert "type" in config
        assert (
            config["type"]
            == "agentic_rag.components.evaluators.readability_evaluator.ReadabilityEvaluator"
        )

        # Deserialize
        new_evaluator = ReadabilityEvaluator.from_dict(config)
        assert new_evaluator is not None

    def test_different_text_complexity_levels(self):
        """Test readability evaluation with texts of different complexity."""
        evaluator = ReadabilityEvaluator()

        # Simple text
        simple_result = evaluator.run(
            query="Test",
            replies=["The cat sat. The dog ran."],
        )

        # Complex text
        complex_result = evaluator.run(
            query="Test",
            replies=[
                "The implementation of sophisticated algorithms necessitates "
                "comprehensive understanding of computational complexity theory."
            ],
        )

        simple_flesch = simple_result["eval_data"]["eval_metrics"]["readability"][
            "flesch_reading_ease"
        ]
        complex_flesch = complex_result["eval_data"]["eval_metrics"]["readability"][
            "flesch_reading_ease"
        ]

        # Simple text should have higher Flesch Reading Ease
        assert simple_flesch > complex_flesch


@pytest.mark.integration
class TestReadabilityEvaluatorIntegration:
    """Integration tests for ReadabilityEvaluator."""

    def test_real_text_analysis(self):
        """Test with real text analysis."""
        evaluator = ReadabilityEvaluator()

        result = evaluator.run(
            query="What is machine learning?",
            replies=[
                "Machine learning is a type of artificial intelligence. "
                "It allows computers to learn from data. "
                "The systems improve their performance over time. "
                "They do this without being explicitly programmed."
            ],
        )

        # Verify response structure
        assert "eval_data" in result
        assert "readability" in result["eval_data"]["eval_metrics"]

        readability = result["eval_data"]["eval_metrics"]["readability"]
        assert 0 <= readability["score"] <= 1
        assert 0 <= readability["flesch_reading_ease"] <= 100
        assert readability["flesch_kincaid_grade"] >= 0
