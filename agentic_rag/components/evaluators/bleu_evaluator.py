"""BLEU metric for lexical overlap evaluation.

BLEU (Bilingual Evaluation Understudy) measures n-gram precision between the
generated answer and reference answer. It computes how many n-grams (1-4 by default)
in the generated answer appear in the reference answer.

Key characteristics:
- Precision-oriented: penalizes missing words from reference
- Uses geometric mean of n-gram precisions
- Applies brevity penalty for short answers
- Smoothing helps with zero n-gram counts (especially for short texts)

Score range: 0.0 (no overlap) to 1.0 (perfect match)

Best for: Comparing literal word/phrase overlap
"""

from typing import Any, Dict, List, Optional

from haystack import component, default_from_dict, default_to_dict
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


@component
class BLEUEvaluator:
    """BLEU score metric measuring n-gram precision overlap.

    Usage:
        ```python
        evaluator = BLEUEvaluator(max_n=4, smoothing=True)
        result = evaluator.run(
            query="What is Python?",
            replies=["Python is a programming language."],
            ground_truth_answer="Python is a high-level programming language."
        )
        ```
    """

    def __init__(self, max_n: int = 4, smoothing: bool = True):
        """Initialize BLEU metric.

        Args:
            max_n: Maximum n-gram size (default: 4)
            smoothing: Apply smoothing for zero counts (default: True)
        """
        self.max_n = max_n
        self.smoothing = smoothing
        self.smooth_fn = SmoothingFunction().method1 if smoothing else None

    @component.output_types(eval_data=Dict[str, Any])  # type: ignore[misc]
    def run(
        self,
        query: str,
        replies: Optional[List[str]] = None,
        eval_data: Optional[Dict[str, Any]] = None,
        ground_truth_answer: Optional[str] = None,
        relevant_doc_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run BLEU evaluation.

        Args:
            query: User query
            replies: Generated answers
            eval_data: Evaluation dict from previous evaluator (optional)
            ground_truth_answer: Reference answer (required for evaluation)
            relevant_doc_ids: Passed through (not used by this evaluator)

        Returns:
            Dict with single key 'eval_data' containing all results
        """
        # Initialize or update eval_data
        if eval_data is None:
            eval_data = {
                "query": query,
                "answer": replies[0] if replies else None,
                "ground_truth_answer": ground_truth_answer,
                "relevant_doc_ids": relevant_doc_ids,
                "eval_metrics": {},
            }
        else:
            # Update with current data if not already set
            if "query" not in eval_data:
                eval_data["query"] = query
            if "answer" not in eval_data and replies:
                eval_data["answer"] = replies[0]
            if ground_truth_answer and "ground_truth_answer" not in eval_data:
                eval_data["ground_truth_answer"] = ground_truth_answer
            if relevant_doc_ids and "relevant_doc_ids" not in eval_data:
                eval_data["relevant_doc_ids"] = relevant_doc_ids

        # Ensure eval_metrics exists
        if "eval_metrics" not in eval_data:
            eval_data["eval_metrics"] = {}

        answer = eval_data.get("answer")
        ground_truth = eval_data.get("ground_truth_answer")

        # Skip evaluation if no ground truth or answer
        if not ground_truth or not answer:
            return {"eval_data": eval_data}

        # Compute BLEU score
        try:
            reference = [ground_truth.split()]
            hypothesis = answer.split()

            # Compute BLEU score with equal weights for all n-grams
            weights = tuple([1.0 / self.max_n] * self.max_n)
            score = sentence_bleu(
                reference,
                hypothesis,
                weights=weights,
                smoothing_function=self.smooth_fn,
            )

            # Add metric to eval_data
            eval_data["eval_metrics"][f"bleu_{self.max_n}"] = {
                "score": float(score),
                "max_n": self.max_n,
                "smoothing": self.smoothing,
                "type": "lexical_overlap",
            }

        except Exception as e:
            print(f"Error computing BLEU score: {e}")
            eval_data["eval_metrics"][f"bleu_{self.max_n}_error"] = {
                "error": str(e),
                "type": "lexical_overlap",
            }

        return {"eval_data": eval_data}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize component to dict."""
        return default_to_dict(  # type: ignore[no-any-return]
            self,
            max_n=self.max_n,
            smoothing=self.smoothing,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BLEUEvaluator":
        """Deserialize component from dict."""
        return default_from_dict(cls, data)  # type: ignore[no-any-return]
