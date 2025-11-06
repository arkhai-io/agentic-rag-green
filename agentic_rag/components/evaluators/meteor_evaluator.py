"""METEOR metric for lexical overlap evaluation.

METEOR (Metric for Evaluation of Translation with Explicit ORdering) is more
sophisticated than BLEU/ROUGE. It considers:

1. Exact word matches
2. Stem matches (e.g., "running" matches "run")
3. Synonym matches (using WordNet)
4. Word order via chunk alignment

Key characteristics:
- Balances precision and recall (configurable via alpha)
- Penalizes fragmentation (words matched out of order)
- Handles paraphrasing better than pure n-gram methods
- More computationally expensive than BLEU/ROUGE

Score range: 0.0 (no match) to 1.0 (perfect match)

Best for: Evaluating semantic similarity with paraphrasing
"""

from typing import Any, Dict, List, Optional

from haystack import component, default_from_dict, default_to_dict
from nltk.translate.meteor_score import meteor_score


@component
class METEOREvaluator:
    """METEOR score metric with synonym and stemming support.

    Usage:
        ```python
        evaluator = METEOREvaluator(alpha=0.9, beta=3.0, gamma=0.5)
        result = evaluator.run(
            query="What is Python?",
            replies=["Python is a programming language."],
            ground_truth_answer="Python is a high-level programming language."
        )
        ```
    """

    def __init__(self, alpha: float = 0.9, beta: float = 3.0, gamma: float = 0.5):
        """Initialize METEOR metric.

        Args:
            alpha: Weight for precision vs recall (default: 0.9)
            beta: Penalty weight for fragmentation (default: 3.0)
            gamma: Weight for fragmentation penalty (default: 0.5)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    @component.output_types(eval_data=Dict[str, Any])  # type: ignore[misc]
    def run(
        self,
        query: str,
        replies: Optional[List[str]] = None,
        eval_data: Optional[Dict[str, Any]] = None,
        ground_truth_answer: Optional[str] = None,
        relevant_doc_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run METEOR evaluation.

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

        # Compute METEOR score
        try:
            reference = ground_truth.split()
            hypothesis = answer.split()

            score = meteor_score(
                [reference],
                hypothesis,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
            )

            # Add metric to eval_data
            eval_data["eval_metrics"]["meteor"] = {
                "score": float(score),
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
                "type": "lexical_overlap",
            }

        except Exception as e:
            print(f"Error computing METEOR score: {e}")
            eval_data["eval_metrics"]["meteor_error"] = {
                "error": str(e),
                "type": "lexical_overlap",
            }

        return {"eval_data": eval_data}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize component to dict."""
        return default_to_dict(  # type: ignore[no-any-return]
            self,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "METEOREvaluator":
        """Deserialize component from dict."""
        return default_from_dict(cls, data)  # type: ignore[no-any-return]
