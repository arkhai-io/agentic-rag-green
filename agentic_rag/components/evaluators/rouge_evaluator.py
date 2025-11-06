"""ROUGE metric for lexical overlap evaluation.

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures recall-based
n-gram overlap. Unlike BLEU's precision focus, ROUGE emphasizes how much of the
reference answer is captured by the generated answer.

Key characteristics:
- Recall-oriented: penalizes missing information from reference
- ROUGE-1: unigram overlap
- ROUGE-2: bigram overlap
- ROUGE-L: longest common subsequence (captures word order)
- Porter stemmer optional (matches word stems, e.g., "running" = "run")

Score range: 0.0 (no overlap) to 1.0 (perfect recall)

Best for: Ensuring generated answer covers reference content
"""

from typing import Any, Dict, List, Optional

from haystack import component, default_from_dict, default_to_dict
from rouge_score import rouge_scorer


@component
class ROUGEEvaluator:
    """ROUGE score metric measuring recall-oriented n-gram overlap.

    Usage:
        ```python
        evaluator = ROUGEEvaluator(rouge_type="rougeL", use_stemmer=True)
        result = evaluator.run(
            query="What is Python?",
            replies=["Python is a programming language."],
            ground_truth_answer="Python is a high-level programming language."
        )
        ```
    """

    def __init__(self, rouge_type: str = "rougeL", use_stemmer: bool = True):
        """Initialize ROUGE metric.

        Args:
            rouge_type: ROUGE variant (rouge1, rouge2, rougeL, rougeLsum)
            use_stemmer: Apply Porter stemmer (default: True)
        """
        self.rouge_type = rouge_type
        self.use_stemmer = use_stemmer
        self.scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=use_stemmer)

    @component.output_types(eval_data=Dict[str, Any])  # type: ignore[misc]
    def run(
        self,
        query: str,
        replies: Optional[List[str]] = None,
        eval_data: Optional[Dict[str, Any]] = None,
        ground_truth_answer: Optional[str] = None,
        relevant_doc_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run ROUGE evaluation.

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

        # Compute ROUGE score
        try:
            scores = self.scorer.score(ground_truth, answer)

            # Use F1 score as primary metric
            f1_score = scores[self.rouge_type].fmeasure
            precision = scores[self.rouge_type].precision
            recall = scores[self.rouge_type].recall

            # Add metric to eval_data
            metric_name = self.rouge_type.lower()
            eval_data["eval_metrics"][metric_name] = {
                "score": float(f1_score),
                "precision": float(precision),
                "recall": float(recall),
                "rouge_type": self.rouge_type,
                "use_stemmer": self.use_stemmer,
                "type": "lexical_overlap",
            }

        except Exception as e:
            print(f"Error computing ROUGE score: {e}")
            eval_data["eval_metrics"][f"{self.rouge_type.lower()}_error"] = {
                "error": str(e),
                "type": "lexical_overlap",
            }

        return {"eval_data": eval_data}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize component to dict."""
        return default_to_dict(  # type: ignore[no-any-return]
            self,
            rouge_type=self.rouge_type,
            use_stemmer=self.use_stemmer,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ROUGEEvaluator":
        """Deserialize component from dict."""
        return default_from_dict(cls, data)  # type: ignore[no-any-return]
