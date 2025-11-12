"""LLM-as-a-judge metric for answer completeness and correctness.

This metric uses an LLM to evaluate two critical dimensions of answer quality:

1. **Completeness**: Does the answer cover all important aspects of the gold answer?
   - Evaluates information coverage, not just fact count
   - Considers whether key topics and concepts are addressed
   - Penalizes missing critical information
   - Range: 1-5 (1=Very Incomplete, 5=Fully Complete)

2. **Correctness**: Is the information in the answer factually accurate?
   - Identifies factual errors and contradictions
   - Checks accuracy against the gold standard
   - Detects hallucinations and misinformation
   - Range: 1-5 (1=Many Errors, 5=Fully Accurate)

Best Practices Implemented:
- Clear, structured prompts with examples
- Granular 1-5 Likert scale for nuance
- Explicit evaluation criteria and rubric
- Chain-of-thought reasoning required
- JSON output for structured parsing
- Temperature=0 for consistency
- Separate scoring for each dimension
- Detailed reasoning for interpretability

The metric returns both individual dimension scores and an overall quality score.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import requests  # type: ignore[import-untyped]
from haystack import component, default_from_dict, default_to_dict

if TYPE_CHECKING:
    from ...config import Config


@component
class AnswerQualityEvaluator:
    """
    LLM-based evaluation of answer completeness and correctness.

    This metric provides a comprehensive assessment of answer quality by evaluating:
    1. Completeness: Coverage of important information from gold answer
    2. Correctness: Factual accuracy compared to gold standard

    The metric uses a carefully designed prompt with explicit rubrics and requires
    the LLM to provide reasoning for its scores, improving reliability and interpretability.

    Usage:
        ```python
        # With explicit API key
        evaluator = AnswerQualityEvaluator(
            api_key="your-openrouter-key",
            model="anthropic/claude-3.5-sonnet"
        )

        # With Config object
        from agentic_rag import Config
        config = Config(openrouter_api_key="your-key")
        evaluator = AnswerQualityEvaluator(config=config, model="anthropic/claude-3.5-sonnet")

        result = evaluator.run(
            query="What is Python?",
            replies=["Python is a programming language."],
            ground_truth_answer="Python is a high-level programming language..."
        )
        ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "anthropic/claude-3.5-sonnet",
        base_url: str = "https://openrouter.ai/api/v1",
        config: Optional["Config"] = None,
    ):
        """
        Initialize answer quality evaluator.

        Args:
            api_key: OpenRouter API key (overrides config)
            model: Model identifier on OpenRouter
            base_url: OpenRouter API base URL
            config: Config object with API key (required if api_key not provided)
        """
        # Priority: explicit api_key > config object
        if config is not None:
            self.api_key = api_key or config.openrouter_api_key
        else:
            self.api_key = api_key

        if not self.api_key:
            raise ValueError(
                "OpenRouter API key required. Provide via config parameter:\n"
                "  config = Config(openrouter_api_key='your-key')\n"
                "  AnswerQualityEvaluator(config=config)"
            )

        self.model = model
        self.base_url = base_url

        # Load prompt template
        prompt_path = Path(__file__).parent / "prompts" / "answer_quality.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt template not found at {prompt_path}. "
                "Please ensure answer_quality.txt exists in the prompts directory."
            )
        self.prompt_template = prompt_path.read_text()

    @component.output_types(eval_data=Dict[str, Any])  # type: ignore[misc]
    def run(
        self,
        query: str,
        replies: Optional[List[str]] = None,
        eval_data: Optional[Dict[str, Any]] = None,
        ground_truth_answer: Optional[str] = None,
        relevant_doc_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run answer quality evaluation using LLM-as-a-judge.

        Args:
            query: User query
            replies: Generated answers
            eval_data: Evaluation dict from previous evaluator (optional)
            ground_truth_answer: Expected/reference answer (required for evaluation)
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

        # Evaluate answer quality
        try:
            result = self._evaluate_answer(
                question=query,
                model_answer=answer,
                gold_answer=ground_truth,
            )

            # Add metrics to eval_data
            eval_data["eval_metrics"]["answer_quality_completeness"] = {
                "score": result["completeness_score"] / 5.0,  # Normalize to 0-1
                "raw_score": result["completeness_score"],
                "reasoning": result["completeness_reasoning"],
                "key_missing_info": result["key_missing_info"],
                "type": "llm_judge",
            }

            eval_data["eval_metrics"]["answer_quality_correctness"] = {
                "score": result["correctness_score"] / 5.0,  # Normalize to 0-1
                "raw_score": result["correctness_score"],
                "reasoning": result["correctness_reasoning"],
                "factual_errors": result["factual_errors"],
                "type": "llm_judge",
            }

            eval_data["eval_metrics"]["answer_quality_overall"] = {
                "score": result["overall_score"] / 5.0,  # Normalize to 0-1
                "raw_score": result["overall_score"],
                "type": "llm_judge",
            }

        except Exception as e:
            print(f"Error evaluating answer quality: {e}")
            # Add error to metadata instead of failing
            eval_data["eval_metrics"]["answer_quality_error"] = {
                "error": str(e),
                "type": "llm_judge",
            }

        return {"eval_data": eval_data}

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """
        Call LLM via OpenRouter API.

        Args:
            prompt: Formatted prompt string

        Returns:
            Dictionary with evaluation results
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/arkhai-io/agentic-rag",
            "X-Title": "Agentic RAG",
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,  # Deterministic for consistency
            },
            timeout=60,
        )
        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"]

        # Parse JSON response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        parsed: Dict[str, Any] = json.loads(content)
        return parsed

    def _evaluate_answer(
        self, question: str, model_answer: str, gold_answer: str
    ) -> Dict[str, Any]:
        """
        Evaluate a single answer for completeness and correctness.

        Args:
            question: The question being answered
            model_answer: The model's generated answer
            gold_answer: The reference (gold) answer

        Returns:
            Dictionary with scores and reasoning
        """
        prompt = self.prompt_template.format(
            question=question,
            model_answer=model_answer,
            gold_answer=gold_answer,
        )

        try:
            result = self._call_llm(prompt)

            # Validate scores are in range
            completeness = max(1, min(5, result.get("completeness_score", 3)))
            correctness = max(1, min(5, result.get("correctness_score", 3)))

            # Overall quality is average of the two dimensions
            overall = (completeness + correctness) / 2.0

            return {
                "completeness_score": completeness,
                "correctness_score": correctness,
                "overall_score": overall,
                "completeness_reasoning": result.get("completeness_reasoning", ""),
                "correctness_reasoning": result.get("correctness_reasoning", ""),
                "key_missing_info": result.get("key_missing_info", []),
                "factual_errors": result.get("factual_errors", []),
            }

        except Exception as e:
            print(f"Error evaluating answer: {e}")
            return {
                "completeness_score": 0,
                "correctness_score": 0,
                "overall_score": 0.0,
                "completeness_reasoning": f"Error: {str(e)}",
                "correctness_reasoning": f"Error: {str(e)}",
                "key_missing_info": [],
                "factual_errors": [],
            }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize component to dict."""
        return default_to_dict(  # type: ignore[no-any-return]
            self,
            api_key=self.api_key,
            model=self.model,
            base_url=self.base_url,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnswerQualityEvaluator":
        """Deserialize component from dict."""
        return default_from_dict(cls, data)  # type: ignore[no-any-return]
