"""LLM-as-judge metric for factual faithfulness evaluation.

Inspired by MORQA: Benchmarking Evaluation Metrics for Medical Open-Ended
Question Answering (Yim et al., 2025).

The LLM extracts up to 10 atomic facts from the model answer and verifies each
against the reference answer. Each fact is labeled as:
- supported_by_reference: Fact is confirmed by reference
- partially_supported: Fact is somewhat aligned but incomplete/imprecise
- contradicted: Fact directly conflicts with reference
- not_in_reference: Fact is not mentioned in reference

Computes atomic_faithfulness score:
(#supported + 0.5*#partially_supported) / max(1, #facts)

Also identifies critical errors (wrong diagnosis, wrong drug/dose, missed red-flags).
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import requests  # type: ignore[import-untyped]
from haystack import component, default_from_dict, default_to_dict

if TYPE_CHECKING:
    from ...config import Config


@component
class MORQAFaithfulnessEvaluator:
    """LLM-as-judge for evaluating factual faithfulness against reference.

    Extracts atomic facts and verifies alignment with gold standard answer.

    Usage:
        ```python
        # With explicit API key
        evaluator = MORQAFaithfulnessEvaluator(api_key="your-openrouter-key")

        # With Config object
        from agentic_rag import Config
        config = Config(openrouter_api_key="your-key")
        evaluator = MORQAFaithfulnessEvaluator(config=config)

        result = evaluator.run(
            query="What is the capital of France?",
            replies=["Paris is the capital of France."],
            ground_truth_answer="The capital of France is Paris."
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
        """Initialize MORQA faithfulness evaluation metric.

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
                "  MORQAFaithfulnessEvaluator(config=config)"
            )

        self.model = model
        self.base_url = base_url

        # Load prompt template
        prompt_path = Path(__file__).parent / "prompts" / "morqa_faithfulness.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt template not found at {prompt_path}. "
                "Please ensure morqa_faithfulness.txt exists in the prompts directory."
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
        """Run MORQA faithfulness evaluation.

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
            if "query" not in eval_data:
                eval_data["query"] = query
            if "answer" not in eval_data and replies:
                eval_data["answer"] = replies[0]
            if ground_truth_answer and "ground_truth_answer" not in eval_data:
                eval_data["ground_truth_answer"] = ground_truth_answer
            if relevant_doc_ids and "relevant_doc_ids" not in eval_data:
                eval_data["relevant_doc_ids"] = relevant_doc_ids

        if "eval_metrics" not in eval_data:
            eval_data["eval_metrics"] = {}

        answer = eval_data.get("answer")
        ground_truth = eval_data.get("ground_truth_answer")

        # Skip evaluation if no ground truth or answer
        if not ground_truth or not answer:
            return {"eval_data": eval_data}

        # Evaluate faithfulness
        try:
            prompt = self.prompt_template.format(
                question=query,
                answer=answer,
                gold_answer=ground_truth,
            )
            result = self._call_llm(prompt)

            # Add metrics to eval_data
            eval_data["eval_metrics"]["morqa_faithfulness"] = {
                "score": result.get("atomic_faithfulness", 0.0),
                "facts": result.get("facts", []),
                "critical_errors": result.get("critical_errors", []),
                "summary": result.get("summary", ""),
                "type": "llm_judge",
            }

        except Exception as e:
            print(f"Error in MORQA faithfulness evaluation: {e}")
            eval_data["eval_metrics"]["morqa_faithfulness_error"] = {
                "error": str(e),
                "type": "llm_judge",
            }

        return {"eval_data": eval_data}

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call LLM via OpenRouter API."""
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
                "temperature": 0.0,
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

    def to_dict(self) -> Dict[str, Any]:
        """Serialize component to dict."""
        return default_to_dict(  # type: ignore[no-any-return]
            self,
            api_key=self.api_key,
            model=self.model,
            base_url=self.base_url,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MORQAFaithfulnessEvaluator":
        """Deserialize component from dict."""
        return default_from_dict(cls, data)  # type: ignore[no-any-return]
