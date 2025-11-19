"""LLM-as-a-judge metric for answer quality evaluation.

Inspired by LONGQAEval: Designing Reliable Evaluations of Long-Form Clinical QA
under Resource Constraints (Bologna et al., 2025).

The LLM evaluates answers across three dimensions using a 5-point Likert scale:

1. Medical Knowledge Alignment (1-5): Assesses factual accuracy, evidence-based
   claims, appropriate uncertainty expression, and absence of contradictions.

2. Question Addressing (1-5): Evaluates relevance, completeness of response,
   inclusion of requested details, and lack of digressions.

3. Risk Communication (1-5): Checks for clear explanation of contraindications,
   side effects, and potential consequences in accessible language.

Scores: 5=Agree, 4=Partially Agree, 3=Neutral, 2=Partially Disagree, 1=Disagree.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import httpx
from haystack import component, default_from_dict, default_to_dict

if TYPE_CHECKING:
    from ...config import Config


@component
class LongQAAnswerEvaluator:
    """LLM-as-judge for evaluating answer quality across three dimensions.

    Evaluates:
    1. Alignment with current medical knowledge
    2. Addressing the specific question
    3. Communicating contraindications or risks

    Usage:
        ```python
        # With explicit API key
        evaluator = LongQAAnswerEvaluator(api_key="your-openrouter-key")

        # With Config object
        from agentic_rag import Config
        config = Config(openrouter_api_key="your-key")
        evaluator = LongQAAnswerEvaluator(config=config)

        result = evaluator.run(
            query="What are the side effects of aspirin?",
            replies=["Aspirin can cause stomach upset..."]
        )
        ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "anthropic/claude-3.5-sonnet",
        base_url: str = "https://openrouter.ai/api/v1",
        config: Optional["Config"] = None,
        timeout: float = 60.0,
    ):
        """Initialize LongQA answer evaluation metric.

        Args:
            api_key: OpenRouter API key (overrides config)
            model: Model identifier on OpenRouter
            base_url: OpenRouter API base URL
            config: Config object with API key (required if api_key not provided)
            timeout: Timeout for API requests in seconds (default: 60.0)
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
                "  LongQAAnswerEvaluator(config=config)"
            )

        self.model = model
        self.base_url = base_url
        self.timeout = timeout

        # Create sync and async HTTP clients
        self.client = httpx.Client(timeout=timeout)
        self.async_client = httpx.AsyncClient(timeout=timeout)

        # Load prompt template
        prompt_path = Path(__file__).parent / "prompts" / "longqa_answer.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt template not found at {prompt_path}. "
                "Please ensure longqa_answer.txt exists in the prompts directory."
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
        """Run LongQA answer quality evaluation.

        Args:
            query: User query
            replies: Generated answers
            eval_data: Evaluation dict from previous evaluator (optional)
            ground_truth_answer: Passed through (not required for this evaluator)
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

        # Skip evaluation if no answer
        if not answer:
            return {"eval_data": eval_data}

        # Evaluate with LongQA
        try:
            prompt = self.prompt_template.format(question=query, answer=answer)
            scores = self._call_llm(prompt)

            # Average across dimensions for overall score (normalize to 0-1)
            avg_score = sum(scores.values()) / len(scores) if scores else 0.0
            normalized_avg = avg_score / 5.0  # Normalize from 1-5 scale to 0-1

            # Add metrics to eval_data
            eval_data["eval_metrics"]["longqa_answer"] = {
                "score": normalized_avg,
                "raw_score": avg_score,
                "dimension_scores": {k: v / 5.0 for k, v in scores.items()},
                "raw_dimension_scores": scores,
                "type": "llm_judge",
            }

        except Exception as e:
            print(f"Error in LongQA answer evaluation: {e}")
            eval_data["eval_metrics"]["longqa_answer_error"] = {
                "error": str(e),
                "type": "llm_judge",
            }

        return {"eval_data": eval_data}

    @component.output_types(eval_data=Dict[str, Any])  # type: ignore[misc]
    async def run_async(
        self,
        query: str,
        replies: Optional[List[str]] = None,
        eval_data: Optional[Dict[str, Any]] = None,
        ground_truth_answer: Optional[str] = None,
        relevant_doc_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Async version of run.

        Run LongQA answer quality evaluation asynchronously.

        Args:
            query: User query
            replies: Generated answers
            eval_data: Evaluation dict from previous evaluator (optional)
            ground_truth_answer: Passed through (not required for this evaluator)
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

        # Skip evaluation if no answer
        if not answer:
            return {"eval_data": eval_data}

        # Evaluate with LongQA (async)
        try:
            prompt = self.prompt_template.format(question=query, answer=answer)
            scores = await self._call_llm_async(prompt)

            # Average across dimensions for overall score (normalize to 0-1)
            avg_score = sum(scores.values()) / len(scores) if scores else 0.0
            normalized_avg = avg_score / 5.0  # Normalize from 1-5 scale to 0-1

            # Add metrics to eval_data
            eval_data["eval_metrics"]["longqa_answer"] = {
                "score": normalized_avg,
                "raw_score": avg_score,
                "dimension_scores": {k: v / 5.0 for k, v in scores.items()},
                "raw_dimension_scores": scores,
                "type": "llm_judge",
            }

        except Exception as e:
            print(f"Error in LongQA answer evaluation: {e}")
            eval_data["eval_metrics"]["longqa_answer_error"] = {
                "error": str(e),
                "type": "llm_judge",
            }

        return {"eval_data": eval_data}

    def _call_llm(self, prompt: str) -> Dict[str, int]:
        """Call LLM via OpenRouter API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/arkhai-io/agentic-rag",
            "X-Title": "Agentic RAG",
        }

        response = self.client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            },
        )
        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"]

        # Parse JSON response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        parsed: Dict[str, int] = json.loads(content)
        return parsed

    async def _call_llm_async(self, prompt: str) -> Dict[str, int]:
        """Async version of _call_llm."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/arkhai-io/agentic-rag",
            "X-Title": "Agentic RAG",
        }

        response = await self.async_client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
            },
        )
        response.raise_for_status()

        content = response.json()["choices"][0]["message"]["content"]

        # Parse JSON response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        parsed: Dict[str, int] = json.loads(content)
        return parsed

    def to_dict(self) -> Dict[str, Any]:
        """Serialize component to dict."""
        return default_to_dict(  # type: ignore[no-any-return]
            self,
            api_key=self.api_key,
            model=self.model,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LongQAAnswerEvaluator":
        """Deserialize component from dict."""
        return default_from_dict(cls, data)  # type: ignore[no-any-return]
