"""LLM-as-judge metric for answer structure and organization evaluation.

Evaluates how well an answer is structured and organized, independent of
factual correctness. Assesses:

1. **Organization** (1-5): Clear introduction, body, conclusion; logical flow
2. **Formatting** (1-5): Appropriate use of paragraphs, lists, headers, emphasis
3. **Information Hierarchy** (1-5): Most important info first, good progression
4. **Clarity of Expression** (1-5): Easy to follow, good transitions, no confusion

Scores: 5=Excellent, 4=Good, 3=Acceptable, 2=Poor, 1=Unacceptable

The overall score is the average across all four dimensions, scaled to 0-1.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import httpx
from haystack import component, default_from_dict, default_to_dict

if TYPE_CHECKING:
    from ...config import Config


@component
class AnswerStructureEvaluator:
    """LLM-as-judge for evaluating answer structure and organization.

    Evaluates:
    1. Organization (logical flow, intro/body/conclusion)
    2. Formatting (paragraphs, lists, emphasis)
    3. Information hierarchy (importance ordering)
    4. Clarity of expression (transitions, comprehensibility)

    Uses OpenRouter API for LLM evaluation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "anthropic/claude-3.5-sonnet",
        base_url: str = "https://openrouter.ai/api/v1",
        config: Optional["Config"] = None,
        timeout: float = 60.0,
    ):
        """Initialize answer structure evaluation metric.

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
                "  AnswerStructureEvaluator(config=config)"
            )
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

        # Create sync and async HTTP clients
        self.client = httpx.Client(timeout=timeout)
        self.async_client = httpx.AsyncClient(timeout=timeout)

        # Load prompt template
        prompt_path = Path(__file__).parent / "prompts" / "answer_structure.txt"
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
        """Evaluate answer structure and organization.

        Args:
            query: The question being answered
            replies: List of generated answers (uses first one)
            eval_data: Existing evaluation data to extend
            ground_truth_answer: Ground truth (not used for this metric)
            relevant_doc_ids: Relevant doc IDs (not used for this metric)

        Returns:
            Dictionary with eval_data containing structure metrics
        """
        # Initialize or extend eval_data
        if eval_data is None:
            eval_data = {
                "query": query,
                "answer": replies[0] if replies else "",
                "eval_metrics": {},
            }
        else:
            # Preserve existing data
            if "query" not in eval_data:
                eval_data["query"] = query
            if "answer" not in eval_data and replies:
                eval_data["answer"] = replies[0]
            if "eval_metrics" not in eval_data:
                eval_data["eval_metrics"] = {}

        # Skip if no answer
        if not replies or not replies[0].strip():
            return {"eval_data": eval_data}

        answer = replies[0]

        try:
            # Format prompt
            prompt = self.prompt_template.format(question=query, answer=answer)

            # Call LLM
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
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

            result = json.loads(content)

            # Extract scores
            organization = result.get("organization", 3)
            formatting = result.get("formatting", 3)
            hierarchy = result.get("hierarchy", 3)
            clarity = result.get("clarity", 3)
            summary = result.get("summary", "")

            # Compute average score (1-5 scale, convert to 0-1)
            avg_score = (organization + formatting + hierarchy + clarity) / 4
            normalized_score = (avg_score - 1) / 4  # Convert 1-5 to 0-1

            # Add to eval_metrics
            eval_data["eval_metrics"]["answer_structure_organization"] = {
                "score": (organization - 1) / 4,
                "raw_score": organization,
                "type": "reference_free",
            }
            eval_data["eval_metrics"]["answer_structure_formatting"] = {
                "score": (formatting - 1) / 4,
                "raw_score": formatting,
                "type": "reference_free",
            }
            eval_data["eval_metrics"]["answer_structure_hierarchy"] = {
                "score": (hierarchy - 1) / 4,
                "raw_score": hierarchy,
                "type": "reference_free",
            }
            eval_data["eval_metrics"]["answer_structure_clarity"] = {
                "score": (clarity - 1) / 4,
                "raw_score": clarity,
                "type": "reference_free",
            }
            eval_data["eval_metrics"]["answer_structure_overall"] = {
                "score": normalized_score,
                "raw_score": avg_score,
                "summary": summary,
                "type": "reference_free",
            }

        except Exception as e:
            print(f"Error evaluating answer structure: {e}")
            # On error, add zero scores
            for dimension in [
                "organization",
                "formatting",
                "hierarchy",
                "clarity",
                "overall",
            ]:
                eval_data["eval_metrics"][f"answer_structure_{dimension}"] = {
                    "score": 0.0,
                    "raw_score": 0,
                    "error": str(e),
                    "type": "reference_free",
                }

        return {"eval_data": eval_data}

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
    def from_dict(cls, data: Dict[str, Any]) -> "AnswerStructureEvaluator":
        """Deserialize component from dict."""
        return default_from_dict(cls, data)  # type: ignore[no-any-return]
