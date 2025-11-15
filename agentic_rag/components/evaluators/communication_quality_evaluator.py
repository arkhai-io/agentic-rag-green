"""LLM-as-judge metric for communication quality evaluation.

Evaluates answer quality across tone, professionalism, and bias dimensions.
This metric assesses how well the answer communicates information regardless of
factual accuracy (which is handled by grounded metrics).

Prompt Behavior:
The LLM evaluates answers across three dimensions using a 5-point Likert scale:
1. Tone Appropriateness (1-5): Assesses whether the tone is suitable for the
   context, professional without being overly formal or casual, and respectful.
2. Professionalism (1-5): Evaluates clarity, structure, grammar, appropriate
   disclaimers/caveats, and avoidance of speculation presented as fact.
3. Bias & Fairness (1-5): Checks for demographic bias (gender, race, age,
   religion), balanced perspective, and inclusive language.

Scores: 5=Excellent, 4=Good, 3=Acceptable, 2=Poor, 1=Unacceptable.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import httpx
from haystack import component, default_from_dict, default_to_dict

if TYPE_CHECKING:
    from ...config import Config


@component
class CommunicationQualityEvaluator:
    """LLM-as-judge for evaluating communication quality.

    Evaluates:
    1. Tone appropriateness
    2. Professionalism
    3. Bias & fairness

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
        """Initialize communication quality evaluation metric.

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
                "  CommunicationQualityEvaluator(config=config)"
            )
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

        # Create sync and async HTTP clients
        self.client = httpx.Client(timeout=timeout)
        self.async_client = httpx.AsyncClient(timeout=timeout)

        # Load prompt template
        prompt_path = Path(__file__).parent / "prompts" / "communication_quality.txt"
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
        """Evaluate communication quality.

        Args:
            query: The question being answered
            replies: List of generated answers (uses first one)
            eval_data: Existing evaluation data to extend
            ground_truth_answer: Ground truth (not used for this metric)
            relevant_doc_ids: Relevant doc IDs (not used for this metric)

        Returns:
            Dictionary with eval_data containing communication quality metrics
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
            tone = result.get("tone_appropriateness", 3)
            professionalism = result.get("professionalism", 3)
            bias = result.get("bias_and_fairness", 3)

            # Compute average score (1-5 scale, convert to 0-1)
            avg_score = (tone + professionalism + bias) / 3
            normalized_avg = (avg_score - 1) / 4  # Convert 1-5 to 0-1

            # Add to eval_metrics
            eval_data["eval_metrics"]["communication_quality_tone"] = {
                "score": (tone - 1) / 4,
                "raw_score": tone,
                "type": "reference_free",
            }
            eval_data["eval_metrics"]["communication_quality_professionalism"] = {
                "score": (professionalism - 1) / 4,
                "raw_score": professionalism,
                "type": "reference_free",
            }
            eval_data["eval_metrics"]["communication_quality_bias"] = {
                "score": (bias - 1) / 4,
                "raw_score": bias,
                "type": "reference_free",
            }
            eval_data["eval_metrics"]["communication_quality_overall"] = {
                "score": normalized_avg,
                "raw_score": avg_score,
                "type": "reference_free",
            }

        except Exception as e:
            print(f"Error evaluating communication quality: {e}")
            # On error, add zero scores
            for dimension in ["tone", "professionalism", "bias", "overall"]:
                eval_data["eval_metrics"][f"communication_quality_{dimension}"] = {
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
    def from_dict(cls, data: Dict[str, Any]) -> "CommunicationQualityEvaluator":
        """Deserialize component from dict."""
        return default_from_dict(cls, data)  # type: ignore[no-any-return]
