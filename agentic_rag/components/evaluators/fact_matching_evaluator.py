"""LLM-based fact extraction with semantic similarity matching.

This metric evaluates answer quality by comparing atomic facts extracted from both
the model's answer and a gold standard reference answer.

Methodology:
1. **Fact Extraction (LLM-based):**
   - Uses an LLM to decompose both model and gold answers into atomic facts
   - Each fact is a discrete, verifiable statement
   - Facts are self-contained and independent of context

2. **Semantic Encoding:**
   - Converts all extracted facts into dense vector embeddings using SentenceTransformers
   - Embeddings capture semantic meaning beyond surface-level text similarity

3. **Similarity Matrix Computation:**
   - Computes cosine similarity between all pairs of model and gold facts
   - Creates an N×M matrix where entry (i,j) is similarity between model_fact[i] and gold_fact[j]

4. **1:1 Bipartite Matching:**
   - Greedy: Iteratively matches highest similarity pairs above threshold
   - Optimal: Uses Hungarian algorithm for globally optimal assignment
   - Each fact can match to at most one fact from the other set

5. **Metrics Calculation:**
   - Precision: What fraction of model facts are correct (matched to gold)?
   - Recall: What fraction of gold facts are covered by model?
   - F1 Score: Harmonic mean of precision and recall (primary metric)
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Set, Tuple

import httpx
import numpy as np
from haystack import component, default_from_dict, default_to_dict
from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from ...config import Config


@component
class FactMatchingEvaluator:
    """LLM-based fact extraction with embedding-based matching.

    This metric provides a comprehensive evaluation of answer quality by extracting
    atomic facts from both the model's generated answer and a reference (gold) answer,
    then performing semantic similarity-based matching to compute precision, recall, and F1.

    Usage:
        ```python
        # With explicit API key
        evaluator = FactMatchingEvaluator(
            api_key="your-openrouter-key",
            similarity_threshold=0.75,
            matching_strategy="greedy"
        )

        # With Config object
        from agentic_rag import Config
        config = Config(openrouter_api_key="your-key")
        evaluator = FactMatchingEvaluator(config=config, similarity_threshold=0.75)

        result = evaluator.run(
            query="What is Python?",
            replies=["Python is a programming language."],
            ground_truth_answer="Python is a high-level programming language."
        )
        ```
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        llm_model: str = "anthropic/claude-3.5-sonnet",
        embedding_model: str = "all-MiniLM-L6-v2",
        base_url: str = "https://openrouter.ai/api/v1",
        similarity_threshold: float = 0.75,
        matching_strategy: Literal["greedy", "optimal"] = "greedy",
        config: Optional["Config"] = None,
        timeout: float = 60.0,
    ):
        """Initialize fact matching metric.

        Args:
            api_key: OpenRouter API key (overrides config)
            llm_model: Model identifier on OpenRouter for fact extraction
            embedding_model: SentenceTransformer model for embeddings
            base_url: OpenRouter API base URL
            similarity_threshold: Minimum cosine similarity to consider a match
            matching_strategy: 'greedy' or 'optimal' (Hungarian algorithm)
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
                "  FactMatchingEvaluator(config=config)"
            )

        self.llm_model = llm_model
        self.base_url = base_url
        self.similarity_threshold = similarity_threshold
        self.matching_strategy = matching_strategy
        self.embedding_model_name = embedding_model
        self.timeout = timeout

        # Create sync and async HTTP clients
        self.client = httpx.Client(timeout=timeout)
        self.async_client = httpx.AsyncClient(timeout=timeout)

        # Load prompt template
        prompt_path = Path(__file__).parent / "prompts" / "fact_extraction.txt"
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt template not found at {prompt_path}. "
                "Please ensure fact_extraction.txt exists in the prompts directory."
            )
        self.prompt_template = prompt_path.read_text()

        # Initialize embedding model
        self.encoder = SentenceTransformer(embedding_model)

    @component.output_types(eval_data=Dict[str, Any])  # type: ignore[misc]
    def run(
        self,
        query: str,
        replies: Optional[List[str]] = None,
        eval_data: Optional[Dict[str, Any]] = None,
        ground_truth_answer: Optional[str] = None,
        relevant_doc_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run fact matching evaluation.

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

        # Evaluate with fact matching
        try:
            # Extract facts from both answers
            model_facts = self._extract_facts(query, answer, "model")
            gold_facts = self._extract_facts(query, ground_truth, "gold")

            # Match facts
            matching_result = self._match_facts(model_facts, gold_facts)

            # Add metrics to eval_data
            eval_data["eval_metrics"]["fact_matching"] = {
                "score": matching_result["f1"],
                "precision": matching_result["precision"],
                "recall": matching_result["recall"],
                "model_facts": model_facts,
                "gold_facts": gold_facts,
                "num_matches": len(matching_result["matches"]),
                "num_unmatched_model": len(matching_result["unmatched_model"]),
                "num_unmatched_gold": len(matching_result["unmatched_gold"]),
                "type": "llm_judge",
            }

        except Exception as e:
            print(f"Error in fact matching evaluation: {e}")
            eval_data["eval_metrics"]["fact_matching_error"] = {
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

        Evaluate answer using fact extraction and semantic matching asynchronously.

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

        # Evaluate with fact matching (async)
        try:
            # Extract facts from both answers (async)
            model_facts = await self._extract_facts_async(query, answer, "model")
            gold_facts = await self._extract_facts_async(query, ground_truth, "gold")

            # Match facts (sync - just computation, no I/O)
            matching_result = self._match_facts(model_facts, gold_facts)

            # Add metrics to eval_data
            eval_data["eval_metrics"]["fact_matching"] = {
                "score": matching_result["f1"],
                "precision": matching_result["precision"],
                "recall": matching_result["recall"],
                "model_facts": model_facts,
                "gold_facts": gold_facts,
                "num_matches": len(matching_result["matches"]),
                "num_unmatched_model": len(matching_result["unmatched_model"]),
                "num_unmatched_gold": len(matching_result["unmatched_gold"]),
                "type": "llm_judge",
            }

        except Exception as e:
            print(f"Error in fact matching evaluation: {e}")
            eval_data["eval_metrics"]["fact_matching_error"] = {
                "error": str(e),
                "type": "llm_judge",
            }

        return {"eval_data": eval_data}

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """Call LLM via OpenRouter API to extract facts."""
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
                "model": self.llm_model,
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

        parsed: Dict[str, Any] = json.loads(content)
        return parsed

    async def _call_llm_async(self, prompt: str) -> Dict[str, Any]:
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
                "model": self.llm_model,
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

        parsed: Dict[str, Any] = json.loads(content)
        return parsed

    def _extract_facts(self, question: str, answer: str, answer_type: str) -> List[str]:
        """Extract atomic facts from an answer using LLM."""
        prompt = self.prompt_template.format(
            question=question,
            answer=answer,
            answer_type=answer_type.upper(),
        )

        try:
            result = self._call_llm(prompt)
            facts: List[str] = result.get("facts", [])
            return facts
        except Exception as e:
            print(f"Error extracting facts from {answer_type} answer: {e}")
            return []

    async def _extract_facts_async(
        self, question: str, answer: str, answer_type: str
    ) -> List[str]:
        """Async version of _extract_facts."""
        prompt = self.prompt_template.format(
            question=question,
            answer=answer,
            answer_type=answer_type.upper(),
        )

        try:
            result = await self._call_llm_async(prompt)
            facts: List[str] = result.get("facts", [])
            return facts
        except Exception as e:
            print(f"Error extracting facts from {answer_type} answer: {e}")
            return []

    def _compute_similarity_matrix(
        self, facts1: List[str], facts2: List[str]
    ) -> np.ndarray:
        """Compute cosine similarity matrix between two fact sets."""
        if not facts1 or not facts2:
            return np.array([])

        # Encode facts to embeddings
        embeddings1 = self.encoder.encode(facts1)
        embeddings2 = self.encoder.encode(facts2)

        # Normalize embeddings
        norm1 = np.linalg.norm(embeddings1, axis=1, keepdims=True)
        norm2 = np.linalg.norm(embeddings2, axis=1, keepdims=True)
        normalized1 = embeddings1 / norm1
        normalized2 = embeddings2 / norm2

        # Compute cosine similarity matrix
        similarity_matrix: np.ndarray = np.dot(normalized1, normalized2.T)
        return similarity_matrix

    def _greedy_matching(
        self, similarity_matrix: np.ndarray
    ) -> Tuple[List[Dict[str, Any]], Set[int]]:
        """Perform greedy matching based on similarity matrix."""
        matches = []
        used_model = set()
        used_gold = set()

        # Flatten and sort by similarity (highest first)
        pairs = []
        for i in range(similarity_matrix.shape[0]):
            for j in range(similarity_matrix.shape[1]):
                if similarity_matrix[i, j] >= self.similarity_threshold:
                    pairs.append((i, j, similarity_matrix[i, j]))

        pairs.sort(key=lambda x: x[2], reverse=True)

        # Greedily match
        for model_idx, gold_idx, similarity in pairs:
            if model_idx not in used_model and gold_idx not in used_gold:
                matches.append(
                    {
                        "model_idx": model_idx,
                        "gold_idx": gold_idx,
                        "similarity": float(similarity),
                    }
                )
                used_model.add(model_idx)
                used_gold.add(gold_idx)

        return matches, used_gold

    def _optimal_matching(
        self, similarity_matrix: np.ndarray
    ) -> Tuple[List[Dict[str, Any]], Set[int]]:
        """Perform optimal bipartite matching using Hungarian algorithm."""
        from scipy.optimize import linear_sum_assignment

        # Convert similarity to cost (higher similarity = lower cost)
        cost_matrix = 1 - similarity_matrix

        # Find optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        used_gold = set()

        for model_idx, gold_idx in zip(row_ind, col_ind, strict=False):
            similarity = similarity_matrix[model_idx, gold_idx]
            if similarity >= self.similarity_threshold:
                matches.append(
                    {
                        "model_idx": model_idx,
                        "gold_idx": gold_idx,
                        "similarity": float(similarity),
                    }
                )
                used_gold.add(gold_idx)

        return matches, used_gold

    def _match_facts(
        self, model_facts: List[str], gold_facts: List[str]
    ) -> Dict[str, Any]:
        """Match facts between model and gold answers."""
        if not model_facts or not gold_facts:
            return {
                "matches": [],
                "unmatched_model": model_facts,
                "unmatched_gold": gold_facts,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }

        # Compute similarity matrix
        similarity_matrix = self._compute_similarity_matrix(model_facts, gold_facts)

        # Find matches
        if self.matching_strategy == "greedy":
            matches, used_gold_indices = self._greedy_matching(similarity_matrix)
        else:
            matches, used_gold_indices = self._optimal_matching(similarity_matrix)

        # Identify unmatched facts
        used_model_indices = {m["model_idx"] for m in matches}
        unmatched_model = [
            model_facts[i]
            for i in range(len(model_facts))
            if i not in used_model_indices
        ]
        unmatched_gold = [
            gold_facts[j] for j in range(len(gold_facts)) if j not in used_gold_indices
        ]

        # Compute metrics
        num_matches = len(matches)
        precision = num_matches / len(model_facts) if model_facts else 0.0
        recall = num_matches / len(gold_facts) if gold_facts else 0.0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return {
            "matches": matches,
            "unmatched_model": unmatched_model,
            "unmatched_gold": unmatched_gold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize component to dict."""
        return default_to_dict(  # type: ignore[no-any-return]
            self,
            api_key=self.api_key,
            llm_model=self.llm_model,
            embedding_model=self.embedding_model_name,
            base_url=self.base_url,
            similarity_threshold=self.similarity_threshold,
            matching_strategy=self.matching_strategy,
            timeout=self.timeout,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FactMatchingEvaluator":
        """Deserialize component from dict."""
        return default_from_dict(cls, data)  # type: ignore[no-any-return]
