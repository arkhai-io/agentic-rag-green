"""Semantic coherence metric using sentence embeddings.

Evaluates the semantic consistency and flow within an answer by measuring
how well consecutive sentences relate to each other.

Methodology:
1. Splits answer into individual sentences
2. Encodes each sentence into a dense vector embedding
3. Computes cosine similarity between consecutive sentence pairs
4. Returns statistics on inter-sentence coherence

Metrics Computed:
- **coherence_score**: Average similarity between consecutive sentences (primary metric)
- **min_coherence**: Lowest similarity (detects jarring transitions)
- **coherence_variance**: Consistency of flow throughout answer
- **topic_consistency**: Similarity of all sentences to the overall answer embedding

High coherence indicates:
- Smooth transitions between ideas
- Consistent topic throughout answer
- Logical progression of thought

Low coherence may indicate:
- Abrupt topic changes
- Disjointed or rambling responses
- Multiple unrelated ideas mixed together
"""

import re
from statistics import mean, variance
from typing import Any, Dict, List, Optional

import numpy as np
from haystack import component, default_from_dict, default_to_dict
from sentence_transformers import SentenceTransformer


@component
class CoherenceEvaluator:
    """Semantic coherence metric using sentence embeddings.

    Measures how well consecutive sentences in an answer relate to each other
    semantically, indicating smooth flow and topical consistency.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize coherence metric.

        Args:
            embedding_model: SentenceTransformer model name
        """
        self.embedding_model_name = embedding_model
        self.encoder = SentenceTransformer(embedding_model)

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Split on sentence-ending punctuation
        sentences = re.split(r"[.!?]+", text)
        # Filter out empty sentences and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 3]
        return sentences

    def _compute_cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            emb1: First embedding vector
            emb2: Second embedding vector

        Returns:
            Cosine similarity (0-1)
        """
        dot_product = np.dot(emb1, emb2)
        norm_product = np.linalg.norm(emb1) * np.linalg.norm(emb2)

        if norm_product == 0:
            return 0.0

        similarity = dot_product / norm_product
        return float(similarity)

    def _analyze_coherence(self, text: str) -> Dict[str, Any]:
        """Analyze semantic coherence of text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with coherence metrics
        """
        if not text.strip():
            return {
                "coherence_score": 0.0,
                "min_coherence": 0.0,
                "coherence_variance": 0.0,
                "topic_consistency": 0.0,
                "num_sentences": 0,
            }

        sentences = self._split_sentences(text)

        # Need at least 2 sentences for coherence
        if len(sentences) < 2:
            return {
                "coherence_score": 1.0,  # Single sentence is perfectly coherent
                "min_coherence": 1.0,
                "coherence_variance": 0.0,
                "topic_consistency": 1.0,
                "num_sentences": len(sentences),
            }

        # Encode all sentences
        embeddings = self.encoder.encode(sentences)

        # Encode full text for topic consistency
        full_text_embedding = self.encoder.encode([text])[0]

        # Compute consecutive sentence similarities
        consecutive_sims = []
        for i in range(len(embeddings) - 1):
            sim = self._compute_cosine_similarity(embeddings[i], embeddings[i + 1])
            consecutive_sims.append(sim)

        # Compute topic consistency (each sentence vs. full text)
        topic_sims = []
        for emb in embeddings:
            sim = self._compute_cosine_similarity(emb, full_text_embedding)
            topic_sims.append(sim)

        # Compute metrics
        coherence_score = mean(consecutive_sims) if consecutive_sims else 0.0
        min_coherence = min(consecutive_sims) if consecutive_sims else 0.0
        coherence_var = variance(consecutive_sims) if len(consecutive_sims) > 1 else 0.0
        topic_consistency = mean(topic_sims) if topic_sims else 0.0

        return {
            "coherence_score": coherence_score,
            "min_coherence": min_coherence,
            "coherence_variance": coherence_var,
            "topic_consistency": topic_consistency,
            "num_sentences": len(sentences),
        }

    @component.output_types(eval_data=Dict[str, Any])  # type: ignore[misc]
    def run(
        self,
        query: str,
        replies: Optional[List[str]] = None,
        eval_data: Optional[Dict[str, Any]] = None,
        ground_truth_answer: Optional[str] = None,
        relevant_doc_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compute coherence scores.

        Args:
            query: The question being answered
            replies: List of generated answers (uses first one)
            eval_data: Existing evaluation data to extend
            ground_truth_answer: Ground truth (not used for this metric)
            relevant_doc_ids: Relevant doc IDs (not used for this metric)

        Returns:
            Dictionary with eval_data containing coherence metrics
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
            # Analyze coherence
            analysis = self._analyze_coherence(answer)

            # Add to eval_metrics
            eval_data["eval_metrics"]["coherence"] = {
                "score": analysis["coherence_score"],
                "min_coherence": analysis["min_coherence"],
                "coherence_variance": analysis["coherence_variance"],
                "topic_consistency": analysis["topic_consistency"],
                "num_sentences": analysis["num_sentences"],
                "type": "reference_free",
            }

        except Exception as e:
            print(f"Error computing coherence: {e}")
            eval_data["eval_metrics"]["coherence"] = {
                "score": 0.0,
                "error": str(e),
                "type": "reference_free",
            }

        return {"eval_data": eval_data}

    def to_dict(self) -> Dict[str, Any]:
        """Serialize component to dict."""
        return default_to_dict(  # type: ignore[no-any-return]
            self,
            embedding_model=self.embedding_model_name,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CoherenceEvaluator":
        """Deserialize component from dict."""
        return default_from_dict(cls, data)  # type: ignore[no-any-return]
