"""
Async Qdrant Retrieval Pipeline Example

This example demonstrates async retrieval with Qdrant, which supports
async operations with BOTH local and remote storage (unlike ChromaDB).

Key advantage: Full async support with local file-based storage!
No Docker server required for async operations.

The retrieval pipeline:
- Queries both Qdrant indexing pipelines in parallel
- Retrieves relevant documents from both local Qdrant stores
- Aggregates and re-ranks results using a cross-encoder
- Generates answers using LLMs
- Evaluates answer quality with multiple metrics
"""

import asyncio
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from agentic_rag import Config, PipelineFactory
from agentic_rag.pipeline import PipelineRunner

# Load environment variables
load_dotenv()

# Configuration
USERNAME = "your_username_2"
PROJECT = "demo_rag_app_qdrant"  # Must match indexing example
RETRIEVAL_PIPELINE_NAME = "qdrant_multi_source_retrieval"

# Indexing pipelines to query (must already exist in Neo4j)
INDEXING_PIPELINES = ["qdrant_fast_index", "qdrant_semantic_index"]

# Create configuration
config = Config(
    neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
    neo4j_password=os.getenv("NEO4J_PASSWORD"),
    openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),  # Required for LLM
    log_level=os.getenv("AGENTIC_RAG_LOG_LEVEL", "INFO"),
)


async def create_qdrant_retrieval_pipeline_async() -> Any:
    """
    Create a multi-source retrieval pipeline using Qdrant.

    This creates branched pipeline architecture:
    - Branch 1: qdrant_fast_index (small chunks, 384-dim embeddings)
    - Branch 2: qdrant_semantic_index (large chunks, 768-dim embeddings)

    Auto-injected per branch from indexing pipelines:
    - Embedder: Converts query to embeddings (model matches indexing)
    - Retriever: Retrieves from local Qdrant (path inherited from indexing)

    Explicitly defined per branch:
    - Prompt builder: Constructs prompts for LLM
    - Generator: Generates answers using LLM
    - Evaluators: Assess answer quality

    Branches execute in parallel, results aggregated automatically.
    """
    factory = PipelineFactory(config=config)

    # Define pipeline components
    # INDEX placeholder auto-injects embedder + retriever from indexing pipelines
    pipeline_spec = [
        {"type": "INDEX"},  # Auto-injected: embedder + retriever
        {"type": "GENERATOR.PROMPT_BUILDER"},
        {"type": "GENERATOR.OPENROUTER"},
        # Evaluators
        {"type": "EVALUATOR.BLEU"},
        {"type": "EVALUATOR.ROUGE"},
        {"type": "EVALUATOR.COHERENCE"},
        {"type": "EVALUATOR.READABILITY"},
    ]

    # Define prompt template
    prompt_template = """Answer the question based only on the provided documents.

        Question: {{query}}

        Documents:
        {% for doc in documents %}
        {{ doc.content }}
        ---
        {% endfor %}

        Answer:"""

    # Configure components
    pipeline_config = {
        "_pipeline_name": RETRIEVAL_PIPELINE_NAME,
        # Query both Qdrant indexing pipelines
        "_indexing_pipelines": INDEXING_PIPELINES,
        # Retriever config (path/collection auto-inherited from indexing)
        "qdrant_embedding_retriever": {
            "top_k": 5,  # Retrieve top 5 from each (10 total)
        },
        # Prompt builder
        "prompt_builder": {
            "template": prompt_template,
        },
        # Generator (API key from config)
        "openrouter_generator": {
            "model": "anthropic/claude-3.5-sonnet",
            "generation_kwargs": {
                "temperature": 0.7,
                "max_tokens": 1000,
            },
        },
        # Evaluators
        "bleu_evaluator": {
            "max_n": 4,
            "smoothing": True,
        },
        "rouge_evaluator": {
            "rouge_type": "rougeL",
            "use_stemmer": True,
        },
        "coherence_evaluator": {
            "embedding_model": "all-MiniLM-L6-v2",
        },
    }

    # Build pipeline asynchronously
    pipeline = await factory.build_pipeline_graphs_from_specs_async(
        pipeline_specs=[pipeline_spec],
        username=USERNAME,
        project=PROJECT,
        configs=[pipeline_config],
        pipeline_types=["retrieval"],
    )

    if factory.graph_store:
        await factory.graph_store.close_async()
    return pipeline[0]


async def run_qdrant_retrieval_pipeline_async(
    query: str,
    ground_truth_answer: Optional[str] = None,
    relevant_doc_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run the Qdrant multi-source retrieval pipeline asynchronously.

    Executes both branch pipelines concurrently:
    1. Fast branch: Retrieves from qdrant_fast_index (local)
    2. Semantic branch: Retrieves from qdrant_semantic_index (local)

    All operations are fully async with local Qdrant storage!
    """
    runner = PipelineRunner(
        config=config,
        enable_caching=False,
    )

    # Load pipelines asynchronously
    await runner.load_pipelines_async(
        pipeline_names=[RETRIEVAL_PIPELINE_NAME], username=USERNAME, project=PROJECT
    )

    # Run retrieval asynchronously
    result = await runner.run_async(
        pipeline_name=RETRIEVAL_PIPELINE_NAME,
        username=USERNAME,
        type="retrieval",
        project=PROJECT,
        query=query,
        ground_truth_answer=ground_truth_answer,
        relevant_doc_ids=relevant_doc_ids or [],
    )

    await runner.graph_store.close_async()
    return result  # type: ignore[no-any-return]


async def main() -> None:
    # Step 1: Create retrieval pipeline
    # Prerequisites: Both Qdrant indexing pipelines must exist
    # Run async_qdrant_indexing_example.py first
    print("Creating Qdrant retrieval pipeline asynchronously...")
    await create_qdrant_retrieval_pipeline_async()
    print("Retrieval pipeline created.")

    # Step 2: Run queries
    query = "What is machine learning?"
    ground_truth = "Machine learning is a method of data analysis that automates analytical model building."

    print(f"\nRunning query: '{query}'")

    result = await run_qdrant_retrieval_pipeline_async(
        query=query,
        ground_truth_answer=ground_truth,
    )

    # Display results
    print("\n" + "=" * 80)
    print("RETRIEVAL RESULTS")
    print("=" * 80)
    print(f"\nQuery: {result.get('query')}")
    print(f"Total documents retrieved: {len(result.get('documents', []))}")
    print(f"Branches executed: {len(result.get('branches', {}))}")

    if "branches" in result:
        for branch_id, branch_result in result["branches"].items():
            print("\n" + "─" * 80)
            print(f"Branch: {branch_id}")
            print("─" * 80)

            # Show generated answer
            if isinstance(branch_result, dict):
                for comp_id, comp_output in branch_result.items():
                    if isinstance(comp_output, dict):
                        if "replies" in comp_output:
                            print("\nGenerated Answer:")
                            print(comp_output["replies"][0])

                        if "eval_data" in comp_output:
                            eval_data = comp_output["eval_data"]
                            if "eval_metrics" in eval_data:
                                print("\nEvaluation Metrics:")
                                for metric_name, metric_value in eval_data[
                                    "eval_metrics"
                                ].items():
                                    if (
                                        isinstance(metric_value, dict)
                                        and "score" in metric_value
                                    ):
                                        print(
                                            f"  {metric_name}: {metric_value['score']:.3f}"
                                        )

    print("\n" + "=" * 80)
    print("All operations completed with local Qdrant + full async support!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
