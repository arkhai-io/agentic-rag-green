"""
Retrieval Pipeline Example

This example demonstrates how to create and run a multi-source retrieval pipeline
that queries two indexing pipelines simultaneously:

1. Fast retrieval index: Quick, precise matching with small chunks
2. Semantic retrieval index: Deep understanding with larger context

The retrieval pipeline:
- Queries both indexing pipelines in parallel
- Retrieves relevant documents from both vector stores
- Aggregates and re-ranks results using a cross-encoder
- Generates answers using LLMs
- Evaluates answer quality with multiple metrics

The pipeline automatically inherits embedders and retrievers from each
indexing pipeline, creating separate branch pipelines for parallel execution.
"""

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from agentic_rag import Config, PipelineFactory
from agentic_rag.pipeline import PipelineRunner

# Load environment variables from .env file
load_dotenv()

# Configuration
USERNAME = "your_username"
PROJECT = "demo_rag_app"  # Must match the project from indexing_pipeline_example.py
RETRIEVAL_PIPELINE_NAME = "multi_source_retrieval"

# Indexing pipelines to query (must already exist in Neo4j)
# Querying both provides diverse retrieval: fast + semantic understanding
INDEXING_PIPELINES = ["fast_retrieval_index", "semantic_retrieval_index"]

# Create configuration from environment variables
config = Config(
    neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
    neo4j_password=os.getenv("NEO4J_PASSWORD"),
    openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),  # Required for LLM generation
    log_level=os.getenv("AGENTIC_RAG_LOG_LEVEL", "INFO"),
)


def create_retrieval_pipeline() -> Any:
    """
    Create a multi-source retrieval pipeline that queries both indexing pipelines.

    This creates a branched pipeline architecture where each indexing pipeline
    gets its own execution branch:
    - Branch 1: fast_retrieval_index (small chunks, fast embeddings)
    - Branch 2: semantic_retrieval_index (large chunks, powerful embeddings)

    Auto-injected components (per branch from indexing pipelines):
    - Embedder: Converts query text to embeddings (model matches indexing)
    - Retriever: Retrieves from ChromaDB (root_dir inherited from indexing)

    Explicitly defined components (per branch):
    - Ranker: Re-ranks retrieved documents using cross-encoder
    - Prompt builder: Constructs prompts for LLM
    - Generator: Generates answers using LLM
    - Evaluators: Assess answer quality with multiple metrics

    The branches execute in parallel and results are aggregated automatically.

    Returns:
        PipelineSpec: The created pipeline specification
    """
    # Initialize pipeline factory with config (singleton)
    # GraphStore will be created automatically from config
    factory = PipelineFactory(config=config)

    # Define pipeline components
    # INDEX is a placeholder that auto-injects embedder + retriever from indexing pipelines
    pipeline_spec = [
        {"type": "INDEX"},  # Auto-injected: embedder + retriever
        {"type": "GENERATOR.PROMPT_BUILDER"},  # Build prompt template
        {"type": "GENERATOR.OPENROUTER"},  # Generate answer with LLM
        # Optional: Add evaluators
        {"type": "EVALUATOR.BLEU"},  # Lexical overlap (requires ground truth)
        {"type": "EVALUATOR.ROUGE"},  # Recall-oriented metric (requires ground truth)
        {"type": "EVALUATOR.COHERENCE"},  # Semantic coherence (no ground truth needed)
        {
            "type": "EVALUATOR.READABILITY"
        },  # Readability metrics (no ground truth needed)
    ]

    # Define the prompt template for answer generation
    prompt_template = """Answer the question based only on the provided documents.

        Question: {{query}}

        Documents:
        {% for doc in documents %}
        {{ doc.content }}
        ---
        {% endfor %}

        Answer:"""

    # Configure component parameters
    pipeline_config = {
        "_pipeline_name": RETRIEVAL_PIPELINE_NAME,
        # Specify which indexing pipelines to query
        # This creates 2 parallel branches, one for each indexing pipeline
        "_indexing_pipelines": INDEXING_PIPELINES,
        # Retriever config (root_dir automatically inherited from each indexing pipeline)
        "chroma_embedding_retriever": {
            "top_k": 5,  # Retrieve top 5 from each pipeline (10 total documents)
        },
        # Ranker config (re-ranks documents from both sources)
        "sentence_transformers_similarity_ranker": {
            "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "top_k": 3,  # Return top 3 after re-ranking 10 documents
        },
        # Prompt builder config
        "prompt_builder": {
            "template": prompt_template,
        },
        # Generator config (API key from config)
        "openrouter_generator": {
            "model": "anthropic/claude-3.5-sonnet",
            "generation_kwargs": {
                "temperature": 0.7,
                "max_tokens": 1000,
            },
        },
        # Evaluator configs
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

    # Build the pipeline and store it in Neo4j
    # Username and project are injected at method level for multi-tenant isolation
    pipeline = factory.build_pipeline_graphs_from_specs(
        pipeline_specs=[pipeline_spec],
        username=USERNAME,
        project=PROJECT,
        configs=[pipeline_config],
        pipeline_types=["retrieval"],
    )

    if factory.graph_store:
        factory.graph_store.close()
    return pipeline[0]


def run_retrieval_pipeline(
    query: str,
    ground_truth_answer: Optional[str] = None,
    relevant_doc_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run the multi-source retrieval pipeline to answer a query.

    This executes both branch pipelines in parallel:
    1. Fast branch: Retrieves from fast_retrieval_index
    2. Semantic branch: Retrieves from semantic_retrieval_index

    Documents from both branches are aggregated, re-ranked, and used
    to generate a single comprehensive answer.

    Args:
        query: The question to answer
        ground_truth_answer: Optional ground truth for evaluation
        relevant_doc_ids: Optional list of relevant document IDs for evaluation

    Returns:
        dict: Pipeline results including:
            - query: The original query
            - branches: Results from each pipeline branch
            - documents: Aggregated and ranked documents
            - Evaluation metrics per branch
    """
    # Initialize runner (singleton - no username needed at init)
    # This creates 2 separate branch pipelines (one per indexing pipeline)
    runner = PipelineRunner(
        config=config,  # Config provides Neo4j connection and API keys
        enable_caching=False,
    )

    # Load pipelines with username and project injection
    runner.load_pipelines(
        pipeline_names=[RETRIEVAL_PIPELINE_NAME], username=USERNAME, project=PROJECT
    )

    # Run the pipeline with the query
    # Username and project are injected at method level
    # Execution flow per branch:
    # 1. Embed the query (using the branch's embedding model)
    # 2. Retrieve top 5 documents from the branch's ChromaDB
    # 3. Re-rank all 10 documents (5 from each branch) using cross-encoder
    # 4. Generate answer using top 3 ranked documents
    # 5. Evaluate answer quality with multiple metrics
    result = runner.run(
        pipeline_name=RETRIEVAL_PIPELINE_NAME,
        username=USERNAME,
        type="retrieval",
        project=PROJECT,
        query=query,
        ground_truth_answer=ground_truth_answer,  # Optional for grounded evaluation
        relevant_doc_ids=relevant_doc_ids or [],  # Optional for document recall
    )

    runner.graph_store.close()
    return result  # type: ignore[no-any-return]


if __name__ == "__main__":
    # Step 1: Create the retrieval pipeline (only needs to be done once)
    # Prerequisites: Both indexing pipelines must already exist in Neo4j
    # Run indexing_pipeline_example.py first to create them
    pipeline = create_retrieval_pipeline()

    # Step 2: Run queries through the pipeline
    # The pipeline will query both indexing pipelines and aggregate results
    query = "What is machine learning?"

    # Optional: Provide ground truth for evaluation
    ground_truth = "Machine learning is a method of data analysis that automates analytical model building."

    # Run the query across both indexing pipelines
    result = run_retrieval_pipeline(
        query=query,
        ground_truth_answer=ground_truth,  # Optional for grounded evaluation
    )

    # Access results from the multi-source retrieval
    # result["query"] - The original query
    # result["branches"]["fast_retrieval_index"] - Results from fast pipeline
    # result["branches"]["semantic_retrieval_index"] - Results from semantic pipeline
    # result["documents"] - Aggregated and re-ranked documents from both sources
    # Each branch contains evaluation metrics (BLEU, ROUGE, coherence, etc.)
