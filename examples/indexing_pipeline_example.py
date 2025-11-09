"""
Indexing Pipeline Example

This example demonstrates how to create and run multiple indexing pipelines with
different configurations for optimal retrieval performance. We create:

1. Fast Pipeline: Small chunks with lightweight embeddings for quick retrieval
2. Semantic Pipeline: Larger chunks with powerful embeddings for deep understanding

Each pipeline:
- Converts documents (PDF/text)
- Chunks them with different strategies
- Generates embeddings using different models
- Stores them in separate ChromaDB collections

The pipelines automatically store metadata in Neo4j and content in IPFS.
"""

import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

from agentic_rag.components import GraphStore
from agentic_rag.pipeline import PipelineFactory, PipelineRunner

# Load environment variables from .env file
load_dotenv()

# Configuration
USERNAME = "your_username"

# Pipeline names for different indexing strategies
FAST_PIPELINE = "fast_retrieval_index"
SEMANTIC_PIPELINE = "semantic_retrieval_index"

# Neo4j connection details
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")


def create_indexing_pipelines() -> List:
    """
    Create two indexing pipelines with different retrieval strategies.

    Pipeline 1 - Fast Retrieval:
    - Small chunks (300 chars) for precise matching
    - Lightweight embeddings (all-MiniLM-L6-v2) for speed
    - Optimized for quick, keyword-like retrieval

    Pipeline 2 - Semantic Retrieval:
    - Larger chunks (800 chars) for context preservation
    - Powerful embeddings (all-mpnet-base-v2) for semantic understanding
    - Optimized for conceptual and contextual retrieval

    Returns:
        List[PipelineSpec]: List of created pipeline specifications
    """
    # Initialize connection to Neo4j graph store
    graph_store = GraphStore(
        uri=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
    )

    # Initialize pipeline factory
    factory = PipelineFactory(graph_store=graph_store, username=USERNAME)

    # Define component specs for both pipelines
    # Both pipelines use the same component types but different configurations
    pipeline_specs = [
        # Fast retrieval pipeline
        [
            {"type": "CONVERTER.TEXT"},
            {"type": "CHUNKER.MARKDOWN_AWARE"},
            {"type": "EMBEDDER.SENTENCE_TRANSFORMERS_DOC"},
            {"type": "WRITER.CHROMA_DOCUMENT_WRITER"},
        ],
        # Semantic retrieval pipeline
        [
            {"type": "CONVERTER.TEXT"},
            {"type": "CHUNKER.MARKDOWN_AWARE"},
            {"type": "EMBEDDER.SENTENCE_TRANSFORMERS_DOC"},
            {"type": "WRITER.CHROMA_DOCUMENT_WRITER"},
        ],
    ]

    # Configure each pipeline with different parameters
    # Note: root_dir is automatically generated as ./data/{username}/{pipeline_name}
    configs = [
        # Fast pipeline configuration
        {
            "_pipeline_name": FAST_PIPELINE,
            "markdown_aware_chunker": {
                "chunk_size": 300,  # Smaller chunks for precise retrieval
                "chunk_overlap": 30,  # 10% overlap
            },
            "document_embedder": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",  # Fast, lightweight
            },
        },
        # Semantic pipeline configuration
        {
            "_pipeline_name": SEMANTIC_PIPELINE,
            "markdown_aware_chunker": {
                "chunk_size": 800,  # Larger chunks preserve context
                "chunk_overlap": 100,  # ~12% overlap
            },
            "document_embedder": {
                "model": "sentence-transformers/all-mpnet-base-v2",  # More powerful
            },
        },
    ]

    # Build both pipelines and store them in Neo4j
    # Each pipeline gets its own ChromaDB collection automatically
    pipelines = factory.build_pipeline_graphs_from_specs(
        pipeline_specs=pipeline_specs,
        configs=configs,
        pipeline_types=["indexing", "indexing"],
        username=USERNAME,
    )

    graph_store.close()
    return pipelines


def run_indexing_pipelines(data_directory: str) -> Dict[str, Any]:
    """
    Run both indexing pipelines on documents in the specified directory.

    Args:
        data_directory: Path to directory containing documents (e.g., "./data/")

    Each pipeline will independently:
    1. Load the documents
    2. Chunk them according to its strategy
    3. Generate embeddings using its model
    4. Store everything in its own ChromaDB collection

    This allows the same documents to be indexed with multiple strategies
    for diverse retrieval capabilities.

    Returns:
        dict: Results from both pipelines
    """
    # Connect to Neo4j
    graph_store = GraphStore(
        uri=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
    )

    # Initialize runner with both indexing pipelines
    runner = PipelineRunner(
        graph_store=graph_store,
        username=USERNAME,
        pipeline_names=[FAST_PIPELINE, SEMANTIC_PIPELINE],
        enable_caching=False,
    )

    results = {}

    # Run each indexing pipeline on the directory
    for pipeline_name in [FAST_PIPELINE, SEMANTIC_PIPELINE]:
        result = runner.run(
            pipeline_name=pipeline_name,
            type="indexing",
            data_path=data_directory,
        )
        results[pipeline_name] = result

    graph_store.close()
    return results


if __name__ == "__main__":
    # Step 1: Create both indexing pipelines (only needs to be done once)
    # The pipeline structures are stored in Neo4j for reuse
    pipelines = create_indexing_pipelines()

    # Step 2: Run both pipelines on your documents
    # This can be called multiple times with different documents
    # Both pipelines will process the same documents with different strategies

    # Specify the data directory (runner will automatically find .txt, .pdf, .md files)
    data_dir = "./data/"

    # Verify directory exists and has files
    if not Path(data_dir).exists():
        print(f"WARNING: Directory {data_dir} not found")
        print("Please run: poetry run python examples/create_sample_data.py")
    else:
        # Check if directory has any supported files
        supported_files = list(Path(data_dir).glob("*.txt")) + list(
            Path(data_dir).glob("*.pdf")
        )
        if not supported_files:
            print(f"WARNING: No documents found in {data_dir}")
            print("Please run: poetry run python examples/create_sample_data.py")
        else:
            print(f"Indexing documents from {data_dir}...")
            results = run_indexing_pipelines(data_dir)
            print("Indexing complete!")
            # results["fast_retrieval_index"] contains fast pipeline results
            # results["semantic_retrieval_index"] contains semantic pipeline results
