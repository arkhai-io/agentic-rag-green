<div align="center">
  <img src="assets/logo.jpg" alt="Arkhai" width="120"/>
  <h1>Agentic RAG</h1>
  <p><strong>Multi-pipeline RAG with automatic component orchestration</strong></p>

  [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
</div>

---

## What is Agentic RAG?

A component-based RAG system that lets you query **multiple vector stores with different embeddings and chunking strategies** simultaneously. Built on Haystack 2.0 with Neo4j for pipeline orchestration.

**Key Idea**: Create multiple indexing pipelines with different configurations (chunk sizes, embedding models), then automatically inject and orchestrate them at retrieval time.

## Architecture

```mermaid
graph LR
    subgraph "1. Indexing"
        Doc[Documents] --> P1[Pipeline A<br/>300 chunks<br/>MiniLM]
        Doc --> P2[Pipeline B<br/>600 chunks<br/>MPNet]
        Doc --> P3[Pipeline C<br/>1000 chunks<br/>E5]
    end

    subgraph "2. Storage"
        P1 --> S1[(Store A)]
        P2 --> S2[(Store B)]
        P3 --> S3[(Store C)]
    end

    subgraph "3. Retrieval"
        Q[Query] --> R[Router]
        R --> B1[Branch A<br/>Auto-injected]
        R --> B2[Branch B<br/>Auto-injected]
        R --> B3[Branch C<br/>Auto-injected]
        S1 -.config.-> B1
        S2 -.config.-> B2
        S3 -.config.-> B3
    end

    subgraph "4. Generation"
        B1 & B2 & B3 --> Agg[Aggregate]
        Agg --> Rank[Rerank]
        Rank --> Gen[Generate]
        Gen --> Ans[Answer]
    end

    style Q fill:#e3f2fd
    style R fill:#e3f2fd
    style Agg fill:#f3e5f5
    style Rank fill:#e8f5e9
    style Gen fill:#fff3e0
```

## How It Works

**1. Create diverse indexing pipelines** with different strategies:
```python
Pipeline A: Small chunks (300) + MiniLM embeddings → Store A
Pipeline B: Medium chunks (600) + MPNet embeddings → Store B
Pipeline C: Large chunks (1000) + E5 embeddings → Store C
```

**2. System stores configuration in Neo4j**:
- Component types, parameters, connections
- Embedding models, storage paths, chunk sizes

**3. At retrieval, specify which pipelines to query**:
```python
retrieval_pipeline = {
    "_indexing_pipelines": ["Pipeline A", "Pipeline B"]  # Query 2 of 3
}
```

**4. System automatically creates parallel branches**:
```python
Branch A: Query Embedder (MiniLM) → Retriever (Store A) → Docs
Branch B: Query Embedder (MPNet) → Retriever (Store B) → Docs
↓
Aggregated → Reranked → Generated Answer
```

Each branch gets the **exact same embedding model and storage path** used during indexing - automatically injected from Neo4j metadata.

## Installation

```bash
git clone https://github.com/arkhai/agentic-rag.git
cd agentic-rag
poetry install
```

**Requirements**: Python 3.10+, Neo4j

## Quick Start

### 1. Index documents with multiple strategies

```python
from agentic_rag.pipeline import PipelineFactory, PipelineRunner
from agentic_rag.components import GraphStore

factory = PipelineFactory(graph_store=GraphStore())

# Create 2 indexing pipelines with different chunk sizes
indexing_specs = [
    [
        {"type": "CONVERTER.MARKITDOWN_PDF"},
        {"type": "CHUNKER.MARKDOWN_AWARE"},
        {"type": "EMBEDDER.SENTENCE_TRANSFORMERS_DOC"},
        {"type": "WRITER.CHROMA_DOCUMENT_WRITER"}
    ]
]

configs = [
    {
        "_pipeline_name": "small_chunks",
        "markdown_aware_chunker": {"chunk_size": 300},
        "document_embedder": {"model": "all-MiniLM-L6-v2"},
        "chroma_document_writer": {"root_dir": "./data/small"}
    },
    {
        "_pipeline_name": "large_chunks",
        "markdown_aware_chunker": {"chunk_size": 1000},
        "document_embedder": {"model": "all-mpnet-base-v2"},
        "chroma_document_writer": {"root_dir": "./data/large"}
    }
]

# Build and store in Neo4j
factory.build_pipeline_graphs_from_specs(
    pipeline_specs=indexing_specs * 2,
    configs=configs,
    pipeline_types=["indexing", "indexing"],
    username="myuser"
)

# Index documents
runner = PipelineRunner(graph_store=factory.graph_store, username="myuser")
runner.run(pipeline_name="small_chunks", type="indexing", data_path="./docs/paper.pdf")
runner.run(pipeline_name="large_chunks", type="indexing", data_path="./docs/paper.pdf")
```

### 2. Query multiple pipelines simultaneously

```python
# Create retrieval pipeline that queries both stores
retrieval_spec = [
    [
        {"type": "INDEX"},  # Auto-injects embedders/retrievers from indexing pipelines
        {"type": "RANKER.SENTENCE_TRANSFORMERS_SIMILARITY"},
        {"type": "GENERATOR.PROMPT_BUILDER"},
        {"type": "GENERATOR.OPENROUTER"}
    ]
]

retrieval_config = [
    {
        "_pipeline_name": "multi_retrieval",
        "_indexing_pipelines": ["small_chunks", "large_chunks"],  # Query both!
        "chroma_embedding_retriever": {"top_k": 5},
        "sentence_transformers_similarity_ranker": {"top_k": 3},
        "openrouter_generator": {"model": "anthropic/claude-3.5-sonnet"}
    }
]

factory.build_pipeline_graphs_from_specs(
    pipeline_specs=retrieval_spec,
    configs=retrieval_config,
    pipeline_types=["retrieval"],
    username="myuser"
)

# Query retrieves from both stores, reranks, generates answer
runner = PipelineRunner(graph_store=factory.graph_store, username="myuser")
result = runner.run(
    pipeline_name="multi_retrieval",
    type="retrieval",
    query="What are the main findings?"
)

print(f"Retrieved {result['total_documents']} documents from {result['branches_count']} branches")
print(f"Answer: {result['replies'][0]}")
```

## Why This Matters

**Problem**: Different chunking strategies and embedding models excel at different tasks:
- Small chunks + lightweight embeddings: Fast, good for facts
- Large chunks + powerful embeddings: Better for context, reasoning

**Traditional approach**: Pick one strategy, hope it works

**Agentic RAG**: Use them all, let reranking select the best results

## Evaluation

Built-in evaluation metrics for answer quality:

```python
# Add evaluators to your retrieval pipeline
retrieval_spec = [
    [
        {"type": "INDEX"},
        {"type": "GENERATOR.OPENROUTER"},
        {"type": "EVALUATOR.BLEU"},           # Lexical overlap
        {"type": "EVALUATOR.ANSWER_QUALITY"}, # LLM-as-judge
        {"type": "EVALUATOR.COHERENCE"},      # Semantic consistency
    ]
]

result = runner.run(
    pipeline_name="multi_retrieval",
    type="retrieval",
    query="What is machine learning?",
    ground_truth_answer="ML is a subset of AI..."  # Optional for grounded metrics
)

print(result['eval_data']['eval_metrics'])
# {
#   "bleu_4": {"score": 0.65},
#   "answer_quality_overall": {"score": 0.85},
#   "coherence": {"score": 0.78}
# }
```

**Grounded metrics** (require gold standard):
- BLEU, ROUGE, METEOR - Lexical overlap
- Answer Quality, Fact Matching - LLM-based evaluation

**Ungrounded metrics** (no gold standard):
- Coherence - Semantic consistency
- Readability - Reading level, complexity
- Answer Structure - Organization, formatting
- Communication Quality - Tone, professionalism

## Development

```bash
# Install dev dependencies
poetry install

# Run tests
make test

# Run specific test
poetry run pytest tests/test_multi_pipeline.py -v

# Type checking
make type-check

# Format code
make format
```

## Components

**Converters**: PDF → Markdown (MarkItDown, Marker)
**Chunkers**: Markdown-aware, Semantic boundary detection
**Embedders**: Sentence Transformers (document/query modes)
**Writers/Retrievers**: ChromaDB
**Generators**: OpenAI, OpenRouter (Claude, etc.)
**Rankers**: Cross-encoder reranking
**Evaluators**: BLEU, ROUGE, METEOR, Answer Quality, Coherence, Readability

See [`agentic_rag/types/component_enums.py`](agentic_rag/types/component_enums.py) for full component list.

## License

MIT License

## Built With

[Haystack 2.0](https://haystack.deepset.ai/) · [ChromaDB](https://www.trychroma.com/) · [Neo4j](https://neo4j.com/)
