# Agentic RAG

A flexible, component-based RAG (Retrieval-Augmented Generation) pipeline system built on Haystack.

## Features

- **Component-based Architecture**: Modular pipeline components for converters, chunkers, embedders, retrievers, and generators
- **Custom Components**: Built-in markdown-aware and semantic chunkers
- **Automatic Document Store Integration**: ChromaDB integration with local persistence
- **Type-safe Configuration**: Strongly typed component specifications and configurations
- **Factory Pattern**: Dynamic pipeline creation from simple specifications

## Quick Start

### Installation

```bash
pip install -e .
```

### Basic Usage

```python
from agentic_rag import PipelineFactory

factory = PipelineFactory()

# Create an indexing pipeline
indexing_spec = [
    {"type": "CONVERTER.TEXT"},
    {"type": "CHUNKER.MARKDOWN_AWARE"},
    {"type": "EMBEDDER.SENTENCE_TRANSFORMERS_DOC"},
    {"type": "WRITER.DOCUMENT_WRITER"}
]

spec, pipeline = factory.create_pipeline_from_spec(
    indexing_spec,
    "my_indexing_pipeline"
)
```

### Available Components

- **Converters**: PDF, DOCX, Markdown, HTML, Text
- **Chunkers**: Document Splitter, Markdown-Aware, Semantic
- **Embedders**: SentenceTransformers (Text & Document)
- **Retrievers**: Chroma Embedding Retriever
- **Generators**: OpenAI Generator
- **Writers**: Document Writer (with Chroma integration)

### Configuration

```python
config = {
    "markdown_aware_chunker": {
        "chunk_size": 1000,
        "chunk_overlap": 100
    },
    "chroma_embedding_retriever": {
        "root_dir": "./data",  # Chroma DB path: ./data/chroma/
        "top_k": 5
    }
}

spec, pipeline = factory.create_pipeline_from_spec(
    pipeline_spec,
    "configured_pipeline",
    config
)
```

## Document Store Integration

ChromaDB is automatically initialized with local persistence:
- Default path: `./data/chroma/`
- Configurable via `root_dir` parameter
- Shared across components in the same pipeline

## Development

```bash
# Install dependencies
poetry install

# Run tests
make test

# Type checking
make type-check

# Linting
make lint
```

## License

MIT License
