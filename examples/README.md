# Agentic RAG Examples

## Setup

```bash
poetry install
```

Configure `.env`:
```bash
NEO4J_URI=your_neo4j_uri
NEO4J_USERNAME=your_username
NEO4J_PASSWORD=your_password
OPENROUTER_API_KEY=your_key
```

## Usage

```bash
# Create sample data
poetry run python examples/create_sample_data.py

# Index documents
poetry run python examples/indexing_pipeline_example.py

# Query indexed documents
poetry run python examples/retrieval_pipeline_example.py
```

## Examples

### `indexing_pipeline_example.py`

Two indexing strategies:
- **Fast**: 300 char chunks, all-MiniLM-L6-v2
- **Semantic**: 800 char chunks, all-mpnet-base-v2

### `retrieval_pipeline_example.py`

Multi-source retrieval with evaluation:
- Queries both indexing pipelines
- Re-ranks results
- Generates answers via OpenRouter
- Evaluates with BLEU, ROUGE, coherence, readability

## Storage

- **Neo4j**: Pipeline graphs, metadata, lineage
- **ChromaDB**: Vector embeddings at `./data/{username}/{pipeline_name}/`
- **IPFS**: Document content
