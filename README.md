# Agentic RAG Green Agent

A Green Agent (Assessor) for the AgentBeats platform that provides a RAG (Retrieval-Augmented Generation) evaluation environment. This agent enables standardized benchmarking of AI agents on RAG tasks through the A2A (Agent-to-Agent) protocol.

## Overview

The Green Agent acts as both an environment provider and evaluator for Purple Agents (participants) in RAG benchmarks. It provides:

- **Agent Registration**: Purple agents register with unique identifiers
- **Project Management**: Organize pipelines into isolated projects
- **Pipeline Creation**: Build indexing and retrieval pipelines from component specifications
- **Document Indexing**: Process and store documents for retrieval
- **Query Execution**: Run RAG queries with automatic evaluation
- **Assessment Orchestration**: Coordinate full benchmark assessments via AgentBeats

## Architecture

```
Purple Agent (Participant)          Green Agent (Assessor)
       |                                    |
       |  A2A Protocol (JSON-RPC)           |
       |----------------------------------->|
       |  {"action": "register", ...}       |
       |                                    |
       |<-----------------------------------|
       |  {"success": true, ...}            |
       |                                    |
                                            |
                              +-------------+-------------+
                              |                           |
                         RAGEnvironment              Neo4j Graph
                              |                           |
                    +---------+---------+                 |
                    |         |         |                 |
                 Factory   Runner   GraphStore -----------+
```

## Directory Structure

```
agentic-rag-green/
├── src/
│   ├── server.py       # A2A server entry point and agent card configuration
│   ├── executor.py     # A2A request handler and context management
│   ├── agent.py        # Core agent logic and action dispatch
│   ├── environment.py  # RAG environment wrapping agentic-rag SDK
│   ├── messenger.py    # A2A client for outbound communication
│   └── models.py       # Pydantic models for request/response validation
├── agentic-rag/        # Submodule: Core RAG pipeline library
│   └── agentic_rag/
│       ├── components/ # Chunkers, embedders, generators, evaluators
│       ├── pipeline/   # PipelineFactory and PipelineRunner
│       └── types/      # Component specifications and enums
├── data/
│   ├── tasks/          # Benchmark task definitions
│   └── ground_truth/   # Ground truth answers for evaluation
├── tests/              # Test suite
├── Dockerfile          # Container configuration for AgentBeats
├── pyproject.toml      # Dependencies and project metadata
└── README.md
```

## Installation

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Neo4j database (local or Aura)
- ChromaDB (for vector storage)
- OpenRouter API key (for LLM-based generation and evaluation)

### Setup

```bash
# Clone repository
git clone https://github.com/arkhai/agentic-rag-green.git
cd agentic-rag-green

# Initialize submodule
git submodule update --init --recursive

# Install dependencies
uv sync

# Configure environment
cp sample.env .env
# Edit .env with your credentials
```

## Usage

### Starting the Server (Local)

```bash
# Start ChromaDB first
docker run -d -p 8000:8000 --name chromadb chromadb/chroma

# Run the server
CHROMA_HOST=localhost CHROMA_PORT=8000 uv run python -m src.server --host 0.0.0.0 --port 9009
```

### Agent Card

Once running, the agent card is available at:
```
http://localhost:9009/.well-known/agent.json
```

### A2A Actions

All actions are sent via JSON-RPC to the server root endpoint.

#### Register Agent

```bash
curl -X POST http://localhost:9009/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "id": "1",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "{\"action\": \"register\", \"params\": {\"agent_name\": \"my_agent\"}}"}],
        "messageId": "msg-001"
      }
    }
  }'
```

#### Create Project

```bash
curl -X POST http://localhost:9009/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "id": "2",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "{\"action\": \"create_project\", \"params\": {\"project_name\": \"my_project\"}}"}],
        "messageId": "msg-002",
        "contextId": "<context_id_from_registration>"
      }
    }
  }'
```

#### Create Pipeline

```bash
curl -X POST http://localhost:9009/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "message/send",
    "id": "3",
    "params": {
      "message": {
        "role": "user",
        "parts": [{"kind": "text", "text": "{\"action\": \"create_pipeline\", \"params\": {\"project_name\": \"my_project\", \"pipeline_name\": \"indexer\", \"pipeline_type\": \"indexing\", \"components\": [{\"type\": \"CONVERTER.TEXT\"}, {\"type\": \"CHUNKER.DOCUMENT_SPLITTER\"}, {\"type\": \"EMBEDDER.SENTENCE_TRANSFORMERS_DOC\"}, {\"type\": \"WRITER.CHROMA_DOCUMENT_WRITER\"}]}}"}],
        "messageId": "msg-003",
        "contextId": "<context_id>"
      }
    }
  }'
```

### Available Components

| Category | Types |
|----------|-------|
| CONVERTER | TEXT, PDF, DOCX, MARKDOWN, HTML, MARKER_PDF, MARKITDOWN_PDF |
| CHUNKER | DOCUMENT_SPLITTER, MARKDOWN_AWARE, SEMANTIC |
| EMBEDDER | SENTENCE_TRANSFORMERS, SENTENCE_TRANSFORMERS_DOC |
| RETRIEVER | CHROMA_EMBEDDING, QDRANT_EMBEDDING |
| RANKER | SENTENCE_TRANSFORMERS_SIMILARITY |
| GENERATOR | PROMPT_BUILDER, OPENAI, OPENROUTER |
| WRITER | CHROMA_DOCUMENT_WRITER, QDRANT_DOCUMENT_WRITER |
| EVALUATOR | BLEU, ROUGE, METEOR, ANSWER_QUALITY, FACT_MATCHING, COHERENCE, READABILITY |

### Available Actions

| Action | Description |
|--------|-------------|
| `register` | Register a new agent with unique name |
| `create_project` | Create a project to organize pipelines |
| `list_projects` | List all projects for the agent |
| `create_pipeline` | Create indexing or retrieval pipeline |
| `list_pipelines` | List pipelines in a project |
| `index_documents` | Index documents using an indexing pipeline |
| `query` | Run a RAG query with optional evaluation |

## Docker Deployment

### Build

```bash
docker build -t agentic-rag-green:latest .
```

### Run with Docker Network (Recommended)

```bash
# Create network
docker network create rag-network

# Start ChromaDB
docker run -d --name chromadb --network rag-network -p 8000:8000 chromadb/chroma

# Run green agent with benchmark data
docker run -d --name green-agent \
  --network rag-network \
  --add-host=host.docker.internal:host-gateway \
  -p 9009:9009 \
  -e CHROMA_HOST=chromadb \
  -e CHROMA_PORT=8000 \
  -e PAPERS_URL=https://github.com/arkhai-io/agentic-rag-green/releases/download/v1.0.0/papers.zip \
  -e QA_PAIRS_URL=https://github.com/arkhai-io/agentic-rag-green/releases/download/v1.0.0/grounded_queries.zip \
  --env-file .env \
  agentic-rag-green:latest
```

### Environment Variables

Create a `.env` file (see `sample.env`):

```bash
# Required
NEO4J_URI=bolt://host.docker.internal:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Optional (for coherence evaluation)
OPENROUTER_API_KEY=your_openrouter_key

# Optional: Download benchmark data at startup
PAPERS_URL=https://github.com/arkhai-io/agentic-rag-green/releases/download/v1.0.0/papers.zip
QA_PAIRS_URL=https://github.com/arkhai-io/agentic-rag-green/releases/download/v1.0.0/grounded_queries.zip
```

### Benchmark Data

Benchmark data (PDF papers + QA pairs) can be:
1. **Downloaded at startup** via `PAPERS_URL` and `QA_PAIRS_URL` environment variables
2. **Baked into the image** by placing files in `data/benchmarks/<domain>/`

Expected structure:
```
data/benchmarks/female_longevity/
├── papers/          # PDF files (supports subfolders)
│   ├── paper1.pdf
│   └── topic/
│       └── paper2.pdf
└── qa_pairs.json    # or any .json file with questions
```

QA JSON format:
```json
{
  "questions": [
    {"id": "q1", "question": "...", "ground_truth": "..."}
  ]
}
```

## AgentBeats Integration

This Green Agent is compatible with the AgentBeats platform for running standardized assessments.

### Assessment Configuration

The agent expects an `EvalRequest` with:

```json
{
  "participants": {
    "rag_agent": "http://purple-agent-url:port"
  },
  "config": {
    "agent_name": "my_agent",
    "project_name": "my_project",
    "indexing_pipeline": "pdf_indexer",
    "retrieval_pipeline": "retriever",
    "domain": "female_longevity"
  }
}
```

### Leaderboard Setup

1. Fork the [leaderboard template](https://github.com/RDI-Foundation/agentbeats-leaderboard-template)
2. Configure `scenario.toml`:

```toml
[green_agent]
agentbeats_id = "your-green-agent-id"
env = { 
  NEO4J_URI = "${NEO4J_URI}",
  CHROMA_HOST = "chromadb",
  PAPERS_URL = "https://github.com/.../papers.zip"
}

[[participants]]
agentbeats_id = ""  # Purple agents fill this
name = "rag_agent"
env = {}

[config]
domain = "female_longevity"
agent_name = ""           # Purple fills
project_name = ""         # Purple fills
indexing_pipeline = ""    # Purple fills
retrieval_pipeline = ""   # Purple fills
```

### Evaluation Metrics

The assessment returns:
- **pass_rate**: Percentage of successful queries
- **time_used**: Total assessment time (seconds)
- **avg_bleu**: Average BLEU score
- **avg_rouge_l**: Average ROUGE-L score  
- **avg_coherence**: Average coherence score

## Development

```bash
# Run tests
uv run pytest

# Format code
uv run black src/
uv run isort src/

# Type checking
uv run mypy src/
```

## License

MIT License

## References

- [AgentBeats Platform](https://agentbeats.dev)
- [A2A Protocol Specification](https://a2a-protocol.org)
- [agentic-rag SDK Documentation](https://github.com/arkhai/agentic-rag)
