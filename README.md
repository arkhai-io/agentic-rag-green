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
- Poetry
- Neo4j database (local or Aura)
- OpenRouter API key (for LLM-based generation and evaluation)

### Setup

```bash
# Clone repository
git clone https://github.com/arkhai/agentic-rag-green.git
cd agentic-rag-green

# Initialize submodule
git submodule update --init --recursive

# Install dependencies
poetry install

# Configure environment
cp .env.example .env
# Edit .env with your credentials:
#   NEO4J_URI=neo4j+s://your-instance.neo4j.io
#   NEO4J_USERNAME=neo4j
#   NEO4J_PASSWORD=your-password
#   OPENROUTER_API_KEY=your-api-key
```

## Usage

### Starting the Server

```bash
poetry run python -m src.server --host 0.0.0.0 --port 9009
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

```bash
# Build image
docker build -t ghcr.io/arkhai/agentic-rag-green:latest .

# Run container
docker run -p 9009:9009 \
  -e NEO4J_URI=neo4j+s://your-instance.neo4j.io \
  -e NEO4J_USERNAME=neo4j \
  -e NEO4J_PASSWORD=your-password \
  -e OPENROUTER_API_KEY=your-api-key \
  ghcr.io/arkhai/agentic-rag-green:latest
```

## AgentBeats Integration

This Green Agent is compatible with the AgentBeats platform for running standardized assessments. Register the agent on AgentBeats and configure a leaderboard to track Purple Agent performance on RAG benchmarks.

For assessment configuration, the agent expects:

```json
{
  "participants": {
    "rag_agent": "http://purple-agent-url:port"
  },
  "config": {
    "domain": "machine_learning",
    "num_tasks": 10,
    "metrics": ["rouge", "bleu", "answer_quality"]
  }
}
```

## Development

```bash
# Run tests
poetry run pytest

# Format code
poetry run black src/
poetry run isort src/

# Type checking
poetry run mypy src/
```

## License

MIT License

## References

- [AgentBeats Platform](https://agentbeats.dev)
- [A2A Protocol Specification](https://a2a-protocol.org)
- [agentic-rag SDK Documentation](https://github.com/arkhai/agentic-rag)
