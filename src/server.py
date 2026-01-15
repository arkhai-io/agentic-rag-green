"""A2A Server Entry Point for the Green Agent.

This is the main entry point for the Green Agent server. It sets up:
1. Agent Card - Metadata describing the agent's capabilities
2. A2A Server - HTTP server that handles A2A protocol requests
3. Request Handler - Routes incoming requests to the Executor

Usage:
------
    # Run locally
    python -m src.server --host 0.0.0.0 --port 9009
    
    # Or with uv
    uv run src/server.py --host 0.0.0.0 --port 9009
    
    # Docker (via Dockerfile ENTRYPOINT)
    docker run -p 9009:9009 agentic-rag-green

Command Line Arguments:
-----------------------
    --host: Host to bind the server (default: 127.0.0.1)
    --port: Port to bind the server (default: 9009)
    --card-url: URL to advertise in the agent card (for external access)

Agent Card:
-----------
The Agent Card is metadata that describes this agent to the A2A protocol.
It includes:
- name: Agent display name
- description: What the agent does
- url: Endpoint URL for A2A communication
- version: Agent version
- skills: List of capabilities/skills the agent provides
- capabilities: Protocol features supported (streaming, etc.)

When another agent or the AgentBeats platform wants to communicate with
this agent, they first fetch the agent card to learn about its capabilities.
"""

import argparse

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from .executor import Executor


# =============================================================================
# AGENT CARD CONFIGURATION
# =============================================================================

def create_agent_card(host: str, port: int, card_url: str | None) -> AgentCard:
    """
    Create the Agent Card for this Green Agent.
    
    The Agent Card describes this agent's capabilities to the A2A protocol.
    Other agents and platforms use this to understand what this agent can do.
    
    Args:
        host: Server host address
        port: Server port number
        card_url: Optional external URL (for Docker/cloud deployments)
    
    Returns:
        AgentCard: Configured agent card
    """
    # Define the RAG Environment skill
    rag_environment_skill = AgentSkill(
        id="rag-environment",
        name="RAG Environment",
        description=(
            "Provides a RAG (Retrieval-Augmented Generation) environment "
            "for AI agents to interact with. Supports agent registration, "
            "project management, pipeline creation, document indexing, "
            "and query execution."
        ),
        tags=[
            "rag",
            "retrieval",
            "generation",
            "evaluation",
            "benchmark",
            "environment",
        ],
        examples=[
            # Registration
            '{"action": "register", "params": {"agent_name": "my_agent"}}',
            # Create project
            '{"action": "create_project", "params": {"project_name": "my_project"}}',
            # Create pipeline
            '{"action": "create_pipeline", "params": {"pipeline_name": "my_pipeline", "project_name": "my_project", "components": [{"type": "chunker", "name": "sentence_chunker"}]}}',
            # Query
            '{"action": "query", "params": {"pipeline_name": "my_pipeline", "query": "What is machine learning?"}}',
        ],
    )
    
    # Define the Assessment skill
    assessment_skill = AgentSkill(
        id="rag-assessment",
        name="RAG Assessment",
        description=(
            "Evaluates RAG agents using standardized benchmarks. "
            "Sends tasks to participant agents, evaluates responses "
            "using metrics like ROUGE, BLEU, and LLM-as-Judge, "
            "and reports scores to AgentBeats leaderboards."
        ),
        tags=[
            "assessment",
            "evaluation",
            "benchmark",
            "leaderboard",
            "agentbeats",
        ],
        examples=[
            # Assessment request format (from AgentBeats)
            '{"participants": {"rag_agent": "http://agent:9010"}, "config": {"domain": "machine_learning", "num_tasks": 10}}',
        ],
    )

    # Build the agent card
    agent_card = AgentCard(
        name="Agentic RAG Green Agent",
        description=(
            "Green Agent for RAG (Retrieval-Augmented Generation) assessment. "
            "Provides a RAG environment for agents to interact with, and "
            "evaluates agent performance using multiple metrics including "
            "ROUGE, BLEU, METEOR, and LLM-as-Judge evaluations. "
            "Part of the AgentBeats competition platform."
        ),
        url=card_url or f"http://{host}:{port}/",
        version="0.1.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[rag_environment_skill, assessment_skill],
    )
    
    return agent_card


# =============================================================================
# SERVER SETUP
# =============================================================================

def create_server(host: str, port: int, card_url: str | None) -> A2AStarletteApplication:
    """
    Create the A2A server application.
    
    Sets up:
    - Agent Card (metadata)
    - Executor (request handling logic)
    - Task Store (in-memory task tracking)
    - Request Handler (A2A protocol handling)
    
    Args:
        host: Server host address
        port: Server port number
        card_url: Optional external URL
    
    Returns:
        A2AStarletteApplication: Configured server application
    """
    # Create agent card
    agent_card = create_agent_card(host, port, card_url)
    
    # Create request handler with executor and task store
    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    
    # Create A2A server application
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    
    return server


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main entry point for the Green Agent server.
    
    Parses command line arguments and starts the uvicorn server.
    """
    parser = argparse.ArgumentParser(
        description="Run the Agentic RAG Green Agent A2A server."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9009,
        help="Port to bind the server (default: 9009)",
    )
    parser.add_argument(
        "--card-url",
        type=str,
        default=None,
        help="URL to advertise in the agent card (for external access)",
    )
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   🟢 Agentic RAG Green Agent                                     ║
║                                                                  ║
║   A2A Server for RAG Evaluation Benchmark                        ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║   Host: {args.host:<54} ║
║   Port: {args.port:<54} ║
║   Card URL: {(args.card_url or f'http://{args.host}:{args.port}/'):<50} ║
║                                                                  ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║   Skills:                                                        ║
║   • RAG Environment - Register, create pipelines, query          ║
║   • RAG Assessment - Evaluate agents with benchmarks             ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """)

    # Create and run server
    server = create_server(args.host, args.port, args.card_url)
    uvicorn.run(server.build(), host=args.host, port=args.port, timeout_keep_alive=300)


if __name__ == "__main__":
    main()
