"""Agentic RAG Green Agent - A2A compatible evaluation benchmark.

This package provides a Green Agent for the AgentBeats competition that:
1. Provides a RAG environment for Purple Agents to interact with
2. Evaluates Purple Agent performance using multiple metrics

Main Components:
---------------
- server: A2A server entry point
- agent: Green Agent business logic
- executor: A2A request handler
- environment: RAG environment (registration, pipelines, queries)
- messenger: A2A communication utilities
- models: Pydantic models for requests/responses
"""

__version__ = "0.1.0"

from .agent import RAGAssessorAgent
from .environment import RAGEnvironment
from .executor import Executor
from .models import (
    ActionRequest,
    ActionResponse,
    ActionType,
    EvalRequest,
    RegistrationResult,
    RegistrationStatus,
)

__all__ = [
    "RAGAssessorAgent",
    "RAGEnvironment",
    "Executor",
    "ActionRequest",
    "ActionResponse",
    "ActionType",
    "EvalRequest",
    "RegistrationResult",
    "RegistrationStatus",
]
