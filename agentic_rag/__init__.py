"""Agentic RAG - A Python package for agentic retrieval-augmented generation."""

from .components import ComponentRegistry, get_default_registry
from .pipeline import PipelineFactory
from .types import (
    DOCUMENT_STORE,
    ComponentSpec,
    ComponentType,
    DataType,
    PipelineSpec,
    list_available_components,
)

__version__: str = "0.1.0"
__author__: str = "Vardhan Shorewala"
__email__: str = "vardhanshorewala@berkeley.edu"

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "PipelineFactory",
    "ComponentRegistry",
    "get_default_registry",
    "ComponentSpec",
    "PipelineSpec",
    "DataType",
    "ComponentType",
    "DOCUMENT_STORE",
    "list_available_components",
]
