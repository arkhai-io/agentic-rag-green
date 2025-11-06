"""Component registry and factory for Haystack components."""

# Custom components
from .chunkers import MarkdownAwareChunker, SemanticChunker
from .evaluators import GoldStandardEvaluator, ReferenceFreeEvaluator
from .generators import OpenRouterGenerator
from .neo4j_manager import GraphStore
from .registry import ComponentRegistry, get_default_registry
from .secrets import Secrets

__all__ = [
    "ComponentRegistry",
    "get_default_registry",
    "GoldStandardEvaluator",
    "GraphStore",
    "MarkdownAwareChunker",
    "OpenRouterGenerator",
    "ReferenceFreeEvaluator",
    "Secrets",
    "SemanticChunker",
]
