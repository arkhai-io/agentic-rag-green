"""Component registry and factory for Haystack components."""

# Custom components
from .chunkers import MarkdownAwareChunker, SemanticChunker
from .registry import ComponentRegistry, get_default_registry

__all__ = [
    "ComponentRegistry",
    "get_default_registry",
    "MarkdownAwareChunker",
    "SemanticChunker",
]
