"""Component registry and factory for Haystack components."""

from .registry import ComponentRegistry, get_default_registry

__all__ = [
    "ComponentRegistry",
    "get_default_registry",
]
