"""Component substitution mappings for pipeline graph building."""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ComponentSubstitution:
    """Defines how one component should be substituted with another."""

    target_component: str  # The component to substitute with
    preserve_config: bool = True  # Whether to preserve original component config
    name_suffix: Optional[str] = None  # Optional suffix for unique naming


# Map of component substitutions: original_component -> substitution_info
COMPONENT_SUBSTITUTIONS: Dict[str, ComponentSubstitution] = {
    "chroma_document_writer": ComponentSubstitution(
        target_component="chroma_embedding_retriever",
        preserve_config=False,
        name_suffix="retriever",
    ),
    "qdrant_document_writer": ComponentSubstitution(
        target_component="qdrant_embedding_retriever",
        preserve_config=False,
        name_suffix="retriever",
    ),
}


def get_component_substitution(component_name: str) -> Optional[ComponentSubstitution]:
    """Get substitution info for a component, if any exists."""
    return COMPONENT_SUBSTITUTIONS.get(component_name)


def should_substitute_component(component_name: str) -> bool:
    """Check if a component should be substituted."""
    return component_name in COMPONENT_SUBSTITUTIONS
