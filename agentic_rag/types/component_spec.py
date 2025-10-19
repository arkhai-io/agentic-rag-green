"""Component specification and instance definitions."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

from .data_types import ComponentType, DataType, PipelineUsage


@dataclass
class ComponentSpec:
    """Specification for a pipeline component."""

    name: str
    component_type: ComponentType
    haystack_class: str
    input_types: List[DataType]
    output_types: List[DataType]
    pipeline_usage: PipelineUsage = PipelineUsage.BOTH
    default_config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    parallelizable: bool = True

    # NEW: Store the final merged config right here
    runtime_config: Dict[str, Any] = field(default_factory=dict)

    # Store the full type string (e.g., "EMBEDDER.SENTENCE_TRANSFORMERS_DOC")
    full_type: str = ""

    def is_compatible_input(self, data_type: DataType) -> bool:
        """Check if a data type is compatible with this component's inputs."""
        return data_type in self.input_types

    def configure(self, user_config: Dict[str, Any]) -> "ComponentSpec":
        """Merge user config with defaults and store as runtime config."""
        self.runtime_config = {**self.default_config, **user_config}
        return self

    def get_config(self) -> Dict[str, Any]:
        """Get the runtime config, falling back to default if not configured."""
        return self.runtime_config if self.runtime_config else self.default_config


def _create_chroma_document_store(root_dir: str = ".") -> Any:
    """Create a ChromaDocumentStore with local persistence."""
    try:
        from haystack_integrations.document_stores.chroma import ChromaDocumentStore

        # Create data/chroma directory if it doesn't exist
        chroma_path = os.path.join(root_dir, "data", "chroma")
        os.makedirs(chroma_path, exist_ok=True)

        # Initialize ChromaDocumentStore with local persistence
        # This matches the Haystack documentation pattern
        return ChromaDocumentStore(persist_path=chroma_path)
    except ImportError as e:
        raise ImportError(
            f"ChromaDocumentStore not available. Install with: pip install chroma-haystack. Error: {e}"
        )


def create_haystack_component(spec: ComponentSpec) -> Any:
    """Create a Haystack component from specification - handles config internally."""
    # Get the final config from ComponentSpec
    config = spec.get_config().copy()  # Copy to avoid modifying original

    # Special handling for Chroma components - inject document store
    if (
        "ChromaEmbeddingRetriever" in spec.haystack_class
        or "DocumentWriter" in spec.haystack_class
    ):
        # Get root directory from config or use current directory
        config.pop("model", None)  # Remove model if present for these components
        root_dir = config.pop("root_dir", ".")
        document_store = _create_chroma_document_store(root_dir)
        config["document_store"] = document_store

    # Dynamic import and instantiation
    module_path, class_name = spec.haystack_class.rsplit(".", 1)

    try:
        import importlib

        module = importlib.import_module(module_path)
        component_class = getattr(module, class_name)
        return component_class(**config)
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Failed to import Haystack component {spec.haystack_class}: {e}"
        )
