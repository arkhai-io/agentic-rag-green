"""Component registry for mapping component names to Haystack classes."""

from typing import Dict, List, Optional

from ..types import ComponentSpec, ComponentType, DataType


class ComponentRegistry:
    """Registry for managing component specifications."""

    def __init__(self) -> None:
        self._components: Dict[str, ComponentSpec] = {}
        self._initialize_default_components()

    def register_component(self, spec: ComponentSpec) -> None:
        """Register a component specification."""
        self._components[spec.name] = spec

    def get_component_spec(self, name: str) -> Optional[ComponentSpec]:
        """Get component specification by name."""
        return self._components.get(name)

    def list_components(self) -> List[str]:
        """List all registered component names."""
        return list(self._components.keys())

    def get_components_by_type(
        self, component_type: ComponentType
    ) -> List[ComponentSpec]:
        """Get all components of a specific type."""
        return [
            spec
            for spec in self._components.values()
            if spec.component_type == component_type
        ]

    def _initialize_default_components(self) -> None:
        """Initialize default Haystack component mappings."""

        # Converters
        self.register_component(
            ComponentSpec(
                name="pdf_converter",
                component_type=ComponentType.CONVERTER,
                haystack_class="haystack.components.converters.PyPDFToDocument",
                input_types=[DataType.LIST_BYTE_STREAM],
                output_types=[DataType.LIST_DOCUMENT],
                default_config={},
            )
        )

        self.register_component(
            ComponentSpec(
                name="docx_converter",
                component_type=ComponentType.CONVERTER,
                haystack_class="haystack.components.converters.DOCXToDocument",
                input_types=[DataType.LIST_BYTE_STREAM],
                output_types=[DataType.LIST_DOCUMENT],
                default_config={},
            )
        )

        self.register_component(
            ComponentSpec(
                name="markdown_converter",
                component_type=ComponentType.CONVERTER,
                haystack_class="haystack.components.converters.MarkdownToDocument",
                input_types=[DataType.LIST_BYTE_STREAM],
                output_types=[DataType.LIST_DOCUMENT],
                default_config={},
            )
        )

        self.register_component(
            ComponentSpec(
                name="html_converter",
                component_type=ComponentType.CONVERTER,
                haystack_class="haystack.components.converters.HTMLToDocument",
                input_types=[DataType.LIST_BYTE_STREAM],
                output_types=[DataType.LIST_DOCUMENT],
                default_config={},
            )
        )

        self.register_component(
            ComponentSpec(
                name="text_converter",
                component_type=ComponentType.CONVERTER,
                haystack_class="haystack.components.converters.TextFileToDocument",
                input_types=[DataType.LIST_BYTE_STREAM],
                output_types=[DataType.LIST_DOCUMENT],
                default_config={},
            )
        )

        # Chunkers/Splitters - Only verified existing ones
        self.register_component(
            ComponentSpec(
                name="chunker",
                component_type=ComponentType.CHUNKER,
                haystack_class="haystack.components.preprocessors.DocumentSplitter",
                input_types=[DataType.LIST_DOCUMENT],
                output_types=[DataType.LIST_DOCUMENT],
                default_config={"split_by": "sentence", "split_length": 512},
            )
        )

        # Custom Chunkers
        self.register_component(
            ComponentSpec(
                name="markdown_aware_chunker",
                component_type=ComponentType.CHUNKER,
                haystack_class="agentic_rag.components.chunkers.MarkdownAwareChunker",
                input_types=[DataType.LIST_DOCUMENT],
                output_types=[DataType.LIST_DOCUMENT],
                default_config={"chunk_size": 1000, "chunk_overlap": 100},
            )
        )

        # Embedders - Only verified existing ones
        self.register_component(
            ComponentSpec(
                name="embedder",
                component_type=ComponentType.EMBEDDER,
                haystack_class="haystack.components.embedders.SentenceTransformersTextEmbedder",
                input_types=[DataType.LIST_STRING],
                output_types=[DataType.LIST_LIST_FLOAT],
                default_config={"model": "sentence-transformers/all-MiniLM-L6-v2"},
            )
        )

        self.register_component(
            ComponentSpec(
                name="document_embedder",
                component_type=ComponentType.EMBEDDER,
                haystack_class="haystack.components.embedders.SentenceTransformersDocumentEmbedder",
                input_types=[DataType.LIST_DOCUMENT],
                output_types=[DataType.LIST_DOCUMENT],  # Documents with embeddings
                default_config={"model": "sentence-transformers/all-MiniLM-L6-v2"},
            )
        )

        # Document Stores are handled separately as dependencies, not pipeline components

        # Retrievers

        # Chroma Retrievers
        self.register_component(
            ComponentSpec(
                name="chroma_embedding_retriever",
                component_type=ComponentType.RETRIEVER,
                haystack_class="haystack_integrations.retrievers.chroma.ChromaEmbeddingRetriever",
                input_types=[DataType.LIST_FLOAT],
                output_types=[DataType.LIST_DOCUMENT],
                dependencies=[],  # Document store will be passed during component creation
                default_config={"top_k": 10},
            )
        )

        # Generators - Only verified existing ones
        self.register_component(
            ComponentSpec(
                name="generator",
                component_type=ComponentType.GENERATOR,
                haystack_class="haystack.components.generators.OpenAIGenerator",
                input_types=[DataType.LIST_DOCUMENT, DataType.STRING],
                output_types=[DataType.STRING],
                dependencies=[],  # Retriever will be connected via pipeline
                default_config={"model": "gpt-3.5-turbo"},
                parallelizable=False,
            )
        )


# Global registry instance
_default_registry: Optional[ComponentRegistry] = None


def get_default_registry() -> ComponentRegistry:
    """Get the default component registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ComponentRegistry()
    return _default_registry
