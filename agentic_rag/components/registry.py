"""Component registry for mapping component names to Haystack classes."""

from typing import Dict, List, Optional

from ..types import ComponentSpec, ComponentType, DataType, PipelineUsage


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
                pipeline_usage=PipelineUsage.INDEXING,
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
                pipeline_usage=PipelineUsage.INDEXING,
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
                pipeline_usage=PipelineUsage.INDEXING,
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
                pipeline_usage=PipelineUsage.INDEXING,
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
                pipeline_usage=PipelineUsage.INDEXING,
                default_config={},
            )
        )

        self.register_component(
            ComponentSpec(
                name="marker_pdf_converter",
                component_type=ComponentType.CONVERTER,
                haystack_class="agentic_rag.components.converters.MarkerPDFToDocument",
                input_types=[DataType.LIST_BYTE_STREAM],
                output_types=[DataType.LIST_DOCUMENT],
                pipeline_usage=PipelineUsage.INDEXING,
                default_config={
                    "languages": "en",
                    "output_format": "markdown",
                    "store_full_path": False,
                },
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
                pipeline_usage=PipelineUsage.INDEXING,
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
                pipeline_usage=PipelineUsage.INDEXING,
                default_config={"chunk_size": 1000, "chunk_overlap": 100},
            )
        )

        self.register_component(
            ComponentSpec(
                name="semantic_chunker",
                component_type=ComponentType.CHUNKER,
                haystack_class="agentic_rag.components.chunkers.SemanticChunker",
                input_types=[DataType.LIST_DOCUMENT],
                output_types=[DataType.LIST_DOCUMENT],
                pipeline_usage=PipelineUsage.INDEXING,
                default_config={
                    "min_chunk_size": 200,
                    "max_chunk_size": 1000,
                    "overlap_size": 50,
                },
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
                pipeline_usage=PipelineUsage.RETRIEVAL,
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
                pipeline_usage=PipelineUsage.INDEXING,
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
                haystack_class="haystack_integrations.components.retrievers.chroma.ChromaEmbeddingRetriever",
                input_types=[DataType.LIST_FLOAT],
                output_types=[DataType.LIST_DOCUMENT],
                pipeline_usage=PipelineUsage.RETRIEVAL,
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
                pipeline_usage=PipelineUsage.RETRIEVAL,
                dependencies=[],  # Retriever will be connected via pipeline
                default_config={"model": "gpt-3.5-turbo"},
                parallelizable=False,
            )
        )

        # Writers - Document indexing components
        self.register_component(
            ComponentSpec(
                name="document_writer",
                component_type=ComponentType.WRITER,
                haystack_class="haystack.components.writers.DocumentWriter",
                input_types=[DataType.LIST_DOCUMENT],
                output_types=[
                    DataType.DICT
                ],  # Returns metadata about written documents
                pipeline_usage=PipelineUsage.INDEXING,
                dependencies=[],  # Document store will be passed during component creation
                default_config={},
                parallelizable=True,
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
