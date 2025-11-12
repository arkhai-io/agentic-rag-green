"""Test custom MarkdownAwareChunker component."""

from agentic_rag import PipelineFactory, list_available_components
from agentic_rag.components.chunkers import MarkdownAwareChunker


class TestCustomComponentsPipeline:
    """Test MarkdownAwareChunker component."""

    def setup_method(self):
        """Set up test fixtures."""
        # Reset singleton instances before each test
        from agentic_rag.components import GraphStore
        from agentic_rag.pipeline import PipelineFactory, PipelineRunner
        from agentic_rag.pipeline.storage import GraphStorage

        PipelineFactory.reset_instance()
        PipelineRunner.reset_instance()
        GraphStore.reset_instance()
        GraphStorage.reset_instance()

    def teardown_method(self):
        """Clean up after each test."""
        # Reset singleton instances after each test
        from agentic_rag.components import GraphStore
        from agentic_rag.pipeline import PipelineFactory, PipelineRunner
        from agentic_rag.pipeline.storage import GraphStorage

        PipelineFactory.reset_instance()
        PipelineRunner.reset_instance()
        GraphStore.reset_instance()
        GraphStorage.reset_instance()

    def test_markdown_chunker_available_in_registry(self):
        """Test that MarkdownAwareChunker is registered and available."""
        from unittest.mock import MagicMock

        from agentic_rag.components import GraphStore

        available = list_available_components()

        # Check that CHUNKER category exists and includes our custom component
        assert "CHUNKER" in available
        assert "MARKDOWN_AWARE" in available["CHUNKER"]

        # Test component can be retrieved from registry
        mock_graph_store = MagicMock(spec=GraphStore)
        factory = PipelineFactory(graph_store=mock_graph_store)
        spec = factory.registry.get_component_spec("markdown_aware_chunker")

        assert spec is not None
        assert spec.name == "markdown_aware_chunker"
        assert (
            spec.haystack_class
            == "agentic_rag.components.chunkers.MarkdownAwareChunker"
        )
        assert spec.component_type.value == "chunker"

    def test_markdown_chunker_direct_import(self):
        """Test that MarkdownAwareChunker can be imported and used directly."""
        from haystack import Document

        # Create test document with markdown content
        markdown_content = """# Main Title

This is some content under the main title that is long enough to exceed the chunk size limit. It has multiple paragraphs and enough text to force splitting.

## Subtitle

More content under the subtitle that should be chunked appropriately. This section also has enough content to warrant its own chunk when we set a small chunk size.

### Sub-subtitle

Final content section with additional text to make sure we have enough content to trigger the chunking algorithm properly."""

        document = Document(content=markdown_content, meta={"source": "test.md"})

        # Create chunker instance with smaller chunk size to force splitting
        chunker = MarkdownAwareChunker(chunk_size=120, chunk_overlap=20)

        # Run chunking
        result = chunker.run(documents=[document])
        chunks = result["documents"]

        # Verify chunking worked
        assert len(chunks) > 1
        assert all(chunk.content.strip() for chunk in chunks)
        assert all(chunk.meta["source"] == "test.md" for chunk in chunks)

    def test_custom_component_enum_parsing(self):
        """Test that CHUNKER.MARKDOWN_AWARE enum is correctly parsed."""
        from unittest.mock import MagicMock

        from agentic_rag.components import GraphStore

        mock_graph_store = MagicMock(spec=GraphStore)
        factory = PipelineFactory(graph_store=mock_graph_store)

        # Parse the enum-style specification
        parsed_name = factory._parse_component_spec({"type": "CHUNKER.MARKDOWN_AWARE"})

        # Should map to the registered component name
        assert parsed_name == "markdown_aware_chunker"

        # Verify spec exists
        spec = factory.registry.get_component_spec(parsed_name)
        assert spec is not None
        assert spec.component_type.value == "chunker"
