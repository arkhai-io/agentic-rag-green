"""Test custom MarkdownAwareChunker component."""

from agentic_rag import PipelineFactory, list_available_components
from agentic_rag.components.chunkers import MarkdownAwareChunker


class TestCustomComponentsPipeline:
    """Test MarkdownAwareChunker component."""

    def test_markdown_chunker_available_in_registry(self):
        """Test that MarkdownAwareChunker is registered and available."""
        available = list_available_components()

        # Check that CHUNKER category exists and includes our custom component
        assert "CHUNKER" in available
        assert "MARKDOWN_AWARE" in available["CHUNKER"]

        # Test component can be retrieved from registry
        factory = PipelineFactory()
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
        factory = PipelineFactory()

        # Parse the enum-style specification
        parsed_name = factory._parse_component_spec({"type": "CHUNKER.MARKDOWN_AWARE"})

        # Should map to the registered component name
        assert parsed_name == "markdown_aware_chunker"

        # Verify spec exists
        spec = factory.registry.get_component_spec(parsed_name)
        assert spec is not None
        assert spec.component_type.value == "chunker"
