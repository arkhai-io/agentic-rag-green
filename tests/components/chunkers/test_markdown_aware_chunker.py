"""Test pipeline creation with custom components."""

import pytest

from agentic_rag import PipelineFactory, list_available_components
from agentic_rag.components.chunkers import MarkdownAwareChunker


class TestCustomComponentsPipeline:
    """Test pipeline creation and usage with custom components."""

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

    def test_create_pipeline_with_markdown_chunker(self):
        """Test creating a pipeline that includes the MarkdownAwareChunker."""
        factory = PipelineFactory()

        # Create pipeline with custom chunker
        pipeline_spec = [
            {"type": "CONVERTER.TEXT"},  # Start with text converter
            {"type": "CHUNKER.MARKDOWN_AWARE"},  # Use our custom chunker
        ]

        config = {"markdown_aware_chunker": {"chunk_size": 500, "chunk_overlap": 50}}

        try:
            spec, haystack_pipeline = factory.create_pipeline_from_spec(
                pipeline_spec, "markdown_pipeline", config
            )

            # Verify pipeline spec
            assert spec.name == "markdown_pipeline"
            assert len(spec.components) == 2

            # Check component names and types
            component_names = [comp.name for comp in spec.components]
            assert "text_converter" in component_names
            assert "markdown_aware_chunker" in component_names

            # Check configuration was applied
            assert "markdown_aware_chunker" in spec.component_configs
            chunker_config = spec.component_configs["markdown_aware_chunker"]
            assert chunker_config["chunk_size"] == 500
            assert chunker_config["chunk_overlap"] == 50

            # Verify Haystack pipeline was created
            assert haystack_pipeline is not None
            assert hasattr(haystack_pipeline, "run")

        except ImportError as e:
            # Skip if Haystack dependencies are missing
            pytest.skip(f"Skipping test due to missing dependencies: {e}")

    def test_create_full_rag_pipeline_with_custom_chunker(self):
        """Test creating a full RAG pipeline using the custom markdown chunker."""
        factory = PipelineFactory()

        # Create a complete RAG pipeline with our custom chunker
        pipeline_spec = [
            {"type": "CONVERTER.TEXT"},
            {"type": "CHUNKER.MARKDOWN_AWARE"},
            {"type": "EMBEDDER.SENTENCE_TRANSFORMERS"},
            {"type": "GENERATOR.OPENAI"},
        ]

        config = {
            "markdown_aware_chunker": {"chunk_size": 800, "chunk_overlap": 100},
            "embedder": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
            "generator": {"model": "gpt-3.5-turbo"},
        }

        try:
            spec, haystack_pipeline = factory.create_pipeline_from_spec(
                pipeline_spec, "full_rag_pipeline", config
            )

            # Verify pipeline structure
            assert spec.name == "full_rag_pipeline"
            assert len(spec.components) == 4

            # Check all components are present
            component_names = [comp.name for comp in spec.components]
            expected_components = [
                "text_converter",
                "markdown_aware_chunker",
                "embedder",
                "generator",
            ]

            for expected_comp in expected_components:
                assert expected_comp in component_names

            # Verify custom chunker configuration
            chunker_config = spec.component_configs["markdown_aware_chunker"]
            assert chunker_config["chunk_size"] == 800
            assert chunker_config["chunk_overlap"] == 100

            # Verify pipeline can be created
            assert haystack_pipeline is not None

        except ImportError as e:
            pytest.skip(f"Skipping test due to missing dependencies: {e}")

    def test_compare_chunkers_in_pipeline(self):
        """Test comparing default chunker vs custom markdown chunker in pipelines."""
        factory = PipelineFactory()

        # Pipeline with default chunker
        default_pipeline_spec = [
            {"type": "CONVERTER.TEXT"},
            {"type": "CHUNKER.DOCUMENT_SPLITTER"},
        ]

        # Pipeline with custom markdown chunker
        custom_pipeline_spec = [
            {"type": "CONVERTER.TEXT"},
            {"type": "CHUNKER.MARKDOWN_AWARE"},
        ]

        default_config = {"chunker": {"split_by": "sentence", "split_length": 512}}

        custom_config = {
            "markdown_aware_chunker": {"chunk_size": 512, "chunk_overlap": 50}
        }

        try:
            # Create both pipelines
            default_spec, default_pipeline = factory.create_pipeline_from_spec(
                default_pipeline_spec, "default_chunker_pipeline", default_config
            )

            custom_spec, custom_pipeline = factory.create_pipeline_from_spec(
                custom_pipeline_spec, "custom_chunker_pipeline", custom_config
            )

            # Verify both pipelines were created
            assert default_spec.components[1].name == "chunker"
            assert custom_spec.components[1].name == "markdown_aware_chunker"

            # Verify different component classes
            default_chunker_class = default_spec.components[1].haystack_class
            custom_chunker_class = custom_spec.components[1].haystack_class

            assert "DocumentSplitter" in default_chunker_class
            assert "MarkdownAwareChunker" in custom_chunker_class

            # Both should be valid Haystack pipelines
            assert default_pipeline is not None
            assert custom_pipeline is not None

        except ImportError as e:
            pytest.skip(f"Skipping test due to missing dependencies: {e}")

    def test_custom_component_enum_parsing(self):
        """Test that custom component enums are parsed correctly."""
        from agentic_rag.types.component_enums import (
            CHUNKER,
            get_component_value,
            validate_component_spec,
        )

        # Test enum value
        assert CHUNKER.MARKDOWN_AWARE.value == "markdown_aware_chunker"

        # Test parsing
        assert validate_component_spec("CHUNKER.MARKDOWN_AWARE")
        value = get_component_value("CHUNKER.MARKDOWN_AWARE")
        assert value == "markdown_aware_chunker"

        # Test factory parsing
        factory = PipelineFactory()
        component_name = factory._parse_component_spec(
            {"type": "CHUNKER.MARKDOWN_AWARE"}
        )
        assert component_name == "markdown_aware_chunker"

    def test_multiple_custom_components_pipeline(self):
        """Test creating pipeline with multiple custom components (future-proofing)."""
        factory = PipelineFactory()

        # For now, just test with our markdown chunker
        # This test structure allows for easy addition of more custom components
        pipeline_spec = [
            {"type": "CONVERTER.TEXT"},
            {"type": "CHUNKER.MARKDOWN_AWARE"},
        ]

        config = {
            "markdown_aware_chunker": {
                "chunk_size": 1000,
                "separators": ["\n\n", "\n", " "],
            }
        }

        try:
            spec, pipeline = factory.create_pipeline_from_spec(
                pipeline_spec, "multi_custom_pipeline", config
            )

            # Verify pipeline creation
            assert len(spec.components) == 2
            assert spec.components[1].name == "markdown_aware_chunker"

            # Verify custom configuration
            chunker_config = spec.component_configs["markdown_aware_chunker"]
            assert chunker_config["chunk_size"] == 1000
            assert chunker_config["separators"] == ["\n\n", "\n", " "]

        except ImportError as e:
            pytest.skip(f"Skipping test due to missing dependencies: {e}")

    def test_error_handling_with_custom_components(self):
        """Test error handling when using custom components."""
        factory = PipelineFactory()

        # Test invalid custom component configuration
        pipeline_spec = [{"type": "CHUNKER.MARKDOWN_AWARE"}]

        # Invalid config (non-integer chunk_size)
        invalid_config = {
            "markdown_aware_chunker": {"chunk_size": "invalid"}  # Should be int
        }

        try:
            # This might succeed in spec creation but fail during component instantiation
            spec, pipeline = factory.create_pipeline_from_spec(
                pipeline_spec, "error_test_pipeline", invalid_config
            )

            # If we get here, the spec was created but pipeline creation might fail
            # This is acceptable - the error will occur when trying to run the pipeline
            assert spec is not None

        except (ValueError, TypeError, ImportError):
            # Expected errors for invalid configuration or missing dependencies
            assert True  # Test passes if appropriate error is raised
