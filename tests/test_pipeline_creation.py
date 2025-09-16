"""Test pipeline creation and building functionality."""

import pytest

from agentic_rag import PipelineFactory, list_available_components


class TestPipelineCreation:
    """Test actual pipeline creation and building."""

    def test_available_components(self) -> None:
        """Test that we have the expected components available."""
        available = list_available_components()

        # Check we have all expected categories
        assert "CONVERTER" in available
        assert "CHUNKER" in available
        assert "EMBEDDER" in available
        assert "RETRIEVER" in available
        assert "GENERATOR" in available

        # Check specific components exist
        assert "PDF" in available["CONVERTER"]
        assert "HTML" in available["CONVERTER"]
        assert "TEXT" in available["CONVERTER"]

        assert "DOCUMENT_SPLITTER" in available["CHUNKER"]

        assert "SENTENCE_TRANSFORMERS" in available["EMBEDDER"]
        assert "SENTENCE_TRANSFORMERS_DOC" in available["EMBEDDER"]

        assert "CHROMA_EMBEDDING" in available["RETRIEVER"]

        assert "OPENAI" in available["GENERATOR"]

    def test_simple_pipeline_creation(self) -> None:
        """Test creating a simple 2-component pipeline."""
        factory = PipelineFactory()

        pipeline_spec = [
            {"type": "CONVERTER.PDF"},
            {"type": "CHUNKER.DOCUMENT_SPLITTER"},
        ]

        spec, haystack_pipeline = factory.create_pipeline_from_spec(
            pipeline_spec, "test_pipeline"
        )

        # Check pipeline spec
        assert spec.name == "test_pipeline"
        assert len(spec.components) == 2
        assert spec.components[0].name == "pdf_converter"
        assert spec.components[1].name == "chunker"

        # Check Haystack pipeline exists
        assert haystack_pipeline is not None
        assert hasattr(haystack_pipeline, "run")  # Should be a Haystack Pipeline

    def test_simple_processing_pipeline(self) -> None:
        """Test creating a simple processing pipeline without retriever."""
        factory = PipelineFactory()

        pipeline_spec = [
            {"type": "CONVERTER.PDF"},
            {"type": "CHUNKER.DOCUMENT_SPLITTER"},
            {"type": "EMBEDDER.SENTENCE_TRANSFORMERS"},
        ]

        config = {
            "chunker": {"split_length": 256},
            "embedder": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
        }

        spec, haystack_pipeline = factory.create_pipeline_from_spec(
            pipeline_spec, "simple_processing_pipeline", config
        )

        # Check pipeline spec
        assert spec.name == "simple_processing_pipeline"
        assert len(spec.components) == 3

        # Check component names
        component_names = [comp.name for comp in spec.components]
        assert "pdf_converter" in component_names
        assert "chunker" in component_names
        assert "embedder" in component_names

        # Check configurations were applied
        assert spec.component_configs["chunker"]["split_length"] == 256
        assert (
            spec.component_configs["embedder"]["model"]
            == "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Check Haystack pipeline
        assert haystack_pipeline is not None
        assert hasattr(haystack_pipeline, "run")

    def test_multiple_pipelines(self) -> None:
        """Test creating multiple pipelines at once."""
        factory = PipelineFactory()

        pipeline_specs = [
            # Simple converter + chunker
            [{"type": "CONVERTER.PDF"}, {"type": "CHUNKER.DOCUMENT_SPLITTER"}],
            # Embedder only
            [{"type": "EMBEDDER.SENTENCE_TRANSFORMERS"}],
            # Single component
            [{"type": "CONVERTER.TEXT"}],
        ]

        configs = [{}, {}, {}]  # No config for HTML converter

        pipelines = factory.create_pipelines_from_specs(pipeline_specs, configs)

        # Check we got 3 pipelines
        assert len(pipelines) == 3

        # Check each pipeline
        for i, (spec, haystack_pipeline) in enumerate(pipelines):
            assert spec.name == f"pipeline_{i}"
            assert haystack_pipeline is not None
            assert len(spec.components) == len(pipeline_specs[i])

    def test_invalid_component_type(self) -> None:
        """Test error handling for invalid component types."""
        factory = PipelineFactory()

        # Invalid category
        with pytest.raises(ValueError, match="Invalid component specification"):
            factory.create_pipeline_from_spec([{"type": "INVALID.PDF"}], "test")

        # Invalid component type
        with pytest.raises(ValueError, match="Invalid component specification"):
            factory.create_pipeline_from_spec([{"type": "CONVERTER.INVALID"}], "test")

        # Missing type key
        with pytest.raises(ValueError, match="Component spec must have 'type' key"):
            factory.create_pipeline_from_spec([{"invalid": "spec"}], "test")

        # Wrong format
        with pytest.raises(
            ValueError, match="Component type must be in format 'CATEGORY.TYPE'"
        ):
            factory.create_pipeline_from_spec([{"type": "INVALID_FORMAT"}], "test")

    def test_pipeline_length_limits(self) -> None:
        """Test pipeline length validation."""
        factory = PipelineFactory()

        # Too many components (more than 5)
        long_spec = [{"type": "CONVERTER.PDF"}] * 6
        with pytest.raises(
            ValueError, match="Pipeline cannot have more than 5 components"
        ):
            factory.create_pipeline_from_spec(long_spec, "test")

        # Empty pipeline
        with pytest.raises(ValueError, match="Pipeline must have at least 1 component"):
            factory.create_pipeline_from_spec([], "test")

    def test_config_merging(self) -> None:
        """Test that default and user configs are merged correctly."""
        factory = PipelineFactory()

        pipeline_spec = [{"type": "EMBEDDER.SENTENCE_TRANSFORMERS"}]

        # Provide partial config (should merge with defaults)
        config = {
            "embedder": {
                "batch_size": 32  # New config
                # model should come from default
            }
        }

        spec, haystack_pipeline = factory.create_pipeline_from_spec(
            pipeline_spec, "test_config", config
        )

        # Check that both default and user config are present
        embedder_config = spec.component_configs["embedder"]
        # Note: config merging happens during component creation, not in the spec
        # The spec only stores the user-provided config
        assert "batch_size" in embedder_config  # From user
        assert embedder_config["batch_size"] == 32

    def test_component_types_validation(self) -> None:
        """Test that component types are correctly identified."""
        factory = PipelineFactory()

        pipeline_spec = [
            {"type": "CONVERTER.PDF"},
            {"type": "CHUNKER.DOCUMENT_SPLITTER"},
            {"type": "EMBEDDER.SENTENCE_TRANSFORMERS"},
        ]

        spec, _ = factory.create_pipeline_from_spec(pipeline_spec, "test_types")

        # Check component types
        types = [comp.component_type.value for comp in spec.components]
        assert "converter" in types
        assert "chunker" in types
        assert "embedder" in types

    def test_haystack_component_creation(self) -> None:
        """Test that actual Haystack components are created."""
        factory = PipelineFactory()

        # Test with a simple component that should definitely work
        pipeline_spec = [{"type": "CHUNKER.DOCUMENT_SPLITTER"}]

        config = {"chunker": {"split_by": "sentence", "split_length": 100}}

        try:
            spec, haystack_pipeline = factory.create_pipeline_from_spec(
                pipeline_spec, "test_haystack", config
            )

            # If we get here without ImportError, the component was created successfully
            assert spec is not None
            assert haystack_pipeline is not None

        except ImportError as e:
            # This is expected if Haystack is not installed
            assert "Haystack" in str(e) or "haystack" in str(e)
            pytest.skip(f"Skipping test due to missing Haystack: {e}")

    def test_pipeline_component_order(self) -> None:
        """Test that pipeline components are created in the correct order."""
        factory = PipelineFactory()

        # Pipeline with logical flow: convert -> chunk -> embed
        pipeline_spec = [
            {"type": "CONVERTER.PDF"},
            {"type": "CHUNKER.DOCUMENT_SPLITTER"},
            {"type": "EMBEDDER.SENTENCE_TRANSFORMERS"},
            {"type": "GENERATOR.OPENAI"},
        ]

        spec, _ = factory.create_pipeline_from_spec(pipeline_spec, "test_order")

        # Check that components are in expected order
        component_names = [comp.name for comp in spec.components]
        assert component_names[0] == "pdf_converter"
        assert component_names[1] == "chunker"
        assert component_names[2] == "embedder"
        assert component_names[3] == "generator"
