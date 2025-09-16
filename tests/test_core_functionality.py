"""Test core functionality without requiring all Haystack dependencies."""

import pytest

from agentic_rag import PipelineFactory, list_available_components


class TestCoreFunctionality:
    """Test core system functionality."""

    def test_available_components(self) -> None:
        """Test that component listing works."""
        available = list_available_components()

        # Check expected categories
        expected_categories = [
            "CONVERTER",
            "CHUNKER",
            "EMBEDDER",
            "RETRIEVER",
            "GENERATOR",
        ]
        for category in expected_categories:
            assert category in available, f"Missing category: {category}"

        # Check specific components
        assert "PDF" in available["CONVERTER"]
        assert "CHROMA_EMBEDDING" in available["RETRIEVER"]

        # Check custom components
        assert "MARKDOWN_AWARE" in available["CHUNKER"]

    def test_factory_creation(self) -> None:
        """Test that factory can be created."""
        factory = PipelineFactory()
        assert factory is not None
        assert factory.registry is not None

    def test_component_parsing(self) -> None:
        """Test parsing component specifications."""
        factory = PipelineFactory()

        # Test valid parsing
        test_cases = [
            ({"type": "CONVERTER.PDF"}, "pdf_converter"),
            ({"type": "CHUNKER.DOCUMENT_SPLITTER"}, "chunker"),
            ({"type": "CHUNKER.MARKDOWN_AWARE"}, "markdown_aware_chunker"),
            ({"type": "EMBEDDER.SENTENCE_TRANSFORMERS"}, "embedder"),
            ({"type": "RETRIEVER.CHROMA_EMBEDDING"}, "chroma_embedding_retriever"),
            ({"type": "GENERATOR.OPENAI"}, "generator"),
        ]

        for spec_dict, expected_name in test_cases:
            result = factory._parse_component_spec(spec_dict)
            assert result == expected_name, f"Expected {expected_name}, got {result}"

    def test_component_lookup(self) -> None:
        """Test looking up component specifications."""
        factory = PipelineFactory()

        # Test valid lookups
        test_components = [
            "pdf_converter",
            "chunker",
            "markdown_aware_chunker",
            "embedder",
            "chroma_embedding_retriever",
            "generator",
        ]

        for component_name in test_components:
            spec = factory.registry.get_component_spec(component_name)
            assert spec is not None, f"Component {component_name} not found"
            assert spec.name == component_name
            assert spec.haystack_class is not None
            assert len(spec.haystack_class) > 0

    def test_invalid_specifications(self) -> None:
        """Test error handling for invalid specifications."""
        factory = PipelineFactory()

        # Test invalid category
        with pytest.raises(ValueError, match="Invalid component specification"):
            factory._parse_component_spec({"type": "INVALID.PDF"})

        # Test invalid component type
        with pytest.raises(ValueError, match="Invalid component specification"):
            factory._parse_component_spec({"type": "CONVERTER.INVALID"})

        # Test missing type key
        with pytest.raises(ValueError, match="Component spec must have 'type' key"):
            factory._parse_component_spec({"invalid": "spec"})

        # Test wrong format
        with pytest.raises(
            ValueError, match="Component type must be in format 'CATEGORY.TYPE'"
        ):
            factory._parse_component_spec({"type": "INVALID_FORMAT"})

    def test_pipeline_spec_creation_without_building(self) -> None:
        """Test creating pipeline specs without building Haystack pipelines."""
        factory = PipelineFactory()

        # Test parsing multiple components
        component_specs = [
            {"type": "CONVERTER.PDF"},
            {"type": "CHUNKER.DOCUMENT_SPLITTER"},
            {"type": "EMBEDDER.SENTENCE_TRANSFORMERS"},
        ]

        # Parse each component manually (like factory does)
        component_specs_list = []
        component_configs = {}

        for spec_item in component_specs:
            component_name = factory._parse_component_spec(spec_item)
            spec = factory.registry.get_component_spec(component_name)
            assert spec is not None
            component_specs_list.append(spec)
            component_configs[component_name] = {}

        # Check we got the right components
        assert len(component_specs_list) == 3
        assert component_specs_list[0].name == "pdf_converter"
        assert component_specs_list[1].name == "chunker"
        assert component_specs_list[2].name == "embedder"

    def test_config_merging_logic(self) -> None:
        """Test configuration merging without building components."""

        factory = PipelineFactory()
        spec = factory.registry.get_component_spec("embedder")

        # Test config merging
        user_config = {"batch_size": 32}

        # Check that default config exists
        assert "model" in spec.default_config
        default_model = spec.default_config["model"]

        # Test the merging logic (without actually creating component)
        merged_config = {**spec.default_config, **user_config}

        assert "model" in merged_config
        assert merged_config["model"] == default_model  # From default
        assert "batch_size" in merged_config
        assert merged_config["batch_size"] == 32  # From user
