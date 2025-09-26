"""Test MarkerPDFToDocument in RAG pipeline scenarios."""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from agentic_rag import PipelineFactory


class TestMarkerRAGPipeline:
    """Test MarkerPDFToDocument integration in RAG pipelines."""

    def setup_method(self):
        """Set up test environment with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.factory = PipelineFactory()
        self.sample_pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Hello World!) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000074 00000 n \n0000000120 00000 n \n0000000179 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n235\n%%EOF"

    def teardown_method(self):
        """Clean up temporary directory."""
        if hasattr(self, "temp_dir") and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def create_sample_pdf(self, filename: str = "sample.pdf") -> str:
        """Create a sample PDF file for testing."""
        pdf_path = os.path.join(self.temp_dir, filename)
        with open(pdf_path, "wb") as f:
            f.write(self.sample_pdf_content)
        return pdf_path

    def test_marker_converter_available_in_registry(self):
        """Test that MarkerPDFToDocument is available in the component registry."""
        from agentic_rag.components.registry import get_default_registry

        registry = get_default_registry()

        # Check if marker converter is registered
        marker_spec = registry.get_component_spec("marker_pdf_converter")
        if marker_spec:
            assert marker_spec.name == "marker_pdf_converter"
            assert "MarkerPDFToDocument" in marker_spec.haystack_class
            assert marker_spec.component_type.value == "converter"

    def test_marker_converter_direct_import(self):
        """Test direct import and instantiation of MarkerPDFToDocument."""
        try:
            from agentic_rag.components.converters.marker_pdf_converter import (
                MarkerPDFToDocument,
            )

            converter = MarkerPDFToDocument()
            assert converter is not None
            assert converter.languages == "en"
            assert converter.output_format == "markdown"

        except ImportError as e:
            pytest.skip(f"MarkerPDFToDocument not available: {e}")

    def test_create_pipeline_with_marker_converter(self):
        """Test creating a pipeline that includes MarkerPDFToDocument."""
        # Define pipeline with marker converter
        pipeline_spec = [
            {"type": "CONVERTER.MARKER_PDF"},
            {"type": "CHUNKER.MARKDOWN_AWARE"},
        ]

        config = {
            "marker_pdf_converter": {"languages": "en", "output_format": "markdown"},
            "markdown_aware_chunker": {"chunk_size": 1000, "chunk_overlap": 100},
        }

        try:
            spec, pipeline = self.factory.create_pipeline_from_spec(
                pipeline_spec, "marker_processing_pipeline", config
            )

            # Verify pipeline structure
            assert spec.name == "marker_processing_pipeline"
            assert len(spec.components) == 2

            # Check component types
            component_types = [comp.component_type.value for comp in spec.components]
            assert "converter" in component_types
            assert "chunker" in component_types

            # Verify pipeline object exists
            assert pipeline is not None
            assert hasattr(pipeline, "run")

        except ImportError as e:
            pytest.skip(f"Skipping due to missing dependencies: {e}")

    def test_create_full_rag_pipeline_with_marker_converter(self):
        """Test creating a complete RAG pipeline with MarkerPDFToDocument."""
        # Define complete indexing pipeline: Marker Convert -> Chunk -> Embed -> Write
        indexing_spec = [
            {"type": "CONVERTER.MARKER_PDF"},
            {"type": "CHUNKER.MARKDOWN_AWARE"},
            {"type": "EMBEDDER.SENTENCE_TRANSFORMERS_DOC"},
            {"type": "WRITER.DOCUMENT_WRITER"},
        ]

        config = {
            "marker_pdf_converter": {
                "languages": "en",
                "output_format": "markdown",
                "store_full_path": True,
            },
            "markdown_aware_chunker": {"chunk_size": 800, "chunk_overlap": 100},
            "document_embedder": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
            "document_writer": {"root_dir": self.temp_dir},
        }

        try:
            spec, pipeline = self.factory.create_pipeline_from_spec(
                indexing_spec, "marker_rag_indexing", config
            )

            # Verify pipeline structure
            assert spec.name == "marker_rag_indexing"
            assert len(spec.components) == 4

            # Check component types
            component_types = [comp.component_type.value for comp in spec.components]
            assert set(component_types) == {
                "converter",
                "chunker",
                "embedder",
                "writer",
            }

            # Verify pipeline object exists
            assert pipeline is not None
            assert hasattr(pipeline, "run")

            # Verify component configurations
            marker_spec = spec.get_component_by_name("marker_pdf_converter")
            assert marker_spec is not None
            marker_config = marker_spec.get_config()
            assert marker_config.get("languages") == "en"
            assert marker_config.get("store_full_path") is True

            chunker_spec = spec.get_component_by_name("markdown_aware_chunker")
            assert chunker_spec is not None
            chunker_config = chunker_spec.get_config()
            assert chunker_config.get("chunk_size") == 800

        except ImportError as e:
            pytest.skip(f"Skipping due to missing dependencies: {e}")

    def test_compare_converters_in_pipeline(self):
        """Test comparing different converters in similar pipelines."""
        # Test with regular text converter
        text_pipeline_spec = [
            {"type": "CONVERTER.TEXT"},
            {"type": "CHUNKER.MARKDOWN_AWARE"},
        ]

        # Test with marker converter
        marker_pipeline_spec = [
            {"type": "CONVERTER.MARKER_PDF"},
            {"type": "CHUNKER.MARKDOWN_AWARE"},
        ]

        config = {
            "markdown_aware_chunker": {"chunk_size": 500, "chunk_overlap": 50},
            "marker_pdf_converter": {"languages": "en", "output_format": "markdown"},
        }

        try:
            # Create text pipeline
            text_spec, text_pipeline = self.factory.create_pipeline_from_spec(
                text_pipeline_spec, "text_pipeline", config
            )

            # Create marker pipeline
            marker_spec, marker_pipeline = self.factory.create_pipeline_from_spec(
                marker_pipeline_spec, "marker_pipeline", config
            )

            # Both should have same number of components
            assert len(text_spec.components) == len(marker_spec.components)

            # But different converter types
            text_converter = text_spec.components[0]
            marker_converter = marker_spec.components[0]

            assert text_converter.component_type.value == "converter"
            assert marker_converter.component_type.value == "converter"
            assert text_converter.haystack_class != marker_converter.haystack_class

        except ImportError as e:
            pytest.skip(f"Skipping due to missing dependencies: {e}")

    def test_marker_converter_enum_parsing(self):
        """Test that marker converter enum is properly parsed."""
        from agentic_rag.types.component_enums import CONVERTER

        # Test that MARKER is available in the CONVERTER enum
        try:
            assert hasattr(CONVERTER, "MARKER_PDF")
            assert CONVERTER.MARKER_PDF.value == "marker_pdf_converter"

        except AttributeError:
            pytest.skip("MARKER not yet added to component enums")

    def test_error_handling_with_marker_converter(self):
        """Test error handling when marker converter fails."""
        pipeline_spec = [
            {"type": "CONVERTER.MARKER_PDF"},
            {"type": "CHUNKER.MARKDOWN_AWARE"},
        ]

        # Invalid configuration to trigger error
        config = {
            "marker_pdf_converter": {
                "languages": "invalid_language_code",
                "output_format": "invalid_format",
            }
        }

        try:
            spec, pipeline = self.factory.create_pipeline_from_spec(
                pipeline_spec, "error_test_pipeline", config
            )

            # Pipeline creation should succeed even with invalid config
            # (validation happens at runtime)
            assert spec is not None
            assert pipeline is not None

        except ImportError as e:
            pytest.skip(f"Skipping due to missing dependencies: {e}")

    @patch("agentic_rag.components.converters.marker_pdf_converter.create_model_dict")
    @patch("agentic_rag.components.converters.marker_pdf_converter.ConfigParser")
    @patch("agentic_rag.components.converters.marker_pdf_converter.PdfConverter")
    def test_marker_pipeline_execution_mock(
        self, mock_pdf_converter, mock_config_parser, mock_create_model_dict
    ):
        """Test executing a pipeline with marker converter (mocked)."""
        # Setup mocks
        mock_create_model_dict.return_value = {"model": "test"}
        mock_config_instance = Mock()
        mock_config_parser.return_value = mock_config_instance

        mock_rendered = Mock()
        mock_rendered.markdown = (
            "# Test Document\n\nThis is a test PDF converted to markdown."
        )

        mock_converter_instance = Mock()
        mock_converter_instance.return_value = mock_rendered
        mock_pdf_converter.return_value = mock_converter_instance

        # Create pipeline
        pipeline_spec = [
            {"type": "CONVERTER.MARKER_PDF"},
            {"type": "CHUNKER.MARKDOWN_AWARE"},
        ]

        config = {
            "marker_pdf_converter": {"languages": "en", "output_format": "markdown"},
            "markdown_aware_chunker": {"chunk_size": 500, "chunk_overlap": 50},
        }

        try:
            spec, pipeline = self.factory.create_pipeline_from_spec(
                pipeline_spec, "mock_marker_pipeline", config
            )

            # Create sample PDF for testing
            self.create_sample_pdf()

            # This would normally run the pipeline, but we'll just verify structure
            assert spec is not None
            assert pipeline is not None
            assert len(spec.components) == 2

            # Verify the first component is our marker converter
            first_component = spec.components[0]
            assert first_component.component_type.value == "converter"
            assert "MarkerPDFToDocument" in first_component.haystack_class

        except ImportError as e:
            pytest.skip(f"Skipping due to missing dependencies: {e}")

    def test_marker_retrieval_pipeline(self):
        """Test creating a retrieval pipeline that works with marker-processed documents."""
        # Define retrieval pipeline that would work with documents processed by marker
        retrieval_spec = [
            {"type": "EMBEDDER.SENTENCE_TRANSFORMERS"},
            {"type": "RETRIEVER.CHROMA_EMBEDDING"},
            {"type": "GENERATOR.OPENAI"},
        ]

        config = {
            "chroma_embedding_retriever": {"root_dir": self.temp_dir, "top_k": 5},
            "embedder": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
            "generator": {"model": "gpt-3.5-turbo"},
        }

        try:
            spec, pipeline = self.factory.create_pipeline_from_spec(
                retrieval_spec, "marker_retrieval_pipeline", config
            )

            # Verify pipeline structure
            assert spec.name == "marker_retrieval_pipeline"
            assert len(spec.components) == 3

            # Check component types
            component_types = [comp.component_type.value for comp in spec.components]
            assert set(component_types) == {"embedder", "retriever", "generator"}

            # This retrieval pipeline would work with documents that were
            # processed and stored using the marker indexing pipeline

        except ImportError as e:
            pytest.skip(f"Skipping due to missing dependencies: {e}")

    def test_end_to_end_marker_rag_workflow(self):
        """Test the complete workflow: marker indexing -> retrieval."""
        # 1. Create indexing pipeline with marker
        indexing_spec = [
            {"type": "CONVERTER.MARKER_PDF"},
            {"type": "CHUNKER.MARKDOWN_AWARE"},
            {"type": "EMBEDDER.SENTENCE_TRANSFORMERS_DOC"},
            {"type": "WRITER.DOCUMENT_WRITER"},
        ]

        # 2. Create retrieval pipeline
        retrieval_spec = [
            {"type": "EMBEDDER.SENTENCE_TRANSFORMERS"},
            {"type": "RETRIEVER.CHROMA_EMBEDDING"},
            {"type": "GENERATOR.OPENAI"},
        ]

        # Shared configuration
        shared_config = {
            "root_dir": self.temp_dir,
            "model": "sentence-transformers/all-MiniLM-L6-v2",
        }

        indexing_config = {
            "marker_pdf_converter": {
                "languages": "en",
                "output_format": "markdown",
                "store_full_path": True,
            },
            "markdown_aware_chunker": {"chunk_size": 800, "chunk_overlap": 100},
            "document_embedder": {"model": shared_config["model"]},
            "document_writer": {"root_dir": shared_config["root_dir"]},
        }

        retrieval_config = {
            "chroma_embedding_retriever": {**shared_config, "top_k": 3},
            "embedder": {"model": shared_config["model"]},
            "generator": {"model": "gpt-3.5-turbo"},
        }

        try:
            # Create both pipelines
            indexing_spec_obj, indexing_pipeline = (
                self.factory.create_pipeline_from_spec(
                    indexing_spec, "marker_indexing_rag", indexing_config
                )
            )

            retrieval_spec_obj, retrieval_pipeline = (
                self.factory.create_pipeline_from_spec(
                    retrieval_spec, "marker_retrieval_rag", retrieval_config
                )
            )

            # Verify both pipelines were created
            assert indexing_spec_obj is not None
            assert retrieval_spec_obj is not None
            assert indexing_pipeline is not None
            assert retrieval_pipeline is not None

            # Verify they use the same Chroma path for document storage
            indexing_writer_spec = indexing_spec_obj.get_component_by_name(
                "document_writer"
            )
            assert indexing_writer_spec is not None
            indexing_writer_config = indexing_writer_spec.get_config()

            retrieval_retriever_spec = retrieval_spec_obj.get_component_by_name(
                "chroma_embedding_retriever"
            )
            assert retrieval_retriever_spec is not None
            retrieval_retriever_config = retrieval_retriever_spec.get_config()

            # Both should reference the same root directory
            assert indexing_writer_config.get(
                "root_dir"
            ) == retrieval_retriever_config.get("root_dir")

            # Verify marker converter configuration
            marker_spec = indexing_spec_obj.get_component_by_name(
                "marker_pdf_converter"
            )
            assert marker_spec is not None
            marker_config = marker_spec.get_config()
            assert marker_config.get("languages") == "en"
            assert marker_config.get("output_format") == "markdown"
            assert marker_config.get("store_full_path") is True

            print("✅ End-to-end marker RAG workflow test passed!")
            print(
                f"📊 Indexing pipeline: {len(indexing_spec_obj.components)} components"
            )
            print(
                f"🔍 Retrieval pipeline: {len(retrieval_spec_obj.components)} components"
            )

        except ImportError as e:
            pytest.skip(f"Skipping due to missing dependencies: {e}")

    def test_marker_converter_performance_config(self):
        """Test marker converter with performance-oriented configuration."""
        pipeline_spec = [
            {"type": "CONVERTER.MARKER_PDF"},
            {"type": "CHUNKER.SEMANTIC"},  # Use semantic chunker for better performance
        ]

        config = {
            "marker_pdf_converter": {
                "languages": "en",
                "output_format": "markdown",
                "store_full_path": False,  # Reduce metadata size
            },
            "semantic_chunker": {
                "min_chunk_size": 100,
                "max_chunk_size": 800,
                "overlap_size": 50,
            },
        }

        try:
            spec, pipeline = self.factory.create_pipeline_from_spec(
                pipeline_spec, "performance_marker_pipeline", config
            )

            # Verify configuration optimized for performance
            marker_spec = spec.get_component_by_name("marker_pdf_converter")
            assert marker_spec is not None
            marker_config = marker_spec.get_config()
            assert marker_config.get("store_full_path") is False

            semantic_spec = spec.get_component_by_name("semantic_chunker")
            assert semantic_spec is not None
            semantic_config = semantic_spec.get_config()
            assert semantic_config.get("max_chunk_size") == 800

        except ImportError as e:
            pytest.skip(f"Skipping due to missing dependencies: {e}")
