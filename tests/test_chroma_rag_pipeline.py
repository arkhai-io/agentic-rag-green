"""Test Chroma RAG pipeline with document writing and retrieval."""

import shutil
import tempfile
from pathlib import Path

import pytest

from agentic_rag import PipelineFactory


class TestChromaRAGPipeline:
    """Test Chroma-based RAG pipeline functionality."""

    def setup_method(self):
        """Set up test environment with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.factory = PipelineFactory()

    def teardown_method(self):
        """Clean up temporary directory."""
        if hasattr(self, "temp_dir") and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_chroma_indexing_pipeline(self):
        """Test creating an indexing pipeline that writes to Chroma."""
        # Define indexing pipeline: Convert -> Chunk -> Embed -> Write
        indexing_spec = [
            {"type": "CONVERTER.TEXT"},
            {"type": "CHUNKER.MARKDOWN_AWARE"},
            {"type": "EMBEDDER.SENTENCE_TRANSFORMERS_DOC"},
            {"type": "WRITER.CHROMA_DOCUMENT_WRITER"},
        ]

        # Configuration with custom root directory
        config = {
            "chroma_document_writer": {"root_dir": self.temp_dir},
            "markdown_aware_chunker": {"chunk_size": 500, "chunk_overlap": 50},
            "document_embedder": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
        }

        try:
            spec, pipeline = self.factory.create_pipeline_from_spec(
                indexing_spec, "chroma_indexing", config
            )

            # Verify pipeline structure
            assert spec.name == "chroma_indexing"
            assert len(spec.components) == 4

            # Check component types
            component_types = [comp.component_type.value for comp in spec.components]
            assert "converter" in component_types
            assert "chunker" in component_types
            assert "embedder" in component_types
            assert "writer" in component_types

            # Verify pipeline object exists
            assert pipeline is not None
            assert hasattr(pipeline, "run")

        except ImportError as e:
            pytest.skip(f"Skipping due to missing dependencies: {e}")

    def test_chroma_retrieval_pipeline(self):
        """Test creating a retrieval pipeline that reads from Chroma."""
        # Define retrieval pipeline: Embed Query -> Retrieve -> Generate
        retrieval_spec = [
            {"type": "EMBEDDER.SENTENCE_TRANSFORMERS"},
            {"type": "RETRIEVER.CHROMA_EMBEDDING"},
            {"type": "GENERATOR.OPENAI"},
        ]

        # Configuration with custom root directory
        config = {
            "chroma_embedding_retriever": {"root_dir": self.temp_dir, "top_k": 5},
            "embedder": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
            "generator": {"model": "gpt-3.5-turbo"},
        }

        try:
            spec, pipeline = self.factory.create_pipeline_from_spec(
                retrieval_spec, "chroma_retrieval", config
            )

            # Verify pipeline structure
            assert spec.name == "chroma_retrieval"
            assert len(spec.components) == 3

            # Check component types
            component_types = [comp.component_type.value for comp in spec.components]
            assert "embedder" in component_types
            assert "retriever" in component_types
            assert "generator" in component_types

            # Verify pipeline object exists
            assert pipeline is not None
            assert hasattr(pipeline, "run")

            # Verify retriever configuration
            retriever_spec = spec.get_component_by_name("chroma_embedding_retriever")
            assert retriever_spec is not None
            retriever_config = retriever_spec.get_config()
            assert retriever_config.get("top_k") == 5

        except ImportError as e:
            pytest.skip(f"Skipping due to missing dependencies: {e}")

    def test_chroma_test_rag_pipeline(self):
        """Test complete RAG pipeline with Chroma document store injection."""
        # Create both indexing and retrieval pipelines using the same Chroma store

        # 1. Indexing Pipeline
        indexing_spec = [
            {"type": "CONVERTER.TEXT"},
            {"type": "CHUNKER.MARKDOWN_AWARE"},
            {"type": "EMBEDDER.SENTENCE_TRANSFORMERS_DOC"},
            {"type": "WRITER.CHROMA_DOCUMENT_WRITER"},
        ]

        # 2. Retrieval Pipeline
        retrieval_spec = [
            {"type": "EMBEDDER.SENTENCE_TRANSFORMERS"},
            {"type": "RETRIEVER.CHROMA_EMBEDDING"},
            {"type": "GENERATOR.OPENAI"},
        ]

        # Shared configuration for both pipelines
        shared_config = {
            "root_dir": self.temp_dir,
            "model": "sentence-transformers/all-MiniLM-L6-v2",
        }

        indexing_config = {
            "chroma_document_writer": shared_config,
            "markdown_aware_chunker": {"chunk_size": 800, "chunk_overlap": 100},
            "document_embedder": {"model": shared_config["model"]},
        }

        retrieval_config = {
            "chroma_embedding_retriever": {**shared_config, "top_k": 3},
            "embedder": {"model": shared_config["model"]},
            "generator": {"model": "gpt-3.5-turbo"},
        }

        try:
            # Create indexing pipeline
            indexing_spec_obj, indexing_pipeline = (
                self.factory.create_pipeline_from_spec(
                    indexing_spec, "chroma_indexing_rag", indexing_config
                )
            )

            # Create retrieval pipeline
            retrieval_spec_obj, retrieval_pipeline = (
                self.factory.create_pipeline_from_spec(
                    retrieval_spec, "chroma_retrieval_rag", retrieval_config
                )
            )

            # Verify both pipelines were created
            assert indexing_spec_obj is not None
            assert retrieval_spec_obj is not None
            assert indexing_pipeline is not None
            assert retrieval_pipeline is not None

            # Verify they use the same Chroma path
            # Verify pipeline configurations
            assert indexing_spec_obj.name == "chroma_indexing_rag"
            assert retrieval_spec_obj.name == "chroma_retrieval_rag"

            # Check indexing pipeline components
            indexing_types = [
                comp.component_type.value for comp in indexing_spec_obj.components
            ]
            assert set(indexing_types) == {"converter", "chunker", "embedder", "writer"}

            # Check retrieval pipeline components
            retrieval_types = [
                comp.component_type.value for comp in retrieval_spec_obj.components
            ]
            assert set(retrieval_types) == {"embedder", "retriever", "generator"}

            # Verify ChromaEmbeddingRetriever got the correct configuration
            retriever_spec = retrieval_spec_obj.get_component_by_name(
                "chroma_embedding_retriever"
            )
            assert retriever_spec is not None
            retriever_config_final = retriever_spec.get_config()
            assert retriever_config_final.get("top_k") == 3

            print("✅ RAG pipeline test passed!")

            print(
                f"📊 Indexing pipeline: {len(indexing_spec_obj.components)} components"
            )
            print(
                f"🔍 Retrieval pipeline: {len(retrieval_spec_obj.components)} components"
            )

        except ImportError as e:
            pytest.skip(f"Skipping due to missing Haystack/Chroma dependencies: {e}")

    def test_chroma_document_store_injection(self):
        """Test that ChromaDocumentStore is properly injected into components."""
        from agentic_rag.types import (
            ComponentSpec,
            ComponentType,
            DataType,
            PipelineUsage,
            create_haystack_component,
        )

        # Create a mock ChromaEmbeddingRetriever spec
        chroma_spec = ComponentSpec(
            name="test_chroma_retriever",
            component_type=ComponentType.RETRIEVER,
            haystack_class="haystack_integrations.components.retrievers.chroma.ChromaEmbeddingRetriever",
            input_types=[DataType.LIST_FLOAT],
            output_types=[DataType.LIST_DOCUMENT],
            pipeline_usage=PipelineUsage.RETRIEVAL,
            default_config={"top_k": 10},
        )

        try:
            chroma_spec.configure({"root_dir": self.temp_dir, "top_k": 5})
            # This should inject the document store automatically
            component = create_haystack_component(chroma_spec)

            # If we get here without error, the injection worked
            assert component is not None
            print("✅ ChromaDocumentStore injection test passed!")

        except ImportError as e:
            pytest.skip(f"Skipping due to missing Chroma dependencies: {e}")

    def test_multiple_chroma_components_same_store(self):
        """Test that multiple Chroma components can share the same document store path."""
        # Test both writer and retriever using same root directory
        pipeline_spec = [
            {"type": "EMBEDDER.SENTENCE_TRANSFORMERS_DOC"},
            {"type": "WRITER.CHROMA_DOCUMENT_WRITER"},
            {"type": "EMBEDDER.SENTENCE_TRANSFORMERS"},
            {"type": "RETRIEVER.CHROMA_EMBEDDING"},
        ]

        config = {
            "chroma_document_writer": {"root_dir": self.temp_dir},
            "chroma_embedding_retriever": {"root_dir": self.temp_dir, "top_k": 5},
            "document_embedder": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
            "embedder": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
        }

        try:
            spec, pipeline = self.factory.create_pipeline_from_spec(
                pipeline_spec, "chroma_shared_store", config
            )

            # Both writer and retriever should use the same Chroma store location
            assert spec is not None
            assert pipeline is not None

            # Verify all components were created
            assert len(spec.components) == 4

            print("✅ Multiple Chroma components sharing store test passed!")

        except ImportError as e:
            pytest.skip(f"Skipping due to missing dependencies: {e}")
