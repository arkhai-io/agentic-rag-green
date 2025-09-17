"""Integration tests for PipelineRunner with real component execution."""

import os
import tempfile

import pytest

from agentic_rag.pipeline.runner import PipelineRunner


class TestPipelineRunnerIntegration:
    """Test PipelineRunner with actual component execution."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.runner = PipelineRunner()

    def teardown_method(self):
        """Clean up test fixtures."""
        # Clean up temp directory
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_simple_chunker_indexing_pipeline(self):
        """Test a simple indexing pipeline with just a chunker component."""
        try:
            from haystack import Document

            # Simple pipeline with just chunker
            component_specs = [{"type": "CHUNKER.MARKDOWN_AWARE"}]

            config = {
                "markdown_aware_chunker": {"chunk_size": 500, "chunk_overlap": 50}
            }

            # Load the pipeline
            self.runner.load_pipeline(component_specs, "chunker_test", config)

            # Create test documents
            documents = [
                Document(
                    content="# Introduction\n\nThis is a test document with markdown content.\n\n## Section 1\n\nSome content here.",
                    meta={"title": "Test Doc 1"},
                ),
                Document(
                    content="# Another Document\n\nThis document has different content.\n\n## Details\n\nMore information here.",
                    meta={"title": "Test Doc 2"},
                ),
            ]

            # Run the indexing pipeline
            results = self.runner.run("indexing", {"documents": documents})

            # Verify results
            assert isinstance(results, dict)
            assert results["success"] is True
            assert "results" in results
            assert results["processed_count"] == 2

            print("✅ Simple chunker pipeline test passed!")
            print(f"📊 Processed {results['processed_count']} documents")

        except ImportError:
            pytest.skip("Haystack not available for integration test")

    def test_full_indexing_pipeline_with_embedder_writer(self):
        """Test complete indexing pipeline: chunker -> embedder -> writer."""
        try:
            from haystack import Document

            # Complete indexing pipeline
            component_specs = [
                {"type": "CHUNKER.MARKDOWN_AWARE"},
                {"type": "EMBEDDER.SENTENCE_TRANSFORMERS_DOC"},
                {"type": "WRITER.DOCUMENT_WRITER"},
            ]

            config = {
                "markdown_aware_chunker": {"chunk_size": 400, "chunk_overlap": 40},
                "document_embedder": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                },
                "document_writer": {"root_dir": self.temp_dir},
            }

            # Load the pipeline
            self.runner.load_pipeline(component_specs, "full_indexing_test", config)

            # Create test documents
            documents = [
                Document(
                    content="# Machine Learning\n\nMachine learning is a subset of artificial intelligence.",
                    meta={"source": "ml_guide.md"},
                )
            ]

            # Run the indexing pipeline
            results = self.runner.run("indexing", {"documents": documents})

            # Verify results
            assert isinstance(results, dict)
            assert results["success"] is True
            assert "results" in results
            assert results["processed_count"] == 1

            print("✅ Full indexing pipeline test passed!")
            print(f"📊 Successfully indexed {results['processed_count']} document")

        except ImportError as e:
            pytest.skip(f"Dependencies not available: {e}")

    def test_end_to_end_chroma_pipeline(self):
        """Test complete end-to-end pipeline: index documents then retrieve them."""
        try:
            from haystack import Document

            # Use a shared datastore location for both indexing and retrieval
            shared_datastore_path = os.path.join(self.temp_dir, "shared_chroma_db")

            # Step 1: Create indexing pipeline to store documents
            indexing_specs = [
                {"type": "CHUNKER.MARKDOWN_AWARE"},
                {"type": "EMBEDDER.SENTENCE_TRANSFORMERS_DOC"},
                {"type": "WRITER.DOCUMENT_WRITER"},
            ]

            indexing_config = {
                "markdown_aware_chunker": {"chunk_size": 400, "chunk_overlap": 40},
                "document_embedder": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                },
                "document_writer": {
                    "root_dir": shared_datastore_path  # Shared location
                },
            }

            # Load indexing pipeline
            indexing_runner = PipelineRunner()
            indexing_runner.load_pipeline(
                indexing_specs, "indexing_test", indexing_config
            )

            # Create test documents
            test_documents = [
                Document(
                    content="# Python Programming\n\nPython is a high-level programming language used for web development, data science, and automation.",
                    meta={"source": "python_guide.md", "topic": "programming"},
                ),
                Document(
                    content="# Machine Learning Basics\n\nMachine learning is a subset of artificial intelligence that enables computers to learn from data.",
                    meta={"source": "ml_basics.md", "topic": "ai"},
                ),
                Document(
                    content="# Web Development\n\nWeb development involves creating websites and web applications using HTML, CSS, and JavaScript.",
                    meta={"source": "web_dev.md", "topic": "web"},
                ),
            ]

            # Index the documents
            indexing_results = indexing_runner.run(
                "indexing", {"documents": test_documents}
            )

            assert indexing_results["success"] is True
            assert indexing_results["processed_count"] == 3
            print(
                f"✅ Indexing completed: {indexing_results['processed_count']} documents indexed"
            )

            # Step 2: Create retrieval pipeline to search the indexed documents
            retrieval_specs = [
                {"type": "EMBEDDER.SENTENCE_TRANSFORMERS"},
                {"type": "RETRIEVER.CHROMA_EMBEDDING"},
            ]

            retrieval_config = {
                "embedder": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
                "chroma_embedding_retriever": {
                    "root_dir": shared_datastore_path,  # Same shared location
                    "top_k": 5,
                },
            }

            # Load retrieval pipeline
            retrieval_runner = PipelineRunner()
            retrieval_runner.load_pipeline(
                retrieval_specs, "retrieval_test", retrieval_config
            )

            # Test retrieval with different queries
            test_queries = [
                "Python programming language",
                "machine learning artificial intelligence",
                "web development HTML CSS",
            ]

            for query in test_queries:
                retrieval_results = retrieval_runner.run(
                    "retrieval", {"query": query, "top_k": 2}
                )

                if not retrieval_results["success"]:
                    print(
                        f"❌ Retrieval failed for query '{query}': {retrieval_results.get('error', 'Unknown error')}"
                    )
                    print(
                        f"Error type: {retrieval_results.get('error_type', 'Unknown')}"
                    )

                assert (
                    retrieval_results["success"] is True
                ), f"Retrieval failed: {retrieval_results.get('error', 'Unknown error')}"
                assert "results" in retrieval_results
                assert retrieval_results["query"] == query

                print(f"✅ Retrieval test passed for query: '{query}'")

            print("🎉 End-to-end Chroma pipeline test completed successfully!")

        except ImportError as e:
            pytest.skip(f"Dependencies not available: {e}")

    def test_retrieval_with_runtime_parameters(self):
        """Test retrieval pipeline with runtime parameter passing using Chroma."""
        try:
            from haystack import Document

            # Use a shared datastore location for both indexing and retrieval
            shared_datastore_path = os.path.join(self.temp_dir, "params_chroma_db")

            # Step 1: Index some test documents first
            indexing_specs = [
                {"type": "CHUNKER.MARKDOWN_AWARE"},
                {"type": "EMBEDDER.SENTENCE_TRANSFORMERS_DOC"},
                {"type": "WRITER.DOCUMENT_WRITER"},
            ]

            indexing_config = {
                "markdown_aware_chunker": {"chunk_size": 300, "chunk_overlap": 30},
                "document_embedder": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                },
                "document_writer": {
                    "root_dir": shared_datastore_path
                },  # Shared location
            }

            indexing_runner = PipelineRunner()
            indexing_runner.load_pipeline(
                indexing_specs, "param_indexing_test", indexing_config
            )

            # Create test documents with different categories
            test_documents = [
                Document(
                    content="Python programming tutorial for beginners",
                    meta={"category": "programming", "difficulty": "beginner"},
                ),
                Document(
                    content="Advanced Java development patterns and practices",
                    meta={"category": "programming", "difficulty": "advanced"},
                ),
                Document(
                    content="Machine learning fundamentals and algorithms",
                    meta={"category": "ai", "difficulty": "intermediate"},
                ),
                Document(
                    content="Deep learning neural networks explained",
                    meta={"category": "ai", "difficulty": "advanced"},
                ),
                Document(
                    content="Web development with modern JavaScript frameworks",
                    meta={"category": "web", "difficulty": "intermediate"},
                ),
            ]

            # Index the documents
            indexing_results = indexing_runner.run(
                "indexing", {"documents": test_documents}
            )
            assert indexing_results["success"] is True
            print(
                f"✅ Indexed {indexing_results['processed_count']} documents for parameter testing"
            )

            # Step 2: Create retrieval pipeline
            retrieval_specs = [
                {"type": "EMBEDDER.SENTENCE_TRANSFORMERS"},
                {"type": "RETRIEVER.CHROMA_EMBEDDING"},
            ]

            retrieval_config = {
                "embedder": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
                "chroma_embedding_retriever": {
                    "root_dir": shared_datastore_path,  # Same shared location
                    "top_k": 10,  # Default, will be overridden at runtime
                },
            }

            retrieval_runner = PipelineRunner()
            retrieval_runner.load_pipeline(
                retrieval_specs, "param_retrieval_test", retrieval_config
            )

            # Test 1: Override top_k at runtime
            results1 = retrieval_runner.run(
                "retrieval",
                {
                    "query": "programming tutorial",
                    "top_k": 2,  # Should limit to 2 results
                },
            )

            if not results1["success"]:
                print(
                    f"❌ Runtime top_k test failed: {results1.get('error', 'Unknown error')}"
                )
                print(f"Error type: {results1.get('error_type', 'Unknown')}")

            assert (
                results1["success"] is True
            ), f"Runtime top_k test failed: {results1.get('error', 'Unknown error')}"
            print(f"✅ Runtime top_k test passed! Query: '{results1['query']}'")

            # Test 2: Test with different top_k values
            results2 = retrieval_runner.run(
                "retrieval",
                {"query": "machine learning", "top_k": 1},  # Should limit to 1 result
            )

            assert results2["success"] is True
            print(f"✅ Runtime top_k=1 test passed! Query: '{results2['query']}'")

            print("🎉 All runtime parameter tests completed successfully!")

        except ImportError as e:
            pytest.skip(f"Dependencies not available: {e}")

    def test_error_handling_missing_query(self):
        """Test error handling when query is missing."""
        # Use a simple embedder component for error testing (no external dependencies)
        component_specs = [{"type": "EMBEDDER.SENTENCE_TRANSFORMERS"}]
        config = {"embedder": {"model": "sentence-transformers/all-MiniLM-L6-v2"}}

        self.runner.load_pipeline(component_specs, "error_test", config)

        # Test missing query
        results = self.runner.run("retrieval", {"top_k": 5})  # No query

        assert results["success"] is False
        assert "error" in results
        assert "query" in results["error"].lower()
        print("✅ Error handling test passed - missing query detected")

    def test_error_handling_invalid_query_type(self):
        """Test error handling when query is not a string."""
        # Use a simple embedder component for error testing
        component_specs = [{"type": "EMBEDDER.SENTENCE_TRANSFORMERS"}]
        config = {"embedder": {"model": "sentence-transformers/all-MiniLM-L6-v2"}}

        self.runner.load_pipeline(component_specs, "error_test2", config)

        # Test invalid query type
        results = self.runner.run("retrieval", {"query": 123})  # Not a string

        assert results["success"] is False
        assert "error" in results
        assert "string" in results["error"].lower()
        print("✅ Error handling test passed - invalid query type detected")

    def test_error_handling_missing_documents(self):
        """Test error handling when documents are missing for indexing."""
        component_specs = [{"type": "CHUNKER.MARKDOWN_AWARE"}]
        config = {"markdown_aware_chunker": {"chunk_size": 500}}

        self.runner.load_pipeline(component_specs, "error_test3", config)

        # Test missing documents
        results = self.runner.run("indexing", {"chunk_size": 500})  # No documents

        assert results["success"] is False
        assert "error" in results
        assert "documents" in results["error"].lower()
        print("✅ Error handling test passed - missing documents detected")

    def test_error_handling_invalid_documents_type(self):
        """Test error handling when documents is not a list."""
        component_specs = [{"type": "CHUNKER.MARKDOWN_AWARE"}]
        config = {"markdown_aware_chunker": {"chunk_size": 500}}

        self.runner.load_pipeline(component_specs, "error_test4", config)

        # Test invalid documents type
        results = self.runner.run("indexing", {"documents": "not a list"})

        assert results["success"] is False
        assert "error" in results
        assert "list" in results["error"].lower()
        print("✅ Error handling test passed - invalid documents type detected")

    def test_no_pipeline_loaded_error(self):
        """Test error when trying to run without loading a pipeline."""
        empty_runner = PipelineRunner()

        # Try to run without loading pipeline
        with pytest.raises(RuntimeError) as exc_info:
            empty_runner.run("retrieval", {"query": "test"})

        assert "No pipeline loaded" in str(exc_info.value)
        print("✅ No pipeline loaded error test passed")

    def test_invalid_pipeline_type_error(self):
        """Test error when using invalid pipeline type."""
        component_specs = [{"type": "CHUNKER.MARKDOWN_AWARE"}]
        self.runner.load_pipeline(component_specs, "test", {})

        # Try invalid pipeline type
        with pytest.raises(ValueError) as exc_info:
            self.runner.run("invalid_type", {"query": "test"})

        assert "Unsupported pipeline type" in str(exc_info.value)
        print("✅ Invalid pipeline type error test passed")


if __name__ == "__main__":
    # Run tests manually for debugging
    test_instance = TestPipelineRunnerIntegration()
    test_instance.setup_method()

    try:
        test_instance.test_simple_chunker_indexing_pipeline()
        test_instance.test_error_handling_missing_query()
        test_instance.test_error_handling_missing_documents()
        test_instance.test_no_pipeline_loaded_error()
        print("\n🎉 All manual tests completed!")
    finally:
        test_instance.teardown_method()
