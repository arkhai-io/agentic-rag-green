"""Test graph-based pipeline architecture (Factory + Runner + Neo4j)."""

from unittest.mock import MagicMock

import pytest

from agentic_rag.components import GraphStore, get_default_registry
from agentic_rag.pipeline import PipelineFactory, PipelineRunner


@pytest.fixture
def mock_graph_store():
    """Create a mock GraphStore for testing without Neo4j."""
    mock = MagicMock(spec=GraphStore)
    # Mock the driver and session
    mock.driver = MagicMock()
    mock.driver.verify_connectivity = MagicMock()
    return mock


@pytest.fixture
def graph_store(test_config):
    """Create a real or mock GraphStore based on availability."""
    # Try to create real GraphStore with test_config
    # If it fails, return a mock
    try:
        if test_config.validate_neo4j():
            return GraphStore(config=test_config)
    except Exception:
        pass
    return mock_graph_store()


class TestGraphPipelineArchitecture:
    """Test the new graph-based pipeline system."""

    def setup_method(self):
        """Set up test fixtures."""
        from agentic_rag.pipeline.storage import GraphStorage

        self.registry = get_default_registry()
        # Reset singleton instances before each test
        PipelineFactory.reset_instance()
        PipelineRunner.reset_instance()
        GraphStore.reset_instance()
        GraphStorage.reset_instance()

    def teardown_method(self):
        """Clean up after each test."""
        from agentic_rag.pipeline.storage import GraphStorage

        # Reset singleton instances after each test
        PipelineFactory.reset_instance()
        PipelineRunner.reset_instance()
        GraphStore.reset_instance()
        GraphStorage.reset_instance()

    def test_factory_builds_pipeline_graph(self, mock_graph_store, test_config):
        """Test that factory builds and stores pipeline graph in Neo4j."""
        factory = PipelineFactory(graph_store=mock_graph_store, config=test_config)

        pipeline_spec = [
            {"type": "CONVERTER.PDF"},
            {"type": "CHUNKER.DOCUMENT_SPLITTER"},
        ]

        # Build pipeline graph (stores in Neo4j) - username and project now injected at method level
        spec = factory.build_pipeline_graph(
            pipeline_spec,
            "test_pipeline",
            username="test_user",
            project="test_project",
            config={},
        )

        assert spec is not None
        assert spec.name == "test_pipeline"
        assert len(spec.components) == 2
        assert spec.components[0].name == "pdf_converter"
        assert spec.components[1].name == "chunker"

    def test_runner_loads_pipeline_graph(self, mock_graph_store, test_config):
        """Test that runner can load pipeline graph from Neo4j."""
        factory = PipelineFactory(graph_store=mock_graph_store, config=test_config)

        # First, create a pipeline
        pipeline_spec = [
            {"type": "CONVERTER.PDF"},
            {"type": "CHUNKER.DOCUMENT_SPLITTER"},
        ]

        # Username and project now injected at method level
        factory.build_pipeline_graph(
            pipeline_spec,
            "load_test_pipeline",
            username="test_user",
            project="test_project",
            config={},
        )

        # Mock the load response
        mock_graph_store.validate_user_exists.return_value = True
        mock_graph_store.get_pipeline_components_by_hash.return_value = [
            {
                "id": "comp_1",
                "component_name": "pdf_converter",
                "component_type": "CONVERTER.PDF",
                "component_config_json": "{}",
                "pipeline_name": "load_test_pipeline",
                "pipeline_type": "indexing",
                "next_components": ["comp_2"],
                "node_labels": ["Component"],
            },
            {
                "id": "comp_2",
                "component_name": "chunker",
                "component_type": "CHUNKER.DOCUMENT_SPLITTER",
                "component_config_json": '{"split_by":"sentence","split_length":512}',
                "pipeline_name": "load_test_pipeline",
                "pipeline_type": "indexing",
                "next_components": [],
                "node_labels": ["Component"],
            },
        ]

        # Now create runner and load pipelines with username injection
        runner = PipelineRunner(
            graph_store=mock_graph_store,
            config=test_config,
        )

        runner.load_pipelines(
            pipeline_names=["load_test_pipeline"],
            username="test_user",
            project="test_project",
        )

        # Verify graph data was loaded
        assert runner._pipeline_graphs
        assert "load_test_pipeline" in runner._pipeline_graphs

    def test_runner_builds_haystack_components(self, mock_graph_store, test_config):
        """Test that runner builds runtime Haystack components from graph."""
        factory = PipelineFactory(graph_store=mock_graph_store, config=test_config)

        # Create pipeline
        pipeline_spec = [
            {"type": "CONVERTER.PDF"},
            {"type": "CHUNKER.DOCUMENT_SPLITTER"},
        ]

        # Username and project now injected at method level
        factory.build_pipeline_graph(
            pipeline_spec,
            "component_test_pipeline",
            username="test_user",
            project="test_project",
            config={},
        )

        # Mock the load response
        mock_graph_store.validate_user_exists.return_value = True
        mock_graph_store.get_pipeline_components_by_hash.return_value = [
            {
                "id": "comp_1",
                "component_name": "pdf_converter",
                "component_type": "CONVERTER.PDF",
                "component_config_json": "{}",
                "pipeline_name": "component_test_pipeline",
                "pipeline_type": "indexing",
                "cache_key": "cache_123",
                "next_components": ["comp_2"],
                "node_labels": ["Component"],
            },
            {
                "id": "comp_2",
                "component_name": "chunker",
                "component_type": "CHUNKER.DOCUMENT_SPLITTER",
                "component_config_json": '{"split_by":"sentence","split_length":512}',
                "pipeline_name": "component_test_pipeline",
                "pipeline_type": "indexing",
                "cache_key": "cache_456",
                "next_components": [],
                "node_labels": ["Component"],
            },
        ]

        # Create runner and load pipelines with username injection
        runner = PipelineRunner(
            graph_store=mock_graph_store,
            config=test_config,
        )

        runner.load_pipelines(
            pipeline_names=["component_test_pipeline"],
            username="test_user",
            project="test_project",
        )

        # Verify components were built
        assert "component_test_pipeline" in runner._haystack_components_by_pipeline
        components = runner._haystack_components_by_pipeline["component_test_pipeline"]
        assert components is not None
        assert len(components) >= 2  # At least the 2 components we added

    def test_runner_requires_parameters(self, mock_graph_store):
        """Test that runner can be initialized without parameters (singleton)."""
        # PipelineRunner is now a singleton and can be initialized without parameters
        # It will use config defaults
        runner = PipelineRunner(graph_store=mock_graph_store)
        assert runner is not None
        # Subsequent calls should return the same instance
        runner2 = PipelineRunner()
        assert runner is runner2

    def test_pipeline_with_config(self, mock_graph_store, test_config):
        """Test pipeline creation with custom configuration."""
        factory = PipelineFactory(graph_store=mock_graph_store, config=test_config)

        pipeline_spec = [
            {"type": "CHUNKER.DOCUMENT_SPLITTER"},
        ]

        config = {
            "chunker": {
                "chunk_size": 500,
                "chunk_overlap": 50,
            }
        }

        # Username and project now injected at method level
        spec = factory.build_pipeline_graph(
            pipeline_spec,
            "config_test_pipeline",
            username="test_user",
            project="test_project",
            config=config,
        )

        assert spec.components[0].get_config()["chunk_size"] == 500
        assert spec.components[0].get_config()["chunk_overlap"] == 50

    def test_component_registry_integration(self):
        """Test that components are properly looked up from registry."""
        spec = self.registry.get_component_spec("pdf_converter")

        assert spec is not None
        assert spec.name == "pdf_converter"
        assert spec.component_type.value == "converter"

    def test_invalid_pipeline_hash_handling(self, mock_graph_store, test_config):
        """Test handling of invalid pipeline hashes."""
        # Mock empty response for nonexistent pipeline
        mock_graph_store.validate_user_exists.return_value = True
        mock_graph_store.get_pipeline_components_by_hash.return_value = []

        # PipelineRunner will try to load pipelines and will log a warning
        runner = PipelineRunner(
            graph_store=mock_graph_store,
            config=test_config,
        )

        runner.load_pipelines(
            pipeline_names=["nonexistent_pipeline"],
            username="test_user",
            project="test_project",
        )

        # Should load empty/missing data for non-existent pipeline
        # Either not in dict or empty list
        if "nonexistent_pipeline" in runner._pipeline_graphs:
            assert len(runner._pipeline_graphs["nonexistent_pipeline"]) == 0
        # If not in dict, that's also fine - pipeline doesn't exist

    def test_project_hierarchy_in_graph(self, mock_graph_store, test_config):
        """Test that project nodes are created in the graph hierarchy."""

        factory = PipelineFactory(graph_store=mock_graph_store, config=test_config)

        pipeline_spec = [
            {"type": "CONVERTER.TEXT"},
            {"type": "CHUNKER.DOCUMENT_SPLITTER"},
        ]

        # Build pipeline with username and project
        spec = factory.build_pipeline_graph(
            pipeline_spec,
            "project_test_pipeline",
            username="alice",
            project="rag_app",
            config={},
        )

        # Verify pipeline was created
        assert spec is not None
        assert spec.name == "project_test_pipeline"

        # Verify that add_nodes_batch was called for User, Project, and Component nodes
        add_nodes_calls = mock_graph_store.add_nodes_batch.call_args_list

        # Should have calls for User, Project, and Component nodes
        assert len(add_nodes_calls) >= 3

        # Check that User node was created
        user_call = [call for call in add_nodes_calls if call[0][1] == "User"]
        assert len(user_call) > 0
        user_data = user_call[0][0][0][0]
        assert user_data["username"] == "alice"

        # Check that Project node was created
        project_call = [call for call in add_nodes_calls if call[0][1] == "Project"]
        assert len(project_call) > 0
        project_data = project_call[0][0][0][0]
        assert project_data["name"] == "rag_app"
        assert project_data["username"] == "alice"

        # Check that Component nodes were created
        component_call = [call for call in add_nodes_calls if call[0][1] == "Component"]
        assert len(component_call) > 0

    def test_project_node_creation(self):
        """Test ProjectNode creation and ID generation."""
        from agentic_rag.types import ProjectNode

        project = ProjectNode(
            name="my_rag_app", username="bob", description="My RAG application"
        )

        assert project.name == "my_rag_app"
        assert project.username == "bob"
        assert project.description == "My RAG application"
        assert project.id is not None
        assert project.id.startswith("proj_")

        # Test dictionary conversion
        project_dict = project.to_dict()
        assert project_dict["name"] == "my_rag_app"
        assert project_dict["username"] == "bob"
        assert project_dict["description"] == "My RAG application"
        assert "id" in project_dict

    def test_storage_paths_include_project(self, mock_graph_store, test_config):
        """Test that storage paths include project in hierarchy."""
        from agentic_rag.config import Config

        config = Config(agentic_root_dir="./test_data")

        # Test project path generation
        path = config.get_project_path("alice", "rag_app")
        assert "alice" in path
        assert "rag_app" in path
        assert path.endswith("alice/rag_app") or path.endswith(
            "alice\\rag_app"
        )  # Handle Windows paths


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
