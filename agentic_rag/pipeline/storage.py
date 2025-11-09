"""Graph storage for creating and managing graph representations of pipelines."""

from typing import Any, Dict, List, Optional, Tuple

from haystack import Pipeline

from ..components import ComponentRegistry, GraphStore
from ..types import (
    ComponentNode,
    ComponentSpec,
    GraphRelationship,
    PipelineSpec,
    PipelineType,
    UserNode,
    create_haystack_component,
)
from ..utils.logger import get_system_logger


class GraphStorage:
    """Stores and manages graph representations of pipeline components and their relationships."""

    def __init__(self, graph_store: GraphStore, registry: ComponentRegistry) -> None:
        self.graph_store = graph_store
        self.registry = registry
        self.logger = get_system_logger(__name__)

    def create_pipeline_graph(
        self,
        spec: PipelineSpec,
        connections: List[Tuple[str, str]],
        username: str,
        branch_id: Optional[str] = None,
    ) -> None:
        """Create graph representation of the pipeline components.

        Args:
            spec: Pipeline specification
            connections: List of (source, target) component connections
            username: Username for pipeline ownership
            branch_id: Optional branch identifier for retrieval pipeline branches
        """

        nodes: List[Dict[str, Any]] = []
        node_id_by_name: Dict[str, str] = {}
        for component_spec in spec.components:
            node = ComponentNode(
                component_name=component_spec.name,
                pipeline_name=spec.name,
                version="1.0.0",
                author=username,
                component_config=component_spec.get_config(),
                component_type=(
                    component_spec.full_type if component_spec.full_type else None
                ),
                pipeline_type=spec.pipeline_type.value if spec.pipeline_type else None,
                branch_id=branch_id,
            )
            node_dict = node.to_dict()
            nodes.append(node_dict)
            node_id_by_name[component_spec.name] = node_dict["id"]

        # Add/update the owning user node
        self.logger.info(
            f"Creating pipeline graph for user '{username}', pipeline '{spec.name}'"
        )
        user_node = UserNode(username=username, display_name=username.title())
        user_dict = user_node.to_dict()
        self.graph_store.add_nodes_batch([user_dict], "User")

        # Add component nodes
        self.logger.debug(f"Adding {len(nodes)} component nodes to graph")
        self.graph_store.add_nodes_batch(nodes, "Component")

        # Connect user to the first component in the pipeline
        if spec.components:
            first_component_id = node_id_by_name.get(spec.components[0].name)
            if first_component_id:
                self.graph_store.add_edges_batch(
                    [
                        (
                            user_dict["id"],
                            first_component_id,
                            GraphRelationship.OWNS.value,
                        )
                    ],
                    source_label="User",
                    target_label="Component",
                )

        # Connect sequential components
        if connections:
            component_edges = []
            for source_name, target_name in connections:
                source_id = node_id_by_name.get(source_name)
                target_id = node_id_by_name.get(target_name)
                if source_id and target_id:
                    component_edges.append(
                        (source_id, target_id, GraphRelationship.FLOWS_TO.value)
                    )
            if component_edges:
                self.graph_store.add_edges_batch(
                    component_edges,
                    source_label="Component",
                    target_label="Component",
                )

        # Retrieval pipelines will be handled separately
        # No need to store graph for retrieval - they're built dynamically from indexing metadata
        if spec.pipeline_type == PipelineType.RETRIEVAL:
            self.logger.info(
                f"Retrieval pipeline '{spec.name}' - will be built dynamically from indexing pipeline metadata"
            )

    def build_pipeline_graph(
        self,
        spec: PipelineSpec,
        username: str = "test_user",
        branch_id: Optional[str] = None,
    ) -> None:
        """Build a graph representation of the pipeline specification.

        Args:
            spec: Pipeline specification
            username: Username for pipeline ownership (defaults to "test_user")
            branch_id: Optional branch identifier for retrieval pipeline branches
        """

        # Determine connections based on component order and dependencies
        connections = self._determine_connections(spec.components)

        # Create graph representation
        self.create_pipeline_graph(spec, connections, username, branch_id)

    def build_haystack_pipeline(self, spec: PipelineSpec) -> Any:
        """Build a Haystack pipeline from a pipeline specification."""

        # Create Haystack pipeline
        pipeline = Pipeline()

        # Create component instances and add to pipeline
        for component_spec in spec.components:
            # ComponentSpec handles its own config internally
            haystack_component = create_haystack_component(component_spec)

            pipeline.add_component(component_spec.name, haystack_component)

        # Determine connections based on component order and dependencies
        connections = self._determine_connections(spec.components)

        # Add connections to pipeline
        for source, target in connections:
            try:
                pipeline.connect(source, target)
            except Exception as e:
                print(f"Warning: Could not connect {source} -> {target}: {e}")

        return pipeline

    def _determine_connections(
        self, components: List[ComponentSpec]
    ) -> List[Tuple[str, str]]:
        """Determine how components should be connected - simple sequential connections."""
        connections: List[Tuple[str, str]] = []

        # Connect each component to the previous one
        for i in range(1, len(components)):
            prev_component = components[i - 1]
            current_component = components[i]
            connections.append((prev_component.name, current_component.name))

        return connections

    def load_pipeline_by_hashes(
        self, pipeline_hashes: List[str], username: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve components for each pipeline hash from Neo4j.

        Args:
            pipeline_hashes: List of pipeline names to load
            username: Username for permissions

        Returns:
            Dictionary mapping pipeline names to their component data
        """
        # 1. Validate username exists in Neo4j
        if not self.graph_store.validate_user_exists(username):
            raise ValueError(f"Username '{username}' not found in Neo4j")

        pipelines_data: Dict[str, List[Dict[str, Any]]] = {}

        # 2. Fetch components for each pipeline hash separately
        for pipeline_hash in pipeline_hashes:
            print(f"\nFetching components for pipeline: {pipeline_hash}")

            # Call Neo4j for this specific pipeline hash (single hash method)
            component_data_list = self.graph_store.get_pipeline_components_by_hash(
                pipeline_hash, username  # Single pipeline hash
            )

            print(f"   Found {len(component_data_list)} components")

            if component_data_list:
                pipelines_data[pipeline_hash] = component_data_list

                # Print details for this pipeline
                print(component_data_list)

        return pipelines_data
