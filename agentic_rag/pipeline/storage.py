"""Graph storage for creating and managing graph representations of pipelines."""

from typing import Any, Dict, List, Tuple

from haystack import Pipeline

from ..components import ComponentRegistry, GraphStore
from ..types import (
    ComponentNode,
    ComponentSpec,
    ComponentType,
    DocumentStoreNode,
    GraphRelationship,
    PipelineSpec,
    PipelineType,
    UserNode,
    create_haystack_component,
)


class GraphStorage:
    """Stores and manages graph representations of pipeline components and their relationships."""

    def __init__(self, graph_store: GraphStore, registry: ComponentRegistry) -> None:
        self.graph_store = graph_store
        self.registry = registry

    def create_pipeline_graph(
        self, spec: PipelineSpec, connections: List[Tuple[str, str]]
    ) -> None:
        """Create graph representation of the pipeline components."""

        nodes: List[Dict[str, Any]] = []
        node_id_by_name: Dict[str, str] = {}
        for component_spec in spec.components:
            node = ComponentNode(
                component_name=component_spec.name,
                pipeline_name=spec.name,
                version="1.0.0",
                author="test_user",
                component_config=component_spec.get_config(),
            )
            node_dict = node.to_dict()
            nodes.append(node_dict)
            node_id_by_name[component_spec.name] = node_dict["id"]

        # Add/update the owning user node
        user_node = UserNode(username="test_user", display_name="Test User")
        user_dict = user_node.to_dict()
        self.graph_store.add_nodes_batch([user_dict], "User")

        # Add component nodes
        self.graph_store.add_nodes_batch(nodes, "Component")

        document_store_nodes: List[Dict[str, Any]] = []
        component_to_store_edges: List[Tuple[str, str, str]] = []

        if spec.pipeline_type == PipelineType.INDEXING:
            # For indexing pipelines, create new DocumentStore nodes
            for index, component_spec in enumerate(spec.components):
                if component_spec.name != "chroma_document_writer":
                    continue

                writer_config = component_spec.get_config()
                root_dir = writer_config.get("root_dir", ".")

                # Collect node IDs in reverse order (writer first, then preceding components)
                related_component_ids: List[str] = []

                # Add writer first
                writer_id = node_id_by_name.get(component_spec.name)
                if writer_id:
                    related_component_ids.append(writer_id)

                # Then add preceding embedders in reverse order
                for i in range(index - 1, -1, -1):
                    related = spec.components[i]
                    if related.component_type == ComponentType.EMBEDDER:
                        related_id = node_id_by_name.get(related.name)
                        if related_id:
                            related_component_ids.append(related_id)

                doc_store_node = DocumentStoreNode(
                    pipeline_name=spec.name,
                    root_dir=root_dir,
                    component_node_ids=related_component_ids,
                )
                doc_store_dict = doc_store_node.to_dict()
                document_store_nodes.append(doc_store_dict)

                if writer_id:
                    component_to_store_edges.append(
                        (
                            writer_id,
                            doc_store_dict["id"],
                            GraphRelationship.WRITES_TO.value,
                        )
                    )

        if document_store_nodes:
            self.graph_store.add_nodes_batch(document_store_nodes, "DocumentStore")

        if component_to_store_edges:
            self.graph_store.add_edges_batch(
                component_to_store_edges,
                source_label="Component",
                target_label="DocumentStore",
            )

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

        # For retrieval pipelines, fetch DocumentStore components and create substitutions
        if spec.pipeline_type == PipelineType.RETRIEVAL and spec.indexing_pipelines:
            retrieval_edges: List[Tuple[str, str, str]] = []
            retriever_to_docstore_edges: List[Tuple[str, str, str]] = []
            node_substitutions: List[Dict[str, Any]] = []

            for store_name, store_id in spec.indexing_pipelines.items():
                # Get component IDs from DocumentStore
                component_ids = self.graph_store.get_document_store_component_ids(
                    store_id
                )

                if component_ids:
                    # Get the actual component data
                    components = self.graph_store.get_component_nodes_by_ids(
                        component_ids
                    )
                    processed_component_ids = []

                    for comp in components:
                        comp_name = str(comp.get("component_name", "unknown"))
                        comp_id = str(comp.get("id", "unknown"))
                        comp_pipeline = str(comp.get("pipeline_name", "unknown"))

                        # Check if this component needs substitution using the mapping
                        from ..types.component_mappings import (
                            get_component_substitution,
                        )

                        substitution = get_component_substitution(comp_name)

                        if substitution:
                            # Create substituted component node from registry
                            from ..components import ComponentRegistry

                            temp_registry = ComponentRegistry()
                            target_spec = temp_registry.get_component_spec(
                                substitution.target_component
                            )

                            if target_spec:

                                # Use indexing pipeline, retrieval pipeline, and component type in ID
                                component_name = (
                                    f"{spec.name}_{substitution.target_component}"
                                )
                                component_id = f"{comp_pipeline}_{substitution.target_component}_for_{spec.name}"

                                # Always use empty config for substituted components
                                config_json = "{}"

                                # Create substituted component dictionary directly
                                substituted_dict = {
                                    "id": component_id,
                                    "component_name": component_name,
                                    "pipeline_name": spec.name,
                                    "version": "1.0.0",
                                    "author": "test_user",
                                    "component_config_json": config_json,
                                }

                                node_substitutions.append(substituted_dict)
                                processed_component_ids.append(component_id)

                                # Connect this component directly to the DocumentStore
                                retriever_to_docstore_edges.append(
                                    (
                                        component_id,
                                        store_id,
                                        GraphRelationship.READS_FROM.value,
                                    )
                                )
                            else:
                                # Fallback to original if retriever spec not found
                                processed_component_ids.append(comp_id)
                        else:
                            # Use original component
                            processed_component_ids.append(comp_id)

                    # Create edges in reverse order with substituted components
                    # Reverse the processed component IDs
                    reversed_component_ids = list(reversed(processed_component_ids))

                    # Connect each component to the next in reverse order
                    for i in range(len(reversed_component_ids) - 1):
                        source_id = reversed_component_ids[i]
                        target_id = reversed_component_ids[i + 1]
                        retrieval_edges.append(
                            (source_id, target_id, GraphRelationship.FLOWS_TO.value)
                        )

                    # Connect the last component (first in original order) to DocumentStore
                    if reversed_component_ids:
                        last_comp_id = reversed_component_ids[-1]
                        retrieval_edges.append(
                            (last_comp_id, store_id, GraphRelationship.READS_FROM.value)
                        )

                    # Connect last retrieval pipeline component to first component in reversed list (last in original)
                    if processed_component_ids and spec.components:
                        first_in_reversed = processed_component_ids[
                            -1
                        ]  # Last in original = first in reversed
                        last_retrieval_comp = spec.components[-1]
                        last_retrieval_id = node_id_by_name.get(
                            last_retrieval_comp.name
                        )

                        if last_retrieval_id:
                            retrieval_edges.append(
                                (
                                    last_retrieval_id,
                                    first_in_reversed,
                                    GraphRelationship.FLOWS_TO.value,
                                )
                            )

            # Add substituted nodes to Neo4j
            if node_substitutions:
                self.graph_store.add_nodes_batch(node_substitutions, "Component")

            # Add all retrieval edges to Neo4j
            if retrieval_edges:
                self.graph_store.add_edges_batch(
                    retrieval_edges,
                    source_label="Component",
                    target_label="Component",
                )

            # Add retriever-to-DocumentStore edges separately
            if retriever_to_docstore_edges:
                self.graph_store.add_edges_batch(
                    retriever_to_docstore_edges,
                    source_label="Component",
                    target_label="DocumentStore",
                )

    def build_pipeline_graph(self, spec: PipelineSpec) -> None:
        """Build a graph representation of the pipeline specification."""

        # Determine connections based on component order and dependencies
        connections = self._determine_connections(spec.components)

        # Create graph representation
        self.create_pipeline_graph(spec, connections)

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
