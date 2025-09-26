"""Pipeline builder for creating Haystack pipelines from specifications."""

import json
from typing import Any, Dict, List, Optional, Tuple

from haystack import Pipeline

from ..components import ComponentRegistry, GraphStore
from ..types import (
    ComponentNode,
    ComponentSpec,
    ComponentType,
    DocumentStoreNode,
    PipelineSpec,
    UserNode,
    create_haystack_component,
)


class PipelineBuilder:
    """Builds Haystack pipelines from pipeline specifications."""

    def __init__(
        self, registry: ComponentRegistry, graph_store: Optional[GraphStore] = None
    ) -> None:
        self.registry = registry
        self.graph_store = graph_store

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

        # Create graph representation if graph store is available
        if self.graph_store:
            self._create_pipeline_graph(spec, connections)

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

    def _create_pipeline_graph(
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

        if self.graph_store is None:
            return

        # Add/update the owning user node
        user_node = UserNode(username="test_user", display_name="Test User")
        user_dict = user_node.to_dict()
        self.graph_store.add_nodes_batch([user_dict], "User")

        # Add component nodes
        self.graph_store.add_nodes_batch(nodes, "Component")

        document_store_nodes: List[Dict[str, Any]] = []
        component_to_store_edges: List[Tuple[str, str, str]] = []
        store_to_component_edges: List[Tuple[str, str, str]] = []

        for index, component_spec in enumerate(spec.components):
            if component_spec.name != "chroma_document_writer":
                continue

            writer_config = component_spec.get_config()
            root_dir = writer_config.get("root_dir", ".")

            related_components: List[Dict[str, Any]] = []
            for related in spec.components[: index + 1]:
                if related.component_type in {
                    ComponentType.EMBEDDER,
                    ComponentType.RETRIEVER,
                }:
                    related_components.append(
                        {
                            "name": related.name,
                            "config": related.get_config(),
                        }
                    )

            retrieval_components_json = json.dumps(
                related_components, sort_keys=True, separators=(",", ":")
            )

            doc_store_node = DocumentStoreNode(
                pipeline_name=spec.name,
                root_dir=root_dir,
                retrieval_components_json=retrieval_components_json,
            )
            doc_store_dict = doc_store_node.to_dict()
            document_store_nodes.append(doc_store_dict)

            writer_id = node_id_by_name.get(component_spec.name)
            if writer_id:
                component_to_store_edges.append(
                    (writer_id, doc_store_dict["id"], "WRITES_TO")
                )

            for related in spec.components[: index + 1]:
                if related.component_type in {
                    ComponentType.EMBEDDER,
                    ComponentType.RETRIEVER,
                }:
                    related_id = node_id_by_name.get(related.name)
                    if related_id:
                        store_to_component_edges.append(
                            (doc_store_dict["id"], related_id, "USES_DOCUMENT_STORE")
                        )

        if document_store_nodes:
            self.graph_store.add_nodes_batch(document_store_nodes, "DocumentStore")

        if component_to_store_edges:
            self.graph_store.add_edges_batch(
                component_to_store_edges,
                source_label="Component",
                target_label="DocumentStore",
            )
        if store_to_component_edges:
            self.graph_store.add_edges_batch(
                store_to_component_edges,
                source_label="DocumentStore",
                target_label="Component",
            )

        # Connect user to the first component in the pipeline
        if spec.components:
            first_component_id = node_id_by_name.get(spec.components[0].name)
            if first_component_id:
                self.graph_store.add_edges_batch(
                    [(user_dict["id"], first_component_id, "STARTS_WITH")],
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
                    component_edges.append((source_id, target_id, "CONNECTED_TO"))
            if component_edges:
                self.graph_store.add_edges_batch(
                    component_edges,
                    source_label="Component",
                    target_label="Component",
                )
