"""Pipeline graph builder for creating graph representations of pipelines."""

from typing import Any, Dict, List, Tuple

from ..components import GraphStore
from ..types import (
    ComponentNode,
    ComponentType,
    DocumentStoreNode,
    PipelineSpec,
    PipelineType,
    UserNode,
)


class PipelineGraphBuilder:
    """Builds graph representations of pipeline components."""

    def __init__(self, graph_store: GraphStore) -> None:
        self.graph_store = graph_store

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
                        (writer_id, doc_store_dict["id"], "WRITES_TO")
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
