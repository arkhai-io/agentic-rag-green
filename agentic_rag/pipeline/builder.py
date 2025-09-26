"""Pipeline builder for creating Haystack pipelines from specifications."""

from typing import Any, List, Optional, Tuple

from haystack import Pipeline

from ..components import ComponentRegistry, GraphStore
from ..types import (
    ComponentNode,
    ComponentSpec,
    PipelineSpec,
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

        nodes = []
        node_id_by_name = {}
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

        if self.graph_store is not None:
            self.graph_store.add_nodes_batch(nodes, "Component")
            if connections:
                edges = []
                for source_name, target_name in connections:
                    source_id = node_id_by_name.get(source_name)
                    target_id = node_id_by_name.get(target_name)
                    if source_id and target_id:
                        edges.append((source_id, target_id, "CONNECTED_TO"))
                if edges:
                    self.graph_store.add_edges_batch(
                        edges, source_label="Component", target_label="Component"
                    )
