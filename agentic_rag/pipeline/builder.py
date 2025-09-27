"""Pipeline builder for creating Haystack pipelines from specifications."""

from typing import Any, List, Optional, Tuple

from haystack import Pipeline

from ..components import ComponentRegistry, GraphStore
from ..types import ComponentSpec, PipelineSpec, create_haystack_component
from .graph_builder import PipelineGraphBuilder


class PipelineBuilder:
    """Builds Haystack pipelines from pipeline specifications."""

    def __init__(
        self, registry: ComponentRegistry, graph_store: Optional[GraphStore] = None
    ) -> None:
        self.registry = registry
        self.graph_store = graph_store
        self.graph_builder = PipelineGraphBuilder(graph_store) if graph_store else None

    def build_pipeline_graph(self, spec: PipelineSpec) -> None:
        """Build a graph representation of the pipeline specification."""

        # Determine connections based on component order and dependencies
        connections = self._determine_connections(spec.components)

        # Create graph representation if graph builder is available
        if self.graph_builder:
            self.graph_builder.create_pipeline_graph(spec, connections)
        else:
            print("Warning: No graph store configured, pipeline graph not created")

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
