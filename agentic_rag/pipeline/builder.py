"""Pipeline builder for creating Haystack pipelines from specifications."""

from typing import Any, Dict, List, Tuple

from haystack import Pipeline

from ..components import ComponentRegistry
from ..types import ComponentSpec, PipelineSpec, create_haystack_component


class PipelineBuilder:
    """Builds Haystack pipelines from pipeline specifications."""

    def __init__(self, registry: ComponentRegistry) -> None:
        self.registry = registry

    def build_haystack_pipeline(self, spec: PipelineSpec) -> Any:
        """Build a Haystack pipeline from a pipeline specification."""

        # Create Haystack pipeline
        pipeline = Pipeline()

        # Create component instances and add to pipeline
        component_map = {}
        for component_spec in spec.components:
            component_config = spec.component_configs.get(component_spec.name, {})
            haystack_component = create_haystack_component(
                component_spec, component_config
            )

            pipeline.add_component(component_spec.name, haystack_component)
            component_map[component_spec.name] = component_spec.name

        # Determine connections based on component order and dependencies
        connections = self._determine_connections(spec.components, component_map)

        # Add connections to pipeline
        for source, target in connections:
            try:
                pipeline.connect(source, target)
            except Exception as e:
                print(f"Warning: Could not connect {source} -> {target}: {e}")

        return pipeline

    def _determine_connections(
        self, components: List[ComponentSpec], component_map: Dict[str, str]
    ) -> List[Tuple[str, str]]:
        """Determine how components should be connected."""
        connections: List[Tuple[str, str]] = []

        if len(components) <= 1:
            return connections

        # Create connections based on dependencies and sequential order
        for i, component in enumerate(components):
            component_name = component_map[component.name]

            # Connect to dependencies first
            for dep_name in component.dependencies:
                if dep_name in component_map:
                    dep_component_name = component_map[dep_name]
                    connections.append((dep_component_name, component_name))

            # If no dependencies, connect to previous component in sequence
            if not component.dependencies and i > 0:
                prev_component = components[i - 1]
                prev_component_name = component_map[prev_component.name]
                connections.append((prev_component_name, component_name))

        return connections
