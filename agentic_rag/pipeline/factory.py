"""Factory for creating pipelines from specifications."""

from typing import Any, Dict, List, Optional

from ..components import GraphStore, get_default_registry
from ..types import PipelineSpec, get_component_value, validate_component_spec
from .storage import GraphStorage


class PipelineFactory:
    """Factory for creating pipelines from component specifications."""

    def __init__(
        self, graph_store: Optional[GraphStore] = None, username: str = "test_user"
    ) -> None:
        self.registry = get_default_registry()
        self.graph_store = graph_store
        self.username = username
        self.graph_storage = (
            GraphStorage(graph_store, self.registry) if graph_store else None
        )

    def build_pipeline_graphs_from_specs(
        self,
        pipeline_specs: List[List[Dict[str, str]]],
        configs: Optional[List[Dict[str, Any]]] = None,
        username: Optional[str] = None,
    ) -> List[PipelineSpec]:
        """
        Build multiple pipeline graphs from dict-based specifications.

        Args:
            pipeline_specs: List of component specifications as dicts.
                Example: [[{"type": "CONVERTER.PDF"}, {"type": "CHUNKER.RECURSIVE"}]]
            configs: Optional list of configuration dicts for each pipeline
            username: Username for pipeline ownership (defaults to factory's username)

        Returns:
            List of PipelineSpec objects with graph representations built
        """
        if configs is None:
            configs = [{}] * len(pipeline_specs)

        if len(configs) != len(pipeline_specs):
            raise ValueError("Number of configs must match number of pipeline specs")

        # Use provided username or fall back to factory's username
        effective_username = username or self.username

        pipeline_specs_list = []

        for i, (spec, config) in enumerate(zip(pipeline_specs, configs)):
            if len(spec) < 1 or len(spec) > 5:
                raise ValueError(
                    f"Pipeline {i} must have 1-5 components, got {len(spec)}"
                )

            pipeline_name = f"pipeline_{i}"
            pipeline_spec = self.build_pipeline_graph(
                spec, pipeline_name, config, effective_username
            )
            pipeline_specs_list.append(pipeline_spec)

        return pipeline_specs_list

    def build_pipeline_graph(
        self,
        component_specs: List[Dict[str, str]],
        pipeline_name: str,
        config: Optional[Dict[str, Any]] = None,
        username: Optional[str] = None,
    ) -> PipelineSpec:
        """
        Build a single pipeline graph from dict-based component specifications.

        Args:
            component_specs: List of component specifications as dicts
                Example: [{"type": "CONVERTER.PDF"}, {"type": "CHUNKER.RECURSIVE"}]
            pipeline_name: Name for the pipeline
            config: Optional configuration dict
            username: Username for pipeline ownership (defaults to factory's username)

        Returns:
            PipelineSpec with graph representation built
        """
        config = config or {}

        # Use provided username or fall back to factory's username
        effective_username = username or self.username

        # Parse component specifications and validate
        component_specs_list = []
        for spec_item in component_specs:
            component_name = self._parse_component_spec(spec_item)

            spec = self.registry.get_component_spec(component_name)
            if spec is None:
                raise ValueError(f"Unknown component: {component_name}")

            # Configure the spec directly with user config
            user_config = config.get(component_name, {})
            configured_spec = spec.configure(user_config)
            component_specs_list.append(configured_spec)

        # Create pipeline specification - no separate component_configs needed!
        pipeline_spec = PipelineSpec(
            name=pipeline_name,
            components=component_specs_list,  # Already configured!
        )

        # Build the graph representation (no Haystack pipeline)
        if self.graph_storage:
            self.graph_storage.build_pipeline_graph(pipeline_spec, effective_username)
        else:
            print("Warning: No graph store configured, pipeline graph not created")

        return pipeline_spec

    def _parse_component_spec(self, spec_item: Dict[str, str]) -> str:
        """
        Parse a component specification dict into a component name.

        Args:
            spec_item: Dict with 'type' key, e.g. {"type": "CONVERTER.PDF"}

        Returns:
            Component registry name

        Raises:
            ValueError: If specification is invalid
        """
        if not isinstance(spec_item, dict):
            raise ValueError(f"Component spec must be dict, got: {type(spec_item)}")

        if "type" not in spec_item:
            raise ValueError("Component spec must have 'type' key")

        type_spec = spec_item["type"]
        if "." not in type_spec:
            raise ValueError(
                f"Component type must be in format 'CATEGORY.TYPE', got: {type_spec}"
            )

        if not validate_component_spec(type_spec):
            raise ValueError(f"Invalid component specification: {type_spec}")

        return get_component_value(type_spec)
