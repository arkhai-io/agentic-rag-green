"""Factory for creating dynamic pipelines from specifications."""

from typing import Any, Dict, List, Optional, Tuple

from ..components import get_default_registry
from ..types import PipelineSpec, get_component_value, validate_component_spec
from .builder import PipelineBuilder


class PipelineFactory:
    """Factory for creating pipelines from component specifications."""

    def __init__(self) -> None:
        self.registry = get_default_registry()
        self.builder = PipelineBuilder(self.registry)

    def create_pipelines_from_specs(
        self,
        pipeline_specs: List[List[Dict[str, str]]],
        configs: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Tuple[PipelineSpec, Any]]:
        """
        Create multiple pipelines from dict-based specifications.

        Args:
            pipeline_specs: List of component specifications as dicts.
                Example: [[{"type": "CONVERTER.PDF"}, {"type": "CHUNKER.RECURSIVE"}]]
            configs: Optional list of configuration dicts for each pipeline

        Returns:
            List of tuples containing (PipelineSpec, haystack.Pipeline)
        """
        if configs is None:
            configs = [{}] * len(pipeline_specs)

        if len(configs) != len(pipeline_specs):
            raise ValueError("Number of configs must match number of pipeline specs")

        pipelines = []

        for i, (spec, config) in enumerate(zip(pipeline_specs, configs)):
            if len(spec) < 1 or len(spec) > 5:
                raise ValueError(
                    f"Pipeline {i} must have 1-5 components, got {len(spec)}"
                )

            pipeline_name = f"pipeline_{i}"
            pipeline_spec, haystack_pipeline = self.create_pipeline_from_spec(
                spec, pipeline_name, config
            )
            pipelines.append((pipeline_spec, haystack_pipeline))

        return pipelines

    def create_pipeline_from_spec(
        self,
        component_specs: List[Dict[str, str]],
        pipeline_name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[PipelineSpec, Any]:
        """
        Create a single pipeline from dict-based component specifications.

        Args:
            component_specs: List of component specifications as dicts
                Example: [{"type": "CONVERTER.PDF"}, {"type": "CHUNKER.RECURSIVE"}]
            pipeline_name: Name for the pipeline
            config: Optional configuration dict

        Returns:
            Tuple of (PipelineSpec, haystack.Pipeline)
        """
        config = config or {}

        # Parse component specifications and validate
        component_specs_list = []
        component_configs = {}
        for spec_item in component_specs:
            component_name = self._parse_component_spec(spec_item)

            spec = self.registry.get_component_spec(component_name)
            if spec is None:
                raise ValueError(f"Unknown component: {component_name}")

            component_specs_list.append(spec)
            # Get component-specific config if provided
            component_configs[component_name] = config.get(component_name, {})

        # Create pipeline specification
        pipeline_spec = PipelineSpec(
            name=pipeline_name,
            components=component_specs_list,
            component_configs=component_configs,
            pipeline_config=config.get("pipeline", {}),
        )

        # Build the actual Haystack pipeline
        haystack_pipeline = self.builder.build_haystack_pipeline(pipeline_spec)

        return pipeline_spec, haystack_pipeline

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
