"""Pipeline specification definitions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

from .component_spec import ComponentSpec


class PipelineType(Enum):
    """Type of pipeline - indexing or retrieval."""

    INDEXING = "indexing"
    RETRIEVAL = "retrieval"


@dataclass
class PipelineSpec:
    """Specification for a complete pipeline."""

    name: str
    components: List[ComponentSpec]
    pipeline_type: PipelineType = PipelineType.INDEXING
    connections: List[Tuple[str, str]] = field(default_factory=list)
    indexing_pipelines: Optional[Union[List[str], Dict[str, str]]] = (
        None  # List of pipeline names or dict mapping store_name -> indexing_pipeline_name
    )

    def __post_init__(self) -> None:
        """Validate pipeline specification."""
        if len(self.components) < 1:
            raise ValueError("Pipeline must have at least 1 component")
        if len(self.components) > 10:
            raise ValueError("Pipeline cannot have more than 10 components")

    def get_component_by_name(self, name: str) -> Optional[ComponentSpec]:
        """Get a component spec by name."""
        for component in self.components:
            if component.name == name:
                return component
        return None

    def validate_dependencies(self) -> bool:
        """Validate that all component dependencies are satisfied."""
        component_names = {comp.name for comp in self.components}

        for component in self.components:
            for dependency in component.dependencies:
                if dependency not in component_names:
                    raise ValueError(
                        f"Component {component.name} depends on {dependency} "
                        f"which is not present in pipeline"
                    )
        return True
