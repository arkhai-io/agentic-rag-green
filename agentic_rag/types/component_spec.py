"""Component specification and instance definitions."""

from dataclasses import dataclass, field
from typing import Any, Dict, List

from .data_types import ComponentType, DataType


@dataclass
class ComponentSpec:
    """Specification for a pipeline component."""

    name: str
    component_type: ComponentType
    haystack_class: str
    input_types: List[DataType]
    output_types: List[DataType]
    default_config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    parallelizable: bool = True

    def is_compatible_input(self, data_type: DataType) -> bool:
        """Check if a data type is compatible with this component's inputs."""
        return data_type in self.input_types


def create_haystack_component(spec: ComponentSpec, config: Dict[str, Any]) -> Any:
    """Create a Haystack component from specification and configuration."""
    # Merge default config with provided config
    merged_config = {**spec.default_config, **config}

    # Dynamic import and instantiation
    module_path, class_name = spec.haystack_class.rsplit(".", 1)

    try:
        import importlib

        module = importlib.import_module(module_path)
        component_class = getattr(module, class_name)
        return component_class(**merged_config)
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Failed to import Haystack component {spec.haystack_class}: {e}"
        )
