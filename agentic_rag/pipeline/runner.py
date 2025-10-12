"""Pipeline runner for executing Haystack pipelines with data.

ARCHITECTURE:
- Factory: Build pipelines once and store in Neo4j (creation time)
- Runner: Load pipelines from Neo4j and execute (runtime)
"""

from typing import Any, Dict, List, Optional, Tuple

from ..components import GraphStore


class PipelineRunner:
    """Executes pipelines with input data."""

    def __init__(
        self,
        graph_store: Optional[GraphStore] = None,
    ) -> None:
        """
        Initialize the pipeline runner.

        Args:
            graph_store: GraphStore for loading pipelines from Neo4j
        """
        self.graph_store = graph_store
        self._active_pipeline_graph: Optional[Tuple[Any, Any]] = (
            None  # Graph representation from Neo4j
        )
        self._active_pipeline: Optional[Dict[str, Any]] = (
            None  # Runtime Haystack components
        )

    def load_pipeline_graph(self, pipeline_hashes: List[str], username: str) -> None:
        """
        Load a pipeline from Neo4j using pipeline hashes and username.

        This method loads pipelines by their hash identifiers and associates them
        with a specific user context.

        Args:
            pipeline_hashes: List of pipeline hash identifiers
            username: Username for pipeline context and permissions

        Raises:
            RuntimeError: If no graph store is configured
            ValueError: If pipeline hashes not found or invalid username
        """
        if not self.graph_store:
            raise RuntimeError(
                "No graph store configured. Pass GraphStore to constructor."
            )

        # Delegate to GraphStorage for actual loading logic
        from ..components import get_default_registry
        from .storage import GraphStorage

        registry = get_default_registry()
        graph_storage = GraphStorage(self.graph_store, registry)

        pipelines_data = graph_storage.load_pipeline_by_hashes(
            pipeline_hashes, username
        )

        # Store the graph representation - call build_haystack_components_from_graph() to build runtime components
        self._active_pipeline_graph = (pipeline_hashes[0], pipelines_data)

    def build_haystack_components_from_graph(
        self, pipeline_name: str
    ) -> Dict[str, Any]:
        """
        Build runtime Haystack components from loaded pipeline graph data.

        Args:
            pipeline_name: Name of the pipeline to build components for

        Returns:
            Dictionary mapping component IDs to instantiated Haystack components

        Raises:
            RuntimeError: If no pipeline graph is loaded
            ValueError: If pipeline name not found in loaded data
        """
        if self._active_pipeline_graph is None:
            raise RuntimeError(
                "No pipeline graph loaded. Call load_pipeline_graph() first."
            )

        _, pipelines_data = self._active_pipeline_graph

        if pipeline_name not in pipelines_data:
            raise ValueError(
                f"Pipeline '{pipeline_name}' not found in loaded data. "
                f"Available: {list(pipelines_data.keys())}"
            )

        components_data = pipelines_data[pipeline_name]

        # Import registry to look up component specs
        from ..components import get_default_registry

        registry = get_default_registry()

        # Build components dynamically
        haystack_components = {}

        print(f"\nBuilding Haystack components for: {pipeline_name}")

        for comp_data in components_data:
            comp_id = comp_data.get("id")
            node_labels = comp_data.get("node_labels", [])

            # Skip DocumentStore nodes - they're created automatically by components
            if "DocumentStore" in node_labels:
                continue

            # Handle Component nodes
            comp_name = comp_data.get("component_name")
            config_json = comp_data.get("component_config_json", "{}")

            if not comp_name:
                continue

            try:
                # Get the base component spec from registry
                base_spec = registry.get_component_spec(comp_name)
                if not base_spec:
                    print(f"  Error: Component '{comp_name}' not found in registry")
                    continue

                # Parse and apply configuration
                import json
                from copy import deepcopy

                from ..types import create_haystack_component

                config = json.loads(config_json) if config_json != "{}" else {}

                # Create a copy to avoid mutating the registry's spec
                spec_copy = deepcopy(base_spec)

                # Configure with loaded config
                if config:
                    configured_spec = spec_copy.configure(config)
                else:
                    configured_spec = spec_copy

                # Instantiate the actual Haystack component
                # For writers/retrievers, create_haystack_component will handle document store creation
                haystack_component = create_haystack_component(configured_spec)
                haystack_components[comp_id] = haystack_component

                print(f"  Built {comp_name} ({type(haystack_component).__name__})")

            except Exception as e:
                print(f"  Error building {comp_name}: {e}")

        print(f"Built {len(haystack_components)} components\n")

        # Store the runtime components as the active pipeline
        self._active_pipeline = haystack_components

        return haystack_components
