"""Pipeline runner for executing Haystack pipelines with data.

ARCHITECTURE:
- Factory: Build pipelines once and store in Neo4j (creation time)
- Runner: Load pipelines from Neo4j and execute (runtime)
"""

from typing import Any, Dict, List, Optional

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
        # Graph representations from Neo4j for multiple pipelines
        self._pipeline_graphs: Dict[str, List[Dict[str, Any]]] = {}
        # Haystack components by pipeline (not yet connected)
        self._haystack_components_by_pipeline: Dict[str, Dict[str, Any]] = {}
        # Actual Haystack Pipeline objects (connected and ready to run)
        self._haystack_pipelines: Dict[str, Any] = {}

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

        # Store the graph representations for all loaded pipelines
        self._pipeline_graphs.update(pipelines_data)

    def build_haystack_components_from_graph(self, pipeline_name: str) -> None:
        """
        Build runtime Haystack components from loaded pipeline graph data.

        Args:
            pipeline_name: Name of the pipeline to build components for

        Raises:
            RuntimeError: If no pipeline graph is loaded
            ValueError: If pipeline name not found in loaded data
        """
        if not self._pipeline_graphs:
            raise RuntimeError(
                "No pipeline graphs loaded. Call load_pipeline_graph() first."
            )

        if pipeline_name not in self._pipeline_graphs:
            raise ValueError(
                f"Pipeline '{pipeline_name}' not found in loaded data. "
                f"Available: {list(self._pipeline_graphs.keys())}"
            )

        components_data = self._pipeline_graphs[pipeline_name]

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

            if not comp_name or not comp_id:
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

        # Store the components for this pipeline
        self._haystack_components_by_pipeline[pipeline_name] = haystack_components

    def create_haystack_pipeline(self, pipeline_name: str) -> Any:
        """
        Create an actual Haystack Pipeline from components by connecting them sequentially.

        Args:
            pipeline_name: Name of the pipeline to create

        Returns:
            Connected Haystack Pipeline object ready to run

        Raises:
            RuntimeError: If components haven't been built yet
        """
        from haystack import Pipeline

        if pipeline_name not in self._haystack_components_by_pipeline:
            raise RuntimeError(
                f"Components for pipeline '{pipeline_name}' not built yet. "
                f"Call build_haystack_components_from_graph() first."
            )

        # Get the graph data to determine component order
        if pipeline_name not in self._pipeline_graphs:
            raise RuntimeError(f"Graph data for pipeline '{pipeline_name}' not found.")

        components_data = self._pipeline_graphs[pipeline_name]
        haystack_components = self._haystack_components_by_pipeline[pipeline_name]

        # Create Haystack pipeline
        pipeline = Pipeline()

        print(f"\nCreating Haystack pipeline for: {pipeline_name}")

        # Add all components to pipeline (filter out DocumentStore nodes)
        component_nodes = [
            comp
            for comp in components_data
            if "DocumentStore" not in comp.get("node_labels", [])
        ]

        # Add components to pipeline
        for comp_data in component_nodes:
            comp_id = comp_data.get("id")
            comp_name = comp_data.get("component_name")

            if comp_id in haystack_components:
                pipeline.add_component(comp_name, haystack_components[comp_id])
                print(f"  Added component: {comp_name}")

        # Connect components sequentially based on graph order
        for i in range(len(component_nodes) - 1):
            current_comp = component_nodes[i]
            next_comp = component_nodes[i + 1]

            current_name = current_comp.get("component_name")
            next_name = next_comp.get("component_name")

            if current_name and next_name:
                try:
                    pipeline.connect(current_name, next_name)
                    print(f"  Connected: {current_name} -> {next_name}")
                except Exception as e:
                    print(
                        f"  Warning: Could not connect {current_name} -> {next_name}: {e}"
                    )

        print(f"Pipeline '{pipeline_name}' created successfully\n")

        # Store the connected pipeline
        self._haystack_pipelines[pipeline_name] = pipeline

        return pipeline
