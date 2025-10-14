"""Pipeline runner for executing Haystack pipelines with data.

ARCHITECTURE:
- Factory: Build pipelines once and store in Neo4j (creation time)
- Runner: Load pipelines from Neo4j and execute (runtime)
"""

from typing import Any, Dict, List, Optional

from ..components import GraphStore
from ..types import PipelineUsage


class PipelineRunner:
    """Executes pipelines with input data."""

    def __init__(
        self,
        graph_store: Optional[GraphStore] = None,
        username: Optional[str] = None,
        pipeline_names: Optional[List[str]] = None,
    ) -> None:
        """
        Initialize the pipeline runner.

        Args:
            graph_store: GraphStore for loading pipelines from Neo4j
            username: Username to load pipelines for (auto-loads pipelines if provided)
            pipeline_names: List of pipeline names to load (e.g., ['pdf_indexing_pipeline'])
                          If None and username is provided, loads all pipelines for user
        """
        self.graph_store = graph_store
        self.username = username

        # Graph representations from Neo4j for multiple pipelines
        self._pipeline_graphs: Dict[str, List[Dict[str, Any]]] = {}
        # Haystack components by pipeline (not yet connected)
        self._haystack_components_by_pipeline: Dict[str, Dict[str, Any]] = {}
        # Actual Haystack Pipeline objects (connected and ready to run)
        self._haystack_pipelines: Dict[str, Any] = {}

        # Auto-load and build pipelines if username is provided
        if username and graph_store:
            self._auto_load_pipelines(pipeline_names)

    def _auto_load_pipelines(self, pipeline_names: Optional[List[str]] = None) -> None:
        """
        Automatically load and build pipelines for the user.

        Args:
            pipeline_names: List of pipeline names to load. If None, loads all user pipelines.
        """
        if not self.username:
            raise ValueError("Username is required for auto-loading pipelines")

        if not pipeline_names:
            # If no specific pipelines provided, try common pipeline names
            pipeline_names = ["pdf_indexing_pipeline", "pdf_retrieval_pipeline"]

        print(f"Auto-loading pipelines for user: {self.username}")
        print(f"Pipeline names: {pipeline_names}\n")

        for pipeline_name in pipeline_names:
            try:
                # Load pipeline graph
                self.load_pipeline_graph([pipeline_name], self.username)

                # Build Haystack components
                self.build_haystack_components_from_graph(pipeline_name)

                # Create connected pipeline
                self.create_haystack_pipeline(pipeline_name)

                print(f"Successfully loaded: {pipeline_name}")

            except Exception as e:
                print(f"Warning: Could not load {pipeline_name}: {e}")

        print(f"\nTotal pipelines loaded: {len(self._haystack_pipelines)}\n")

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

        # Add all components from the graph (filter out only DocumentStore nodes)
        # Components may come from connected pipelines (e.g., retrievers accessing other pipeline's stores)
        component_nodes = [
            comp
            for comp in components_data
            if "DocumentStore" not in comp.get("node_labels", [])
        ]

        # Add components to pipeline using comp_id as unique names
        for comp_data in component_nodes:
            comp_id = comp_data.get("id")
            comp_name = comp_data.get("component_name")

            if comp_id in haystack_components:
                # Use comp_id as the component name to ensure uniqueness
                pipeline.add_component(comp_id, haystack_components[comp_id])
                print(f"  Added component: {comp_name} (id: {comp_id})")

        # Connect components based on graph edges
        for comp_data in component_nodes:
            current_id = comp_data.get("id")
            current_name = comp_data.get("component_name")
            next_ids = comp_data.get("next_components", [])

            if current_id in haystack_components:
                for next_id in next_ids:
                    # Only connect if next component exists in haystack_components
                    if next_id in haystack_components:
                        next_data = next(
                            (c for c in component_nodes if c.get("id") == next_id), None
                        )
                        if next_data:
                            next_name = next_data.get("component_name")
                            try:
                                pipeline.connect(current_id, next_id)
                                print(f"  Connected: {current_name} -> {next_name}")
                            except Exception as e:
                                print(
                                    f"  Warning: Could not connect {current_name} -> {next_name}: {e}"
                                )

        print(f"Pipeline '{pipeline_name}' created successfully\n")

        # Store the connected pipeline
        self._haystack_pipelines[pipeline_name] = pipeline

        return pipeline

    def run(self, pipeline_name: str, type: str, **kwargs: Any) -> Any:
        """
        Run a pipeline by name, dispatching to the appropriate execution method.

        Args:
            pipeline_name: Name of the pipeline to run (e.g., 'pdf_indexing_pipeline', 'pdf_retrieval_pipeline')
            type: Pipeline type - "indexing" or "retrieval"
            **kwargs: Pipeline-specific arguments

        Returns:
            Pipeline execution results
        """
        if type == "indexing" or type == PipelineUsage.INDEXING.value:
            return self._run_indexing_pipeline(pipeline_name, **kwargs)
        elif type == "retrieval" or type == PipelineUsage.RETRIEVAL.value:
            return self._run_retrieval_pipeline(pipeline_name, **kwargs)
        else:
            raise ValueError(
                f"Unknown pipeline type: {type}. " "Must be 'indexing' or 'retrieval'"
            )

    def _run_indexing_pipeline(self, pipeline_name: str, **kwargs: Any) -> Any:
        """
        Execute an indexing pipeline.

        Args:
            pipeline_name: Name of the indexing pipeline
            **kwargs: Pipeline-specific arguments
                - data_path: Path to directory containing PDFs (required)
                - sources: Optional list of specific file paths to process

        Returns:
            Indexing results
        """
        from pathlib import Path

        # Get the pipeline
        if pipeline_name not in self._haystack_pipelines:
            raise ValueError(
                f"Pipeline '{pipeline_name}' not found. "
                f"Call create_haystack_pipeline('{pipeline_name}') first."
            )

        pipeline = self._haystack_pipelines[pipeline_name]

        # Get data path from kwargs
        data_path = kwargs.get("data_path")
        if not data_path:
            raise ValueError("data_path is required in kwargs")

        data_dir = Path(data_path)
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        # Try to detect file type based on pipeline name or find any supported files
        input_files: List[Any] = []
        file_patterns = ["*.pdf", "*.txt", "*.md", "*.docx", "*.html"]

        for pattern in file_patterns:
            input_files.extend(data_dir.glob(pattern))

        if not input_files:
            raise FileNotFoundError(f"No supported files found in {data_dir}")

        print(f"\nRunning indexing pipeline: {pipeline_name}")
        print(f"Found {len(input_files)} files in {data_dir}")

        # Run the pipeline with the file sources
        sources = [str(file_path) for file_path in input_files]
        result = pipeline.run({"sources": sources})

        print(f"Indexing completed for {len(input_files)} files\n")

        return result

    def _run_retrieval_pipeline(self, pipeline_name: str, **kwargs: Any) -> Any:
        """
        Execute a retrieval pipeline.

        Args:
            pipeline_name: Name of the retrieval pipeline
            **kwargs: Pipeline-specific arguments (e.g., query, top_k)

        Returns:
            Retrieval results
        """
        # TODO: Implement retrieval pipeline execution
        pass
