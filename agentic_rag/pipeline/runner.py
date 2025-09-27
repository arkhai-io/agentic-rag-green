"""Pipeline runner for executing Haystack pipelines with data."""

from typing import Any, Dict, List, Optional, Tuple

from ..types import PipelineSpec
from .factory import PipelineFactory


class PipelineRunner:
    """Executes pipelines with input data."""

    def __init__(self, factory: Optional[PipelineFactory] = None) -> None:
        """
        Initialize the pipeline runner.

        Args:
            factory: Optional PipelineFactory instance. If None, creates a new one.
        """
        self.factory = factory or PipelineFactory()
        self._active_pipeline: Optional[Tuple[PipelineSpec, Any]] = None

    def load_pipeline(
        self,
        component_specs: List[Dict[str, str]],
        pipeline_name: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Load a pipeline from component specifications.

        Args:
            component_specs: List of component specifications
                Example: [{"type": "CONVERTER.PDF"}, {"type": "CHUNKER.MARKDOWN_AWARE"}]
            pipeline_name: Name for the pipeline
            config: Optional configuration dict
        """
        # Create pipeline graph
        spec = self.factory.build_pipeline_graph(component_specs, pipeline_name, config)

        # Build Haystack pipeline for execution
        haystack_pipeline = self.factory.pipeline_manager.build_haystack_pipeline(spec)

        self._active_pipeline = (spec, haystack_pipeline)

    def load_pipeline_from_spec(
        self, spec: PipelineSpec, haystack_pipeline: Any
    ) -> None:
        """
        Load a pipeline from existing spec and Haystack pipeline.

        Args:
            spec: Pipeline specification
            haystack_pipeline: Built Haystack pipeline
        """
        self._active_pipeline = (spec, haystack_pipeline)

    def run(self, pipeline_type: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the active pipeline with the given inputs.

        Args:
            pipeline_type: Type of pipeline execution ("indexing" or "retrieval")
            inputs: Input data for the pipeline
                For indexing: {"documents": [Document(...), ...]}
                For retrieval: {"query": "search query", "top_k": 5}

        Returns:
            Pipeline execution results

        Raises:
            RuntimeError: If no pipeline is loaded
            ValueError: If inputs are invalid or pipeline type unsupported
        """
        if self._active_pipeline is None:
            raise RuntimeError("No pipeline loaded. Call load_pipeline() first.")

        spec, haystack_pipeline = self._active_pipeline

        if pipeline_type == "indexing":
            return self._run_indexing_pipeline(spec, haystack_pipeline, inputs)
        elif pipeline_type == "retrieval":
            return self._run_retrieval_pipeline(spec, haystack_pipeline, inputs)
        else:
            raise ValueError(
                f"Unsupported pipeline type: {pipeline_type}. Use 'indexing' or 'retrieval'"
            )

    def _run_indexing_pipeline(
        self, spec: PipelineSpec, haystack_pipeline: Any, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run an indexing pipeline to process and store documents.

        Args:
            spec: Pipeline specification
            haystack_pipeline: Haystack pipeline instance
            inputs: Input data with documents or file paths

        Returns:
            Results from the indexing pipeline
        """
        try:
            # Handle direct document input only
            if "documents" not in inputs:
                raise ValueError(
                    "Indexing inputs must contain 'documents' key with list of Document objects"
                )

            documents = inputs["documents"]
            if not isinstance(documents, list):
                raise ValueError("'documents' must be a list of Document objects")

            # Map documents to the first component in the pipeline
            pipeline_inputs = {spec.components[0].name: {"documents": documents}}

            # Execute the pipeline
            results = haystack_pipeline.run(data=pipeline_inputs)

            return {
                "success": True,
                "results": results,
                "processed_count": len(documents),
            }

        except Exception as e:
            return {"success": False, "error": str(e), "error_type": type(e).__name__}

    def _run_retrieval_pipeline(
        self, spec: PipelineSpec, haystack_pipeline: Any, inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run a retrieval pipeline to search for relevant documents.

        Args:
            spec: Pipeline specification
            haystack_pipeline: Haystack pipeline instance
            inputs: Input data with query and search parameters

        Returns:
            Results from the retrieval pipeline
        """
        try:
            # Handle query input - required for retrieval
            if "query" not in inputs:
                raise ValueError(
                    "Retrieval inputs must contain 'query' key with search query string"
                )

            query = inputs["query"]
            if not isinstance(query, str):
                raise ValueError("'query' must be a string")

            # Extract optional parameters
            top_k = inputs.get("top_k", 10)  # Default to 10 results
            filters = inputs.get("filters", None)

            # Build pipeline inputs with all component parameters
            # Note: SentenceTransformersTextEmbedder expects 'text' parameter, not 'query'
            pipeline_inputs = {spec.components[0].name: {"text": query}}

            # Find retriever component and add its parameters to pipeline_inputs
            for component in spec.components:
                if component.component_type.value == "retriever":
                    component_params = {}
                    if top_k is not None:
                        component_params["top_k"] = top_k
                    if filters is not None:
                        component_params["filters"] = filters
                    if component_params:
                        pipeline_inputs[component.name] = component_params
                    break

            # Execute the pipeline
            results = haystack_pipeline.run(data=pipeline_inputs)

            return {"success": True, "results": results, "query": query}

        except Exception as e:
            return {"success": False, "error": str(e), "error_type": type(e).__name__}
