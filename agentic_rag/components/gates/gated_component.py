"""Generic wrapper that adds caching gates to any Haystack component."""

import time
from typing import Any, Dict, List, Optional

from ...utils.logger import get_logger
from ...utils.metrics import MetricsCollector
from ..neo4j_manager import GraphStore
from .ingate import InGate
from .outgate import OutGate


class GatedComponent:
    """
    Generic wrapper that adds InGate/OutGate caching to any component.

    Wraps any Haystack component and automatically:
    1. Checks cache before processing (InGate)
    2. Runs component only on uncached items
    3. Stores results to cache (OutGate)
    4. Merges cached + new results

    Example:
        >>> chunker = MarkdownAwareChunker(chunk_size=500)
        >>> gated_chunker = GatedComponent(
        ...     component=chunker,
        ...     component_id="chunker_1",
        ...     component_name="markdown_chunker",
        ...     graph_store=graph_store,
        ...     username="alice"
        ... )
        >>>
        >>> # Use like normal component
        >>> result = gated_chunker.run(documents=[doc1, doc2, doc3])
        >>> # Automatically cached!
    """

    def __init__(
        self,
        component: Any,
        component_id: str,
        component_name: str,
        graph_store: GraphStore,
        username: str,
        retrieve_from_ipfs: bool = True,
    ):
        """
        Initialize gated component wrapper.

        Args:
            component: The Haystack component to wrap
            component_id: Unique ID for this component instance
            component_name: Human-readable name
            graph_store: Neo4j graph store
            username: User who owns this pipeline/data
            retrieve_from_ipfs: If True, retrieves cached data from IPFS (default: False)
        """
        self.component = component
        self.component_id = component_id
        self.component_name = component_name
        self.graph_store = graph_store
        self.username = username
        self.logger = get_logger(f"{__name__}.{component_name}", username=username)
        self.metrics = MetricsCollector(username=username)

        # Create gates with IPFS retrieval enabled
        self.ingate = InGate(
            graph_store=graph_store,
            component_id=component_id,
            component_name=component_name,
            username=username,
            retrieve_from_ipfs=True,  # Always retrieve from IPFS for cached results
        )

        self.outgate = OutGate(
            graph_store=graph_store,
            component_id=component_id,
            component_name=component_name,
            username=username,
        )

    def run(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Run component with caching.

        Flow:
        1. Check cache with InGate
        2. If all inputs cached → skip component, return cached metadata
        3. If some/all uncached → run component on uncached, store results
        4. Return component output

        Args:
            **kwargs: Component inputs (e.g., documents=[...], query="...", etc.)

        Returns:
            Component output
        """
        start_time = time.time()

        # Extract component config
        component_config = self._extract_component_config()

        # Find the main input parameter
        input_param, input_items = self._extract_input_data(kwargs)

        # If no cacheable inputs, just run normally
        if not input_items:
            result: Dict[str, Any] = self.component.run(**kwargs)
            return result

        # Check cache with InGate
        cache_result = self.ingate.check_cache_batch(
            input_items=input_items,
            component_config=component_config,
        )

        cached_items = cache_result["cached"]
        uncached_items = cache_result["uncached"]
        cache_hits = len(cached_items)
        cache_misses = len(uncached_items)

        # If everything is cached, skip component execution
        if not uncached_items:
            end_time = time.time()
            self.metrics.log_component_execution(
                component_name=self.component_name,
                component_id=self.component_id,
                start_time=start_time,
                end_time=end_time,
                input_count=len(input_items),
                output_count=len(cached_items),
                cache_hits=cache_hits,
                cache_misses=cache_misses,
            )
            return self._format_cached_output(cached_items)

        # Run component on uncached items only
        if len(uncached_items) < len(input_items) and input_param is not None:
            # Partial cache - run only on uncached
            kwargs_uncached = kwargs.copy()
            kwargs_uncached[input_param] = uncached_items
            component_output = self.component.run(**kwargs_uncached)
        else:
            # No cache - run on all
            component_output = self.component.run(**kwargs)

        # Extract outputs
        output_items = self._extract_output_data(component_output)

        # Store new results to cache
        if output_items and uncached_items:
            # For now, assume 1→1 mapping (each input → corresponding output)
            if len(uncached_items) == len(output_items):
                # 1→1: zip inputs and outputs
                for input_item, output_item in zip(uncached_items, output_items):
                    try:
                        self.outgate.store(
                            input_data=input_item,
                            output_data=[output_item],  # Single output per input
                            component_config=component_config,
                        )
                    except Exception:
                        pass  # Continue caching other items
            else:
                # 1→N: Store all outputs for each input (chunking case)
                try:
                    self.outgate.store(
                        input_data=uncached_items[0] if uncached_items else None,
                        output_data=output_items,
                        component_config=component_config,
                    )
                except Exception:
                    pass  # Cache storage failed, but component executed successfully

        end_time = time.time()

        # Log metrics
        self.metrics.log_component_execution(
            component_name=self.component_name,
            component_id=self.component_id,
            start_time=start_time,
            end_time=end_time,
            input_count=len(input_items),
            output_count=len(output_items),
            cache_hits=cache_hits,
            cache_misses=cache_misses,
        )

        # Return component output
        typed_output: Dict[str, Any] = component_output
        return typed_output

    def _extract_component_config(self) -> Dict[str, Any]:
        """
        Extract component configuration for cache keying.

        Returns:
            Dict of component config parameters
        """
        config = {}

        # Try to get component init params
        if hasattr(self.component, "__dict__"):
            for key, value in self.component.__dict__.items():
                # Skip internal/private attributes
                if key.startswith("_"):
                    continue
                # Only include serializable config
                if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    config[key] = value

        return config

    def _extract_input_data(
        self, kwargs: Dict[str, Any]
    ) -> tuple[Optional[str], List[Any]]:
        """
        Find the main input parameter from kwargs.

        Common Haystack params: documents, queries, text, files, etc.

        Returns:
            (param_name, list_of_items) or (None, [])
        """
        # Common input parameter names in Haystack
        INPUT_PARAMS = ["documents", "queries", "text", "files", "data", "inputs"]

        for param in INPUT_PARAMS:
            if param in kwargs:
                value = kwargs[param]
                # Ensure it's a list
                if not isinstance(value, list):
                    value = [value]
                return param, value

        return None, []

    def _extract_output_data(self, component_output: Dict[str, Any]) -> List[Any]:
        """
        Extract output items from component result.

        Args:
            component_output: Component's return value

        Returns:
            List of output items
        """
        # Common output keys in Haystack
        OUTPUT_KEYS = ["documents", "answers", "results", "embeddings", "chunks"]

        for key in OUTPUT_KEYS:
            if key in component_output:
                value = component_output[key]
                if isinstance(value, list):
                    return value
                return [value]

        # Fallback: return the whole output
        return [component_output]

    def _format_cached_output(self, cached_items: List[tuple]) -> Dict[str, Any]:
        """
        Format cached results for pipeline continuation.

        InGate retrieves actual data from IPFS (not metadata), so we format it
        to match the component's expected output format.

        Args:
            cached_items: List of (input, cached_data) tuples
                         where cached_data is List of actual objects retrieved from IPFS

        Returns:
            Dict with documents key containing cached data
        """
        all_cached_data = []
        for _, cached_data in cached_items:
            if isinstance(cached_data, list):
                all_cached_data.extend(cached_data)
            else:
                all_cached_data.append(cached_data)

        # Return in Haystack format
        # Most components use "documents": converters, chunkers, embedders, retrievers
        # Components that DON'T: generators ("replies"/"answers"), text embedders ("embedding")
        return {"documents": all_cached_data}

    def __getattr__(self, name: str) -> Any:
        """
        Proxy all attributes to the wrapped component.

        This allows Haystack to access __haystack_input__, __haystack_output__, etc.
        """
        return getattr(self.component, name)
