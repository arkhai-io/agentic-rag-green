"""InGate component for cache lookup and input filtering.

The InGate checks if input data has already been processed by a component
with a specific configuration. It splits inputs into cached vs uncached items.
"""

import hashlib
import json
from typing import Any, Dict, List, Optional

from ...utils.ipfs_client import LighthouseClient
from ...utils.logger import get_logger
from ..neo4j_manager import GraphStore


class InGate:
    """
    InGate handles cache lookups before component processing.

    Responsibilities:
    - Fingerprint input data
    - Query Neo4j for cached results
    - Split data into cached (bypass component) and uncached (process)
    - Track cache hit/miss statistics

    Flow:
        Input Data → [Fingerprint] → [Neo4j Lookup] → Split:
                                                        ├─> Cached (from graph)
                                                        └─> Uncached (to component)
    """

    def __init__(
        self,
        graph_store: GraphStore,
        component_id: str,
        component_name: str,
        username: Optional[str] = None,
        ipfs_client: Optional[LighthouseClient] = None,
        retrieve_from_ipfs: bool = False,
    ):
        """
        Initialize InGate.

        Args:
            graph_store: Neo4j graph store for cache lookups
            component_id: Unique ID of the component this gate protects
            component_name: Human-readable component name
            username: Username for per-user logging
            ipfs_client: Optional Lighthouse IPFS client (creates one if not provided)
            retrieve_from_ipfs: If True, retrieves actual data from IPFS (default: False, returns metadata only)
        """
        self.graph_store = graph_store
        self.component_id = component_id
        self.component_name = component_name
        self.ipfs_client = ipfs_client or LighthouseClient()
        self.retrieve_from_ipfs = retrieve_from_ipfs
        self.logger = get_logger(f"{__name__}.{component_name}", username=username)

    def check_cache_batch(
        self, input_items: List[Any], component_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check cache for multiple items and split into cached vs uncached.

        Handles both single items and batches - just pass a list.

        Args:
            input_items: List of data items to check (or single item in a list)
            component_config: Component configuration dict

        Returns:
            Dictionary with:
            {
                "cached": [(item, cached_result), ...],
                "uncached": [item, ...],
                "fingerprints": {item_idx: fingerprint, ...}
            }

            cached_result format:
            - If retrieve_from_ipfs=False: List[{fingerprint, ipfs_hash, data_type}]
            - If retrieve_from_ipfs=True: List of actual data retrieved from IPFS

        Example (batch):
            >>> result = ingate.check_cache_batch(
            ...     input_items=[doc1, doc2, doc3],
            ...     component_config={"model": "minilm"}
            ... )
            >>> # Process only uncached
            >>> if result['uncached']:
            ...     new_results = component.run(result['uncached'])

        Example (single item):
            >>> result = ingate.check_cache_batch(
            ...     input_items=[single_doc],  # Just wrap in list
            ...     component_config={"chunk_size": 500}
            ... )
        """
        # Hash the config once
        config_hash = self.hash_config(component_config)

        # Fingerprint all inputs
        fingerprints = {}
        for idx, item in enumerate(input_items):
            fingerprints[idx] = self.fingerprint_data(item)

        # Batch lookup in Neo4j
        fingerprint_list = list(fingerprints.values())
        cache_map = self.graph_store.lookup_cached_transformations_batch(
            input_fingerprints=fingerprint_list,
            component_id=self.component_id,
            config_hash=config_hash,
        )

        # Split into cached vs uncached
        cached = []
        uncached = []

        cache_hits = 0
        for idx, item in enumerate(input_items):
            fp = fingerprints[idx]
            if fp in cache_map:
                # Found cached result
                cached_metadata = cache_map[fp]

                if self.retrieve_from_ipfs:
                    try:
                        cached_data = []
                        for output_meta in cached_metadata:
                            ipfs_hash = output_meta["ipfs_hash"]
                            data_type = output_meta.get("data_type")
                            data = self._retrieve_from_ipfs(ipfs_hash, data_type)
                            cached_data.append(data)
                        cached.append((item, cached_data))
                        cache_hits += 1
                    except (ConnectionError, Exception) as e:
                        # IPFS retrieval failed - treat as cache miss
                        self.logger.warning(f"IPFS retrieval failed for {fp}: {e}")
                        uncached.append(item)
                else:
                    # Just return metadata (fingerprints + IPFS hashes)
                    cached.append((item, cached_metadata))
                    cache_hits += 1
            else:
                # No cache, needs processing
                uncached.append(item)

        self.logger.info(
            f"Cache check: {cache_hits} hits, {len(uncached)} misses (total: {len(input_items)})"
        )

        return {
            "cached": cached,
            "uncached": uncached,
            "fingerprints": fingerprints,
        }

    def fingerprint_data(self, data: Any) -> str:
        """
        Create stable fingerprint hash for data.

        Args:
            data: Data to fingerprint (Document, ByteStream, str, List, etc.)

        Returns:
            SHA256 hash string (e.g., "fp_abc123...")

        Strategy:
            - Document: hash(content + metadata + id)
            - List[Document]: hash of individual hashes
            - ByteStream: hash(file_content + filename)
            - str: hash(content)
            - List[float]: hash(vector values)
        """
        # Serialize data to string for hashing
        data_str = self._serialize_for_fingerprint(data)

        # Create SHA256 hash
        hash_obj = hashlib.sha256(data_str.encode("utf-8"))
        return f"fp_{hash_obj.hexdigest()[:16]}"

    def hash_config(self, config: Dict[str, Any]) -> str:
        """
        Create stable hash of component configuration.

        Args:
            config: Component configuration dictionary

        Returns:
            Hash string (e.g., "cfg_abc123...")

        Notes:
            - Sorts keys for stability
            - Only hashes "semantic" config (affects output)
            - Excludes performance-only settings (batch_size, num_workers, etc.)
        """
        # Performance-only settings that don't affect output
        PERF_ONLY_KEYS = {
            "batch_size",
            "num_workers",
            "device",
            "show_progress",
            "verbose",
            "debug",
            "workers",
            "threads",
        }

        # Filter to semantic config only
        semantic_config = {k: v for k, v in config.items() if k not in PERF_ONLY_KEYS}

        # Sort keys for stability and create JSON
        config_str = json.dumps(semantic_config, sort_keys=True, separators=(",", ":"))

        # Hash and return short version
        hash_obj = hashlib.sha256(config_str.encode("utf-8"))
        return f"cfg_{hash_obj.hexdigest()[:16]}"

    def _retrieve_from_ipfs(
        self, ipfs_hash: str, data_type: Optional[str] = None
    ) -> Any:
        """Retrieve from IPFS and reconstruct as Document."""
        from haystack import Document

        text = self.ipfs_client.retrieve_text(ipfs_hash)
        return Document(content=text)

    def _serialize_for_fingerprint(self, data: Any) -> str:
        """
        Serialize data to string for hashing.

        Args:
            data: Data to serialize

        Returns:
            String representation for hashing
        """
        # Check for embedding first (embedder output)
        if hasattr(data, "embedding") and data.embedding is not None:
            # Hash the embedding vector (the new data added by embedder)
            import numpy as np

            if isinstance(data.embedding, np.ndarray):
                return hashlib.sha256(data.embedding.tobytes()).hexdigest()
            else:
                # List of floats
                return str(data.embedding)

        # Handle content attribute (Document, etc.)
        if hasattr(data, "content"):
            return str(data.content) if data.content else ""

        # Handle data attribute (ByteStream)
        if hasattr(data, "data"):
            if isinstance(data.data, bytes):
                return hashlib.sha256(data.data).hexdigest()
            return str(data.data)

        # Handle lists
        if isinstance(data, list):
            return "|".join(str(item) for item in data)

        # Handle dicts
        if isinstance(data, dict):
            return json.dumps(data, sort_keys=True)

        # Basic types: just string it
        return str(data)
