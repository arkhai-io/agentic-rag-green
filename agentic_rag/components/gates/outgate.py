"""OutGate component for storing results and updating the knowledge graph.

Simple flow:
1. Upload data to IPFS → get IPFS hash
2. Create DataPiece nodes in Neo4j
3. Create TRANSFORMED_BY edges connecting them
"""

import hashlib
import json
from typing import Any, Dict, List, Optional

from ...utils.ipfs_client import LighthouseClient
from ..neo4j_manager import GraphStore


class OutGate:
    """
    OutGate stores component outputs to IPFS and Neo4j.

    Simple responsibilities:
    - Upload data to IPFS (placeholder)
    - Create DataPiece nodes with IPFS hashes
    - Create TRANSFORMED_BY edges (input → output)

    Flow:
        Output → [Upload to IPFS] → [Create DataPiece] → [Create TRANSFORMED_BY edge]
    """

    def __init__(
        self,
        graph_store: GraphStore,
        component_id: str,
        component_name: str,
        username: str,
        ipfs_client: Optional[LighthouseClient] = None,
    ):
        """
        Initialize OutGate.

        Args:
            graph_store: Neo4j graph store
            component_id: Component doing the transformation
            component_name: Human-readable name
            username: Data owner
            ipfs_client: Optional Lighthouse IPFS client (creates one if not provided)
        """
        self.graph_store = graph_store
        self.component_id = component_id
        self.component_name = component_name
        self.username = username
        self.ipfs_client = ipfs_client or LighthouseClient()

    def store(
        self,
        input_data: Any,
        output_data: List[Any],
        component_config: Dict[str, Any],
        processing_time_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Store transformation: input → component → outputs.

        Handles both:
        - 1→1: [single_output]
        - 1→N: [output1, output2, output3, ...]

        Steps:
        1. Fingerprint input
        2. For each output: upload to IPFS, create DataPiece, create edge

        Args:
            input_data: Input to component
            output_data: List of outputs (even if just one)
            component_config: Component configuration
            processing_time_ms: Processing time

        Returns:
            {
                "input_fingerprint": str,
                "output_fingerprints": List[str],
                "output_ipfs_hashes": List[str]
            }

        Example:
            >>> # Single output
            >>> outgate.store(markdown, [chunk], config)

            >>> # Multiple outputs
            >>> outgate.store(markdown, [chunk1, chunk2, chunk3], config)
        """
        # Hash config
        config_hash = self.hash_config(component_config)

        # Fingerprint input
        input_fingerprint = self.fingerprint_data(input_data)

        # Process all outputs in batch
        output_records = []
        for output_item in output_data:
            # Upload to IPFS
            ipfs_hash = self._upload_to_ipfs(output_item)

            # Fingerprint output
            output_fingerprint = self.fingerprint_data(output_item)

            # Detect data type
            data_type = type(output_item).__name__

            output_records.append(
                {
                    "fingerprint": output_fingerprint,
                    "ipfs_hash": ipfs_hash,
                    "data_type": data_type,
                }
            )

        # Store everything to Neo4j in one batch
        self.graph_store.store_transformation_batch(
            input_fingerprint=input_fingerprint,
            input_ipfs_hash=self._upload_to_ipfs(input_data),
            input_data_type=type(input_data).__name__,
            output_records=output_records,
            component_id=self.component_id,
            component_name=self.component_name,
            config_hash=config_hash,
            username=self.username,
            processing_time_ms=processing_time_ms,
        )

        return {
            "input_fingerprint": input_fingerprint,
            "output_fingerprints": [r["fingerprint"] for r in output_records],
            "output_ipfs_hashes": [r["ipfs_hash"] for r in output_records],
        }

    def fingerprint_data(self, data: Any) -> str:
        """Create SHA256 fingerprint of data."""
        data_str = self._serialize_for_fingerprint(data)
        hash_obj = hashlib.sha256(data_str.encode("utf-8"))
        return f"fp_{hash_obj.hexdigest()[:16]}"

    def hash_config(self, config: Dict[str, Any]) -> str:
        """Create hash of component configuration."""
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

    def _upload_to_ipfs(self, data: Any) -> str:
        """
        Upload data to IPFS via Lighthouse and return CID.

        Args:
            data: Any data type (Document, dict, str, bytes, etc.)

        Returns:
            IPFS CID (hash)
        """
        result = self.ipfs_client.upload_any(data)
        ipfs_hash: str = result["Hash"]
        return ipfs_hash

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
