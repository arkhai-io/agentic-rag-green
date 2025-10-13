"""OutGate component for storing results and updating the knowledge graph.

Simple flow:
1. Upload data to IPFS → get IPFS hash
2. Create DataPiece nodes in Neo4j
3. Create TRANSFORMED_BY edges connecting them
"""

from typing import Any, Dict, List, Optional

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
    ):
        """
        Initialize OutGate.

        Args:
            graph_store: Neo4j graph store
            component_id: Component doing the transformation
            component_name: Human-readable name
            username: Data owner
        """
        ...

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
        raise NotImplementedError("OutGate.store() not yet implemented")

    def fingerprint_data(self, data: Any) -> str:
        """Create SHA256 fingerprint of data."""
        raise NotImplementedError("OutGate.fingerprint_data() not yet implemented")

    def hash_config(self, config: Dict[str, Any]) -> str:
        """Create hash of component configuration."""
        raise NotImplementedError("OutGate.hash_config() not yet implemented")

    def _upload_to_ipfs(self, data: Any) -> str:
        """
        Upload data to IPFS and return CID.

        Placeholder for now - actual IPFS SDK integration pending.
        """
        raise NotImplementedError("IPFS integration not yet implemented")

    def _create_data_piece_node(
        self,
        fingerprint: str,
        ipfs_hash: str,
        data_type: str,
        content_preview: Optional[str] = None,
    ) -> None:
        """
        Create DataPiece node in Neo4j.

        Cypher:
            MERGE (d:DataPiece {fingerprint: $fp})
            ON CREATE SET
                d.ipfs_hash = $ipfs_hash,
                d.type = $type,
                d.username = $username,
                d.content_preview = $preview,
                d.created_at = datetime()
        """
        ...

    def _create_transformation_edge(
        self,
        input_fingerprint: str,
        output_fingerprint: str,
        config_hash: str,
        processing_time_ms: Optional[int] = None,
    ) -> None:
        """
        Create TRANSFORMED_BY edge.

        Cypher:
            MATCH (input:DataPiece {fingerprint: $input_fp})
            MATCH (output:DataPiece {fingerprint: $output_fp})
            MERGE (input)-[t:TRANSFORMED_BY {
                component_id: $comp_id,
                config_hash: $cfg_hash
            }]->(output)
            ON CREATE SET
                t.component_name = $comp_name,
                t.processing_time_ms = $proc_time,
                t.created_at = datetime()
        """
        ...
