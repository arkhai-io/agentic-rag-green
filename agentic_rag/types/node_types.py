"""Types for Neo4j nodes."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple


@dataclass
class ComponentNode:
    """Represents a component node in Neo4j."""

    component_name: str
    pipeline_name: str
    version: str
    author: str
    component_config: Dict[str, Any]
    component_type: Optional[str] = None  # e.g., "EMBEDDER.SENTENCE_TRANSFORMERS_DOC"
    pipeline_type: Optional[str] = None  # "indexing" or "retrieval"
    branch_id: Optional[str] = (
        None  # For retrieval pipelines: identifies which indexing pipeline branch
    )
    id: Optional[str] = None  # Auto-generated if not provided
    cache_key: Optional[str] = None  # Pipeline-agnostic key for cache lookups

    def __post_init__(self) -> None:
        """Generate ID and cache_key if not provided."""
        if self.id is None:
            # Create deterministic hash from: component_name__pipeline_name__version__author__branch_id
            import hashlib
            import json

            combined = f"{self.component_name}__{self.pipeline_name}__{self.version}__{self.author}"

            # Include branch_id if provided (for retrieval pipeline branches)
            if self.branch_id:
                combined += f"__{self.branch_id}"

            # Generate SHA-256 hash and take first 12 characters for readability
            hash_obj = hashlib.sha256(combined.encode("utf-8"))
            self.id = f"comp_{hash_obj.hexdigest()[:12]}"

        # Generate pipeline-agnostic cache key (component_type + config + author)
        if self.cache_key is None:
            import hashlib

            # Use component_type + config for cache sharing across pipelines
            config_str = json.dumps(self.component_config, sort_keys=True)
            cache_combined = f"{self.component_type}__{config_str}__{self.author}"

            cache_hash = hashlib.sha256(cache_combined.encode("utf-8"))
            self.cache_key = f"cache_{cache_hash.hexdigest()[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Neo4j insertion."""
        import json

        # Convert component_config to JSON string for Neo4j compatibility
        config_json = json.dumps(
            self.component_config, sort_keys=True, separators=(",", ":")
        )

        result = {
            "id": self.id,
            "component_name": self.component_name,
            "pipeline_name": self.pipeline_name,
            "version": self.version,
            "author": self.author,
            "component_config_json": config_json,
            "cache_key": self.cache_key,
        }

        if self.component_type:
            result["component_type"] = self.component_type

        if self.pipeline_type:
            result["pipeline_type"] = self.pipeline_type

        if self.branch_id:
            result["branch_id"] = self.branch_id

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComponentNode":
        """Create ComponentNode from dictionary."""
        import json

        # Always expect JSON format
        component_config = json.loads(data["component_config_json"])

        return cls(
            component_name=data["component_name"],
            pipeline_name=data["pipeline_name"],
            version=data["version"],
            author=data["author"],
            component_config=component_config,
            component_type=data.get("component_type"),
            pipeline_type=data.get("pipeline_type"),
            branch_id=data.get("branch_id"),
            id=data.get("id"),
            cache_key=data.get("cache_key"),
        )


@dataclass
class ComponentRelationship:
    """Represents a relationship between component nodes."""

    source_id: str
    target_id: str
    relationship_type: str
    properties: Optional[Dict[str, Any]] = None

    def to_tuple(self) -> Tuple[str, str, str] | Tuple[str, str, str, Dict[str, Any]]:
        """Convert to tuple for batch edge insertion."""
        if self.properties:
            return (
                self.source_id,
                self.target_id,
                self.relationship_type,
                self.properties,
            )
        return (self.source_id, self.target_id, self.relationship_type)


@dataclass
class UserNode:
    """Represents a user who owns pipelines."""

    username: str
    email: Optional[str] = None
    display_name: Optional[str] = None
    id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.id is None:
            base = self.username if not self.email else f"{self.username}__{self.email}"
            self.id = base.replace(" ", "_")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "display_name": self.display_name,
        }


@dataclass
class DataPiece:
    """
    Represents a piece of data in the Neo4j graph.

    Used by InGate/OutGate to track data transformations and enable caching.
    Each unique piece of content gets one DataPiece node (deduplicated by fingerprint).
    Content is stored on IPFS, only hash is stored in Neo4j.
    """

    # Identity (required)
    fingerprint: str  # SHA256 hash of content (PRIMARY KEY)

    # Content storage (required)
    ipfs_hash: str  # IPFS hash where actual content is stored
    data_type: str  # "Document", "ByteStream", "List[Document]", etc.

    # Authorship (required)
    username: str  # Who created/owns this data

    # Metadata (optional)
    content_preview: Optional[str] = None  # First 200 chars for quick viewing
    size_bytes: Optional[int] = None  # Size of content
    created_at: Optional[datetime] = None  # When first created
    source: Optional[str] = None  # Original source (file path, URL, etc.)

    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j node properties."""
        props: Dict[str, Any] = {
            "fingerprint": self.fingerprint,
            "ipfs_hash": self.ipfs_hash,
            "type": self.data_type,
            "username": self.username,
        }

        if self.content_preview:
            props["content_preview"] = self.content_preview
        if self.size_bytes is not None:
            props["size_bytes"] = self.size_bytes
        if self.source:
            props["source"] = self.source

        return props

    @classmethod
    def from_neo4j_node(cls, node: dict) -> "DataPiece":
        """Create DataPiece from Neo4j node properties."""
        return cls(
            fingerprint=node["fingerprint"],
            ipfs_hash=node["ipfs_hash"],
            data_type=node["type"],
            username=node["username"],
            content_preview=node.get("content_preview"),
            size_bytes=node.get("size_bytes"),
            source=node.get("source"),
        )


@dataclass
class TransformedByRelationship:
    """
    Properties for TRANSFORMED_BY relationship.

    Connects input DataPiece to output DataPiece.
    Stores information about the transformation that occurred.
    """

    component_id: str  # Which component did the transformation
    component_name: str  # Human-readable component name
    config_hash: str  # Hash of component configuration

    # Optional metadata
    processing_time_ms: Optional[int] = None
    created_at: Optional[datetime] = None

    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j relationship properties."""
        props: Dict[str, Any] = {
            "component_id": self.component_id,
            "component_name": self.component_name,
            "config_hash": self.config_hash,
        }

        if self.processing_time_ms is not None:
            props["processing_time_ms"] = self.processing_time_ms

        return props


@dataclass
class ProcessedByRelationship:
    """
    Properties for PROCESSED_BY relationship.

    Connects DataPiece to Component for tracking/statistics.
    """

    config_hash: str  # Configuration used
    processing_time_ms: Optional[int] = None
    created_at: Optional[datetime] = None

    def to_neo4j_properties(self) -> Dict[str, Any]:
        """Convert to Neo4j relationship properties."""
        props: Dict[str, Any] = {"config_hash": self.config_hash}

        if self.processing_time_ms is not None:
            props["processing_time_ms"] = self.processing_time_ms

        return props
