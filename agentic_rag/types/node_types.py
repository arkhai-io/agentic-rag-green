"""Types for Neo4j nodes."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ComponentNode:
    """Represents a component node in Neo4j."""

    component_name: str
    pipeline_name: str
    version: str
    author: str
    component_config: Dict[str, Any]
    id: Optional[str] = None  # Auto-generated if not provided

    def __post_init__(self) -> None:
        """Generate ID if not provided."""
        if self.id is None:
            # Create deterministic hash from: component_name__pipeline_name__version__author__component_config
            import hashlib
            import json

            config_str = json.dumps(
                self.component_config, sort_keys=True, separators=(",", ":")
            )
            combined = f"{self.component_name}__{self.pipeline_name}__{self.version}__{self.author}__{config_str}"

            # Generate SHA-256 hash and take first 12 characters for readability
            hash_obj = hashlib.sha256(combined.encode("utf-8"))
            self.id = f"comp_{hash_obj.hexdigest()[:12]}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Neo4j insertion."""
        import json

        # Convert component_config to JSON string for Neo4j compatibility
        config_json = json.dumps(
            self.component_config, sort_keys=True, separators=(",", ":")
        )

        return {
            "id": self.id,
            "component_name": self.component_name,
            "pipeline_name": self.pipeline_name,
            "version": self.version,
            "author": self.author,
            "component_config_json": config_json,
        }

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
            id=data.get("id"),
        )


@dataclass
class DocumentStoreNode:
    """Represents a document store node with retrieval component information."""

    store_name: str
    store_type: str  # "chroma", "elasticsearch", etc.
    retrieval_components: List[
        Dict[str, Any]
    ]  # Ordered list of components with configs needed for retrieval
    version: str = "1.0.0"
    creator: str = "system"
    root_dir: str = "."  # Directory where the document store is persisted
    id: Optional[str] = None

    def __post_init__(self) -> None:
        """Generate ID if not provided."""
        if self.id is None:
            self.id = f"docstore_{self.store_name}_{self.store_type}".replace(" ", "_")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Neo4j insertion."""
        # Convert complex objects to JSON strings for Neo4j compatibility
        import json

        retrieval_components_json = (
            json.dumps(self.retrieval_components) if self.retrieval_components else "[]"
        )

        return {
            "id": self.id,
            "store_name": self.store_name,
            "store_type": self.store_type,
            "retrieval_components_json": retrieval_components_json,
            "version": self.version,
            "creator": self.creator,
            "root_dir": self.root_dir,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentStoreNode":
        """Create DocumentStoreNode from dictionary."""
        import json

        # Handle both old format (direct list) and new format (JSON string)
        retrieval_components = data.get("retrieval_components", [])
        if isinstance(retrieval_components, str):
            # New format: JSON string
            retrieval_components = json.loads(
                data.get("retrieval_components_json", "[]")
            )
        elif "retrieval_components_json" in data:
            # New format: JSON string in separate field
            retrieval_components = json.loads(data["retrieval_components_json"])

        return cls(
            store_name=data["store_name"],
            store_type=data["store_type"],
            retrieval_components=retrieval_components,
            version=data.get("version", "1.0.0"),
            creator=data.get("creator", "system"),
            root_dir=data.get("root_dir", "."),
            id=data.get("id"),
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
