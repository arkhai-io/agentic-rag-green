"""Centralized graph relationship definitions for Neo4j edges."""

from enum import Enum


class GraphRelationship(Enum):
    """Defines all relationship types used in the Neo4j graph."""

    # Component to Component relationships
    FLOWS_TO = "FLOWS_TO"  # Data flows from one component to another

    # Component to DocumentStore relationships
    WRITES_TO = "WRITES_TO"  # Writer components write to DocumentStores
    READS_FROM = "READS_FROM"  # Retriever components read from DocumentStores

    # User to Pipeline relationships
    OWNS = "OWNS"  # User owns/created a pipeline

    # Pipeline to Component relationships
    CONTAINS = "CONTAINS"  # Pipeline contains components

    # DataPiece transformation relationships (for InGate/OutGate)
    TRANSFORMED_BY = "TRANSFORMED_BY"  # DataPiece transformed to another DataPiece
    PROCESSED_BY = "PROCESSED_BY"  # DataPiece processed by Component


def get_relationship_name(relationship: GraphRelationship) -> str:
    """Get the string name of a relationship for Neo4j queries."""
    return relationship.value


def get_safe_relationship_name(relationship: GraphRelationship) -> str:
    """Get a safe relationship name for Neo4j (replaces special characters)."""
    return relationship.value.replace("-", "_").replace(" ", "_").upper()
