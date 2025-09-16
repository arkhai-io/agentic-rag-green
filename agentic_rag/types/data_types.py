"""Core data types and enums for the pipeline system."""

from enum import Enum
from typing import Any, Dict


class DataType(Enum):
    """Data types that flow through Haystack pipeline components."""

    # Basic types
    STRING = "str"
    LIST_STRING = "List[str]"

    # Document types (Haystack's main data structure)
    DOCUMENT = "Document"
    LIST_DOCUMENT = "List[Document]"

    # Embedding types
    LIST_FLOAT = "List[float]"
    LIST_LIST_FLOAT = "List[List[float]]"

    # File types
    BYTE_STREAM = "ByteStream"
    LIST_BYTE_STREAM = "List[ByteStream]"

    # Dictionary types
    DICT = "Dict[str, Any]"
    LIST_DICT = "List[Dict[str, Any]]"


class ComponentType(Enum):
    """Categories of pipeline components."""

    # Input/Output
    CONVERTER = "converter"

    # Processing
    CHUNKER = "chunker"
    EMBEDDER = "embedder"

    # Storage
    DOCUMENT_STORE = "document_store"

    # Retrieval
    RETRIEVER = "retriever"
    RERANKER = "reranker"

    # Generation
    GENERATOR = "generator"


class PipelineUsage(Enum):
    """Usage types for pipeline components."""

    INDEXING = "indexing"  # Used when adding/processing documents
    RETRIEVAL = "retrieval"  # Used when querying/retrieving documents
    BOTH = "both"  # Used in both indexing and retrieval operations


# Type aliases
ComponentData = Dict[str, Any]
PipelineData = Dict[str, Any]
