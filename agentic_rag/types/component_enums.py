"""Typed enums for all component categories."""

from enum import Enum
from typing import Dict, List, Type, Union


class CONVERTER(Enum):
    """Document converter components."""

    PDF = "pdf_converter"
    DOCX = "docx_converter"
    MARKDOWN = "markdown_converter"
    HTML = "html_converter"
    TEXT = "text_converter"
    MARKER_PDF = "marker_pdf_converter"
    MARKITDOWN_PDF = "markitdown_pdf_converter"


class CHUNKER(Enum):
    """Document chunking/splitting components."""

    DOCUMENT_SPLITTER = "chunker"
    MARKDOWN_AWARE = "markdown_aware_chunker"
    SEMANTIC = "semantic_chunker"


class EMBEDDER(Enum):
    """Text and document embedding components."""

    SENTENCE_TRANSFORMERS = "embedder"
    SENTENCE_TRANSFORMERS_DOC = "document_embedder"


class RETRIEVER(Enum):
    """Document retrieval components."""

    CHROMA_EMBEDDING = "chroma_embedding_retriever"


class GENERATOR(Enum):
    """Text generation components."""

    OPENAI = "generator"


class WRITER(Enum):
    """Document writing/indexing components."""

    CHROMA_DOCUMENT_WRITER = "chroma_document_writer"


class DOCUMENT_STORE(Enum):
    """Document storage components."""

    CHROMA = "chroma_document_store"


# Union type for all component enums
ComponentEnum = Union[
    CONVERTER, CHUNKER, EMBEDDER, DOCUMENT_STORE, RETRIEVER, GENERATOR, WRITER
]


# Mapping from enum classes to their names for validation
COMPONENT_ENUM_MAP: Dict[str, Type[Enum]] = {
    "CONVERTER": CONVERTER,
    "CHUNKER": CHUNKER,
    "EMBEDDER": EMBEDDER,
    "DOCUMENT_STORE": DOCUMENT_STORE,
    "RETRIEVER": RETRIEVER,
    "GENERATOR": GENERATOR,
    "WRITER": WRITER,
}

# Reverse mapping from component values to enum classes
VALUE_TO_ENUM_MAP: Dict[str, Type[Enum]] = {}
for enum_class in COMPONENT_ENUM_MAP.values():
    for member in enum_class:
        VALUE_TO_ENUM_MAP[member.value] = enum_class


def parse_component_spec(spec: str) -> tuple[Type[Enum], Enum]:
    """
    Parse a component specification like 'CHUNKER.RECURSIVE' into enum class and member.

    Args:
        spec: Component specification string (e.g., 'CHUNKER.RECURSIVE')

    Returns:
        Tuple of (enum_class, enum_member)

    Raises:
        ValueError: If specification is invalid
    """
    if "." not in spec:
        raise ValueError(
            f"Component spec must be in format 'CATEGORY.TYPE', got: {spec}"
        )

    category, component_type = spec.split(".", 1)

    if category not in COMPONENT_ENUM_MAP:
        available = ", ".join(COMPONENT_ENUM_MAP.keys())
        raise ValueError(
            f"Unknown component category '{category}'. Available: {available}"
        )

    enum_class = COMPONENT_ENUM_MAP[category]

    try:
        enum_member = enum_class[component_type]
        return enum_class, enum_member
    except KeyError:
        available = ", ".join([member.name for member in enum_class])
        raise ValueError(
            f"Unknown {category} type '{component_type}'. Available: {available}"
        )


def get_component_value(spec: str) -> str:
    """
    Get the component registry value from an enum specification.

    Args:
        spec: Component specification string (e.g., 'CHUNKER.RECURSIVE')

    Returns:
        Component registry value (e.g., 'recursive_chunker')
    """
    _, enum_member = parse_component_spec(spec)
    return str(enum_member.value)


def validate_component_spec(spec: str) -> bool:
    """
    Validate that a component specification is valid.

    Args:
        spec: Component specification string

    Returns:
        True if valid, False otherwise
    """
    try:
        parse_component_spec(spec)
        return True
    except ValueError:
        return False


def list_available_components() -> Dict[str, List[str]]:
    """
    List all available components by category.

    Returns:
        Dictionary mapping category names to lists of available component types
    """
    result = {}
    for category, enum_class in COMPONENT_ENUM_MAP.items():
        result[category] = [member.name for member in enum_class]
    return result
