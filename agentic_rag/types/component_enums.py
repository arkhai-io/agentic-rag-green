"""Typed enums for all component categories."""

from enum import Enum
from typing import Dict, List, Set, Type, Union


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


class RANKER(Enum):
    """Document ranking/reranking components."""

    SENTENCE_TRANSFORMERS_SIMILARITY = "sentence_transformers_similarity_ranker"


class GENERATOR(Enum):
    """Text generation components."""

    PROMPT_BUILDER = "prompt_builder"
    OPENAI = "generator"
    OPENROUTER = "openrouter_generator"


class WRITER(Enum):
    """Document writing/indexing components."""

    CHROMA_DOCUMENT_WRITER = "chroma_document_writer"


class DOCUMENT_STORE(Enum):
    """Document storage components."""

    CHROMA = "chroma_document_store"


class EvaluationMode(Enum):
    """Mode for evaluation - determines if gold standard is required."""

    REFERENCE_FREE = "reference_free"  # No gold standard needed
    WITH_GOLD_STANDARD = "with_gold_standard"  # Requires ground truth data


class EVALUATOR(Enum):
    """Evaluation components for assessing retrieval and generation quality."""

    # Without gold standard (reference-free evaluation)
    REFERENCE_FREE = "reference_free_evaluator"  # Faithfulness + Context Relevance

    # With gold standard (requires ground truth)
    GOLD_STANDARD = "gold_standard_evaluator"  # Answer Similarity + Document Recall


# Union type for all component enums
ComponentEnum = Union[
    CONVERTER,
    CHUNKER,
    EMBEDDER,
    DOCUMENT_STORE,
    RETRIEVER,
    RANKER,
    GENERATOR,
    WRITER,
    EVALUATOR,
]


# Mapping from enum classes to their names for validation
COMPONENT_ENUM_MAP: Dict[str, Type[Enum]] = {
    "CONVERTER": CONVERTER,
    "CHUNKER": CHUNKER,
    "EMBEDDER": EMBEDDER,
    "DOCUMENT_STORE": DOCUMENT_STORE,
    "RETRIEVER": RETRIEVER,
    "RANKER": RANKER,
    "GENERATOR": GENERATOR,
    "WRITER": WRITER,
    "EVALUATOR": EVALUATOR,
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


# Mapping evaluators to their evaluation modes
EVALUATOR_MODE_MAP: Dict[str, EvaluationMode] = {
    # Reference-free evaluators (no gold standard needed)
    "reference_free_evaluator": EvaluationMode.REFERENCE_FREE,
    # Gold standard evaluators (require ground truth)
    "gold_standard_evaluator": EvaluationMode.WITH_GOLD_STANDARD,
}

# Sets for quick lookup
REFERENCE_FREE_EVALUATORS: Set[str] = {
    evaluator
    for evaluator, mode in EVALUATOR_MODE_MAP.items()
    if mode == EvaluationMode.REFERENCE_FREE
}

GOLD_STANDARD_EVALUATORS: Set[str] = {
    evaluator
    for evaluator, mode in EVALUATOR_MODE_MAP.items()
    if mode == EvaluationMode.WITH_GOLD_STANDARD
}


def requires_gold_standard(evaluator_name: str) -> bool:
    """
    Check if an evaluator requires gold standard data.

    Args:
        evaluator_name: Name of the evaluator component

    Returns:
        True if evaluator requires ground truth data, False otherwise
    """
    return evaluator_name in GOLD_STANDARD_EVALUATORS


def get_evaluator_mode(evaluator_name: str) -> EvaluationMode:
    """
    Get the evaluation mode for an evaluator.

    Args:
        evaluator_name: Name of the evaluator component

    Returns:
        EvaluationMode enum value

    Raises:
        ValueError: If evaluator name is not recognized
    """
    if evaluator_name not in EVALUATOR_MODE_MAP:
        raise ValueError(f"Unknown evaluator: {evaluator_name}")
    return EVALUATOR_MODE_MAP[evaluator_name]
