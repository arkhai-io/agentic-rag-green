"""Test the main package initialization."""

import agentic_rag


def test_version() -> None:
    """Test that version is accessible."""
    assert hasattr(agentic_rag, "__version__")
    assert isinstance(agentic_rag.__version__, str)
    assert agentic_rag.__version__ == "0.1.0"


def test_author() -> None:
    """Test that author information is accessible."""
    assert hasattr(agentic_rag, "__author__")
    assert isinstance(agentic_rag.__author__, str)


def test_email() -> None:
    """Test that email information is accessible."""
    assert hasattr(agentic_rag, "__email__")
    assert isinstance(agentic_rag.__email__, str)
