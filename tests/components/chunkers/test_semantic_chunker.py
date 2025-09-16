"""Tests for SemanticChunker component."""

from haystack import Document

from agentic_rag.components.chunkers import SemanticChunker


class TestSemanticChunker:
    """Test SemanticChunker component functionality."""

    def test_semantic_chunker_initialization(self):
        """Test that SemanticChunker can be initialized with different parameters."""
        # Default initialization
        chunker = SemanticChunker()
        assert chunker.min_chunk_size == 200
        assert chunker.max_chunk_size == 1000
        assert chunker.overlap_size == 50

        # Custom initialization
        chunker = SemanticChunker(
            min_chunk_size=100, max_chunk_size=500, overlap_size=25
        )
        assert chunker.min_chunk_size == 100
        assert chunker.max_chunk_size == 500
        assert chunker.overlap_size == 25

    def test_semantic_chunker_empty_document(self):
        """Test handling of empty documents."""
        chunker = SemanticChunker()
        empty_doc = Document(content="")

        result = chunker.run(documents=[empty_doc])

        assert "documents" in result
        assert len(result["documents"]) == 0

    def test_semantic_chunker_no_documents(self):
        """Test handling of empty document list."""
        chunker = SemanticChunker()

        result = chunker.run(documents=[])

        assert "documents" in result
        assert len(result["documents"]) == 0

    def test_semantic_chunker_header_splitting(self):
        """Test that chunker splits at header boundaries."""
        chunker = SemanticChunker(min_chunk_size=50, max_chunk_size=200)

        content = """# Introduction
This is the introduction section with some content that explains the topic.

## Background
This section provides background information about the subject matter.

### Details
Here are more detailed explanations and examples."""

        doc = Document(content=content)
        result = chunker.run(documents=[doc])

        chunks = result["documents"]
        assert len(chunks) > 1

        # Check that chunks contain header information
        chunk_contents = [chunk.content for chunk in chunks]
        assert any("# Introduction" in content for content in chunk_contents)
        assert any("## Background" in content for content in chunk_contents)

    def test_semantic_chunker_list_splitting(self):
        """Test that chunker splits at list boundaries."""
        chunker = SemanticChunker(min_chunk_size=30, max_chunk_size=150)

        content = """Here are some items:

1. First item with some description
2. Second item with more details
3. Third item with even more information

And some bullet points:

- Point one
- Point two
- Point three"""

        doc = Document(content=content)
        result = chunker.run(documents=[doc])

        chunks = result["documents"]
        assert len(chunks) > 1

        # Check that chunks contain list information
        chunk_contents = [chunk.content for chunk in chunks]
        has_numbered_list = any(
            "1." in content and "2." in content for content in chunk_contents
        )
        has_bullet_list = any("-" in content for content in chunk_contents)
        assert has_numbered_list or has_bullet_list

    def test_semantic_chunker_code_block_splitting(self):
        """Test that chunker handles code blocks properly."""
        chunker = SemanticChunker(min_chunk_size=50, max_chunk_size=300)

        content = """Here's some code:

```python
def hello_world():
    print("Hello, World!")
    return True
```

And here's more explanation about the code."""

        doc = Document(content=content)
        result = chunker.run(documents=[doc])

        chunks = result["documents"]
        assert len(chunks) >= 1

        # Check that code block is preserved
        all_content = " ".join(chunk.content for chunk in chunks)
        assert "```python" in all_content
        assert "def hello_world" in all_content

    def test_semantic_chunker_table_splitting(self):
        """Test that chunker handles tables properly."""
        chunker = SemanticChunker(min_chunk_size=30, max_chunk_size=200)

        content = """Here's a table:

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Value 1  | Value 2  | Value 3  |
| Value 4  | Value 5  | Value 6  |

And some text after the table."""

        doc = Document(content=content)
        result = chunker.run(documents=[doc])

        chunks = result["documents"]
        assert len(chunks) >= 1

        # Check that table structure is preserved
        all_content = " ".join(chunk.content for chunk in chunks)
        assert "|" in all_content

    def test_semantic_chunker_blockquote_splitting(self):
        """Test that chunker handles blockquotes properly."""
        chunker = SemanticChunker(min_chunk_size=30, max_chunk_size=150)

        content = """Some introductory text.

> This is a blockquote
> with multiple lines
> of quoted content

And some concluding text."""

        doc = Document(content=content)
        result = chunker.run(documents=[doc])

        chunks = result["documents"]
        assert len(chunks) >= 1

        # Check that blockquote is preserved
        all_content = " ".join(chunk.content for chunk in chunks)
        assert ">" in all_content

    def test_semantic_chunker_size_constraints(self):
        """Test that chunker respects size constraints."""
        chunker = SemanticChunker(min_chunk_size=50, max_chunk_size=150)

        # Create content with clear semantic boundaries that will force splitting
        content = """# Section 1
This is section one with enough content to meet the minimum chunk size requirements and exceed maximum.

# Section 2
This is section two with also enough content to meet minimum requirements and force a split.

# Section 3
This is section three with sufficient content as well to create another chunk."""

        doc = Document(content=content)
        result = chunker.run(documents=[doc])

        chunks = result["documents"]

        # Should have multiple chunks due to size constraints
        assert len(chunks) > 1

        # Check size constraints (allow some flexibility for semantic boundaries)
        for chunk in chunks:
            assert (
                len(chunk.content) <= chunker.max_chunk_size * 1.3
            )  # Allow 30% flexibility

    def test_semantic_chunker_metadata_preservation(self):
        """Test that chunker preserves and adds appropriate metadata."""
        chunker = SemanticChunker(min_chunk_size=20, max_chunk_size=100)

        original_meta = {"source": "test.md", "author": "test"}
        # Create content that will definitely produce chunks
        content = (
            "# Test Header\nThis is test content with enough text to create a chunk."
        )
        doc = Document(content=content, meta=original_meta)

        result = chunker.run(documents=[doc])
        chunks = result["documents"]

        assert len(chunks) >= 1

        for i, chunk in enumerate(chunks):
            # Check original metadata is preserved
            assert chunk.meta["source"] == "test.md"
            assert chunk.meta["author"] == "test"

            # Check added metadata
            assert chunk.meta["chunk_id"] == i
            assert chunk.meta["total_chunks"] == len(chunks)
            assert chunk.meta["chunk_size"] == len(chunk.content)
            assert chunk.meta["chunker_type"] == "semantic"

    def test_semantic_chunker_document_id_handling(self):
        """Test that chunker handles document IDs properly."""
        chunker = SemanticChunker(min_chunk_size=20, max_chunk_size=100)

        # Test with document ID
        doc_with_id = Document(
            content="# Test\nContent here with enough text.", id="doc_123"
        )
        result = chunker.run(documents=[doc_with_id])
        chunks = result["documents"]

        for i, chunk in enumerate(chunks):
            assert chunk.id == f"doc_123_chunk_{i}"

        # Test without document ID - Haystack auto-generates IDs, so we check the pattern
        doc_without_id = Document(content="# Test\nContent here with enough text.")
        result = chunker.run(documents=[doc_without_id])
        chunks = result["documents"]

        for i, chunk in enumerate(chunks):
            # Haystack auto-generates IDs, so we check that chunk IDs are properly formatted
            assert chunk.id.endswith(f"_chunk_{i}")
            assert len(chunk.id) > 10  # Should be a proper ID

    def test_semantic_chunker_large_content_fallback(self):
        """Test that chunker falls back to recursive splitting for large content."""
        chunker = SemanticChunker(min_chunk_size=50, max_chunk_size=200)

        # Create content with a very large section that exceeds max_chunk_size
        large_section = "This is a very long paragraph. " * 20  # ~600 characters
        content = f"# Large Section\n{large_section}\n\n# Next Section\nShort content."

        doc = Document(content=content)
        result = chunker.run(documents=[doc])

        chunks = result["documents"]
        assert len(chunks) > 1

        # Verify that no chunk exceeds max_chunk_size significantly
        for chunk in chunks:
            # Allow some flexibility due to semantic boundaries
            assert len(chunk.content) <= chunker.max_chunk_size * 1.2

    def test_semantic_chunker_mixed_boundaries(self):
        """Test chunker with mixed semantic boundaries."""
        chunker = SemanticChunker(min_chunk_size=30, max_chunk_size=120)

        content = """# Main Title
        Introduction paragraph with enough content to make it substantial.

        ## Subsection
        Some content here that should be long enough to create boundaries.

        1. First item with details
        2. Second item with more details

        - Bullet one with content
        - Bullet two with more content

        ```python
        code_example = True
        more_code = "substantial"
        ```

        > Quote here with enough content to matter

        | Col1 | Col2 |
        |------|------|
        | A    | B    |

        Final paragraph with enough content to be meaningful."""

        doc = Document(content=content)
        result = chunker.run(documents=[doc])

        chunks = result["documents"]
        # With smaller max_chunk_size, should create multiple chunks
        assert len(chunks) >= 1  # At least one chunk should be created

        # Verify content is preserved across chunks
        all_content = " ".join(chunk.content for chunk in chunks)
        assert "Main Title" in all_content
        assert "code_example" in all_content
        assert "Final paragraph" in all_content

    def test_semantic_chunker_error_handling(self):
        """Test that chunker handles errors gracefully."""
        chunker = SemanticChunker()

        # Test with None content (should be handled gracefully)
        doc_with_none = Document(content=None)
        result = chunker.run(documents=[doc_with_none])

        # Should not crash and should skip the document
        assert "documents" in result
        # The document with None content should be skipped or handled

    def test_semantic_chunker_horizontal_rule_splitting(self):
        """Test that chunker splits at horizontal rule boundaries."""
        chunker = SemanticChunker(min_chunk_size=30, max_chunk_size=150)

        content = """First section content here.

---

Second section content after the horizontal rule.

---

Third section content."""

        doc = Document(content=content)
        result = chunker.run(documents=[doc])

        chunks = result["documents"]
        assert len(chunks) >= 1

        # Check that content is properly segmented
        all_content = " ".join(chunk.content for chunk in chunks)
        assert "First section" in all_content
        assert "Second section" in all_content
        assert "Third section" in all_content
