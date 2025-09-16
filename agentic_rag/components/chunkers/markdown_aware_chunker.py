"""Markdown-aware document chunker component."""

import re
from typing import Dict, List, Optional

from haystack import Document, component, logging

logger = logging.getLogger(__name__)


@component
class MarkdownAwareChunker:
    """
    A chunker that preserves markdown structure when splitting documents.

    This chunker splits text by markdown headers and ensures that content
    remains associated with its relevant headers. It falls back to recursive
    character splitting for oversized sections.

    Usage example:
    ```python
    from agentic_rag.components.chunkers import MarkdownAwareChunker

    chunker = MarkdownAwareChunker(chunk_size=1000)
    result = chunker.run(documents=[document])
    chunks = result["documents"]
    ```
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None,
    ):
        """
        Create a MarkdownAwareChunker component.

        :param chunk_size: Maximum size of each chunk in characters.
        :param chunk_overlap: Number of characters to overlap between chunks.
        :param separators: List of separators for recursive splitting (fallback).
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    @component.output_types(documents=List[Document])  # type: ignore[misc]
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """
        Split documents into chunks while preserving markdown structure.

        :param documents: List of Haystack Documents to chunk.
        :return: Dictionary with the following keys:
            - `documents`: List of chunked Haystack Documents.
        """
        chunked_docs = []

        for doc in documents:
            if not doc.content:
                continue

            try:
                chunks = self._markdown_aware_split(doc.content)

                for i, chunk_content in enumerate(chunks):
                    # Create new document with chunk content
                    chunk_meta = doc.meta.copy() if doc.meta else {}
                    chunk_meta.update(
                        {
                            "chunk_id": i,
                            "total_chunks": len(chunks),
                            "chunk_size": len(chunk_content),
                        }
                    )

                    chunk_id = f"{doc.id}_chunk_{i}" if doc.id else f"chunk_{i}"
                    chunked_doc = Document(
                        content=chunk_content,
                        meta=chunk_meta,
                        id=chunk_id,
                    )
                    chunked_docs.append(chunked_doc)

            except Exception as e:
                logger.error(f"Error chunking document: {e}")
                # If chunking fails, keep original document
                chunked_docs.append(doc)

        return {"documents": chunked_docs}

    def _markdown_aware_split(self, text: str) -> List[str]:
        """
        Chunk text while preserving markdown structure.

        :param text: Text content to chunk.
        :return: List of text chunks.
        """
        # Split by markdown sections (headers)
        sections = re.split(r"\n(#{1,6}\s+.*?\n)", text)

        chunks = []
        current_chunk = ""
        current_header = ""

        for i, section in enumerate(sections):
            if re.match(r"#{1,6}\s+", section):
                # This is a header
                if current_chunk and len(current_chunk) > self.chunk_size:
                    # Current chunk is too big, split it recursively
                    chunks.extend(self._recursive_character_split(current_chunk))
                    current_chunk = ""
                elif current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                current_header = section.strip()
                current_chunk = current_header
            else:
                # This is content
                if len(current_chunk + section) <= self.chunk_size:
                    current_chunk += section
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())

                    # If section is too big, split it
                    if len(section) > self.chunk_size:
                        section_chunks = self._recursive_character_split(section)
                        # Add header to first chunk if we have one
                        if section_chunks and current_header:
                            section_chunks[0] = (
                                current_header + "\n" + section_chunks[0]
                            )
                        chunks.extend(section_chunks)
                        current_chunk = ""
                    else:
                        current_chunk = (
                            current_header + "\n" + section
                            if current_header
                            else section
                        )

        if current_chunk:
            chunks.append(current_chunk.strip())

        return [chunk for chunk in chunks if chunk.strip()]

    def _recursive_character_split(self, text: str) -> List[str]:
        """
        Recursively split text using different separators.

        :param text: Text to split.
        :return: List of text chunks.
        """

        def _split_text(text: str, separators: List[str]) -> List[str]:
            if not separators or len(text) <= self.chunk_size:
                return [text] if text.strip() else []

            separator = separators[0]
            remaining_separators = separators[1:]

            if separator not in text:
                return _split_text(text, remaining_separators)

            chunks = []
            current_chunk = ""

            for part in text.split(separator):
                test_chunk = current_chunk + separator + part if current_chunk else part

                if len(test_chunk) <= self.chunk_size:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk)

                    if len(part) > self.chunk_size:
                        chunks.extend(_split_text(part, remaining_separators))
                        current_chunk = ""
                    else:
                        current_chunk = part

            if current_chunk:
                chunks.append(current_chunk)

            return chunks

        return _split_text(text, self.separators)
