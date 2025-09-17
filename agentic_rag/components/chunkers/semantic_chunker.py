"""Semantic boundary-aware document chunker component."""

import re
from typing import Dict, List

from haystack import Document, component, logging

logger = logging.getLogger(__name__)


@component
class SemanticChunker:
    """
    A chunker that splits text at semantic boundaries like headers, lists, and code blocks.

    This chunker identifies semantic boundaries in text (headers, lists, tables, code blocks, etc.)
    and splits content at these natural break points while respecting size constraints.
    It falls back to recursive character splitting for oversized segments.

    Usage example:
    ```python
    from agentic_rag.components.chunkers import SemanticChunker

    chunker = SemanticChunker(min_chunk_size=200, max_chunk_size=1000)
    result = chunker.run(documents=[document])
    chunks = result["documents"]
    ```
    """

    def __init__(
        self,
        min_chunk_size: int = 200,
        max_chunk_size: int = 1000,
        overlap_size: int = 50,
    ):
        """
        Create a SemanticChunker component.

        :param min_chunk_size: Minimum size of each chunk in characters.
        :param max_chunk_size: Maximum size of each chunk in characters.
        :param overlap_size: Number of characters to overlap between chunks when splitting.
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, List[Document]]:
        """
        Split documents into chunks at semantic boundaries.

        :param documents: List of Haystack Documents to chunk.
        :return: Dictionary with the following keys:
            - `documents`: List of chunked Haystack Documents.
        """
        chunked_docs = []

        for doc in documents:
            if not doc.content:
                continue

            try:
                chunks = self._semantic_split(doc.content)

                for i, chunk_content in enumerate(chunks):
                    # Create new document with chunk content
                    chunk_meta = doc.meta.copy() if doc.meta else {}
                    chunk_meta.update(
                        {
                            "chunk_id": i,
                            "total_chunks": len(chunks),
                            "chunk_size": len(chunk_content),
                            "chunker_type": "semantic",
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

    def _semantic_split(self, text: str) -> List[str]:
        """
        Split text at semantic boundaries (sections, lists, etc.).

        :param text: Text content to chunk.
        :return: List of text chunks.
        """
        # Patterns that indicate semantic boundaries
        semantic_patterns = [
            r"(^|\n)(#{1,6}\s+.*?)(?=\n|$)",  # Headers
            r"(^|\n)(\d+\.\s+)",  # Numbered lists
            r"(^|\n)([*-]\s+)",  # Bullet points
            r"(^|\n)(\|\s*.*?\s*\|)",  # Tables
            r"(^|\n)(```.*?```)",  # Code blocks
            r"(^|\n)(>\s+)",  # Blockquotes
            r"(^|\n)(---+)(?=\n|$)",  # Horizontal rules
        ]

        # Find all semantic boundaries
        boundaries = [0]
        for pattern in semantic_patterns:
            for match in re.finditer(pattern, text, re.MULTILINE | re.DOTALL):
                boundaries.append(match.start())

        boundaries.append(len(text))
        boundaries = sorted(set(boundaries))

        # If no boundaries found, use fallback splitting
        if len(boundaries) <= 2:  # Only start and end
            return self._recursive_character_split(text)

        chunks = []
        current_chunk = ""

        for i in range(len(boundaries) - 1):
            start, end = boundaries[i], boundaries[i + 1]
            segment = text[start:end]

            # Try to add segment to current chunk
            if len(current_chunk + segment) <= self.max_chunk_size:
                current_chunk += segment
            else:
                # Current chunk + segment exceeds max size
                if current_chunk and len(current_chunk.strip()) >= self.min_chunk_size:
                    chunks.append(current_chunk.strip())
                    current_chunk = segment
                elif len(segment) > self.max_chunk_size:
                    # Segment itself is too large, split it
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                    chunks.extend(self._recursive_character_split(segment))
                else:
                    # Current chunk is too small, combine with segment anyway
                    current_chunk += segment

        # Add final chunk if it exists
        if current_chunk.strip():
            if len(current_chunk.strip()) >= self.min_chunk_size:
                chunks.append(current_chunk.strip())
            elif chunks:
                # If final chunk is too small, merge with last chunk if possible
                last_chunk = chunks.pop()
                if (
                    len(last_chunk + " " + current_chunk.strip())
                    <= self.max_chunk_size * 1.2
                ):
                    chunks.append((last_chunk + " " + current_chunk.strip()).strip())
                else:
                    chunks.append(last_chunk)
                    chunks.append(current_chunk.strip())
            else:
                # Only chunk and it's too small, but keep it anyway
                chunks.append(current_chunk.strip())

        return [chunk for chunk in chunks if chunk.strip()]

    def _recursive_character_split(self, text: str) -> List[str]:
        """
        Recursively split text using different separators.

        :param text: Text to split.
        :return: List of text chunks.
        """
        separators = ["\n\n", "\n", " ", ""]

        def _split_text(text: str, separators: List[str]) -> List[str]:
            if not separators or len(text) <= self.max_chunk_size:
                return [text] if text.strip() else []

            separator = separators[0]
            remaining_separators = separators[1:]

            if separator not in text:
                return _split_text(text, remaining_separators)

            chunks = []
            current_chunk = ""

            for part in text.split(separator):
                test_chunk = current_chunk + separator + part if current_chunk else part

                if len(test_chunk) <= self.max_chunk_size:
                    current_chunk = test_chunk
                else:
                    if current_chunk:
                        chunks.append(current_chunk)

                    if len(part) > self.max_chunk_size:
                        chunks.extend(_split_text(part, remaining_separators))
                        current_chunk = ""
                    else:
                        current_chunk = part

            if current_chunk:
                chunks.append(current_chunk)

            return chunks

        return _split_text(text, separators)
