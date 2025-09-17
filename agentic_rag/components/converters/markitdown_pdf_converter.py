"""MarkItDown PDF converter component for Haystack pipelines."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from haystack import Document, component, default_from_dict, default_to_dict
from haystack.components.converters.utils import (
    get_bytestream_from_source,
    normalize_metadata,
)
from haystack.dataclasses import ByteStream
from markitdown import MarkItDown

logger = logging.getLogger(__name__)


@component
class MarkItDownPDFToDocument:
    """
    Converts PDF files to documents using Microsoft's MarkItDown library.

    MarkItDown provides good PDF to markdown conversion with support for various document formats.
    This component focuses specifically on PDF conversion to markdown format.

    ### Usage example

    ```python
    from agentic_rag.components.converters import MarkItDownPDFToDocument

    converter = MarkItDownPDFToDocument()
    results = converter.run(sources=["sample.pdf"], meta={"date_added": "2024-01-01"})
    documents = results["documents"]
    print(documents[0].content)
    # 'This is markdown text extracted from the PDF file.'
    ```
    """

    def __init__(
        self,
        *,
        store_full_path: bool = False,
    ):
        """
        Create a MarkItDownPDFToDocument component.

        :param store_full_path:
            If True, the full path of the file is stored in the metadata of the document.
            If False, only the file name is stored.
        """
        self.store_full_path = store_full_path
        self._markitdown_instance: Optional["MarkItDown"] = None

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(  # type: ignore[no-any-return]
            self,
            store_full_path=self.store_full_path,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MarkItDownPDFToDocument":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary with serialized data.

        :returns:
            Deserialized component.
        """
        return default_from_dict(cls, data)  # type: ignore[no-any-return]

    def _initialize_markitdown(self) -> None:
        """Initialize MarkItDown instance if not already initialized."""
        if self._markitdown_instance is None:
            try:
                from markitdown import MarkItDown

                logger.info("Loading MarkItDown instance...")
                self._markitdown_instance = MarkItDown(enable_plugins=False)
                logger.info("MarkItDown instance loaded successfully")
            except ImportError as e:
                raise ImportError(
                    "MarkItDown library is not installed. "
                    "Please install it with: pip install markitdown"
                ) from e

    def _convert_pdf_with_markitdown(self, file_path: Path) -> str:
        """
        Convert a single PDF file using MarkItDown.

        :param file_path: Path to the PDF file to convert
        :returns: Extracted markdown content
        :raises: Various exceptions for file access or conversion errors
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")

        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {file_path}")

        try:
            logger.info(f"Converting {file_path} using MarkItDown")
            if self._markitdown_instance is None:
                raise RuntimeError("MarkItDown instance not initialized")

            result = self._markitdown_instance.convert(str(file_path))

            if result and hasattr(result, "text_content"):
                content: str = result.text_content.strip()
                if content:
                    return content
                else:
                    logger.warning(f"MarkItDown returned empty content for {file_path}")
                    return ""
            else:
                logger.warning(f"MarkItDown returned invalid result for {file_path}")
                return ""

        except Exception as e:
            logger.error(f"Error converting {file_path} with MarkItDown: {e}")
            raise RuntimeError(
                f"MarkItDown conversion failed for {file_path}: {e}"
            ) from e

    @component.output_types(documents=List[Document])  # type: ignore[misc]
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
    ) -> Dict[str, List[Document]]:
        """
        Converts PDF files to Documents using MarkItDown.

        :param sources: List of file paths or ByteStream objects to convert.
        :param meta: Optional metadata to attach to the Documents.
                    This value can be either a single dictionary or a list of dictionaries.
                    If it's a single dictionary, its content is added to all produced Documents.
                    If it's a list, its length must match the number of sources.
                    For ByteStream objects, their `meta` is added to the output documents.
        :returns: Dictionary with the following keys:
            - `documents`: List of created Documents
        """
        self._initialize_markitdown()

        documents = []
        meta_list = normalize_metadata(meta=meta, sources_count=len(sources))

        for source, metadata in zip(sources, meta_list):
            try:
                # Convert source to ByteStream (handles file paths, ByteStream objects, etc.)
                bytestream = get_bytestream_from_source(source)
            except Exception as e:
                logger.warning(f"Could not read {source}. Skipping it. Error: {e}")
                continue

            try:
                # Write ByteStream data to temporary file for MarkItDown
                import tempfile

                with tempfile.NamedTemporaryFile(
                    suffix=".pdf", delete=False
                ) as temp_file:
                    temp_file.write(bytestream.data)
                    temp_file_path = Path(temp_file.name)

                try:
                    # Convert the temporary PDF file
                    markdown_content = self._convert_pdf_with_markitdown(temp_file_path)
                finally:
                    # Clean up temporary file
                    temp_file_path.unlink(missing_ok=True)

            except Exception as e:
                logger.warning(
                    f"Could not convert {source} to Document, skipping. Error: {e}"
                )
                continue

            if markdown_content is None or markdown_content.strip() == "":
                logger.warning(
                    f"MarkItDown could not extract text from {source}. Returning an empty document."
                )

            # Merge ByteStream metadata with provided metadata
            merged_metadata = {**bytestream.meta, **metadata}

            # Handle file path storage
            if not self.store_full_path and (
                file_path := bytestream.meta.get("file_path")
            ):
                merged_metadata["file_path"] = os.path.basename(file_path)

            # Add conversion method info
            merged_metadata["converter"] = "markitdown"

            document = Document(content=markdown_content or "", meta=merged_metadata)
            documents.append(document)

        logger.info(
            f"Successfully converted {len(documents)} PDF files using MarkItDown"
        )
        return {"documents": documents}
