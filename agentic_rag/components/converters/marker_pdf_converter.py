# SPDX-FileCopyrightText: 2022-present deepset GmbH <info@deepset.ai>
#
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from typing import Any, List, Optional, Union

from haystack import Document, component, default_from_dict, default_to_dict, logging
from haystack.components.converters.utils import (
    get_bytestream_from_source,
    normalize_metadata,
)
from haystack.dataclasses import ByteStream
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

logger = logging.getLogger(__name__)


@component
class MarkerPDFToDocument:
    """
    Converts PDF files to documents using the Marker library for enhanced text extraction.

    Marker provides better text extraction compared to standard PDF parsers,
    especially for complex layouts and academic papers.

    ### Usage example

    ```python
    from agentic_rag.components.converters import MarkerPDFToDocument

    converter = MarkerPDFToDocument()
    results = converter.run(sources=["sample.pdf"], meta={"date_added": datetime.now().isoformat()})
    documents = results["documents"]
    print(documents[0].content)
    # 'This is markdown text extracted from the PDF file.'
    ```
    """

    def __init__(
        self,
        *,
        languages: str = "en",
        output_format: str = "markdown",
        store_full_path: bool = False,
    ):
        """
        Create a MarkerPDFToDocument component.

        :param languages:
            Languages to use for text extraction. Defaults to "en" (English).
        :param output_format:
            Output format for extracted text. Defaults to "markdown".
        :param store_full_path:
            If True, the full path of the file is stored in the metadata of the document.
            If False, only the file name is stored.
        """

        self.languages = languages
        self.output_format = output_format
        self.store_full_path = store_full_path

        # Initialize marker models - these will be loaded lazily
        self._marker_models = None
        self._marker_converter = None

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes the component to a dictionary.

        :returns:
            Dictionary with serialized data.
        """
        return default_to_dict(  # type: ignore[no-any-return]
            self,
            languages=self.languages,
            output_format=self.output_format,
            store_full_path=self.store_full_path,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MarkerPDFToDocument":
        """
        Deserializes the component from a dictionary.

        :param data:
            Dictionary with serialized data.

        :returns:
            Deserialized component.
        """
        return default_from_dict(cls, data)  # type: ignore[no-any-return]

    def _initialize_marker(self) -> None:
        """Initialize marker models and converter if not already initialized."""
        if self._marker_converter is None:
            try:
                logger.info("Loading marker models (this may take a moment)...")

                self._marker_models = create_model_dict()
                config_parser = ConfigParser(
                    {
                        "languages": self.languages,
                        "output_format": self.output_format,
                    }
                )
                self._marker_converter = PdfConverter(
                    config=config_parser.generate_config_dict(),
                    artifact_dict=self._marker_models,
                    processor_list=config_parser.get_processors(),
                    renderer=config_parser.get_renderer(),
                )

                logger.info("Marker models loaded successfully")

            except ImportError as e:
                raise ImportError(
                    f"Failed to import marker components. Make sure marker-pdf is installed: {e}"
                )

    def _convert_pdf_with_marker(self, file_path: str) -> str:
        """
        Convert a PDF file using marker.

        :param file_path: Path to the PDF file
        :returns: Extracted text content
        """
        self._initialize_marker()

        # Ensure the input_path is a valid file
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input path not found: {file_path}")

        # Check if the path is a file and a PDF
        if not os.path.isfile(file_path):
            raise ValueError(f"Path is not a file: {file_path}")

        if not file_path.lower().endswith(".pdf"):
            raise ValueError(f"File at {file_path} is not a PDF.")

        try:
            # Use the cached converter for conversion
            rendered = self._marker_converter(file_path)  # type: ignore[misc]
            return str(rendered.markdown)
        except Exception as e:
            logger.warning(f"Failed to convert PDF {file_path} with marker: {e}")
            return ""

    @component.output_types(documents=List[Document])  # type: ignore[misc]
    def run(
        self,
        sources: List[Union[str, Path, ByteStream]],
        meta: Optional[Union[dict[str, Any], List[dict[str, Any]]]] = None,
    ) -> dict[str, List[Document]]:
        """
        Converts PDF files to documents using Marker.

        :param sources:
            List of file paths or ByteStream objects to convert.
        :param meta:
            Optional metadata to attach to the documents.
            This value can be a list of dictionaries or a single dictionary.
            If it's a single dictionary, its content is added to the metadata of all produced documents.
            If it's a list, its length must match the number of sources, as they are zipped together.
            For ByteStream objects, their `meta` is added to the output documents.

        :returns:
            A dictionary with the following keys:
            - `documents`: A list of converted documents.
        """
        documents = []
        meta_list = normalize_metadata(meta, sources_count=len(sources))

        for source, metadata in zip(sources, meta_list):
            try:
                # Convert all input types (str, Path, ByteStream) to ByteStream for uniform processing
                # This ensures consistent handling regardless of whether the PDF comes from:
                # - A file path: "/path/to/file.pdf"
                # - A Path object: Path("/path/to/file.pdf")
                # - Raw bytes: ByteStream(data=pdf_bytes) from previous pipeline components
                bytestream = get_bytestream_from_source(source)
            except Exception as e:
                logger.warning(
                    "Could not read {source}. Skipping it. Error: {error}",
                    source=source,
                    error=e,
                )
                continue

            # Write ByteStream to temporary file for marker processing
            # WHY TEMPORARY FILE IS NEEDED:
            # - We now have PDF data as bytes in memory (bytestream.data)
            # - But Marker library requires a file path, not raw bytes
            # - So we create a temporary file as a "bridge" between memory and Marker
            # - Even if source was originally a file path, we go through ByteStream for pipeline compatibility
            import tempfile

            try:
                # Create temporary file with .pdf extension (delete=False means we control cleanup)
                # delete=False is crucial - if True, file would be deleted when 'with' block ends,
                # but we need it to exist when we call Marker later
                with tempfile.NamedTemporaryFile(
                    suffix=".pdf", delete=False
                ) as temp_file:
                    temp_file.write(
                        bytestream.data
                    )  # Write PDF bytes from memory to disk
                    temp_file_path = (
                        temp_file.name
                    )  # Get the temporary file path for Marker

                # Convert using marker - now Marker can read from the temporary file path
                text = self._convert_pdf_with_marker(temp_file_path)

                # Clean up temporary file manually (since we used delete=False)
                # This prevents accumulation of temp files on disk
                os.unlink(temp_file_path)

            except Exception as e:
                logger.warning(
                    "Could not read {source} and convert it to Document, skipping. {error}",
                    source=source,
                    error=e,
                )
                # Clean up temp file if it exists
                try:
                    if "temp_file_path" in locals():
                        os.unlink(temp_file_path)
                except OSError:
                    pass
                continue

            if text is None or text.strip() == "":
                logger.warning(
                    "MarkerPDFToDocument could not extract text from the file {source}. Returning an empty document.",
                    source=source,
                )

            merged_metadata = {**bytestream.meta, **metadata}

            if not self.store_full_path and (
                file_path := bytestream.meta.get("file_path")
            ):
                merged_metadata["file_path"] = os.path.basename(file_path)

            document = Document(content=text, meta=merged_metadata)
            documents.append(document)

        return {"documents": documents}
