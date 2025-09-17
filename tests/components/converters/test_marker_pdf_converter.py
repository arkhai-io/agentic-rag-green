"""Test MarkerPDFToDocument converter component."""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from agentic_rag.components.converters.marker_pdf_converter import MarkerPDFToDocument


class TestMarkerPDFToDocument:
    """Test MarkerPDFToDocument functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.sample_pdf_content = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n2 0 obj\n<<\n/Type /Pages\n/Kids [3 0 R]\n/Count 1\n>>\nendobj\n3 0 obj\n<<\n/Type /Page\n/Parent 2 0 R\n/MediaBox [0 0 612 792]\n/Contents 4 0 R\n>>\nendobj\n4 0 obj\n<<\n/Length 44\n>>\nstream\nBT\n/F1 12 Tf\n72 720 Td\n(Hello World!) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f \n0000000009 00000 n \n0000000074 00000 n \n0000000120 00000 n \n0000000179 00000 n \ntrailer\n<<\n/Size 5\n/Root 1 0 R\n>>\nstartxref\n235\n%%EOF"

    def teardown_method(self):
        """Clean up test environment."""
        if hasattr(self, "temp_dir") and Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def create_sample_pdf(self, filename: str = "sample.pdf") -> str:
        """Create a sample PDF file for testing."""
        pdf_path = os.path.join(self.temp_dir, filename)
        with open(pdf_path, "wb") as f:
            f.write(self.sample_pdf_content)
        return pdf_path

    def test_marker_converter_initialization(self):
        """Test that MarkerPDFToDocument initializes correctly."""
        converter = MarkerPDFToDocument()

        assert converter.languages == "en"
        assert converter.output_format == "markdown"
        assert converter.store_full_path is False
        assert converter._marker_models is None
        assert converter._marker_converter is None

    def test_marker_converter_custom_initialization(self):
        """Test MarkerPDFToDocument with custom parameters."""
        converter = MarkerPDFToDocument(
            languages="es", output_format="html", store_full_path=True
        )

        assert converter.languages == "es"
        assert converter.output_format == "html"
        assert converter.store_full_path is True

    def test_to_dict_serialization(self):
        """Test component serialization."""
        converter = MarkerPDFToDocument(
            languages="fr", output_format="text", store_full_path=True
        )

        serialized = converter.to_dict()

        assert "type" in serialized
        assert serialized["init_parameters"]["languages"] == "fr"
        assert serialized["init_parameters"]["output_format"] == "text"
        assert serialized["init_parameters"]["store_full_path"] is True

    def test_from_dict_deserialization(self):
        """Test component deserialization."""
        data = {
            "type": "agentic_rag.components.converters.marker_pdf_converter.MarkerPDFToDocument",
            "init_parameters": {
                "languages": "de",
                "output_format": "markdown",
                "store_full_path": False,
            },
        }

        converter = MarkerPDFToDocument.from_dict(data)

        assert converter.languages == "de"
        assert converter.output_format == "markdown"
        assert converter.store_full_path is False

    @patch("agentic_rag.components.converters.marker_pdf_converter.create_model_dict")
    @patch("agentic_rag.components.converters.marker_pdf_converter.ConfigParser")
    @patch("agentic_rag.components.converters.marker_pdf_converter.PdfConverter")
    def test_marker_initialization_success(
        self, mock_pdf_converter, mock_config_parser, mock_create_model_dict
    ):
        """Test successful marker model initialization."""
        # Setup mocks
        mock_models = {"model1": "test_model"}
        mock_create_model_dict.return_value = mock_models

        mock_config_instance = Mock()
        mock_config_instance.generate_config_dict.return_value = {"config": "test"}
        mock_config_instance.get_processors.return_value = ["processor1"]
        mock_config_instance.get_renderer.return_value = "renderer1"
        mock_config_parser.return_value = mock_config_instance

        mock_converter_instance = Mock()
        mock_pdf_converter.return_value = mock_converter_instance

        converter = MarkerPDFToDocument()
        converter._initialize_marker()

        # Verify initialization
        assert converter._marker_models == mock_models
        assert converter._marker_converter == mock_converter_instance

        # Verify calls
        mock_create_model_dict.assert_called_once()
        mock_config_parser.assert_called_once_with(
            {"languages": "en", "output_format": "markdown"}
        )
        mock_pdf_converter.assert_called_once()

    @patch("agentic_rag.components.converters.marker_pdf_converter.create_model_dict")
    def test_marker_initialization_import_error(self, mock_create_model_dict):
        """Test marker initialization with import error."""
        mock_create_model_dict.side_effect = ImportError("marker not installed")

        converter = MarkerPDFToDocument()

        with pytest.raises(ImportError, match="Failed to import marker components"):
            converter._initialize_marker()

    @patch("agentic_rag.components.converters.marker_pdf_converter.create_model_dict")
    @patch("agentic_rag.components.converters.marker_pdf_converter.ConfigParser")
    @patch("agentic_rag.components.converters.marker_pdf_converter.PdfConverter")
    def test_convert_pdf_with_marker_file_not_found(
        self, mock_pdf_converter, mock_config_parser, mock_create_model_dict
    ):
        """Test PDF conversion with non-existent file."""
        # Setup mocks to prevent model loading
        mock_create_model_dict.return_value = {"model": "test"}
        mock_config_instance = Mock()
        mock_config_parser.return_value = mock_config_instance
        mock_pdf_converter.return_value = Mock()

        converter = MarkerPDFToDocument()

        with pytest.raises(FileNotFoundError, match="Input path not found"):
            converter._convert_pdf_with_marker("/nonexistent/file.pdf")

    @patch("agentic_rag.components.converters.marker_pdf_converter.create_model_dict")
    @patch("agentic_rag.components.converters.marker_pdf_converter.ConfigParser")
    @patch("agentic_rag.components.converters.marker_pdf_converter.PdfConverter")
    def test_convert_pdf_with_marker_not_a_file(
        self, mock_pdf_converter, mock_config_parser, mock_create_model_dict
    ):
        """Test PDF conversion with directory path."""
        # Setup mocks to prevent model loading
        mock_create_model_dict.return_value = {"model": "test"}
        mock_config_instance = Mock()
        mock_config_parser.return_value = mock_config_instance
        mock_pdf_converter.return_value = Mock()

        converter = MarkerPDFToDocument()

        with pytest.raises(ValueError, match="Path is not a file"):
            converter._convert_pdf_with_marker(self.temp_dir)

    @patch("agentic_rag.components.converters.marker_pdf_converter.create_model_dict")
    @patch("agentic_rag.components.converters.marker_pdf_converter.ConfigParser")
    @patch("agentic_rag.components.converters.marker_pdf_converter.PdfConverter")
    def test_convert_pdf_with_marker_not_pdf_file(
        self, mock_pdf_converter, mock_config_parser, mock_create_model_dict
    ):
        """Test PDF conversion with non-PDF file."""
        txt_path = os.path.join(self.temp_dir, "sample.txt")
        with open(txt_path, "w") as f:
            f.write("This is not a PDF")

        # Setup mocks to prevent model loading
        mock_create_model_dict.return_value = {"model": "test"}
        mock_config_instance = Mock()
        mock_config_parser.return_value = mock_config_instance
        mock_pdf_converter.return_value = Mock()

        converter = MarkerPDFToDocument()

        with pytest.raises(ValueError, match="is not a PDF"):
            converter._convert_pdf_with_marker(txt_path)

    @patch("agentic_rag.components.converters.marker_pdf_converter.create_model_dict")
    @patch("agentic_rag.components.converters.marker_pdf_converter.ConfigParser")
    @patch("agentic_rag.components.converters.marker_pdf_converter.PdfConverter")
    def test_convert_pdf_with_marker_success(
        self, mock_pdf_converter, mock_config_parser, mock_create_model_dict
    ):
        """Test successful PDF conversion with marker."""
        # Create sample PDF
        pdf_path = self.create_sample_pdf()

        # Setup mocks
        mock_create_model_dict.return_value = {"model": "test"}
        mock_config_instance = Mock()
        mock_config_parser.return_value = mock_config_instance

        mock_rendered = Mock()
        mock_rendered.markdown = "# Test Document\n\nHello World!"

        mock_converter_instance = Mock()
        mock_converter_instance.return_value = mock_rendered
        mock_pdf_converter.return_value = mock_converter_instance

        converter = MarkerPDFToDocument()
        result = converter._convert_pdf_with_marker(pdf_path)

        assert result == "# Test Document\n\nHello World!"
        mock_converter_instance.assert_called_once_with(pdf_path)

    @patch("agentic_rag.components.converters.marker_pdf_converter.create_model_dict")
    @patch("agentic_rag.components.converters.marker_pdf_converter.ConfigParser")
    @patch("agentic_rag.components.converters.marker_pdf_converter.PdfConverter")
    def test_convert_pdf_with_marker_conversion_error(
        self, mock_pdf_converter, mock_config_parser, mock_create_model_dict
    ):
        """Test PDF conversion with marker error."""
        # Create sample PDF
        pdf_path = self.create_sample_pdf()

        # Setup mocks
        mock_create_model_dict.return_value = {"model": "test"}
        mock_config_instance = Mock()
        mock_config_parser.return_value = mock_config_instance

        mock_converter_instance = Mock()
        mock_converter_instance.side_effect = Exception("Conversion failed")
        mock_pdf_converter.return_value = mock_converter_instance

        converter = MarkerPDFToDocument()
        result = converter._convert_pdf_with_marker(pdf_path)

        assert result == ""  # Should return empty string on error

    def test_run_empty_sources(self):
        """Test run method with empty sources list."""
        converter = MarkerPDFToDocument()
        result = converter.run(sources=[])

        assert result == {"documents": []}

    @patch(
        "agentic_rag.components.converters.marker_pdf_converter.get_bytestream_from_source"
    )
    def test_run_invalid_source(self, mock_get_bytestream):
        """Test run method with invalid source."""
        mock_get_bytestream.side_effect = Exception("Invalid source")

        converter = MarkerPDFToDocument()
        result = converter.run(sources=["invalid_source"])

        assert result == {"documents": []}

    @patch(
        "agentic_rag.components.converters.marker_pdf_converter.get_bytestream_from_source"
    )
    @patch("agentic_rag.components.converters.marker_pdf_converter.create_model_dict")
    @patch("agentic_rag.components.converters.marker_pdf_converter.ConfigParser")
    @patch("agentic_rag.components.converters.marker_pdf_converter.PdfConverter")
    def test_run_successful_conversion(
        self,
        mock_pdf_converter,
        mock_config_parser,
        mock_create_model_dict,
        mock_get_bytestream,
    ):
        """Test successful PDF conversion through run method."""
        from haystack import Document
        from haystack.dataclasses import ByteStream

        # Setup mocks
        mock_bytestream = ByteStream(
            data=self.sample_pdf_content, meta={"file_path": "/test/sample.pdf"}
        )
        mock_get_bytestream.return_value = mock_bytestream

        mock_create_model_dict.return_value = {"model": "test"}
        mock_config_instance = Mock()
        mock_config_parser.return_value = mock_config_instance

        mock_rendered = Mock()
        mock_rendered.markdown = "# Test Document\n\nConverted content"

        mock_converter_instance = Mock()
        mock_converter_instance.return_value = mock_rendered
        mock_pdf_converter.return_value = mock_converter_instance

        converter = MarkerPDFToDocument()
        result = converter.run(sources=["test_source"])

        assert "documents" in result
        assert len(result["documents"]) == 1

        document = result["documents"][0]
        assert isinstance(document, Document)
        assert document.content == "# Test Document\n\nConverted content"
        assert "file_path" in document.meta

    @patch(
        "agentic_rag.components.converters.marker_pdf_converter.get_bytestream_from_source"
    )
    @patch("agentic_rag.components.converters.marker_pdf_converter.create_model_dict")
    @patch("agentic_rag.components.converters.marker_pdf_converter.ConfigParser")
    @patch("agentic_rag.components.converters.marker_pdf_converter.PdfConverter")
    def test_run_with_metadata(
        self,
        mock_pdf_converter,
        mock_config_parser,
        mock_create_model_dict,
        mock_get_bytestream,
    ):
        """Test run method with custom metadata."""
        from haystack.dataclasses import ByteStream

        # Setup mocks
        mock_bytestream = ByteStream(
            data=self.sample_pdf_content,
            meta={"file_path": "/test/sample.pdf", "source": "test"},
        )
        mock_get_bytestream.return_value = mock_bytestream

        mock_create_model_dict.return_value = {"model": "test"}
        mock_config_instance = Mock()
        mock_config_parser.return_value = mock_config_instance

        mock_rendered = Mock()
        mock_rendered.markdown = "# Test Document"

        mock_converter_instance = Mock()
        mock_converter_instance.return_value = mock_rendered
        mock_pdf_converter.return_value = mock_converter_instance

        converter = MarkerPDFToDocument()
        custom_meta = {"author": "Test Author", "category": "Research"}
        result = converter.run(sources=["test_source"], meta=custom_meta)

        document = result["documents"][0]
        assert document.meta["author"] == "Test Author"
        assert document.meta["category"] == "Research"
        assert document.meta["source"] == "test"  # From bytestream
        assert "file_path" in document.meta

    @patch(
        "agentic_rag.components.converters.marker_pdf_converter.get_bytestream_from_source"
    )
    @patch("agentic_rag.components.converters.marker_pdf_converter.create_model_dict")
    @patch("agentic_rag.components.converters.marker_pdf_converter.ConfigParser")
    @patch("agentic_rag.components.converters.marker_pdf_converter.PdfConverter")
    def test_run_store_full_path_false(
        self,
        mock_pdf_converter,
        mock_config_parser,
        mock_create_model_dict,
        mock_get_bytestream,
    ):
        """Test run method with store_full_path=False."""
        from haystack.dataclasses import ByteStream

        # Setup mocks
        mock_bytestream = ByteStream(
            data=self.sample_pdf_content,
            meta={"file_path": "/very/long/path/to/document.pdf"},
        )
        mock_get_bytestream.return_value = mock_bytestream

        mock_create_model_dict.return_value = {"model": "test"}
        mock_config_instance = Mock()
        mock_config_parser.return_value = mock_config_instance

        mock_rendered = Mock()
        mock_rendered.markdown = "# Test Document"

        mock_converter_instance = Mock()
        mock_converter_instance.return_value = mock_rendered
        mock_pdf_converter.return_value = mock_converter_instance

        converter = MarkerPDFToDocument(store_full_path=False)
        result = converter.run(sources=["test_source"])

        document = result["documents"][0]
        assert document.meta["file_path"] == "document.pdf"  # Only basename

    @patch(
        "agentic_rag.components.converters.marker_pdf_converter.get_bytestream_from_source"
    )
    @patch("agentic_rag.components.converters.marker_pdf_converter.create_model_dict")
    @patch("agentic_rag.components.converters.marker_pdf_converter.ConfigParser")
    @patch("agentic_rag.components.converters.marker_pdf_converter.PdfConverter")
    def test_run_store_full_path_true(
        self,
        mock_pdf_converter,
        mock_config_parser,
        mock_create_model_dict,
        mock_get_bytestream,
    ):
        """Test run method with store_full_path=True."""
        from haystack.dataclasses import ByteStream

        # Setup mocks
        mock_bytestream = ByteStream(
            data=self.sample_pdf_content,
            meta={"file_path": "/very/long/path/to/document.pdf"},
        )
        mock_get_bytestream.return_value = mock_bytestream

        mock_create_model_dict.return_value = {"model": "test"}
        mock_config_instance = Mock()
        mock_config_parser.return_value = mock_config_instance

        mock_rendered = Mock()
        mock_rendered.markdown = "# Test Document"

        mock_converter_instance = Mock()
        mock_converter_instance.return_value = mock_rendered
        mock_pdf_converter.return_value = mock_converter_instance

        converter = MarkerPDFToDocument(store_full_path=True)
        result = converter.run(sources=["test_source"])

        document = result["documents"][0]
        assert (
            document.meta["file_path"] == "/very/long/path/to/document.pdf"
        )  # Full path

    @patch(
        "agentic_rag.components.converters.marker_pdf_converter.get_bytestream_from_source"
    )
    @patch("agentic_rag.components.converters.marker_pdf_converter.create_model_dict")
    @patch("agentic_rag.components.converters.marker_pdf_converter.ConfigParser")
    @patch("agentic_rag.components.converters.marker_pdf_converter.PdfConverter")
    def test_run_empty_content_warning(
        self,
        mock_pdf_converter,
        mock_config_parser,
        mock_create_model_dict,
        mock_get_bytestream,
    ):
        """Test run method with empty conversion result."""
        from haystack.dataclasses import ByteStream

        # Setup mocks
        mock_bytestream = ByteStream(
            data=self.sample_pdf_content, meta={"file_path": "/test/empty.pdf"}
        )
        mock_get_bytestream.return_value = mock_bytestream

        mock_create_model_dict.return_value = {"model": "test"}
        mock_config_instance = Mock()
        mock_config_parser.return_value = mock_config_instance

        mock_rendered = Mock()
        mock_rendered.markdown = ""  # Empty content

        mock_converter_instance = Mock()
        mock_converter_instance.return_value = mock_rendered
        mock_pdf_converter.return_value = mock_converter_instance

        converter = MarkerPDFToDocument()
        result = converter.run(sources=["test_source"])

        # Should still create document even with empty content
        assert len(result["documents"]) == 1
        assert result["documents"][0].content == ""

    @patch("agentic_rag.components.converters.marker_pdf_converter.create_model_dict")
    @patch("agentic_rag.components.converters.marker_pdf_converter.ConfigParser")
    @patch("agentic_rag.components.converters.marker_pdf_converter.PdfConverter")
    def test_run_multiple_sources(
        self, mock_pdf_converter, mock_config_parser, mock_create_model_dict
    ):
        """Test run method with multiple sources."""
        # Create multiple sample PDFs
        pdf1_path = self.create_sample_pdf("doc1.pdf")
        pdf2_path = self.create_sample_pdf("doc2.pdf")

        # Setup mocks properly
        mock_create_model_dict.return_value = {"model": "test"}
        mock_config_instance = Mock()
        mock_config_parser.return_value = mock_config_instance

        mock_rendered = Mock()
        mock_rendered.markdown = "# Test Document\n\nConverted content"

        mock_converter_instance = Mock()
        mock_converter_instance.return_value = mock_rendered
        mock_pdf_converter.return_value = mock_converter_instance

        converter = MarkerPDFToDocument()
        result = converter.run(sources=[pdf1_path, pdf2_path])

        # Should get multiple documents
        assert len(result["documents"]) == 2
        assert all(
            doc.content == "# Test Document\n\nConverted content"
            for doc in result["documents"]
        )
