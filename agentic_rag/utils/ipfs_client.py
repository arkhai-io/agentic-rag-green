"""Lighthouse IPFS client for uploading and retrieving data."""

import io
import json
import os
from typing import Any, Dict, Optional, Union

import requests  # type: ignore
from dotenv import load_dotenv

load_dotenv()


class LighthouseClient:
    """
    Client for interacting with Lighthouse IPFS storage.

    Supports:
    - Upload files/text/JSON
    - Upload raw bytes/buffers
    - Retrieve data by CID
    """

    BASE_URL = "https://upload.lighthouse.storage/api/v0"
    GATEWAY_URL = "https://gateway.lighthouse.storage/ipfs"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Lighthouse client.

        Args:
            api_key: Lighthouse API key (defaults to LIGHTHOUSE_API_KEY env var)
        """
        self.api_key = api_key or os.getenv("LIGHTHOUSE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "LIGHTHOUSE_API_KEY not found. "
                "Set it in .env or pass it to LighthouseClient(api_key=...)"
            )

    def upload_text(self, text: str, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Upload text/JSON to Lighthouse.

        Args:
            text: Text content to upload (use json.dumps() for JSON)
            name: Optional name for the uploaded content

        Returns:
            {
                "Name": str,
                "Hash": str (IPFS CID),
                "Size": str
            }
        """
        # Convert text to bytes
        text_bytes = text.encode("utf-8")

        # Upload as buffer
        return self.upload_buffer(text_bytes, name=name)

    def upload_buffer(
        self, data: Union[bytes, io.BytesIO], name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upload raw bytes/buffer to Lighthouse.

        Args:
            data: Bytes or BytesIO buffer
            name: Optional filename

        Returns:
            {
                "Name": str,
                "Hash": str (IPFS CID),
                "Size": str
            }
        """
        # Convert BytesIO to bytes if needed
        if isinstance(data, io.BytesIO):
            data = data.getvalue()

        # Create file-like object
        if name is None:
            name = "data"

        files = {"file": (name, data)}
        headers = {"Authorization": f"Bearer {self.api_key}"}

        response = requests.post(
            f"{self.BASE_URL}/add",
            files=files,
            headers=headers,
            timeout=300,  # 5 min timeout for large files
        )
        response.raise_for_status()

        result = response.json()
        return {
            "Name": result.get("Name"),
            "Hash": result.get("Hash"),
            "Size": result.get("Size"),
        }

    def upload_json(
        self, data: Dict[str, Any], name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Upload JSON object to Lighthouse.

        Args:
            data: Dictionary to upload as JSON
            name: Optional name

        Returns:
            {
                "Name": str,
                "Hash": str (IPFS CID),
                "Size": str
            }
        """
        json_str = json.dumps(data, sort_keys=True)
        return self.upload_text(json_str, name=name)

    def retrieve(self, cid: str) -> bytes:
        """
        Retrieve raw bytes from IPFS via Lighthouse API.

        Uses Lighthouse API first (authenticated), falls back to public gateways.
        """
        errors = []

        # Try Lighthouse API first (authenticated, more reliable)
        if self.api_key:
            try:
                headers = {"Authorization": f"Bearer {self.api_key}"}
                url = f"https://gateway.lighthouse.storage/ipfs/{cid}"
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                return bytes(response.content)
            except requests.exceptions.RequestException as e:
                errors.append(f"Lighthouse: {type(e).__name__}: {str(e)}")

        # Fallback to public gateways (no auth needed)
        gateways = [
            "https://ipfs.io/ipfs/",
            "https://dweb.link/ipfs/",
        ]
        for gateway in gateways:
            try:
                url = f"{gateway}{cid}"
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                return bytes(response.content)
            except requests.exceptions.RequestException as e:
                errors.append(f"{gateway}: {type(e).__name__}: {str(e)}")

        error_msg = (
            f"All IPFS retrieval methods failed for CID: {cid}\nErrors:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )
        raise ConnectionError(error_msg)

    def retrieve_text(self, cid: str) -> str:
        """
        Retrieve text data from IPFS.

        Args:
            cid: IPFS content identifier

        Returns:
            Text content as string
        """
        data = self.retrieve(cid)
        return data.decode("utf-8")

    def retrieve_json(self, cid: str) -> Dict[str, Any]:
        """
        Retrieve JSON data from IPFS.

        Args:
            cid: IPFS content identifier

        Returns:
            Parsed JSON as dictionary
        """
        text = self.retrieve_text(cid)
        result: Dict[str, Any] = json.loads(text)
        return result

    def upload_haystack_document(
        self, document: Any, as_text: bool = True
    ) -> Dict[str, Any]:
        """
        Upload Haystack Document to IPFS.

        Args:
            document: Haystack Document object
            as_text: If True, upload content as .txt file (more readable).
                     If False, upload as JSON with full metadata.

        Returns:
            Upload response with CID
        """
        if as_text:
            # Upload just the content as readable .txt
            content = (
                document.content if hasattr(document, "content") else str(document)
            )
            doc_id = document.id if hasattr(document, "id") else "unknown"
            return self.upload_text(content, name=f"document_{doc_id}.txt")
        else:
            # Upload full document as JSON (includes metadata)
            doc_dict = {
                "content": (
                    document.content if hasattr(document, "content") else str(document)
                ),
                "meta": document.meta if hasattr(document, "meta") else {},
                "id": document.id if hasattr(document, "id") else None,
            }
            return self.upload_json(
                doc_dict, name=f"document_{doc_dict.get('id', 'unknown')}.json"
            )

    def upload_any(
        self, data: Any, name: Optional[str] = None, as_text: bool = True
    ) -> Dict[str, Any]:
        """
        Smart upload that handles any data type.

        Args:
            data: Any data (Document, dict, str, bytes, etc.)
            name: Optional name
            as_text: For Documents, upload as readable .txt (default True)

        Returns:
            Upload response with CID
        """
        # Handle Haystack Document - upload as .txt for readability
        if hasattr(data, "content"):
            return self.upload_haystack_document(data, as_text=as_text)

        # Handle bytes (PDFs, etc.) - avoid uploading large binaries
        if isinstance(data, bytes):
            # Skip large binary files (>5MB) to save IPFS space
            if len(data) > 5 * 1024 * 1024:
                return {"Hash": "skipped_large_binary", "Size": str(len(data))}
            return self.upload_buffer(data, name=name)

        # Handle dict/list (JSON)
        if isinstance(data, dict):
            return self.upload_json(data, name=name)
        if isinstance(data, list):
            return self.upload_json({"data": data}, name=name)

        # Handle string
        if isinstance(data, str):
            return self.upload_text(data, name=name)

        # Fallback: convert to string
        return self.upload_text(str(data), name=name)
