"""Simple secrets management for environment variables."""

import os
from typing import Optional


class Secrets:
    """Simple secrets manager for environment variables."""

    @staticmethod
    def get(key: str, default: Optional[str] = None) -> Optional[str]:
        """Get environment variable value."""
        return os.getenv(key, default)

    @staticmethod
    def require(key: str) -> str:
        """Get required environment variable, raise error if missing."""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable '{key}' is missing")
        return value

    @staticmethod
    def set(key: str, value: str) -> None:
        """Set environment variable."""
        os.environ[key] = value

    @staticmethod
    def load_from_file(filepath: str) -> None:
        """Load environment variables from a file (simple .env format)."""
        if not os.path.exists(filepath):
            return

        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key.strip()] = value.strip()

    # Common secrets for this project
    @staticmethod
    def neo4j_uri() -> Optional[str]:
        """Get Neo4j URI."""
        return Secrets.get("NEO4J_URI")

    @staticmethod
    def neo4j_username() -> Optional[str]:
        """Get Neo4j username."""
        return Secrets.get("NEO4J_USERNAME")

    @staticmethod
    def neo4j_password() -> Optional[str]:
        """Get Neo4j password."""
        return Secrets.get("NEO4J_PASSWORD")

    @staticmethod
    def neo4j_database() -> Optional[str]:
        """Get Neo4j database name."""
        return Secrets.get("NEO4J_DATABASE")

    @staticmethod
    def openai_api_key() -> Optional[str]:
        """Get OpenAI API key."""
        return Secrets.get("OPENAI_API_KEY")
