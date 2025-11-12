"""Configuration management for agentic-rag SDK.

This module provides a centralized configuration system that supports both:
1. Programmatic configuration (for SDK usage)
2. Environment variable fallback (for backward compatibility)

Usage as SDK:
    ```python
    from agentic_rag import Config, PipelineFactory

    # Initialize with explicit configuration
    config = Config(
        neo4j_uri="bolt://localhost:7687",
        neo4j_username="neo4j",
        neo4j_password="password",
        openrouter_api_key="your-key",
        lighthouse_api_key="your-key"
    )

    # Use with PipelineFactory
    factory = PipelineFactory(config=config, username="alice")
    ```

Environment Variables (fallback):
    - NEO4J_URI
    - NEO4J_USERNAME
    - NEO4J_PASSWORD
    - NEO4J_DATABASE
    - OPENROUTER_API_KEY
    - LIGHTHOUSE_API_KEY
    - AGENTIC_RAG_LOG_LEVEL
    - AGENTIC_RAG_LOG_FILE
"""

import os
from typing import Optional


class Config:
    """
    Central configuration for agentic-rag SDK.

    Attributes:
        neo4j_uri: Neo4j database URI
        neo4j_username: Neo4j username
        neo4j_password: Neo4j password
        neo4j_database: Neo4j database name
        openrouter_api_key: OpenRouter API key for LLM access
        lighthouse_api_key: Lighthouse IPFS API key
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_username: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        neo4j_database: Optional[str] = None,
        openrouter_api_key: Optional[str] = None,
        lighthouse_api_key: Optional[str] = None,
        log_level: Optional[str] = None,
        log_file: Optional[str] = None,
    ):
        """
        Initialize configuration.

        All parameters are optional and will fall back to environment variables
        if not provided.

        Args:
            neo4j_uri: Neo4j connection URI (defaults to NEO4J_URI env var)
            neo4j_username: Neo4j username (defaults to NEO4J_USERNAME env var)
            neo4j_password: Neo4j password (defaults to NEO4J_PASSWORD env var)
            neo4j_database: Neo4j database name (defaults to NEO4J_DATABASE env var)
            openrouter_api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            lighthouse_api_key: Lighthouse IPFS key (defaults to LIGHTHOUSE_API_KEY env var)
            log_level: Log level (defaults to AGENTIC_RAG_LOG_LEVEL env var or INFO)
            log_file: Log file path (defaults to AGENTIC_RAG_LOG_FILE env var)
        """
        # Neo4j configuration
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI")
        self.neo4j_username = neo4j_username or os.getenv("NEO4J_USERNAME")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD")
        self.neo4j_database = neo4j_database or os.getenv("NEO4J_DATABASE")

        # API keys
        self.openrouter_api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.lighthouse_api_key = lighthouse_api_key or os.getenv("LIGHTHOUSE_API_KEY")

        # Logging configuration
        self.log_level = log_level or os.getenv("AGENTIC_RAG_LOG_LEVEL") or "INFO"
        self.log_file = log_file or os.getenv("AGENTIC_RAG_LOG_FILE")

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a configuration value by key.

        Args:
            key: Configuration key (e.g., 'neo4j_uri', 'openrouter_api_key')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return getattr(self, key, default)

    def validate_neo4j(self) -> bool:
        """
        Check if Neo4j configuration is complete.

        Returns:
            True if all required Neo4j settings are present
        """
        return all(
            [
                self.neo4j_uri,
                self.neo4j_username,
                self.neo4j_password,
            ]
        )

    def validate_openrouter(self) -> bool:
        """
        Check if OpenRouter API key is configured.

        Returns:
            True if OpenRouter API key is present
        """
        return self.openrouter_api_key is not None

    def validate_lighthouse(self) -> bool:
        """
        Check if Lighthouse API key is configured.

        Returns:
            True if Lighthouse API key is present
        """
        return self.lighthouse_api_key is not None

    def to_dict(self) -> dict:
        """
        Export configuration as dictionary.

        Returns:
            Dictionary with all configuration values (passwords masked)
        """
        return {
            "neo4j_uri": self.neo4j_uri,
            "neo4j_username": self.neo4j_username,
            "neo4j_password": "***" if self.neo4j_password else None,
            "neo4j_database": self.neo4j_database,
            "openrouter_api_key": "***" if self.openrouter_api_key else None,
            "lighthouse_api_key": "***" if self.lighthouse_api_key else None,
            "log_level": self.log_level,
            "log_file": self.log_file,
        }

    def __repr__(self) -> str:
        """String representation with masked secrets."""
        return f"Config({self.to_dict()})"


# Global configuration instance (can be set once for the entire application)
_global_config: Optional[Config] = None


def set_global_config(config: Config) -> None:
    """
    Set the global configuration instance.

    This allows setting configuration once and having it used throughout
    the application without passing it to every component.

    Args:
        config: Configuration instance to use globally

    Example:
        ```python
        from agentic_rag import Config, set_global_config

        config = Config(
            neo4j_uri="bolt://localhost:7687",
            neo4j_username="neo4j",
            neo4j_password="password"
        )
        set_global_config(config)

        # Now all components will use this config by default
        ```
    """
    global _global_config
    _global_config = config


def get_global_config() -> Optional[Config]:
    """
    Get the global configuration instance.

    Returns:
        Global Config instance or None if not set
    """
    return _global_config


def get_config(config: Optional[Config] = None) -> Config:
    """
    Get configuration instance with fallback logic.

    Priority:
    1. Explicitly passed config
    2. Global config (if set via set_global_config)
    3. New Config() with environment variable fallback

    Args:
        config: Optional explicit configuration

    Returns:
        Configuration instance to use
    """
    if config is not None:
        return config

    if _global_config is not None:
        return _global_config

    # Fallback: create new config from environment variables
    return Config()
