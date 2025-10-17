"""Simple logging configuration for agentic-rag with per-user log files.

Usage in your modules:
    from agentic_rag.utils.logger import get_logger

    logger = get_logger(__name__, username="alice")
    logger.info("Processing started")
    logger.error("Something went wrong", exc_info=True)
"""

import logging
import os
from pathlib import Path
from typing import Optional


def get_logger(
    name: str, username: Optional[str] = None, level: Optional[str] = None
) -> logging.Logger:
    """
    Get a configured logger instance with per-user, per-module log files.

    Creates separate log files for each module category:
    - ./.logs/users/{username}/factory.log
    - ./.logs/users/{username}/runner.log
    - ./.logs/users/{username}/storage.log
    - ./.logs/users/{username}/components.log
    - ./.logs/users/{username}/gates.log

    Args:
        name: Logger name (typically __name__ from calling module)
        username: Username for per-user log file
        level: Log level override (DEBUG, INFO, WARNING, ERROR, CRITICAL)
               Defaults to AGENTIC_RAG_LOG_LEVEL env var or INFO

    Returns:
        Configured logger instance

    Examples:
        >>> # Console only
        >>> logger = get_logger(__name__)
        >>> logger.info("General system message")
        >>>
        >>> # With per-user, per-module file logging
        >>> logger = get_logger(__name__, username="alice")
        >>> logger.info("User alice created pipeline")
    """
    # Create logger name with username suffix if provided
    logger_name = f"{name}.{username}" if username else name
    logger = logging.getLogger(logger_name)

    # Only configure if not already configured
    if not logger.handlers:
        # Get log level from args, env var, or default to INFO
        log_level_str: str = level or os.getenv("AGENTIC_RAG_LOG_LEVEL") or "INFO"
        logger.setLevel(getattr(logging, log_level_str.upper()))

        # Console handler with simple format
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logger.level)

        # Format: timestamp - logger_name - level - message
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Per-user, per-module file logging
        if username:
            # Use ./.logs hidden directory relative to project root
            project_root = Path(__file__).parent.parent.parent
            user_logs_dir = project_root / ".logs" / "users" / username
            user_logs_dir.mkdir(parents=True, exist_ok=True)

            # Determine which log file based on module name
            log_filename = _get_log_filename_for_module(name)
            user_log_file = user_logs_dir / log_filename

            file_handler = logging.FileHandler(user_log_file)
            file_handler.setLevel(logger.level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Optional: General log file if env var is set
        log_file = os.getenv("AGENTIC_RAG_LOG_FILE")
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            general_handler = logging.FileHandler(log_file)
            general_handler.setLevel(logger.level)
            general_handler.setFormatter(formatter)
            logger.addHandler(general_handler)

    return logger


def _get_log_filename_for_module(module_name: str) -> str:
    """
    Determine the appropriate log filename based on module name.

    Args:
        module_name: Python module name (e.g., 'agentic_rag.pipeline.factory')

    Returns:
        Log filename (e.g., 'factory.log')
    """
    # Extract the key part of the module name
    if "factory" in module_name:
        return "factory.log"
    elif "runner" in module_name:
        return "runner.log"
    elif "storage" in module_name:
        return "storage.log"
    elif "gates" in module_name or "ingate" in module_name or "outgate" in module_name:
        return "gates.log"
    elif (
        "components" in module_name
        or "chunker" in module_name
        or "converter" in module_name
    ):
        return "components.log"
    else:
        # Default for unknown modules
        return "general.log"


def get_system_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a system-wide logger (logs to ./.logs/system.log).

    Use this for system-level operations not tied to a specific user.

    Args:
        name: Logger name (typically __name__ from calling module)
        level: Log level override

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(f"system.{name}")

    if not logger.handlers:
        log_level_str: str = level or os.getenv("AGENTIC_RAG_LOG_LEVEL") or "INFO"
        logger.setLevel(getattr(logging, log_level_str.upper()))

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logger.level)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # System log file
        project_root = Path(__file__).parent.parent.parent
        system_log = project_root / ".logs" / "system.log"
        system_log.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(system_log)
        file_handler.setLevel(logger.level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def configure_haystack_logging(
    username: Optional[str] = None, level: str = "DEBUG"
) -> None:
    """
    Configure Haystack's built-in logging to use our log files.

    This redirects all Haystack component logs (PDF converters, embedders, etc.)
    to either per-user haystack.log or system logs.

    Args:
        username: Username for per-user logging. If None, logs to system.log
        level: Log level for Haystack components (DEBUG, INFO, WARNING, ERROR)

    Example:
        >>> from agentic_rag.utils.logger import configure_haystack_logging
        >>>
        >>> # Per-user Haystack logging
        >>> configure_haystack_logging(username="alice", level="DEBUG")
        >>>
        >>> # System-wide Haystack logging
        >>> configure_haystack_logging(level="INFO")
    """
    # Get the root Haystack logger
    haystack_logger = logging.getLogger("haystack")
    haystack_logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers to avoid duplicates
    haystack_logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(haystack_logger.level)
    console_handler.setFormatter(formatter)
    haystack_logger.addHandler(console_handler)

    # File handler - per-user or system
    project_root = Path(__file__).parent.parent.parent

    if username:
        # Per-user haystack log file in user's directory
        logs_dir = project_root / ".logs" / "users" / username
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "haystack.log"
    else:
        # System log file
        logs_dir = project_root / ".logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "system.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(haystack_logger.level)
    file_handler.setFormatter(formatter)
    haystack_logger.addHandler(file_handler)

    # Don't propagate to root logger (avoid duplicate logs)
    haystack_logger.propagate = False
