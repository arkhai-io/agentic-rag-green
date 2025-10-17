"""Metrics collection for pipeline and component performance tracking."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class MetricsCollector:
    """
    Collects and stores metrics for pipelines and components.

    Stores metrics as JSONL (JSON Lines) for easy appending and analysis.

    Directory structure:
    .logs/metrics/{username}/
        ├── components.jsonl  # Per-component timing
        └── pipelines.jsonl   # Per-pipeline timing
    """

    def __init__(self, username: str):
        """
        Initialize metrics collector for a user.

        Args:
            username: Username for metrics isolation
        """
        self.username = username

        # Get project root and create metrics directory
        project_root = Path(__file__).parent.parent.parent
        self.metrics_dir = project_root / ".logs" / "metrics" / username
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.components_file = self.metrics_dir / "components.jsonl"
        self.pipelines_file = self.metrics_dir / "pipelines.jsonl"

    def log_component_execution(
        self,
        component_name: str,
        component_id: str,
        start_time: float,
        end_time: float,
        input_count: int,
        output_count: int,
        cache_hits: int = 0,
        cache_misses: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log component execution metrics.

        Args:
            component_name: Human-readable component name
            component_id: Unique component ID
            start_time: Start timestamp (from time.time())
            end_time: End timestamp (from time.time())
            input_count: Number of inputs processed
            output_count: Number of outputs produced
            cache_hits: Number of cache hits
            cache_misses: Number of cache misses
            metadata: Additional metadata (config hash, etc.)
        """
        duration_ms = (end_time - start_time) * 1000

        metric = {
            "timestamp": datetime.fromtimestamp(start_time).isoformat(),
            "username": self.username,
            "component_name": component_name,
            "component_id": component_id,
            "duration_ms": round(duration_ms, 2),
            "input_count": input_count,
            "output_count": output_count,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_rate": (
                round(cache_hits / (cache_hits + cache_misses), 2)
                if (cache_hits + cache_misses) > 0
                else 0.0
            ),
            "metadata": metadata or {},
        }

        self._append_jsonl(self.components_file, metric)

    def log_pipeline_execution(
        self,
        pipeline_name: str,
        start_time: float,
        end_time: float,
        total_components: int,
        success: bool,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log pipeline execution metrics.

        Args:
            pipeline_name: Pipeline identifier
            start_time: Start timestamp (from time.time())
            end_time: End timestamp (from time.time())
            total_components: Number of components executed
            success: Whether execution succeeded
            error: Error message if failed
            metadata: Additional metadata (input files, output counts, etc.)
        """
        duration_ms = (end_time - start_time) * 1000

        metric = {
            "timestamp": datetime.fromtimestamp(start_time).isoformat(),
            "username": self.username,
            "pipeline_name": pipeline_name,
            "duration_ms": round(duration_ms, 2),
            "duration_seconds": round(duration_ms / 1000, 2),
            "total_components": total_components,
            "success": success,
            "error": error,
            "metadata": metadata or {},
        }

        self._append_jsonl(self.pipelines_file, metric)

    def _append_jsonl(self, file_path: Path, data: Dict[str, Any]) -> None:
        """Append a JSON object as a line to a JSONL file."""
        with open(file_path, "a") as f:
            f.write(json.dumps(data) + "\n")

    @staticmethod
    def start_timer() -> float:
        """Start a timer and return the start time."""
        return time.time()

    @staticmethod
    def get_duration_ms(start_time: float) -> float:
        """Get duration in milliseconds from start time."""
        return (time.time() - start_time) * 1000


# Convenience context manager for timing
class TimedExecution:
    """
    Context manager for timing code execution.

    Example:
        >>> with TimedExecution() as timer:
        ...     # do work
        ...     pass
        >>> print(f"Took {timer.duration_ms}ms")
    """

    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.duration_ms: float = 0.0

    def __enter__(self) -> "TimedExecution":
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end_time = time.time()
        if self.start_time:
            self.duration_ms = (self.end_time - self.start_time) * 1000
        return None
