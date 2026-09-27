"""Simple JSONL metrics logger."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any


class MetricsLogger:
    """Appends metric dictionaries as JSON lines to a file."""

    def __init__(self, log_path: str) -> None:
        """Create logger writing to *log_path* (a .jsonl file)."""
        self.log_path = log_path
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)

    def log(self, metrics: dict[str, Any] | None = None, **kwargs: Any) -> None:
        data = dict(metrics) if metrics else {}
        data.update(kwargs)
        entry = {"timestamp": datetime.now().isoformat(), **data}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
