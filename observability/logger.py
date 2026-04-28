"""
NexusMind v2 — Observability / Logging
Structured JSON logging for requests, errors, and agent events.
"""

import json
import logging
import time
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


class JSONFormatter(logging.Formatter):
    """Emit log records as newline-delimited JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "ts":      self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
        }
        if record.exc_info:
            log_obj["exc"] = self.formatException(record.exc_info)
        return json.dumps(log_obj)


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:
        return  # already configured

    root.setLevel(level)

    # Console handler (human-readable)
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s"))
    root.addHandler(console)

    # Rotating file handler (JSON)
    file_handler = RotatingFileHandler(
        LOG_DIR / "nexusmind.jsonl",
        maxBytes=10 * 1024 * 1024,   # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(JSONFormatter())
    root.addHandler(file_handler)


def get_request_logger() -> logging.Logger:
    return logging.getLogger("nexusmind.requests")


def get_agent_logger() -> logging.Logger:
    return logging.getLogger("nexusmind.agent")
