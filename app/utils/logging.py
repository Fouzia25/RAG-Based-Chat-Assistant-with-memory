"""Structured (JSON) logging with secret redaction.

Usage::

    logger = get_logger(__name__)
    logger.info("chat_completed", extra={"session_id": sid, "sources": 3})

Anything passed via ``extra`` becomes a top-level field in the JSON line.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone

# Attributes every LogRecord has; anything else came from `extra=`.
_RESERVED = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}

# Matches OpenAI-style keys (sk-..., sk-proj-...) so they never reach log output.
_SECRET_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-]{10,}")


def redact(text: str) -> str:
    return _SECRET_PATTERN.sub("sk-***REDACTED***", text)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return redact(json.dumps(payload, default=str))


class RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def setup_logging(level: str = "INFO", json_format: bool = True) -> None:
    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(RedactingFormatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    # Quiet noisy libraries; httpx logs every OpenAI request URL at INFO.
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
