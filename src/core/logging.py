from __future__ import annotations

import logging
import sys
from typing import Any

from pythonjsonlogger import jsonlogger

from src.core.config import settings


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(settings.log_level.upper())


def get_logger(name: str, extra: dict[str, Any] | None = None) -> logging.LoggerAdapter:
    logger = logging.getLogger(name)
    payload = extra if extra is not None else {}
    return logging.LoggerAdapter(logger, payload)
