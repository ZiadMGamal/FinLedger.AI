from __future__ import annotations

from fastapi import FastAPI

from src.core.config import settings
from src.core.logging import configure_logging, get_logger

configure_logging()
logger = get_logger("finledger.api")

app = FastAPI(title=settings.project_name, version=settings.project_version)


@app.get("/health", tags=["system"])
async def health_check() -> dict[str, str]:
    logger.info("health_check_called")
    return {"status": "ok", "service": settings.project_name, "environment": settings.project_env}
