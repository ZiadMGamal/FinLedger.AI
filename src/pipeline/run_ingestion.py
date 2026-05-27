from __future__ import annotations

import typer

from src.core.config import settings
from src.core.logging import configure_logging, get_logger
from src.pipeline.ingestion_service import DocFinQAIngestionService

app = typer.Typer(help="FinLedger.AI DocFinQA ingestion pipeline runner")


@app.command()
def ingest(
    input_path: str = typer.Option(default=settings.docfinqa_train_path, help="Path to DocFinQA JSON file"),
) -> None:
    configure_logging()
    logger = get_logger("finledger.pipeline.ingestion_runner")
    service = DocFinQAIngestionService()
    result = service.ingest(input_path=input_path)
    logger.info("ingestion_result", extra=result)
    typer.echo(result)


if __name__ == "__main__":
    app()
