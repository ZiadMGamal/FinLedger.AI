from __future__ import annotations

import typer

from src.core.logging import configure_logging
from src.retrieval.rag_service import FinancialRAGService

app = typer.Typer(help="FinLedger.AI RAG query runner")


@app.command()
def query(question: str = typer.Option(..., help="Financial question to answer")) -> None:
    configure_logging()
    service = FinancialRAGService()
    result = service.answer(question=question)
    typer.echo(result)


if __name__ == "__main__":
    app()
