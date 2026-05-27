from __future__ import annotations

import typer

from src.core.config import settings
from src.core.logging import configure_logging, get_logger
from src.evaluation.dataset_builder import EvaluationDatasetBuilder
from src.evaluation.ragas_evaluator import RagasEvaluationService

app = typer.Typer(help="FinLedger.AI Ragas evaluation runner")


@app.command()
def evaluate(
    input_path: str = typer.Option(default=settings.docfinqa_validation_path, help="Path to DocFinQA validation JSON"),
    sample_size: int = typer.Option(default=settings.evaluation_sample_size, help="Number of QA rows to evaluate"),
    output_path: str = typer.Option(default=settings.evaluation_output_path, help="Evaluation report output path"),
) -> None:
    configure_logging()
    logger = get_logger("finledger.evaluation.runner")
    builder = EvaluationDatasetBuilder()
    evaluator = RagasEvaluationService()
    samples = builder.build_samples(input_path=input_path, sample_size=sample_size)
    report = evaluator.run(samples=samples, output_path=output_path)
    logger.info("evaluation_report_created", extra={"output_path": output_path, "sample_size": sample_size})
    typer.echo(
        {
            "output_path": output_path,
            "total_samples": report["summary"]["total_samples"],
            "faithfulness": report["summary"]["faithfulness"],
            "answer_relevancy": report["summary"]["answer_relevancy"],
            "context_recall": report["summary"]["context_recall"],
        }
    )


if __name__ == "__main__":
    app()
