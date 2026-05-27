from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_recall, faithfulness

from src.core.config import settings
from src.core.logging import get_logger
from src.evaluation.schemas import EvaluationSample, EvaluationSummary
from src.retrieval.rag_service import FinancialRAGService


class RagasEvaluationService:
    def __init__(self) -> None:
        self.logger = get_logger("finledger.evaluation.ragas_service")
        self.rag_service = FinancialRAGService()

    def run(self, samples: list[EvaluationSample], output_path: str) -> dict[str, Any]:
        try:
            records: list[dict[str, Any]] = []
            for sample in samples:
                rag_result = self.rag_service.answer(sample.question)
                contexts = [str(source.get("content", "")) for source in rag_result["sources"] if str(source.get("content", "")).strip()]
                records.append(
                    {
                        "question": sample.question,
                        "answer": rag_result["answer"],
                        "contexts": contexts,
                        "ground_truth": sample.ground_truth,
                        "sources": rag_result["sources"],
                    }
                )

            dataset = Dataset.from_list(
                [
                    {
                        "question": item["question"],
                        "answer": item["answer"],
                        "contexts": item["contexts"],
                        "ground_truth": item["ground_truth"],
                    }
                    for item in records
                ]
            )
            score = evaluate(dataset=dataset, metrics=[faithfulness, answer_relevancy, context_recall])
            score_dict = score.to_pandas().mean(numeric_only=True).to_dict()
            summary = EvaluationSummary(
                total_samples=len(records),
                faithfulness=float(score_dict.get("faithfulness", 0.0)),
                answer_relevancy=float(score_dict.get("answer_relevancy", 0.0)),
                context_recall=float(score_dict.get("context_recall", 0.0)),
            )
            payload = {
                "project_name": settings.project_name,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "retrieval_mode": settings.retrieval_mode,
                "vector_backend": settings.vector_backend,
                "summary": {
                    "total_samples": summary.total_samples,
                    "faithfulness": summary.faithfulness,
                    "answer_relevancy": summary.answer_relevancy,
                    "context_recall": summary.context_recall,
                },
                "rows": records,
            }
            self._write_report(path=output_path, payload=payload)
            self.logger.info("ragas_evaluation_completed", extra={"output_path": output_path, "samples_count": len(records)})
            return payload
        except Exception as exc:
            self.logger.exception("ragas_evaluation_failed", extra={"error": str(exc)})
            raise

    def _write_report(self, path: str, payload: dict[str, Any]) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
