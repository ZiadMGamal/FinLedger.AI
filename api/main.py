from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException

from api.schemas import EvaluateRequest, EvaluateResponse, QueryRequest, QueryResponse
from src.core.config import settings
from src.core.logging import configure_logging, get_logger
from src.evaluation.dataset_builder import EvaluationDatasetBuilder
from src.evaluation.ragas_evaluator import RagasEvaluationService
from src.retrieval.rag_service import FinancialRAGService

configure_logging()
logger = get_logger("finledger.api")

app = FastAPI(title=settings.project_name, version=settings.project_version)


@lru_cache(maxsize=1)
def get_rag_service() -> FinancialRAGService:
    return FinancialRAGService()


@lru_cache(maxsize=1)
def get_dataset_builder() -> EvaluationDatasetBuilder:
    return EvaluationDatasetBuilder()


@lru_cache(maxsize=1)
def get_evaluation_service() -> RagasEvaluationService:
    return RagasEvaluationService()


@app.get("/health", tags=["system"])
async def health_check() -> dict[str, str]:
    logger.info("health_check_called")
    return {"status": "ok", "service": settings.project_name, "environment": settings.project_env}


@app.post("/query", response_model=QueryResponse, tags=["rag"])
async def query(payload: QueryRequest) -> QueryResponse:
    try:
        result = get_rag_service().answer(question=payload.question)
        return QueryResponse(**result)
    except Exception as exc:
        logger.exception("query_endpoint_failed", extra={"error": str(exc)})
        raise HTTPException(status_code=500, detail="Query processing failed.") from exc


@app.post("/evaluate", response_model=EvaluateResponse, tags=["evaluation"])
async def evaluate(payload: EvaluateRequest) -> EvaluateResponse:
    try:
        samples = get_dataset_builder().build_samples(input_path=payload.input_path, sample_size=payload.sample_size)
        report = get_evaluation_service().run(samples=samples, output_path=payload.output_path)
        summary = report["summary"]
        return EvaluateResponse(
            output_path=payload.output_path,
            total_samples=int(summary["total_samples"]),
            faithfulness=float(summary["faithfulness"]),
            answer_relevancy=float(summary["answer_relevancy"]),
            context_recall=float(summary["context_recall"]),
        )
    except Exception as exc:
        logger.exception("evaluate_endpoint_failed", extra={"error": str(exc), "input_path": payload.input_path})
        raise HTTPException(status_code=500, detail="Evaluation processing failed.") from exc
