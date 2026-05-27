from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=3)


class QueryResponse(BaseModel):
    question: str
    answer: str
    mode: str
    sources: list[dict[str, Any]]


class EvaluateRequest(BaseModel):
    input_path: str
    sample_size: int = Field(default=50, ge=1)
    output_path: str


class EvaluateResponse(BaseModel):
    output_path: str
    total_samples: int
    faithfulness: float
    answer_relevancy: float
    context_recall: float
