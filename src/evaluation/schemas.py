from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EvaluationSample:
    question: str
    ground_truth: str


@dataclass(slots=True)
class EvaluationSummary:
    total_samples: int
    faithfulness: float
    answer_relevancy: float
    context_recall: float
