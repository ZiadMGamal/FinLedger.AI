from __future__ import annotations

import random

from src.core.logging import get_logger
from src.evaluation.schemas import EvaluationSample
from src.pipeline.docfinqa_loader import DocFinQALoader


class EvaluationDatasetBuilder:
    def __init__(self) -> None:
        self.logger = get_logger("finledger.evaluation.dataset_builder")
        self.loader = DocFinQALoader()

    def build_samples(self, input_path: str, sample_size: int, seed: int = 42) -> list[EvaluationSample]:
        try:
            records = self.loader.load(input_path=input_path)
            filtered = [record for record in records if record.answer.strip()]
            if not filtered:
                raise ValueError("No evaluation records found with non-empty answers.")
            random.seed(seed)
            random.shuffle(filtered)
            selected = filtered[: min(sample_size, len(filtered))]
            samples = [EvaluationSample(question=item.question, ground_truth=item.answer) for item in selected]
            self.logger.info(
                "evaluation_samples_built",
                extra={"input_path": input_path, "requested_sample_size": sample_size, "actual_sample_size": len(samples)},
            )
            return samples
        except Exception as exc:
            self.logger.exception("evaluation_samples_build_failed", extra={"input_path": input_path, "error": str(exc)})
            raise
