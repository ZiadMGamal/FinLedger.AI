from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.core.logging import get_logger
from src.pipeline.base import BaseDocLoader
from src.pipeline.schemas import DocFinQARecord


class DocFinQALoader(BaseDocLoader):
    def __init__(self) -> None:
        self.logger = get_logger("finledger.pipeline.loader")

    def load(self, input_path: str) -> list[DocFinQARecord]:
        try:
            path = Path(input_path)
            if not path.exists():
                raise FileNotFoundError(f"Dataset path does not exist: {input_path}")

            with path.open("r", encoding="utf-8") as file_handle:
                payload: Any = json.load(file_handle)

            records_raw = self._extract_records(payload)
            records: list[DocFinQARecord] = []
            for index, item in enumerate(records_raw):
                context = str(item.get("Context", "")).strip()
                question = str(item.get("Question", "")).strip()
                answer = str(item.get("Answer", "")).strip()
                if not context or not question:
                    continue
                record = DocFinQARecord(
                    record_id=f"{path.stem}-{index}",
                    context=context,
                    question=question,
                    answer=answer,
                    source_file=str(path),
                    raw_payload=item,
                )
                records.append(record)

            self.logger.info("docfinqa_records_loaded", extra={"records_count": len(records), "input_path": input_path})
            return records
        except Exception as exc:
            self.logger.exception("docfinqa_load_failed", extra={"input_path": input_path, "error": str(exc)})
            raise

    def _extract_records(self, payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [entry for entry in payload if isinstance(entry, dict)]
        if isinstance(payload, dict):
            if "data" in payload and isinstance(payload["data"], list):
                return [entry for entry in payload["data"] if isinstance(entry, dict)]
            if "records" in payload and isinstance(payload["records"], list):
                return [entry for entry in payload["records"] if isinstance(entry, dict)]
            if all(key in payload for key in ("Context", "Question", "Answer")):
                return [payload]
        return []
