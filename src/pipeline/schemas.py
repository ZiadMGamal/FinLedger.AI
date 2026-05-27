from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document


@dataclass(slots=True)
class DocFinQARecord:
    record_id: str
    context: str
    question: str
    answer: str
    source_file: str
    raw_payload: dict[str, Any]


@dataclass(slots=True)
class ChunkingResult:
    parent_documents: list[Document]
    child_documents: list[Document]
