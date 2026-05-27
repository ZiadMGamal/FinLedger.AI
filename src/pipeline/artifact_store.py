from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document

from src.core.logging import get_logger


class ChunkArtifactStore:
    def __init__(self) -> None:
        self.logger = get_logger("finledger.pipeline.artifact_store")

    def save(self, path: str, documents: list[Document]) -> None:
        try:
            target_path = Path(path)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with target_path.open("w", encoding="utf-8") as handle:
                for document in documents:
                    payload = {"page_content": document.page_content, "metadata": document.metadata}
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.logger.info("chunk_artifact_saved", extra={"path": path, "documents_count": len(documents)})
        except Exception as exc:
            self.logger.exception("chunk_artifact_save_failed", extra={"path": path, "error": str(exc)})
            raise

    def load(self, path: str) -> list[Document]:
        try:
            source_path = Path(path)
            if not source_path.exists():
                return []
            documents: list[Document] = []
            with source_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    documents.append(Document(page_content=row["page_content"], metadata=row.get("metadata", {})))
            self.logger.info("chunk_artifact_loaded", extra={"path": path, "documents_count": len(documents)})
            return documents
        except Exception as exc:
            self.logger.exception("chunk_artifact_load_failed", extra={"path": path, "error": str(exc)})
            raise
