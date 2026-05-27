from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS, Qdrant
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.core.config import settings
from src.core.logging import get_logger
from src.pipeline.base import BaseVectorWriter


class QdrantVectorWriter(BaseVectorWriter):
    def __init__(self, embeddings: OpenAIEmbeddings) -> None:
        self.logger = get_logger("finledger.retrieval.qdrant_writer")
        self.embeddings = embeddings

    def upsert_documents(self, documents: list[Document]) -> None:
        try:
            if not documents:
                raise ValueError("No documents provided for Qdrant upsert.")
            Qdrant.from_documents(
                documents=documents,
                embedding=self.embeddings,
                location=None,
                host=settings.qdrant_host,
                port=settings.qdrant_port,
                prefer_grpc=False,
                https=settings.qdrant_use_https,
                api_key=settings.qdrant_api_key if settings.qdrant_api_key else None,
                collection_name=settings.qdrant_collection,
                force_recreate=True,
            )
            self.logger.info(
                "qdrant_upsert_completed",
                extra={
                    "collection_name": settings.qdrant_collection,
                    "documents_count": len(documents),
                    "qdrant_host": settings.qdrant_host,
                    "qdrant_port": settings.qdrant_port,
                },
            )
        except Exception as exc:
            self.logger.exception("qdrant_upsert_failed", extra={"error": str(exc)})
            raise


class FaissVectorWriter(BaseVectorWriter):
    def __init__(self, embeddings: OpenAIEmbeddings) -> None:
        self.logger = get_logger("finledger.retrieval.faiss_writer")
        self.embeddings = embeddings

    def upsert_documents(self, documents: list[Document]) -> None:
        try:
            if not documents:
                raise ValueError("No documents provided for FAISS upsert.")
            vector_store = FAISS.from_documents(documents=documents, embedding=self.embeddings)
            index_path = Path(settings.faiss_index_path)
            index_path.parent.mkdir(parents=True, exist_ok=True)
            vector_store.save_local(folder_path=str(index_path.parent), index_name=index_path.stem)
            self.logger.info(
                "faiss_upsert_completed",
                extra={"documents_count": len(documents), "index_path": str(index_path)},
            )
        except Exception as exc:
            self.logger.exception("faiss_upsert_failed", extra={"error": str(exc)})
            raise
