from __future__ import annotations

from src.core.config import settings
from src.core.logging import get_logger
from src.pipeline.artifact_store import ChunkArtifactStore
from src.pipeline.docfinqa_loader import DocFinQALoader
from src.pipeline.embedding_service import EmbeddingService
from src.pipeline.table_aware_chunker import TableAwareDocFinQAChunker
from src.retrieval.vector_store import FaissVectorWriter, QdrantVectorWriter


class DocFinQAIngestionService:
    def __init__(self) -> None:
        self.logger = get_logger("finledger.pipeline.ingestion_service")
        self.loader = DocFinQALoader()
        self.chunker = TableAwareDocFinQAChunker()
        self.embedding_service = EmbeddingService()
        self.artifact_store = ChunkArtifactStore()

    def ingest(self, input_path: str) -> dict[str, int | str]:
        try:
            records = self.loader.load(input_path=input_path)
            chunking = self.chunker.chunk(records=records)
            self.artifact_store.save(path=settings.parent_chunks_path, documents=chunking.parent_documents)
            self.artifact_store.save(path=settings.child_chunks_path, documents=chunking.child_documents)

            writer = self._select_writer()
            writer.upsert_documents(chunking.child_documents)

            result = {
                "input_path": input_path,
                "vector_backend": settings.vector_backend,
                "records_count": len(records),
                "parent_chunks_count": len(chunking.parent_documents),
                "child_chunks_count": len(chunking.child_documents),
            }
            self.logger.info("docfinqa_ingestion_completed", extra=result)
            return result
        except Exception as exc:
            self.logger.exception("docfinqa_ingestion_failed", extra={"input_path": input_path, "error": str(exc)})
            raise

    def _select_writer(self) -> QdrantVectorWriter | FaissVectorWriter:
        embeddings = self.embedding_service.get_client()
        if settings.vector_backend == "qdrant":
            return QdrantVectorWriter(embeddings=embeddings)
        return FaissVectorWriter(embeddings=embeddings)
