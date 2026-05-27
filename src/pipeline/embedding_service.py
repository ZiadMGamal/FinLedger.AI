from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from src.core.config import settings
from src.core.logging import get_logger


class EmbeddingService:
    def __init__(self) -> None:
        self.logger = get_logger("finledger.pipeline.embeddings")
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required to create embeddings.")
        self.embeddings = OpenAIEmbeddings(model=settings.openai_embed_model, api_key=settings.openai_api_key)
        self.logger.info("embedding_service_initialized", extra={"embedding_model": settings.openai_embed_model})

    def get_client(self) -> OpenAIEmbeddings:
        return self.embeddings
