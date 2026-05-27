from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.documents import Document

from src.pipeline.schemas import ChunkingResult, DocFinQARecord


class BaseDocLoader(ABC):
    @abstractmethod
    def load(self, input_path: str) -> list[DocFinQARecord]:
        raise NotImplementedError


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, records: list[DocFinQARecord]) -> ChunkingResult:
        raise NotImplementedError


class BaseVectorWriter(ABC):
    @abstractmethod
    def upsert_documents(self, documents: list[Document]) -> None:
        raise NotImplementedError
