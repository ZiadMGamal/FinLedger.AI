from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.documents import Document


class BaseRetrieverEngine(ABC):
    @abstractmethod
    def retrieve(self, query: str) -> list[Document]:
        raise NotImplementedError


class BaseRAGEngine(ABC):
    @abstractmethod
    def answer(self, question: str) -> dict[str, Any]:
        raise NotImplementedError
