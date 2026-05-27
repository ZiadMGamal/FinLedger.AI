from __future__ import annotations

import re
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import settings
from src.core.logging import get_logger
from src.pipeline.base import BaseChunker
from src.pipeline.schemas import ChunkingResult, DocFinQARecord


class TableAwareDocFinQAChunker(BaseChunker):
    def __init__(self, parent_chunk_size: int | None = None, child_chunk_size: int | None = None, overlap: int | None = None) -> None:
        self.logger = get_logger("finledger.pipeline.chunker")
        self.parent_chunk_size = parent_chunk_size or settings.chunk_parent_size
        self.child_chunk_size = child_chunk_size or settings.chunk_child_size
        self.overlap = overlap or settings.chunk_overlap
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size,
            chunk_overlap=self.overlap,
            separators=["\n\n", "\n", "\t", " ", ""],
        )

    def chunk(self, records: list[DocFinQARecord]) -> ChunkingResult:
        try:
            parent_docs: list[Document] = []
            child_docs: list[Document] = []
            for record in records:
                record_parents = self._build_parent_documents(record)
                parent_docs.extend(record_parents)
                for parent_doc in record_parents:
                    split_children = self.child_splitter.split_documents([parent_doc])
                    for child_index, child in enumerate(split_children):
                        child.metadata["child_chunk_index"] = child_index
                        child.metadata["chunk_level"] = "child"
                    child_docs.extend(split_children)

            self.logger.info(
                "docfinqa_chunking_completed",
                extra={
                    "records_count": len(records),
                    "parent_docs_count": len(parent_docs),
                    "child_docs_count": len(child_docs),
                    "parent_chunk_size": self.parent_chunk_size,
                    "child_chunk_size": self.child_chunk_size,
                },
            )
            return ChunkingResult(parent_documents=parent_docs, child_documents=child_docs)
        except Exception as exc:
            self.logger.exception("docfinqa_chunking_failed", extra={"error": str(exc)})
            raise

    def _build_parent_documents(self, record: DocFinQARecord) -> list[Document]:
        blocks = self._segment_context_to_blocks(record.context)
        chunks = self._pack_blocks_into_chunks(blocks)
        documents: list[Document] = []
        for chunk_index, chunk_text in enumerate(chunks):
            metadata = {
                "record_id": record.record_id,
                "question": record.question,
                "answer": record.answer,
                "source_file": record.source_file,
                "parent_chunk_index": chunk_index,
                "chunk_level": "parent",
            }
            documents.append(Document(page_content=chunk_text, metadata=metadata))
        return documents

    def _segment_context_to_blocks(self, context: str) -> list[str]:
        lines = context.splitlines()
        blocks: list[str] = []
        current: list[str] = []
        current_mode = "text"
        for line in lines:
            stripped = line.strip()
            line_mode = self._line_mode(line)
            if stripped == "":
                if current:
                    blocks.append("\n".join(current).strip())
                    current = []
                current_mode = "text"
                continue
            if not current:
                current_mode = line_mode
                current.append(line)
                continue
            if line_mode == current_mode:
                current.append(line)
                continue
            blocks.append("\n".join(current).strip())
            current = [line]
            current_mode = line_mode
        if current:
            blocks.append("\n".join(current).strip())
        return [block for block in blocks if block]

    def _line_mode(self, line: str) -> str:
        if "\t" in line:
            return "table"
        if re.match(r"^\s*\|.*\|\s*$", line):
            return "table"
        if re.match(r"^\s*[-:| ]+\s*$", line) and "|" in line:
            return "table"
        if line.lstrip().startswith("#"):
            return "heading"
        return "text"

    def _pack_blocks_into_chunks(self, blocks: Iterable[str]) -> list[str]:
        chunks: list[str] = []
        current = ""
        for block in blocks:
            if not current:
                if len(block) <= self.parent_chunk_size:
                    current = block
                    continue
                chunks.extend(self._split_oversized_block(block))
                continue
            candidate = f"{current}\n\n{block}"
            if len(candidate) <= self.parent_chunk_size:
                current = candidate
                continue
            chunks.append(current)
            if len(block) <= self.parent_chunk_size:
                current = block
                continue
            chunks.extend(self._split_oversized_block(block))
            current = ""
        if current:
            chunks.append(current)
        return chunks

    def _split_oversized_block(self, block: str) -> list[str]:
        if "\t" in block or "|" in block:
            return [block]
        split_docs = self.child_splitter.split_text(block)
        return [segment for segment in split_docs if segment.strip()]
