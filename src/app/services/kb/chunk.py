"""Chunking: markdown-aware for .md, recursive character splitter otherwise."""

from __future__ import annotations

from pathlib import Path

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

_HEADERS_TO_SPLIT_ON = [("#", "H1"), ("##", "H2"), ("###", "H3")]


def _split_markdown(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split markdown on headers, prefixing each section with its header path,
    then split oversized sections with the recursive splitter."""
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_HEADERS_TO_SPLIT_ON)
    sections = splitter.split_text(text)
    chunks: list[str] = []
    for section in sections:
        headers = list(section.metadata.values())
        prefix = " / ".join(headers) if headers else ""
        body = section.page_content.strip()
        if not body:
            continue
        content = f"{prefix}\n\n{body}" if prefix else body
        if len(content) <= chunk_size:
            chunks.append(content)
            continue
        for piece in RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        ).split_text(body):
            chunks.append(f"{prefix}\n\n{piece}" if prefix else piece)
    return chunks


def split_text(path: str, text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split extracted text into chunks for embedding."""
    text = text.strip()
    if not text:
        return []
    if Path(path).suffix.lower() == ".md":
        chunks = _split_markdown(text, chunk_size, chunk_overlap)
        if chunks:
            return chunks
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    ).split_text(text)
