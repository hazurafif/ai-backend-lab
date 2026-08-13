"""Knowledge base API models (Pydantic v2)."""

from __future__ import annotations

from pydantic import BaseModel, Field

# Slug-ish, permissive: letters, digits, spaces, dots, dashes, underscores.
KB_NAME_RE = r"^[A-Za-z0-9][A-Za-z0-9 _.-]{0,63}$"


class KBIn(BaseModel):
    name: str = Field(min_length=1, max_length=64, pattern=KB_NAME_RE)
    description: str | None = Field(default=None, max_length=500)


class KBUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=64, pattern=KB_NAME_RE)
    description: str | None = Field(default=None, max_length=500)


class KBOut(BaseModel):
    id: str
    name: str
    description: str | None = None
    created_at: str
    updated_at: str
    document_count: int = 0
    chunk_count: int = 0


class DocumentOut(BaseModel):
    id: str
    kb_id: str
    path: str
    mime_type: str | None = None
    size_bytes: int
    status: str
    error: str | None = None
    chunk_count: int = 0
    created_at: str
    updated_at: str


class UploadResult(BaseModel):
    path: str
    ok: bool
    doc_id: str | None = None
    error: str | None = None


class UploadResponse(BaseModel):
    kb_id: str
    results: list[UploadResult]


class SearchHit(BaseModel):
    path: str
    content: str
    score: float
    chunk_index: int
    doc_id: str


class SearchOut(BaseModel):
    query: str
    hits: list[SearchHit]
