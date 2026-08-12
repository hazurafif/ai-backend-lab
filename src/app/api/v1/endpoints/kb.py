"""Knowledge base routes: per-user KB CRUD, file/folder upload, search.

Files are uploaded as multipart pairs (`file` + `path`), so a folder upload
from the browser (HTML5 `webkitdirectory`) becomes N pairs with relative
paths. Each file is ingested synchronously (parse -> chunk -> embed -> Weaviate)
and reported per-file in the response; failures leave the document with
`status=failed` and a readable `error`.

Auth: `get_current_user` (owner-only, unlike the admin-only /agent resources).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool

from ....core.config import settings
from ....core.database import persistence
from ....core.dependencies import get_current_user
from ....core.exceptions import Conflict, NotFound
from ....schema.kb_schema import (
    DocumentOut,
    KBIn,
    KBOut,
    KBUpdate,
    SearchOut,
    UploadResponse,
    UploadResult,
)
from ....services.kb.ingest import ingest_document
from ....services.kb.ingest import reindex_kb as reindex_kb_documents
from ....services.kb.vectorstore import KbUnavailableError, KbVectorStore, get_vector_store

router = APIRouter(prefix="/kb", tags=["knowledge base"])

_MAX_PATH_LEN = 500


def _vector_store_or_503() -> KbVectorStore:
    """The configured vector store, or a 503 with a helpful message."""
    store = get_vector_store()
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store is not configured (set WEAVIATE_URL and restart)",
        )
    return store


def _safe_path(raw: str | None) -> str | None:
    """Normalize a client-supplied relative path; None when unsafe/invalid."""
    if not raw:
        return None
    path = raw.replace("\\", "/").strip().lstrip("/")
    if not path or len(path) > _MAX_PATH_LEN:
        return None
    parts = [p for p in path.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        return None
    if any(ch in path for ch in "\x00\r\n\t"):
        return None
    return "/".join(parts)


def _check_extension(path: str) -> None:
    if Path(path).suffix.lower() not in settings.kb_allowed_extensions:
        allowed = ", ".join(settings.kb_allowed_extensions)
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type for '{path}'. Allowed: {allowed}",
        )


async def _get_owned_kb(kb_id: str, username: str) -> dict:
    kb = await persistence.kb.get_kb(username, kb_id)
    if kb is None:
        raise NotFound("Knowledge base not found")
    return kb


# ---------------------------------------------------------------------------
# knowledge bases
# ---------------------------------------------------------------------------


@router.post("", response_model=KBOut, status_code=201)
async def create_kb(body: KBIn, current_user: dict = Depends(get_current_user)):
    kb = await persistence.kb.create_kb(current_user["username"], body.name, body.description)
    if kb is None:
        raise Conflict(f"Knowledge base '{body.name}' already exists")
    return kb


@router.get("", response_model=list[KBOut])
async def list_kbs(current_user: dict = Depends(get_current_user)):
    return await persistence.kb.list_kbs(current_user["username"])


@router.get("/{kb_id}", response_model=KBOut)
async def get_kb(kb_id: str, current_user: dict = Depends(get_current_user)):
    return await _get_owned_kb(kb_id, current_user["username"])


@router.patch("/{kb_id}", response_model=KBOut)
async def update_kb(kb_id: str, body: KBUpdate, current_user: dict = Depends(get_current_user)):
    await _get_owned_kb(kb_id, current_user["username"])
    kb = await persistence.kb.update_kb(
        current_user["username"], kb_id, name=body.name, description=body.description
    )
    if kb is None:
        raise NotFound("Knowledge base not found")
    return kb


@router.delete("/{kb_id}", status_code=204)
async def delete_kb(kb_id: str, current_user: dict = Depends(get_current_user)):
    """Delete a KB: documents (cascade) + all its vectors."""
    await _get_owned_kb(kb_id, current_user["username"])
    store = get_vector_store()
    if store is not None:
        try:
            await run_in_threadpool(store.delete_kb, kb_id)
        except KbUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not await persistence.kb.delete_kb(current_user["username"], kb_id):
        raise NotFound("Knowledge base not found")
    return None


# ---------------------------------------------------------------------------
# documents / upload
# ---------------------------------------------------------------------------


@router.post("/{kb_id}/files", response_model=UploadResponse)
async def upload_files(
    kb_id: str,
    current_user: dict = Depends(get_current_user),
    files: list[UploadFile] = File(...),
    paths: list[str] = Form(default=[]),
):
    """Upload one or more files (with relative `path` per file for folders).

    Multipart form: repeat `file` parts, each optionally paired with a `path`
    form field of the same index (defaults to the file's basename). Files are
    validated (extension, size, safe path), stored as blobs, then ingested.
    """
    username = current_user["username"]
    await _get_owned_kb(kb_id, username)
    if not files:
        raise HTTPException(status_code=422, detail="No files provided")
    if len(files) > settings.kb_max_upload_batch:
        raise HTTPException(
            status_code=422,
            detail=f"Too many files (max {settings.kb_max_upload_batch} per request)",
        )
    if paths and len(paths) != len(files):
        raise HTTPException(status_code=422, detail="'paths' must match the number of files")

    max_bytes = settings.kb_max_file_size_mb * 1024 * 1024
    results: list[UploadResult] = []
    for index, upload in enumerate(files):
        raw_path = paths[index] if paths else (upload.filename or "")
        path = _safe_path(raw_path)
        if path is None:
            results.append(
                UploadResult(path=raw_path or "(unnamed)", ok=False, error="Invalid file path")
            )
            continue
        try:
            _check_extension(path)
        except HTTPException as exc:
            results.append(UploadResult(path=path, ok=False, error=exc.detail))
            continue
        data = await upload.read()
        if len(data) > max_bytes:
            results.append(
                UploadResult(
                    path=path,
                    ok=False,
                    error=f"File too large (max {settings.kb_max_file_size_mb} MB)",
                )
            )
            continue
        doc = await persistence.kb.add_document(
            username, kb_id, path, upload.content_type, len(data), data
        )
        if doc is None:
            results.append(
                UploadResult(path=path, ok=False, error="Duplicate path or knowledge base missing")
            )
            continue
        try:
            await ingest_document(doc, data)
            results.append(UploadResult(path=path, ok=True, doc_id=doc["id"]))
        except KbUnavailableError as exc:
            results.append(UploadResult(path=path, ok=False, error=str(exc)))
        except Exception as exc:
            results.append(UploadResult(path=path, ok=False, doc_id=doc["id"], error=str(exc)))
    return UploadResponse(kb_id=kb_id, results=results)


@router.get("/{kb_id}/files", response_model=list[DocumentOut])
async def list_documents(kb_id: str, current_user: dict = Depends(get_current_user)):
    await _get_owned_kb(kb_id, current_user["username"])
    return await persistence.kb.list_documents(current_user["username"], kb_id)


@router.get("/{kb_id}/files/{doc_id}", response_model=DocumentOut)
async def get_document(kb_id: str, doc_id: str, current_user: dict = Depends(get_current_user)):
    await _get_owned_kb(kb_id, current_user["username"])
    doc = await persistence.kb.get_document(current_user["username"], doc_id)
    if doc is None:
        raise NotFound("Document not found")
    return doc


@router.delete("/{kb_id}/files/{doc_id}", status_code=204)
async def delete_document(kb_id: str, doc_id: str, current_user: dict = Depends(get_current_user)):
    await _get_owned_kb(kb_id, current_user["username"])
    store = get_vector_store()
    if store is not None:
        try:
            await run_in_threadpool(store.delete_document, doc_id)
        except KbUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    if not await persistence.kb.delete_document(current_user["username"], doc_id):
        raise NotFound("Document not found")
    return None


@router.post("/{kb_id}/reindex")
async def reindex_kb(kb_id: str, current_user: dict = Depends(get_current_user)):
    """Re-parse + re-embed every document of the KB (e.g. new embedding model)."""
    await _get_owned_kb(kb_id, current_user["username"])
    try:
        result = await reindex_kb_documents(kb_id, current_user["username"])
    except KbUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"kb_id": kb_id, **result}


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@router.get("/{kb_id}/search", response_model=SearchOut)
async def search_kb(
    kb_id: str,
    current_user: dict = Depends(get_current_user),
    q: str = Query(min_length=1, max_length=500),
    limit: int = Query(default=5, ge=1, le=20),
):
    """Hybrid search (vector + keyword) within one knowledge base."""
    await _get_owned_kb(kb_id, current_user["username"])
    store = _vector_store_or_503()
    try:
        hits = await run_in_threadpool(
            store.search, q, owner=current_user["username"], kb_id=kb_id, limit=limit
        )
    except KbUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return SearchOut(query=q, hits=hits)
