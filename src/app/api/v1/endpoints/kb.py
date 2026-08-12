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
from ....services.kb.paths import safe_path
from ....services.kb.vectorstore import KbUnavailableError, KbVectorStore, get_vector_store
from ....services.kb.zip_upload import ZipValidationError, extract_zip_entries

router = APIRouter(prefix="/kb", tags=["knowledge base"])


def _vector_store_or_503() -> KbVectorStore:
    """The configured vector store, or a 503 with a helpful message."""
    store = get_vector_store()
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="Vector store is not configured (set WEAVIATE_URL and restart)",
        )
    return store


def _check_extension(path: str) -> None:
    if Path(path).suffix.lower() not in settings.kb_allowed_extensions:
        allowed = ", ".join(settings.kb_allowed_extensions)
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type for '{path}'. Allowed: {allowed}",
        )


async def _quota_remaining(username: str) -> int:
    """Bytes of storage quota still available for `username`."""
    used = await persistence.kb.total_bytes(username)
    return settings.kb_quota_mb * 1024 * 1024 - used


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


@router.get("/search", response_model=SearchOut)
async def search_all_kbs(
    current_user: dict = Depends(get_current_user),
    q: str = Query(min_length=1, max_length=500),
    limit: int = Query(default=5, ge=1, le=20),
    alpha: float | None = Query(default=None, ge=0.0, le=1.0),
):
    """Hybrid search across all knowledge bases of the current user.

    Declared before `/{kb_id}` so the literal `search` segment wins.
    `alpha` overrides KB_HYBRID_ALPHA per request (0 = keyword, 1 = vectors).
    """
    store = _vector_store_or_503()
    effective_alpha = alpha if alpha is not None else settings.kb_hybrid_alpha
    try:
        hits = await run_in_threadpool(
            store.search,
            q,
            owner=current_user["username"],
            limit=limit,
            alpha=effective_alpha,
        )
    except KbUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return SearchOut(query=q, hits=hits)


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
    remaining = await _quota_remaining(username)
    results: list[UploadResult] = []
    for index, upload in enumerate(files):
        raw_path = paths[index] if paths else (upload.filename or "")
        path = safe_path(raw_path)
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
        if len(data) > remaining:
            results.append(UploadResult(path=path, ok=False, error="Storage quota exceeded"))
            continue
        doc = await persistence.kb.add_document(
            username, kb_id, path, upload.content_type, len(data), data
        )
        if doc is None:
            results.append(
                UploadResult(path=path, ok=False, error="Duplicate path or knowledge base missing")
            )
            continue
        remaining -= len(data)
        try:
            await ingest_document(doc, data)
            results.append(UploadResult(path=path, ok=True, doc_id=doc["id"]))
        except KbUnavailableError as exc:
            results.append(UploadResult(path=path, ok=False, error=str(exc)))
        except Exception as exc:
            results.append(UploadResult(path=path, ok=False, doc_id=doc["id"], error=str(exc)))
    return UploadResponse(kb_id=kb_id, results=results)


@router.post("/{kb_id}/zip", response_model=UploadResponse)
async def upload_zip(
    kb_id: str,
    current_user: dict = Depends(get_current_user),
    file: UploadFile = File(...),
):
    """Upload a .zip archive of documents (folder upload in one request).

    Structural safety (valid zip, entry count, per-entry size, total size,
    path traversal) is enforced before anything is stored; per-entry checks
    (extension allowlist, quota) produce per-entry results.
    """
    username = current_user["username"]
    await _get_owned_kb(kb_id, username)
    data = await file.read()
    if len(data) > settings.kb_max_file_size_mb * 1024 * 1024:
        return UploadResponse(
            kb_id=kb_id,
            results=[
                UploadResult(
                    path=file.filename or "archive.zip",
                    ok=False,
                    error=f"Archive too large (max {settings.kb_max_file_size_mb} MB)",
                )
            ],
        )
    try:
        entries = await run_in_threadpool(extract_zip_entries, data)
    except ZipValidationError as exc:
        return UploadResponse(
            kb_id=kb_id,
            results=[UploadResult(path=file.filename or "archive.zip", ok=False, error=str(exc))],
        )

    remaining = await _quota_remaining(username)
    results: list[UploadResult] = []
    for entry in entries:
        path = entry["path"]
        try:
            _check_extension(path)
        except HTTPException as exc:
            results.append(UploadResult(path=path, ok=False, error=exc.detail))
            continue
        entry_data = entry["data"]
        if len(entry_data) > remaining:
            results.append(UploadResult(path=path, ok=False, error="Storage quota exceeded"))
            continue
        doc = await persistence.kb.add_document(
            username, kb_id, path, entry["mime_type"], len(entry_data), entry_data
        )
        if doc is None:
            results.append(
                UploadResult(path=path, ok=False, error="Duplicate path or knowledge base missing")
            )
            continue
        remaining -= len(entry_data)
        try:
            await ingest_document(doc, entry_data)
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


@router.get("/{kb_id}/files/{doc_id}/content")
async def download_document(
    kb_id: str, doc_id: str, current_user: dict = Depends(get_current_user)
):
    """Serve the raw uploaded bytes (inline preview; `Content-Disposition` set)."""
    from urllib.parse import quote

    from fastapi import Response

    await _get_owned_kb(kb_id, current_user["username"])
    fetched = await persistence.kb.get_document_content(current_user["username"], doc_id)
    if fetched is None:
        raise NotFound("Document not found")
    meta, data = fetched
    filename = Path(meta["path"]).name
    return Response(
        content=data,
        media_type=meta["mime_type"] or "application/octet-stream",
        headers={"Content-Disposition": f"inline; filename*=UTF-8''{quote(filename)}"},
    )


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
    alpha: float | None = Query(default=None, ge=0.0, le=1.0),
):
    """Hybrid search (vector + keyword) within one knowledge base.

    `alpha` overrides KB_HYBRID_ALPHA per request (0 = keyword, 1 = vectors).
    """
    await _get_owned_kb(kb_id, current_user["username"])
    store = _vector_store_or_503()
    effective_alpha = alpha if alpha is not None else settings.kb_hybrid_alpha
    try:
        hits = await run_in_threadpool(
            store.search,
            q,
            owner=current_user["username"],
            kb_id=kb_id,
            limit=limit,
            alpha=effective_alpha,
        )
    except KbUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return SearchOut(query=q, hits=hits)
