"""Chat routes: /chat (SSE), /api/chat (AI SDK), /threads."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from langgraph.graph.state import CompiledStateGraph

from ....core.constants import thread_metadata_ns
from ....core.database import persistence
from ....core.dependencies import get_current_user
from ....core.exceptions import NotFound
from ....core.run_registry import runs
from ....core.security import decode_access_token
from ....schema.chat_schema import (
    AiSdkChatRequest,
    ChatRequest,
    ResumeRequest,
    SharedChatOut,
    ShareOut,
    ThreadOut,
    ThreadUpdate,
)
from ....services import ai_sdk_chat
from ....services import share as share_service
from ....services.chat import _serialize_message, agent_stream, sse_response
from ....services.searxng import set_search_enabled

router = APIRouter(tags=["chat"])


def _share_url(request: Request, share_token: str) -> str:
    base = str(request.base_url).rstrip("/")
    return f"{base}/shared/{share_token}"


async def _assert_thread_owner(thread_id: str, username: str) -> None:
    """404 when the thread belongs to another user; legacy threads pass.

    Thread metadata lives under ("threads", <owner>) in the store. Threads
    without metadata predate the metadata feature (or were started by a
    guest) — keep serving those to avoid breaking old threads; only a
    positive match under another user's namespace is rejected.
    """
    item = await persistence.store.aget(thread_metadata_ns(username), thread_id)
    if item is not None:
        return
    for other in await persistence.users.list_users():
        other_name = other["username"]
        if other_name == username:
            continue
        if await persistence.store.aget(thread_metadata_ns(other_name), thread_id) is not None:
            raise NotFound(detail=f"Thread '{thread_id}' not found")


@router.post("/chat")
async def chat(
    request: Request,
    body: ChatRequest,
    current_user: dict = Depends(get_current_user),
):
    agent: CompiledStateGraph = request.app.state.agent
    set_search_enabled(body.enable_search)
    return sse_response(
        agent_stream(
            agent, current_user["username"], message=body.message, thread_id=body.thread_id
        )
    )


@router.post("/api/chat")
async def ai_sdk_chat_endpoint(
    request: Request,
    body: AiSdkChatRequest,
):
    """AI SDK data-stream endpoint for the frontend (useChat).

    Body: {"id": <chat uuid>, "messages": [UIMessage...],
    "selectedChatModel": ...}. The last user message is run through the
    agent; the stream is translated to AI SDK chunks (see
    `ai_sdk_chat.sdk_stream`). Auth is optional: a Bearer JWT scopes
    thread metadata to that user, otherwise a "guest" namespace is used
    (starter mode, matching the frontend which has no login yet).
    """
    agent: CompiledStateGraph = request.app.state.agent
    set_search_enabled(body.enable_search)
    text = ai_sdk_chat.extract_user_message(body.messages)

    username = "guest"
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        token_data = decode_access_token(auth_header[7:])
        if token_data is not None and token_data.username:
            username = token_data.username

    if body.decisions is not None or body.decision is not None:
        # HITL resume: continue a paused run. `id` is the thread_id of the
        # interrupted run; decisions follow ResumeRequest semantics.
        if not body.id:
            raise HTTPException(status_code=422, detail="'id' is required to resume a run")
        thread_id = body.id
        if username != "guest":
            await _assert_thread_owner(thread_id, username)
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await agent.aget_state(config)
        if snapshot is None or not snapshot.values.get("messages"):
            raise HTTPException(status_code=404, detail="Thread not found")
        waiting = any(getattr(t, "interrupts", None) for t in (snapshot.tasks or []))
        if not waiting:
            raise HTTPException(status_code=409, detail="Thread is not waiting for input")
        decisions = body.decisions if body.decisions is not None else [body.decision]
        events = agent_stream(agent, username, thread_id=thread_id, resume={"decisions": decisions})
        return sse_response(ai_sdk_chat.sdk_stream(events))

    if not text:
        raise HTTPException(status_code=422, detail="No user message found in request")

    events = agent_stream(agent, username, message=text, thread_id=body.id)
    return sse_response(ai_sdk_chat.sdk_stream(events))


async def _thread_is_waiting(agent: CompiledStateGraph, thread_id: str) -> bool:
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await agent.aget_state(config)
    if snapshot is None or not snapshot.values.get("messages"):
        return False
    return any(getattr(t, "interrupts", None) for t in (snapshot.tasks or []))


@router.post("/threads/{thread_id}/resume")
async def resume_thread(
    request: Request,
    thread_id: str,
    body: ResumeRequest,
    current_user: dict = Depends(get_current_user),
):
    """Resume a run paused on a human-in-the-loop interrupt."""
    agent: CompiledStateGraph = request.app.state.agent
    await _assert_thread_owner(thread_id, current_user["username"])

    if not await _thread_is_waiting(agent, thread_id):
        raise HTTPException(status_code=409, detail="Thread is not waiting for input")

    if body.decisions is not None:
        decisions = body.decisions
    elif body.decision is not None:
        decisions = [body.decision]
    else:
        raise HTTPException(status_code=422, detail="Provide 'decision' or 'decisions'")

    # HITL middleware expects the resume value as {"decisions": [...]}.
    resume_value = {"decisions": decisions}
    return sse_response(
        agent_stream(agent, current_user["username"], thread_id=thread_id, resume=resume_value)
    )


@router.post("/threads/{thread_id}/cancel")
async def cancel_thread(
    request: Request,
    thread_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Abort the active run of a thread; the SSE stream ends with a `done` event carrying `cancelled: true`."""
    await _assert_thread_owner(thread_id, current_user["username"])
    if not runs.cancel(thread_id):
        raise HTTPException(status_code=409, detail="Thread has no active run")
    return {"status": "cancelled", "thread_id": thread_id}


@router.delete("/threads/{thread_id}", status_code=204)
async def delete_thread(
    request: Request,
    thread_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete a thread: checkpoint state, history rows and metadata."""
    await _assert_thread_owner(thread_id, current_user["username"])

    checkpointer = persistence.checkpointer
    delete_thread = getattr(checkpointer, "adelete_thread", None)
    if delete_thread is not None:
        await delete_thread(thread_id)
    await persistence.chat_history.delete_thread(thread_id)
    # Drop the share link (if any) before removing the metadata that holds the token.
    item = await persistence.store.aget(thread_metadata_ns(current_user["username"]), thread_id)
    await share_service.revoke_by_thread(
        thread_id, current_user["username"], item.value.get("share_token") if item else None
    )
    await persistence.store.adelete(thread_metadata_ns(current_user["username"]), thread_id)
    return None


@router.patch("/threads/{thread_id}", response_model=ThreadOut)
async def rename_thread(
    thread_id: str,
    body: ThreadUpdate,
    current_user: dict = Depends(get_current_user),
):
    """Rename a thread (its metadata title)."""
    ns = thread_metadata_ns(current_user["username"])
    item = await persistence.store.aget(ns, thread_id)
    if item is None:
        raise NotFound(detail=f"Thread '{thread_id}' not found")
    value = dict(item.value)
    value["title"] = body.title
    await persistence.store.aput(ns, thread_id, value)
    return ThreadOut(thread_id=thread_id, **value)


@router.post("/threads/{thread_id}/share", response_model=ShareOut, status_code=201)
async def share_thread(
    request: Request,
    thread_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Create a public share link for a thread (owner only).

    Idempotent: sharing an already-shared thread returns the existing token.
    """
    await _assert_thread_owner(thread_id, current_user["username"])
    share = await share_service.create_share(thread_id, current_user["username"])
    return ShareOut(share_token=share["share_token"], url=_share_url(request, share["share_token"]))


@router.get("/threads/{thread_id}/share", response_model=ShareOut)
async def get_thread_share(
    request: Request,
    thread_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Current share link of a thread (owner only); 404 when not shared."""
    await _assert_thread_owner(thread_id, current_user["username"])
    share = await share_service.get_share(thread_id, current_user["username"])
    if share is None:
        raise NotFound(detail=f"Thread '{thread_id}' is not shared")
    return ShareOut(share_token=share["share_token"], url=_share_url(request, share["share_token"]))


@router.delete("/threads/{thread_id}/share", status_code=204)
async def unshare_thread(
    thread_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Revoke the thread's share link (owner only)."""
    await _assert_thread_owner(thread_id, current_user["username"])
    if not await share_service.revoke_share(thread_id, current_user["username"]):
        raise NotFound(detail=f"Thread '{thread_id}' is not shared")
    return None


@router.get("/shared/{share_token}", response_model=SharedChatOut)
async def view_shared_chat(share_token: str):
    """Public, unauthenticated read-only view of a shared thread."""
    share = await share_service.lookup_share(share_token)
    if share is None:
        raise NotFound(detail="Share link not found")
    history = await persistence.chat_history.list_messages(share["thread_id"])
    item = await persistence.store.aget(thread_metadata_ns(share["username"]), share["thread_id"])
    return SharedChatOut(
        thread_id=share["thread_id"],
        title=item.value.get("title") if item else None,
        username=share["username"],
        created_at=share.get("created_at"),
        messages=history,
    )


@router.get("/threads", response_model=list[ThreadOut])
async def list_threads(
    current_user: dict = Depends(get_current_user),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    items = await persistence.store.asearch(thread_metadata_ns(current_user["username"]))
    threads = [ThreadOut(thread_id=it.key, **it.value) for it in items]
    threads.sort(key=lambda t: t.updated_at or t.created_at, reverse=True)
    return threads[offset : offset + limit]


@router.get("/threads/{thread_id}/messages")
async def thread_messages(
    request: Request,
    thread_id: str,
    current_user: dict = Depends(get_current_user),
):
    # Prefer the readable chat_messages table; fall back to checkpoint
    # rehydration for threads created before the table existed (or when the
    # run ended in an error before history was written).
    await _assert_thread_owner(thread_id, current_user["username"])
    history = await persistence.chat_history.list_messages(thread_id)
    if history:
        return history

    # DeepAgentState stores messages via a DeltaChannel, so raw checkpoint
    # values don't contain the full list — rehydrate through the graph.
    agent: CompiledStateGraph = request.app.state.agent
    snapshot = await agent.aget_state({"configurable": {"thread_id": thread_id}})
    if snapshot is None or not snapshot.values.get("messages"):
        raise HTTPException(status_code=404, detail="Thread not found")
    messages = snapshot.values["messages"]
    return [_serialize_message(m) for m in messages]
