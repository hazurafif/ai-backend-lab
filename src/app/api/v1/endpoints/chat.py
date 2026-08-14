"""Chat routes: /chat (SSE), /api/chat (AI SDK), /threads."""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from langgraph.graph.state import CompiledStateGraph
from starlette.datastructures import UploadFile

from ....core.constants import thread_metadata_ns
from ....core.database import persistence
from ....core.dependencies import get_current_user
from ....core.exceptions import NotFound
from ....core.run_registry import runs
from ....core.security import decode_access_token
from ....schema.chat_schema import (
    AiSdkChatRequest,
    ChatRequest,
    FollowUpIn,
    FollowUpOut,
    ResumeRequest,
    SharedChatOut,
    ShareOut,
    ThreadOut,
    ThreadUpdate,
    ThreadUsageOut,
)
from ....services import agent_configs, ai_sdk_chat, session_stats
from ....services import share as share_service
from ....services.chat import _serialize_message, agent_stream, sse_response
from ....services.searxng import set_search_enabled
from ....services.title_generator import generate_title
from ....services.uploads import file_notes, save_uploads
from ....util.date import now_iso

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


async def _resolve_agent(request: Request, name: str | None, username: str) -> CompiledStateGraph:
    """Resolve the compiled graph for an agent config (404 when unknown).

    The built-in 'default' agent is served by `app.state.agent` (lifespan-
    built, and overridable in tests); named agents come from the registry.
    """
    name = name or "default"
    if name == "default" and request.app.state.agent is not None:
        return request.app.state.agent
    try:
        return await request.app.state.agents.resolve(name, username)
    except KeyError:
        raise NotFound(detail=f"Agent '{name}' not found") from None


async def _thread_agent(request: Request, thread_id: str, username: str) -> CompiledStateGraph:
    """The graph the thread was last run with (thread metadata agent, else default)."""
    item = await persistence.store.aget(thread_metadata_ns(username), thread_id)
    name = (item.value or {}).get("agent") if item is not None else None
    return await _resolve_agent(request, name, username)


def _parse_bool(value: str | None) -> bool | None:
    """'true'/'false' form value -> bool (anything else -> None)."""
    if value is None:
        return None
    lowered = str(value).strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    return None


def _form_files(form) -> list[UploadFile]:
    return [f for f in form.getlist("files") if isinstance(f, UploadFile)]


def _augment_message(message: str, results: list[dict]) -> str:
    """Append the uploaded-file notes to the user message (or use them alone)."""
    notes = file_notes(results)
    if not notes:
        return message
    return f"{message}\n\n{notes}" if message else notes


async def _read_chat_input(
    request: Request, username: str
) -> tuple[str, str | None, str | None, bool | None]:
    """Read (message, thread_id, agent, enable_search) from JSON or multipart.

    Multipart (file uploads): the same fields arrive as form values plus
    `files`; uploaded files are saved to the user's uploads dir and their
    paths are appended to the message so the agent can work with them.
    """
    ctype = request.headers.get("content-type", "")
    if ctype.startswith("multipart/form-data") or ctype.startswith(
        "application/x-www-form-urlencoded"
    ):
        form = await request.form()
        message = str(form.get("message") or "").strip()
        thread_id = form.get("thread_id") or None
        agent_name = form.get("agent") or None
        enable_search = _parse_bool(form.get("enable_search"))
        message = _augment_message(message, await save_uploads(username, _form_files(form)))
        if not message:
            raise HTTPException(status_code=422, detail="No user message found in request")
        return message, thread_id, agent_name, enable_search
    body = ChatRequest.model_validate(await request.json())
    return body.message, body.thread_id, body.agent, body.enable_search


@router.post("/chat")
async def chat(
    request: Request,
    current_user: dict = Depends(get_current_user),
):
    message, thread_id, agent_name, enable_search = await _read_chat_input(
        request, current_user["username"]
    )
    agent = await _resolve_agent(request, agent_name, current_user["username"])
    set_search_enabled(enable_search)
    return sse_response(
        agent_stream(
            agent,
            current_user["username"],
            message=message,
            thread_id=thread_id,
            agent_name=agent_name or "default",
        )
    )


@router.post("/api/chat")
async def ai_sdk_chat_endpoint(
    request: Request,
):
    """AI SDK data-stream endpoint for the frontend (useChat).

    JSON body: {"id": <chat uuid>, "messages": [UIMessage...],
    "selectedChatModel": ...} — or multipart/form-data with a JSON-encoded
    `messages` field + `files` uploads (useChat attachments). The last user
    message is run through the agent; the stream is translated to AI SDK
    chunks (see `ai_sdk_chat.sdk_stream`). Auth is optional: a Bearer JWT
    scopes thread metadata to that user, otherwise a "guest" namespace is
    used (starter mode, matching the frontend which has no login yet).
    """
    agent: CompiledStateGraph = request.app.state.agent
    is_multipart = request.headers.get("content-type", "").startswith("multipart/form-data")
    if not is_multipart:
        body = AiSdkChatRequest.model_validate(await request.json())
        set_search_enabled(body.enable_search)
        text = ai_sdk_chat.extract_user_message(body.messages)
    else:
        body = None
        text = ""

    username = "guest"
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        token_data = decode_access_token(auth_header[7:])
        if token_data is not None and token_data.username:
            username = token_data.username

    # Multipart: file uploads (AI SDK useChat attachments). `messages` is a
    # JSON-encoded form field; uploaded files are saved and their paths are
    # appended to the extracted user text. HITL resume stays JSON-only.
    if is_multipart:
        form = await request.form()
        try:
            messages = json.loads(str(form.get("messages") or "[]"))
        except json.JSONDecodeError:
            raise HTTPException(status_code=422, detail="Invalid 'messages' form field") from None
        if not isinstance(messages, list):
            raise HTTPException(status_code=422, detail="Invalid 'messages' form field")
        set_search_enabled(_parse_bool(form.get("enable_search")))
        text = ai_sdk_chat.extract_user_message(messages)
        text = _augment_message(text, await save_uploads(username, _form_files(form)))
        if not text:
            raise HTTPException(status_code=422, detail="No user message found in request")
        agent = await _resolve_agent(request, form.get("agent") or None, username)
        events = agent_stream(
            agent,
            username,
            message=text,
            thread_id=form.get("id") or None,
            agent_name=form.get("agent") or "default",
        )
        return sse_response(ai_sdk_chat.sdk_stream(events))

    if body is not None and (body.decisions is not None or body.decision is not None):
        # HITL resume: continue a paused run. `id` is the thread_id of the
        # interrupted run; decisions follow ResumeRequest semantics.
        if not body.id:
            raise HTTPException(status_code=422, detail="'id' is required to resume a run")
        thread_id = body.id
        if username != "guest":
            await _assert_thread_owner(thread_id, username)
        agent = await _thread_agent(request, thread_id, username)
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

    agent = await _resolve_agent(request, body.agent, username)
    events = agent_stream(
        agent, username, message=text, thread_id=body.id, agent_name=body.agent or "default"
    )
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
    agent = await _thread_agent(request, thread_id, current_user["username"])
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


async def _title_payload(request: Request, thread_id: str, username: str) -> tuple:
    """Shared resolution for title endpoints: (messages, metadata, agent_name, model).

    Raises 404 for unknown/empty threads or unknown agents.
    """
    await _assert_thread_owner(thread_id, username)
    agent = await _thread_agent(request, thread_id, username)
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await agent.aget_state(config)
    messages = (snapshot.values or {}).get("messages") if snapshot is not None else None
    if not messages:
        raise NotFound(detail=f"Thread '{thread_id}' has no messages yet")
    metadata = await persistence.store.aget(thread_metadata_ns(username), thread_id)
    value = dict(metadata.value) if metadata is not None else {}
    agent_name = value.get("agent") or "default"
    try:
        model = await request.app.state.agents.model_for(agent_name, username)
    except KeyError:
        raise NotFound(detail=f"Agent '{agent_name}' not found") from None
    return messages, value, agent_name, model


async def _upsert_title(thread_id: str, username: str, value: dict, agent_name: str) -> None:
    """Store the metadata row (title already set in `value`)."""
    now = now_iso()
    value.setdefault("created_at", now)
    value["updated_at"] = now
    value["agent"] = agent_name
    await persistence.store.aput(thread_metadata_ns(username), thread_id, value)


def _default_title(messages: list) -> str | None:
    """The auto title the chat service sets on first message (truncation)."""
    from ....services.title_generator import _message_text

    for m in messages:
        text = _message_text(m)
        if text:
            return text.strip()[:80]
    return None


@router.post("/threads/{thread_id}/title", response_model=ThreadOut)
async def generate_thread_title(
    request: Request,
    thread_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Generate a title from the LLM (template) and upsert it on the thread.

    The conversation is rendered into the title template and run through the
    thread's own agent model; the result is stored as the metadata title,
    creating the metadata row when the thread has none (legacy threads).
    Fails with 404 when the thread has no messages yet.
    """
    username = current_user["username"]
    messages, value, agent_name, model = await _title_payload(request, thread_id, username)
    title = await generate_title(model, messages)
    value["title"] = title
    await _upsert_title(thread_id, username, value, agent_name)
    return ThreadOut(thread_id=thread_id, **value)


@router.post("/threads/{thread_id}/followup", response_model=FollowUpOut)
async def thread_followup(
    request: Request,
    thread_id: str,
    body: FollowUpIn | None = None,
    current_user: dict = Depends(get_current_user),
):
    """Post-run follow-up for the frontend: auto-title the thread.

    Call this after a chat run's `done` event. It generates an LLM title only
    when the thread has no title yet or it is still the raw first-message
    truncation (`force: true` always regenerates); otherwise it returns the
    existing title without spending tokens. Response: {thread_id, title,
    generated}.
    """
    force = bool(body.force) if body is not None else False
    username = current_user["username"]
    messages, value, agent_name, model = await _title_payload(request, thread_id, username)

    current = value.get("title")
    needs_title = force or not current or current == _default_title(messages)
    if not needs_title:
        return FollowUpOut(thread_id=thread_id, title=current, generated=False)

    title = await generate_title(model, messages)
    value["title"] = title
    await _upsert_title(thread_id, username, value, agent_name)
    return FollowUpOut(thread_id=thread_id, title=title, generated=True)


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
    await _assert_thread_owner(thread_id, current_user["username"])
    return await _thread_history(request, thread_id)


async def _thread_history(request: Request, thread_id: str) -> list[dict]:
    """Readable history rows; falls back to checkpoint rehydration.

    Prefers the `chat_messages` table; falls back to checkpoint rehydration
    for threads created before the table existed (or when the run ended in
    an error before history was written).
    """
    history = await persistence.chat_history.list_messages(thread_id)
    if history:
        return history

    # DeepAgentState stores messages via a DeltaChannel, so raw checkpoint
    # values don't contain the full list — rehydrate through the graph.
    agent: CompiledStateGraph = request.app.state.agent
    snapshot = await agent.aget_state({"configurable": {"thread_id": thread_id}})
    if snapshot is None or not snapshot.values.get("messages"):
        raise HTTPException(status_code=404, detail="Thread not found")
    return [_serialize_message(m) for m in snapshot.values["messages"]]


@router.get("/threads/{thread_id}/usage", response_model=ThreadUsageOut)
async def thread_usage(
    request: Request,
    thread_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Context + token usage of a thread (the "current session" view).

    - `messages`: stored message count + content size
    - `usage`: cumulative input/output/total tokens from usage_metadata
      (null when the provider reports none)
    - `context`: the last run's input tokens (= context the model currently
      sees) vs the model's context window (utilization, remaining) — null
      before the first run
    - `active_run`: true while a run is in progress on this thread
    """
    await _assert_thread_owner(thread_id, current_user["username"])
    history = await _thread_history(request, thread_id)

    item = await persistence.store.aget(thread_metadata_ns(current_user["username"]), thread_id)
    agent_name = (item.value or {}).get("agent") if item is not None else None
    spec = await agent_configs.load_spec(
        persistence.store, agent_name or "default", current_user["username"]
    )
    model = spec.model if spec is not None else None
    return ThreadUsageOut(
        thread_id=thread_id,
        agent=agent_name,
        model=model,
        messages=session_stats.message_counts(history),
        usage=session_stats.compute_usage(history),
        context=session_stats.build_context(
            session_stats.current_context_input_tokens(history), model
        ),
        active_run=runs.is_running(thread_id),
    )
