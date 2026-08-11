"""Chat routes: /chat (SSE), /api/chat (AI SDK), /threads."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from langgraph.graph.state import CompiledStateGraph

from .. import ai_sdk_chat, auth, schemas
from ..db import persistence
from ..services.chat import _serialize_message, agent_stream, sse_response
from ..services.searxng import set_search_enabled
from .deps import get_current_user

router = APIRouter(tags=["chat"])


@router.post("/chat")
async def chat(
    request: Request,
    body: schemas.ChatRequest,
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
    body: schemas.AiSdkChatRequest,
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
    if not text:
        raise HTTPException(status_code=422, detail="No user message found in request")

    username = "guest"
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.lower().startswith("bearer "):
        token_data = auth.decode_access_token(auth_header[7:])
        if token_data is not None and token_data.username:
            username = token_data.username

    events = agent_stream(agent, username, message=text, thread_id=body.id)
    return sse_response(ai_sdk_chat.sdk_stream(events))


@router.post("/threads/{thread_id}/resume")
async def resume_thread(
    request: Request,
    thread_id: str,
    body: schemas.ResumeRequest,
    current_user: dict = Depends(get_current_user),
):
    """Resume a run paused on a human-in-the-loop interrupt."""
    agent: CompiledStateGraph = request.app.state.agent
    config = {"configurable": {"thread_id": thread_id}}

    snapshot = await agent.aget_state(config)
    if snapshot is None or not snapshot.values.get("messages"):
        raise HTTPException(status_code=404, detail="Thread not found")
    waiting = any(getattr(t, "interrupts", None) for t in (snapshot.tasks or []))
    if not waiting:
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


@router.get("/threads", response_model=list[schemas.ThreadOut])
async def list_threads(current_user: dict = Depends(get_current_user)):
    items = await persistence.store.asearch(("threads", current_user["username"]))
    threads = [schemas.ThreadOut(thread_id=it.key, **it.value) for it in items]
    threads.sort(key=lambda t: t.updated_at or t.created_at, reverse=True)
    return threads


@router.get("/threads/{thread_id}/messages")
async def thread_messages(
    request: Request,
    thread_id: str,
    current_user: dict = Depends(get_current_user),
):
    # DeepAgentState stores messages via a DeltaChannel, so raw checkpoint
    # values don't contain the full list — rehydrate through the graph.
    agent: CompiledStateGraph = request.app.state.agent
    snapshot = await agent.aget_state({"configurable": {"thread_id": thread_id}})
    if snapshot is None or not snapshot.values.get("messages"):
        raise HTTPException(status_code=404, detail="Thread not found")
    messages = snapshot.values["messages"]
    return [_serialize_message(m) for m in messages]
