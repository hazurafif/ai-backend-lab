"""AI Backend API: Deep Agents harness behind a streaming (SSE) chat endpoint.

Endpoints:
  POST /chat                      -> SSE stream of agent events
  POST /api/chat                  -> AI SDK data-stream protocol (frontend useChat)
  GET  /threads                   -> conversations for the current user
  GET  /threads/{id}/messages     -> full message history of a thread
  POST /threads/{id}/resume       -> resume an interrupted (HITL) run
  POST /login                     -> JWT
  GET  /users/me                  -> current user
  GET  /health                    -> status

SSE events (event: <name>, data: <json>):
  message_delta   {"id", "delta"}                           token chunk
  message         {"id", "message"}                         finalized message (langchain serialized)
  tool_start      {"id", "name", "args"}                    tool call started
  tool_delta      {"id", "name", "delta"}                   tool output chunk
  tool_end        {"id", "name", "output", "is_error"}      tool call finished
  subagent        {"name", "status", "output"?, "error"?}   delegated task lifecycle
  subagent_delta  {"subagent", "delta"}                     subagent token chunk
  interrupt       {"thread_id", "interrupts"[{value, when}]} run paused for human input
  error           {"source", "message"}                     recoverable stream error
  done            {"thread_id", "messages"[], "interrupted"?} final state

Human-in-the-loop: when a run interrupts (e.g. `interrupt_on={"write_file": true}`),
the stream emits `interrupt` with the HITLRequest payloads, then `done` with
`interrupted: true`. The frontend shows an approval UI and calls
POST /threads/{id}/resume with a decision:
  {"decision": {"type": "approve"}}
  {"decision": {"type": "edit", "edited_action": {"name": ..., "args": {...}}}}
  {"decision": {"type": "reject", "message": "..."}}
  {"decision": {"type": "respond", "message": "..."}}
or a full list: {"decisions": [...]} (must match the number of action_requests).
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from langchain_core.messages import HumanMessage
from langgraph.errors import GraphInterrupt
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from . import ai_sdk_chat, auth, schemas
from .agent import build_agent
from .config import settings
from .db import persistence
from .dependencies import get_current_user
from .mcp_client import mcp_servers

logger = logging.getLogger(__name__)

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _field(obj: Any, *names: str, default: Any = None) -> Any:
    """Read a field from an object or dict, trying several names."""
    for n in names:
        if isinstance(obj, dict):
            if n in obj:
                return obj[n]
        else:
            try:
                v = getattr(obj, n, None)
            except Exception:
                v = None
            if v is not None:
                return v
    return default


def _serialize_message(msg: Any) -> dict:
    try:
        return msg.model_dump(mode="json", exclude_none=True)
    except Exception:
        return {
            "type": getattr(msg, "type", "unknown"),
            "content": str(getattr(msg, "content", "")),
        }


def _jsonable(v: Any) -> Any:
    """Coerce arbitrary values (messages, tool outputs, ...) to JSON-safe data."""
    if v is None or isinstance(v, str | int | float | bool):
        return v
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [_jsonable(x) for x in v]
    if hasattr(v, "model_dump"):
        try:
            return v.model_dump(mode="json", exclude_none=True)
        except Exception:
            pass
    return str(v)


def _sse(event: str, data: dict) -> str:
    return (
        f"event: {event}\ndata: {json.dumps(_jsonable(data), ensure_ascii=False, default=str)}\n\n"
    )


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _collect_interrupts(snapshot: Any) -> list[Any]:
    """Extract interrupt payloads from a state snapshot's pending tasks."""
    out: list[Any] = []
    for task in getattr(snapshot, "tasks", None) or []:
        for it in getattr(task, "interrupts", None) or []:
            if hasattr(it, "value"):  # langgraph Interrupt dataclass
                out.append(it.value)
            elif isinstance(it, dict):
                out.append(it.get("value"))
            else:
                out.append(it)
    return out


def _sse_response(gen: AsyncIterator[str]) -> StreamingResponse:
    return StreamingResponse(gen, media_type="text/event-stream", headers=SSE_HEADERS)


# ---------------------------------------------------------------------------
# agent streaming
# ---------------------------------------------------------------------------


async def _agent_stream(
    agent: CompiledStateGraph,
    username: str,
    *,
    message: str | None = None,
    thread_id: str | None = None,
    resume: Any = None,
) -> AsyncIterator[str]:
    """Run the agent and forward v3 stream projections as normalized SSE events.

    - `message`: new user message for /chat
    - `resume`: value for `Command(resume=...)` (HITL continuation, /resume)
    """
    thread_id = thread_id or f"thread-{uuid.uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": thread_id}}

    if message is not None:
        # Record thread metadata in the store so GET /threads can list conversations.
        now = _now_iso()
        try:
            await persistence.store.aput(
                ("threads", username),
                thread_id,
                {"title": message[:80], "created_at": now, "updated_at": now},
            )
        except Exception:
            logger.exception("failed to record thread metadata")

    if resume is not None:
        run_input: Any = Command(resume=resume)
    else:
        run_input = {"messages": [HumanMessage(content=message)]}

    queue: asyncio.Queue[dict | None] = asyncio.Queue(maxsize=256)

    async def put(event: str, data: dict) -> None:
        await queue.put({"event": event, "data": data})

    async def consume_messages(run) -> None:
        counter = 0
        try:
            async for ms in run.messages:
                counter += 1
                ms_id = _field(ms, "message_id", "id", default=f"llm-{counter}")
                try:
                    async for delta in ms.text:
                        await put("message_delta", {"id": ms_id, "delta": delta})
                except Exception as exc:
                    await put("error", {"source": "messages", "message": str(exc)})
                try:
                    out_attr = ms.output  # awaitable property (async) / property (sync)
                    out = await out_attr() if callable(out_attr) else await out_attr
                except Exception:
                    out = None
                if out is not None:
                    await put("message", {"id": ms_id, "message": _serialize_message(out)})
        except Exception as exc:
            await put("error", {"source": "messages", "message": str(exc)})

    async def consume_tool_calls(run) -> None:
        try:
            async for tc in run.tool_calls:
                # ToolCallStream: stable fields at start, terminal fields after drain.
                tc_id = _field(tc, "tool_call_id", "id", default=f"call-{uuid.uuid4().hex[:8]}")
                name = _field(tc, "tool_name", "name", default="tool")
                await put(
                    "tool_start",
                    {"id": tc_id, "name": name, "args": _field(tc, "input", "args", default={})},
                )
                try:
                    async for delta in tc.output_deltas:
                        await put("tool_delta", {"id": tc_id, "name": name, "delta": delta})
                except Exception:
                    pass  # no deltas
                error = _field(tc, "error", default=None)
                await put(
                    "tool_end",
                    {
                        "id": tc_id,
                        "name": name,
                        "output": _field(tc, "output", default=None),
                        "is_error": error is not None
                        or bool(_field(tc, "completed", default=True) is False),
                        "error": error,
                    },
                )
        except Exception as exc:
            await put("error", {"source": "tool_calls", "message": str(exc)})

    async def consume_subagents(run) -> None:
        try:
            async for sub in run.subagents:
                name = _field(sub, "name", default="subagent")
                await put("subagent", {"name": name, "status": "started"})
                try:
                    async for ms in sub.messages:
                        async for delta in ms.text:
                            await put("subagent_delta", {"subagent": name, "delta": delta})
                except Exception:
                    pass  # subagent message stream is best-effort
                status = _field(sub, "status", default="completed")
                output = None
                with suppress(Exception):
                    output = await sub.output()
                await put(
                    "subagent",
                    {
                        "name": name,
                        "status": status,
                        "output": output,
                        "error": _field(sub, "error"),
                    },
                )
        except Exception as exc:
            await put("error", {"source": "subagents", "message": str(exc)})

    async def consume_values(run) -> None:
        # Drain state snapshots so the pump never blocks; not emitted for now.
        with suppress(Exception):
            async for _ in run.values:
                pass

    interrupted = {"flag": False}

    # Start the run.
    try:
        run = await agent.astream_events(run_input, config=config, version="v3")
    except Exception as exc:
        yield _sse("error", {"source": "run", "message": str(exc)})
        yield _sse("done", {"thread_id": thread_id, "messages": []})
        return

    async def drain() -> None:
        results = await asyncio.gather(
            consume_messages(run),
            consume_tool_calls(run),
            consume_subagents(run),
            consume_values(run),
            return_exceptions=True,
        )
        for r in results:
            if isinstance(r, GraphInterrupt):
                interrupted["flag"] = True
            elif isinstance(r, Exception):
                logger.error("stream consumer failed: %r", r)
        await queue.put(None)

    producer = asyncio.create_task(drain())
    try:
        while True:
            item = await queue.get()
            if item is None:
                break
            yield _sse(item["event"], item["data"])
    finally:
        producer.cancel()

    # Final state. With v3 streaming a HITL interrupt does not raise — the
    # run ends normally, so we detect it from the persisted state's pending
    # tasks instead.
    messages: list[dict] = []
    try:
        final = await run.output()
        if final:
            messages = [_serialize_message(m) for m in final.get("messages", [])]
    except GraphInterrupt:
        pass  # older runtimes raise; handled via snapshot below
    except Exception as exc:
        yield _sse("error", {"source": "final", "message": str(exc)})
        yield _sse("done", {"thread_id": thread_id, "messages": messages})
        return

    snapshot = await agent.aget_state(config)
    interrupts = _collect_interrupts(snapshot)
    if interrupts:
        # Run paused for human input: surface the HITL requests.
        yield _sse("interrupt", {"thread_id": thread_id, "interrupts": interrupts})
        yield _sse("done", {"thread_id": thread_id, "messages": messages, "interrupted": True})
    else:
        yield _sse("done", {"thread_id": thread_id, "messages": messages})


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------


def create_app(*, agent: CompiledStateGraph | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await persistence.start()
        try:
            await mcp_servers.connect()
        except Exception:
            logger.exception("MCP connect failed; continuing without MCP tools")
        app.state.agent = agent or build_agent(
            checkpointer=persistence.checkpointer,
            store=persistence.store,
            mcp_tools=mcp_servers.tools,
            model=settings.model,
            system_prompt=settings.system_prompt,
            interrupt_on=settings.interrupt_on,
        )
        logger.info(
            "Agent ready: model=%s persistence=%s mcp=%s",
            settings.model,
            persistence.backend_name,
            mcp_servers.names or "none",
        )
        yield
        await persistence.stop()

    app = FastAPI(title="AI Backend", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "persistence": persistence.backend_name,
            "mcp_servers": mcp_servers.names,
            "model": settings.model,
            "interrupt_on": settings.interrupt_on,
        }

    @app.post("/login", response_model=schemas.Token)
    async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
        # Demo user store; replace with a real database in production.
        from .main_fake_users import fake_users_db

        user = fake_users_db.get(form_data.username)
        if not user or not auth.verify_password(form_data.password, user["hashed_password"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
        access_token = auth.create_access_token(
            data={"sub": form_data.username},
            expires_delta=timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES),
        )
        return {"access_token": access_token, "token_type": "bearer"}

    @app.get("/users/me/", response_model=schemas.User)
    async def read_users_me(current_user: dict = Depends(get_current_user)):
        return {"username": current_user["username"]}

    @app.post("/chat")
    async def chat(
        request: Request,
        body: schemas.ChatRequest,
        current_user: dict = Depends(get_current_user),
    ):
        agent: CompiledStateGraph = request.app.state.agent
        return _sse_response(
            _agent_stream(
                agent, current_user["username"], message=body.message, thread_id=body.thread_id
            )
        )

    @app.post("/api/chat")
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
        text = ai_sdk_chat.extract_user_message(body.messages)
        if not text:
            raise HTTPException(status_code=422, detail="No user message found in request")

        username = "guest"
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.lower().startswith("bearer "):
            token_data = auth.decode_access_token(auth_header[7:])
            if token_data is not None and token_data.username:
                username = token_data.username

        events = _agent_stream(agent, username, message=text, thread_id=body.id)
        return _sse_response(ai_sdk_chat.sdk_stream(events))

    @app.post("/threads/{thread_id}/resume")
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
        return _sse_response(
            _agent_stream(agent, current_user["username"], thread_id=thread_id, resume=resume_value)
        )

    @app.get("/threads", response_model=list[schemas.ThreadOut])
    async def list_threads(current_user: dict = Depends(get_current_user)):
        items = await persistence.store.asearch(("threads", current_user["username"]))
        threads = [schemas.ThreadOut(thread_id=it.key, **it.value) for it in items]
        threads.sort(key=lambda t: t.updated_at or t.created_at, reverse=True)
        return threads

    @app.get("/threads/{thread_id}/messages")
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

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
