"""Offline tests for provider connections (base URL + API token instead of .env).

Verifies:

  - CRUD via HTTP: create/list/detail/replace/delete, 409 on duplicate name,
    404 on unknown, 403 for non-admins
  - api_token is write-only: masked in every response, PUT without a token
    keeps the stored one
  - is_default semantics: one default per kind; the resolved cache follows
    the store (refresh after mutations)
  - the agent LLM model resolves the default `llm` connection (base_url +
    api_key on the chat model) and the KB embeddings factory resolves the
    default `embeddings` connection
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import pytest_asyncio
from langchain_core.language_models import BaseChatModel

from app.core import config, database
from app.core.security import create_access_token
from app.main import create_app
from app.services import connections as connection_service
from app.services.agent import AgentRegistry
from app.services.agent_configs import AgentSpec
from app.services.kb.embeddings import LocalEmbeddings, build_embeddings

pytestmark = pytest.mark.filterwarnings(
    r"ignore:The v3 streaming protocol on Pregel is experimental."
)


@pytest.fixture(autouse=True)
def _offline_env():
    """Force the in-memory backend: a local .env may set DATABASE_URI."""
    config.settings.database_uri = None


@pytest_asyncio.fixture
async def persistence():
    config.settings.database_uri = None
    await database.persistence.start()
    yield database.persistence
    await database.persistence.stop()


async def client_for(
    username: str = "tester", role: str = "admin"
) -> tuple[httpx.AsyncClient, Any]:
    """An admin client with the app lifespan running (sets app.state.agents).

    The user is seeded after the lifespan starts: persistence.start()
    re-initializes the in-memory stores on entry.
    """
    app = create_app()
    await app.router.lifespan_context(app).__aenter__()
    await database.persistence.users.create_user(username=username, hashed_password="x", role=role)
    token = create_access_token(data={"sub": username})
    return (
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
            headers={"Authorization": f"Bearer {token}"},
        ),
        app,
    )


def payload(name: str = "my-vllm", **overrides) -> dict:
    body = {
        "name": name,
        "kind": "llm",
        "base_url": "http://localhost:9999/v1",
        "api_token": "sk-test-token-1234",
        "extra": {"model": "qwen-72b"},
        "is_default": False,
    }
    body.update(overrides)
    return body


# ---------------------------------------------------------------------------
# CRUD + masking
# ---------------------------------------------------------------------------


async def test_crud_cycle_and_masking():
    client, _app = await client_for()
    async with client:
        # create -> masked token in the response
        r = await client.post("/connections", json=payload())
        assert r.status_code == 201, r.text
        created = r.json()
        assert created["name"] == "my-vllm"
        assert created["api_token"] == "sk-t…1234"
        assert created["has_token"] is True
        assert created["base_url"] == "http://localhost:9999/v1"
        assert created["extra"] == {"model": "qwen-72b"}

        # duplicate name -> 409
        r = await client.post("/connections", json=payload())
        assert r.status_code == 409

        # list + detail also mask
        r = await client.get("/connections")
        assert r.status_code == 200
        assert [c["name"] for c in r.json()] == ["my-vllm"]
        assert all(c["api_token"] != "sk-test-token-1234" for c in r.json())
        r = await client.get("/connections/my-vllm")
        assert r.status_code == 200
        assert r.json()["has_token"] is True

        # replace: api_token omitted -> stored token kept
        r = await client.put("/connections/my-vllm", json=payload(base_url="http://x:1/v1"))
        assert r.status_code == 200, r.text
        assert r.json()["base_url"] == "http://x:1/v1"
        assert r.json()["has_token"] is True

        # replace with a new token -> rotated
        r = await client.put("/connections/my-vllm", json=payload(api_token="sk-rotated-9999"))
        assert r.json()["api_token"] == "sk-r…9999"

        # delete -> 204, then 404
        r = await client.delete("/connections/my-vllm")
        assert r.status_code == 204
        r = await client.get("/connections/my-vllm")
        assert r.status_code == 404
        r = await client.put("/connections/my-vllm", json=payload())
        assert r.status_code == 404
        r = await client.delete("/connections/my-vllm")
        assert r.status_code == 404


async def test_non_admin_forbidden():
    client, _app = await client_for(username="regular", role="user")
    async with client:
        r = await client.get("/connections")
        assert r.status_code == 403
        r = await client.post("/connections", json=payload())
        assert r.status_code == 403


# ---------------------------------------------------------------------------
# default resolution + cache refresh
# ---------------------------------------------------------------------------


async def test_default_connection_per_kind_and_cache():
    client, _app = await client_for()
    async with client:
        # first created llm connection becomes the implicit default
        await client.post("/connections", json=payload(name="llm-a", kind="llm"))
        await client.post(
            "/connections",
            json=payload(name="emb-a", kind="embeddings", is_default=True),
        )
        # explicit default wins over the earlier one
        await client.post(
            "/connections",
            json=payload(name="llm-b", kind="llm", is_default=True),
        )

        llm = await database.persistence.connections.get_default("llm")
        assert llm["name"] == "llm-b"
        emb = await database.persistence.connections.get_default("embeddings")
        assert emb["name"] == "emb-a"

        # cache refresh after mutations picks the same defaults
        await connection_service.refresh_resolved_connections()
        assert connection_service.resolved_llm()["name"] == "llm-b"
        assert connection_service.resolved_embeddings()["name"] == "emb-a"
        assert connection_service.llm_model_kwargs() == {
            "base_url": "http://localhost:9999/v1",
            "api_key": "sk-test-token-1234",
        }

        # making llm-a default demotes llm-b
        r = await client.put("/connections/llm-a", json=payload(name="llm-a", is_default=True))
        assert r.json()["is_default"] is True
        llm = await database.persistence.connections.get_default("llm")
        assert llm["name"] == "llm-a"
        await connection_service.refresh_resolved_connections()
        assert connection_service.resolved_llm()["name"] == "llm-a"

        # deleting the default falls back to the remaining connection
        await client.delete("/connections/llm-a")
        await connection_service.refresh_resolved_connections()
        assert connection_service.resolved_llm()["name"] == "llm-b"

        # deleting the last one clears the cache
        await client.delete("/connections/llm-b")
        await connection_service.refresh_resolved_connections()
        assert connection_service.resolved_llm() is None
        assert connection_service.llm_model_kwargs() == {}


async def test_mask_token_short():
    assert connection_service.mask_token(None) is None
    assert connection_service.mask_token("short") == "••••"
    assert connection_service.mask_token("sk-abcdefghijkl") == "sk-a…ijkl"


# ---------------------------------------------------------------------------
# consumers: agent LLM + KB embeddings
# ---------------------------------------------------------------------------


async def test_agent_model_uses_llm_connection(persistence):
    await database.persistence.connections.create(
        {
            "name": "openai",
            "kind": "llm",
            "base_url": "http://localhost:9999/v1",
            "api_token": "sk-conn-token",
            "extra": {},
            "is_default": True,
        }
    )
    await connection_service.refresh_resolved_connections()
    registry = AgentRegistry(
        checkpointer=persistence.checkpointer,
        store=persistence.store,
        backend=None,
    )
    spec = AgentSpec(
        name="default",
        model="openai:gpt-4o-mini",
        system_prompt=None,
        skills=None,
        tools=None,
        temperature=None,
        interrupt_on=None,
        thinking=None,
        builtin=True,
    )
    model = registry._resolve_model(spec)
    assert isinstance(model, BaseChatModel)
    assert getattr(model, "openai_api_base", None) == "http://localhost:9999/v1"
    assert model.openai_api_key.get_secret_value() == "sk-conn-token"
    # No env fallback: without a connection the model refuses to build.
    await database.persistence.connections.delete("openai")
    await connection_service.refresh_resolved_connections()
    with pytest.raises(ValueError, match="No default 'llm' connection"):
        registry._resolve_model(spec)


async def test_embeddings_uses_embeddings_connection(monkeypatch):
    # Deterministic regardless of a local .env: no env key -> the fallback path.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    await database.persistence.connections.create(
        {
            "name": "emb",
            "kind": "embeddings",
            "base_url": "http://localhost:9999/emb",
            "api_token": "sk-emb-token",
            "extra": {},
            "is_default": True,
        }
    )
    await connection_service.refresh_resolved_connections()
    embeddings = build_embeddings()
    assert not isinstance(embeddings, LocalEmbeddings)
    assert getattr(embeddings, "openai_api_base", None) == "http://localhost:9999/emb"
    assert embeddings.openai_api_key.get_secret_value() == "sk-emb-token"
    # falls back to the local embedder when no connection is saved
    await database.persistence.connections.delete("emb")
    await connection_service.refresh_resolved_connections()
    assert isinstance(build_embeddings(), LocalEmbeddings)
