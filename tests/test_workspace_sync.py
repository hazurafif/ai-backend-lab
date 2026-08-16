"""Offline tests for the skills two-way sync (approach B).

The agent authors skills directly in ``skills/`` (its filesystem is the
authoring surface); ``sync_skills_to_store`` persists changes into the
store at run end while ``materialize_skills`` mirrors store -> disk at run
start. These tests drive the pair directly — no network, no API key, no
Postgres (in-memory store + temp workspace).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from langgraph.store.memory import InMemoryStore

from app.core.config import settings
from app.core.constants import user_skills_ns
from app.schema.agent_schema import SkillFileIn, SkillIn
from app.services import resources
from app.services.workspace import materialize_skills, sync_skills_to_store

SKILL_MD = "---\nname: alpha\ndescription: Alpha skill.\n---\n\nBody.\n"


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


def skills_dir(username: str = "tester") -> Path:
    return Path(settings.workspace_root) / username / "skills"


async def test_writeback_creates_skill_from_direct_write(store):
    """A skill the agent writes straight into skills/ lands in the store."""
    gen = await materialize_skills(store, "tester")
    alpha = skills_dir() / "alpha"
    alpha.mkdir(parents=True)
    (alpha / "SKILL.md").write_text(SKILL_MD)
    (alpha / "scripts").mkdir()
    (alpha / "scripts" / "run.py").write_text("print('hi')\n")

    await sync_skills_to_store(store, "tester", gen)

    ns = user_skills_ns("tester")
    skill = await resources.get_skill(store, "alpha", ns)
    assert skill is not None
    # Raw agent-authored frontmatter is preserved verbatim (no re-serialization).
    assert skill.content == SKILL_MD
    assert [f.path for f in skill.files] == ["scripts/run.py"]
    assert [s.name for s in await resources.list_skills(store, ns)] == ["alpha"]


async def test_writeback_updates_edited_skill(store):
    """Editing a materialized SKILL.md on disk updates the store copy."""
    ns = user_skills_ns("tester")
    await resources.create_skill(
        store, SkillIn(name="alpha", description="Alpha skill.", content="Body.\n"), ns
    )
    gen = await materialize_skills(store, "tester")
    md = skills_dir() / "alpha" / "SKILL.md"
    md.write_text(
        "---\nname: alpha\ndescription: Alpha skill, v2.\nlicense: MIT\n---\n\nBody v2.\n"
    )

    await sync_skills_to_store(store, "tester", gen)

    skill = await resources.get_skill(store, "alpha", ns)
    assert skill is not None
    assert "Body v2." in skill.content
    assert "license: MIT" in skill.content  # extra frontmatter fields survive


async def test_writeback_removes_deleted_files_and_skills(store):
    """Deleting bundled files / a whole skill dir on disk removes them from the store."""
    ns = user_skills_ns("tester")
    await resources.create_skill(
        store,
        SkillIn(
            name="alpha",
            description="Alpha skill.",
            content="Body.\n",
            files=[SkillFileIn(path="scripts/run.py", content="print(1)\n")],
        ),
        ns,
    )
    gen = await materialize_skills(store, "tester")
    alpha = skills_dir() / "alpha"

    (alpha / "scripts" / "run.py").unlink()
    await sync_skills_to_store(store, "tester", gen)
    skill = await resources.get_skill(store, "alpha", ns)
    assert skill is not None and skill.files == []

    shutil.rmtree(alpha)
    await sync_skills_to_store(store, "tester", gen)
    assert await resources.get_skill(store, "alpha", ns) is None


async def test_writeback_skips_invalid_frontmatter(store):
    """A malformed SKILL.md is logged and skipped — no store write, no crash."""
    gen = await materialize_skills(store, "tester")
    broken = skills_dir() / "broken"
    broken.mkdir(parents=True)
    (broken / "SKILL.md").write_text("no frontmatter here\n")

    await sync_skills_to_store(store, "tester", gen)

    ns = user_skills_ns("tester")
    assert await resources.get_skill(store, "broken", ns) is None
    assert await resources.list_skills(store, ns) == []


async def test_writeback_skips_stale_generation(store):
    """A writeback whose materialization was superseded is a no-op."""
    gen1 = await materialize_skills(store, "tester")
    gen2 = await materialize_skills(store, "tester")
    alpha = skills_dir() / "alpha"
    alpha.mkdir(parents=True)
    (alpha / "SKILL.md").write_text(SKILL_MD)

    await sync_skills_to_store(store, "tester", gen1)  # stale
    ns = user_skills_ns("tester")
    assert await resources.get_skill(store, "alpha", ns) is None

    await sync_skills_to_store(store, "tester", gen2)  # current
    assert await resources.get_skill(store, "alpha", ns) is not None


async def test_writeback_leaves_untouched_skills_alone(store):
    """Unchanged skills are never rewritten (no per-run store churn)."""
    ns = user_skills_ns("tester")
    await resources.create_skill(
        store, SkillIn(name="alpha", description="Alpha skill.", content="Body.\n"), ns
    )
    before = ((await store.aget(ns, "/alpha/SKILL.md")).value or {}).get("modified_at")

    gen = await materialize_skills(store, "tester")
    await sync_skills_to_store(store, "tester", gen)

    after = ((await store.aget(ns, "/alpha/SKILL.md")).value or {}).get("modified_at")
    assert after == before


async def test_materialize_without_scaffold_is_idempotent(store):
    """materialize on a bare workspace (no user scaffold) must not raise."""
    gen = await materialize_skills(store, "tester")
    assert gen is not None
    await sync_skills_to_store(store, "tester", gen)
    assert await resources.list_skills(store, user_skills_ns("tester")) == []
