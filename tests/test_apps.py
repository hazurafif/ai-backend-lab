"""Offline smoke tests for the FastMCP app servers in apps/.

Apps are standalone modules (not part of the src/app package), so they are
loaded via importlib and their tools are exercised directly — no network,
no MCP host.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

APPS_ROOT = Path(__file__).resolve().parents[1] / "apps"


def _load_app(rel_path: str):
    path = APPS_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


quiz = _load_app("quiz/quiz_server.py")


def test_quiz_submit_answer_grades_correctly() -> None:
    assert quiz.submit_answer(
        question_index=0, selected=2, correct=2, total_questions=5, current_score=0
    ) == {"is_correct": True, "new_score": 1, "answered_index": 0, "finished": False}


def test_quiz_submit_answer_wrong_keeps_score() -> None:
    assert quiz.submit_answer(
        question_index=1, selected=0, correct=1, total_questions=5, current_score=2
    ) == {"is_correct": False, "new_score": 2, "answered_index": 1, "finished": False}


def test_quiz_submit_answer_last_question_finishes() -> None:
    assert quiz.submit_answer(
        question_index=4, selected=3, correct=3, total_questions=5, current_score=4
    ) == {"is_correct": True, "new_score": 5, "answered_index": 4, "finished": True}


def test_quiz_ui_renders_prefab_app() -> None:
    ui = quiz.take_quiz(topic="World Capitals")
    assert ui.state["score"] == 0
    assert ui.state["current_question"] == 0


async def test_quiz_mcp_exposes_take_quiz_tool() -> None:
    tools = await quiz.mcp.list_tools()
    names = {t.name for t in tools}
    assert "take_quiz" in names
    # submit_answer is a UI-internal backend tool: reachable on the server
    # under its hashed name, but not advertised to the LLM.
    assert "submit_answer" not in names
