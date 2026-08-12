# MCP apps (FastMCP)

Standalone **FastMCP app servers** — Python MCP servers whose tools return
interactive UIs (Prefab components: buttons, cards, progress, charts) that
render inside any MCP host, plus backend tools the UI calls back into.

Ported from the upstream [fastmcp examples/apps](https://github.com/PrefectHQ/fastmcp/tree/main/examples/apps).

## Apps

| App | Tools | What it demonstrates |
| --- | --- | --- |
| `quiz/quiz_server.py` | `take_quiz` (UI), `submit_answer` (backend) | Multi-turn state: the LLM generates questions, the user answers via buttons, each click grades through a backend tool, the final score is sent back to the conversation |

## Running

### Browser dev UI (`fastmcp dev apps`)

```bash
uv sync          # installs fastmcp[apps] (dev group)
uv run fastmcp dev apps apps/quiz/quiz_server.py --mcp-port 8091
```

- MCP server: `http://127.0.0.1:8091/mcp` (auto-reload on save)
- Dev UI: `http://localhost:8080` — pick the `take_quiz` tool, fill in a topic,
  and play the rendered quiz in a new tab
- The left inspector panel shows the JSON-RPC traffic (including the hashed
  `submit_answer` calls the UI makes)

Default ports are 8000 (MCP) and 8080 (dev UI); pass `--mcp-port`/`--dev-port`
to change them — 8091 keeps clear of the backend app (:8000) and the weather
demo server (:8090).

### Plain streamable-HTTP server

```bash
uv run python apps/quiz/quiz_server.py   # streamable HTTP at http://127.0.0.1:8091/mcp
```

## Wiring into the backend agent

The apps speak the same streamable-HTTP protocol the backend already supports,
so register them like any other MCP server — via `mcp_servers.json`,
`MCP_SERVERS_JSON`, or the `POST /agent/tools` API:

```json
{
  "quiz": {
    "url": "http://127.0.0.1:8091/mcp",
    "transport": "streamable_http"
  }
}
```

Then ask the agent something like *"give me a quiz about Python"* — it will
call `take_quiz`, and the agent's reply streams a structured tool event the
frontend can render as the interactive quiz.

## How an app server is structured

```python
from fastmcp import FastMCP, FastMCPApp

app = FastMCPApp("Quiz")            # a UI app: named, owns its tools

@app.tool()                         # backend tool — called by the UI via CallTool
def submit_answer(...) -> dict: ...

@app.ui()                           # LLM-facing entry point — returns a PrefabApp
def take_quiz(...) -> PrefabApp: ...

mcp = FastMCP("Quiz Server", providers=[app])   # expose the app on an MCP server
```

- `@app.ui()` tools are the only ones advertised to the LLM; their result is a
  Prefab UI the host renders (the model sees a text summary).
- `@app.tool()` tools are UI-internal: the renderer calls them over the MCP
  server under a hashed name (`<hash>_<name>`), so the UI can grade, search,
  or persist without the LLM being in the loop.
- `providers=[...]` lets one server host several apps.

## Adding a new app

1. Create `apps/<name>/<name>_server.py` following the quiz layout.
2. Keep it self-contained (no imports from `src/app` — apps are standalone
   servers) and lint-clean: `uv run ruff check apps/`.
3. Add a row to the table above and a smoke test in `tests/test_apps.py`
   (import the module via `importlib`, call the tools directly).
