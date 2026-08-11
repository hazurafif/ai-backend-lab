"""Live API test against a running server (uvicorn on :8000)."""

import json
import urllib.parse
import urllib.request

BASE = "http://127.0.0.1:8000"


def req(method, path, body=None, token=None, timeout=180, form=False):
    headers = {}
    data = None
    if body is not None:
        if form:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
            data = urllib.parse.urlencode(body).encode()
        else:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = urllib.request.Request(BASE + path, method=method, headers=headers, data=data)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            return resp.status, resp.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def chat(token, message, thread_id=None):
    """POST /chat and parse SSE events."""
    body = {"message": message}
    if thread_id:
        body["thread_id"] = thread_id
    return _post_sse("/chat", body, token)


def _post_sse(path, body, token):
    status, text = req("POST", path, body=body, token=token)
    assert status == 200, (status, text[:300])
    events = []
    for block in text.split("\n\n"):
        ev, _, data = block.partition("\n")
        if ev.startswith("event: "):
            events.append((ev[7:], json.loads(data[len("data: ") :])))
    return events


# login (OAuth2 form)
status, text = req("POST", "/login", body={"username": "johndoe", "password": "secret"}, form=True)
token = json.loads(text)["access_token"]
print("[OK] login")

# first turn
events = chat(token, "Say hello and tell me your model name.")
assert any(e == "done" for e, _ in events), "no done event"
thread_id = next(d["thread_id"] for e, d in events if e == "done")
print(f"[OK] turn 1 done, thread={thread_id}")

# history after turn 1
status, text = req("GET", f"/threads/{thread_id}/messages", token=token)
msgs = json.loads(text)
assert status == 200 and isinstance(msgs, list) and len(msgs) >= 2, (status, text[:300])
print(f"[OK] history after turn 1: {len(msgs)} messages ({[m['type'] for m in msgs]})")

# follow-up turn in the SAME thread (write + read a file)
events = chat(
    token,
    "Create hello.txt containing 'hi from deepseek' using write_file, then read it back.",
    thread_id,
)
tools = sorted({d["name"] for e, d in events if e == "tool_start"})
print(f"[OK] turn 2 done, tools used: {tools}")
assert "write_file" in tools and "read_file" in tools, tools

# history after turn 2
status, text = req("GET", f"/threads/{thread_id}/messages", token=token)
msgs = json.loads(text)
assert len(msgs) >= 6, len(msgs)
types = [m["type"] for m in msgs]
print(f"[OK] history after turn 2: {len(msgs)} messages ({types})")

# thread listing
status, text = req("GET", "/threads", token=token)
threads = json.loads(text)
assert threads and threads[0]["thread_id"] == thread_id, threads
print(f"[OK] /threads: {len(threads)} thread(s), newest = {threads[0]['title'][:40]!r}")
