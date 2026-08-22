# Riset: Kandidat Fitur untuk AI Backend

Status: **RISET — belum ada keputusan implementasi.** Dokumen ini memetakan
kandidat fitur berikutnya untuk stack FastAPI + Deep Agents + Postgres + MCP,
berdasarkan (1) review kapabilitas `deepagents 0.7.5` yang terpasang (banyak
param `create_deep_agent` belum diekspos), (2) tren platform agent infra
(LangChain, OpenAI, Microsoft, AWS, ekosistem MCP), dan (3) kesenjangan di
codebase. Blueprint pendamping (tetap valid): `docs/scheduled-runs-plan.md`,
`docs/skill-import-plan.md`, `docs/knowledge-base-plan.md`.

---

## 1. Snapshot kondisi sekarang

| Sudah ada | Belum ada |
|---|---|
| SSE streaming v3 (`astream_events(version="v3")`), AI SDK bridge | Cron/scheduled runs (hanya plan) |
| Checkpoint+history+users di Postgres, fallback in-memory | Skill import dari skills.sh/GitHub (hanya plan) |
| MCP servers (HTTP/stdio), per-user BYOK connections | Rate limit & kuota chat (hanya login yang di-limit) |
| RAG KB (Weaviate hybrid + rerank + query rewrite) | Trace/audit trail tool-call yang persisten |
| Skills per-user (skill-creator layout), agent configs | Feedback/evals loop; eval offline untuk skenario |
| HITL (interrupt + resume), run lifecycle notifications | Input multimodal (gambar/video) ke dalam context model |
| Web search SearXNG, sharing thread, usage/cost report | `permissions`/`response_format` deepagents di agent schema |
| Summarization + prompt-caching middleware (auto, bawaan deepagents) | Semantic memory (vektor atas `memories/`) |

Catatan penting dari review `create_deep_agent` (0.7.5): banyak kapabilitas
bawaan sudah aktif otomatis — summarization middleware di semua jalur graph
(`graph.py:758,843`), prompt caching untuk Anthropic/Bedrock/Fireworks
(`append_prompt_caching_middleware`), files/state via `DeltaChannel` (storage
O(N) bukan O(N²)). Yang **belum** diekspos: `permissions`, `response_format`,
`state_schema`, `context_schema`, `name`, `cache`.

---

## 2. Fitur yang sudah direncanakan — validasi ekosistem

Kedua blueprint ini tetap relevan dan sejalan dengan pola industri; tidak ada
alasan untuk menunda selain prioritas.

### 2.1 Scheduled agent runs (cron) — `docs/scheduled-runs-plan.md`
- Pola "Scheduled Agent" sudah jadi pola katalog resmi (Agent Patterns Catalog:
  scheduler + durable state untuk agent state).
- Praktik industri menyoroti hal yang persis sudah dianalisis di plan:
  timezone/DST, overlap guard (sudah gratis via `RunManager` → `Conflict`),
  missed-window catch-up, idempotensi per tick.
- Komponen runtime (background run, `NotificationHub`, cache graph, resolusi
  agent) semua sudah ada — tinggal trigger + tabel `schedules` + CRUD admin.

### 2.2 Skill import (skills.sh / open Agent Skills) — `docs/skill-import-plan.md`
- skills.sh (Vercel Labs) tetap ekosistem aktif; `npx skills add` adalah
  standar de-facto untuk install skill.
- Progresi alami: `POST /skills/import` (URL/repo → ekstrak → store skill +
  provenance) lalu `POST /skills/import/update` untuk update versi.
- Tanpa dependensi baru: zip extraction guard sudah ada di jalur KB `/zip`.

---

## 3. Kandidat baru — quick wins (kapabilitas deepagents yang belum diekspos)

### 3.A Ekspos `permissions` (FilesystemPermission) di agent configs
- **Apa**: `create_deep_agent(permissions=[...])` menerima aturan
  `FilesystemPermission` (allow/deny/interrupt per path glob) yang diterapkan
  `FilesystemMiddleware` di level tool, termasuk subagent (inherit).
- **Kenapa**: `EXECUTE_ENABLED=false` sekarang adalah refuse global; HITL via
  `interrupt_on` adalah per-tool binary. Permissions memberi guardrail
  per-path yang persis dibutuhkan untuk mode "trusted tapi dibatasi":
  mis. `allow: /workspace/**`, `deny: /etc/**`, `interrupt: .env*`.
  Ini jawaban industri atas governance (Bedrock AgentCore policy controls,
  Microsoft Agent Framework) tanpa service eksternal.
- **Sentuh**: `schema/agent_config_schema.py` (field `permissions`),
  `services/agent.py::build_agent` (teruskan param), mapping ke
  `interrupt_on` otomatis via `_build_interrupt_on_from_permissions`.
- **Estimasi**: kecil (1-2 hari). Tidak ada migrasi DB.

### 3.B Ekspos `response_format` (structured output) per agent config
- **Apa**: parameter `response_format` + `state_schema` di deepagents untuk
  output terstruktur (JSON schema) untuk main agent.
- **Kenapa**: berguna untuk task non-UI (extract, report, transform) —
  hasil langsung jadi `dict` valid, bukan teks bebas; cocok dipasangkan dengan
  scheduled runs (output run = structured report).
- **Sentuh**: agent config schema + `build_agent`; validasi JSON schema di
  input; uji offline dengan scripted model.
- **Estimasi**: kecil-sedang. Risiko: interaksi dengan HITL/subagents perlu
  dicek di test.

### 3.C Toggle & laporan prompt caching / summarization
- **Apa**: middleware caching + summarization aktif otomatis tapi tidak
  terlihat dan tidak bisa diatur per agent config (threshold, model khusus
  untuk summary).
- **Kenapa**: laporan `/threads/{id}/usage` sudah ada — menambahkan indikator
  "cache hit / context reduced" membuat biaya bisa diprediksi; per-config
  `summarization: {"threshold": ..., "model": ...}` untuk agent long-horizon.
- **Estimasi**: kecil untuk laporan; sedang bila jadi field config.

---

## 4. Kandidat baru — nilai menengah

### 4.1 Rate limit & kuota chat per user (token/cost budget)
- **Kenapa**: satu-satunya rate limiter adalah login (`core/rate_limit.py`,
  sliding window). Semua platform agent produksi memberlakukan gating di
  inference (OpenAI AgentKit, Azure governance). `services/session_stats.py`
  sudah menghitung cost — tinggal dipakai untuk budget bulanan per user
  + limit RPS chat.
- **Sentuh**: extend `core/rate_limit.py` (factory per-namespace), dependency
  baru di `api/v1/endpoints/chat.py`, kolom budget di user preferences.
- **Estimasi**: sedang. Risiko: false-positive rate limit di live chat —
  butuh konfigurasi longgar + header `Retry-After`.

### 4.2 Persist trace tool-call (audit trail) + endpoint `/threads/{id}/trace`
- **Kenapa**: arus event sudah lengkap (run_manager + NotificationHub + SSE
  `tool_*`), tapi tidak disimpan — sesudah stream selesai tidak ada jejak
  tool apa yang dipanggil, berapa lama, input/output apa, dan keputusan HITL
  apa yang diambil. "Full session traces for every run" adalah baseline
  observability agent (LangSmith, Guild AI, MLflow AI Monitoring).
- **Sentuh**: tabel `run_traces` (tool_call_id, tool, args, result, durasi,
  decision HITL), tulis dari consumer existing di `services/chat.py`,
  endpoint GET (owner + admin). Tanpa OTel/LangSmith — self-contained.
- **Estimasi**: sedang. Ini juga jadi fondasi eval (4.3).

### 4.3 Feedback loop + eval harness offline
- **Kenapa**: arah industri = eval code-first (OpenAI menarik Evals →
  code; Bedrock AgentCore Evaluations baru GA). Repo sudah punya budaya
  `tests/` + `scripts/` (kb_eval.py) — tinggal (a) `POST /threads/{id}/feedback`
  (👍/👎 + catatan) disimpan di chat_messages, (b) script eval multi-skenario
  yang mereplay prompt ke `build_agent` scripted dan menilai keberhasilan run.
- **Sentuh**: migrasi kolom feedback, endpoint kecil, `scripts/agent_eval.py`.
- **Estimasi**: sedang (eval) + kecil (endpoint).

### 4.4 Chat multimodal (gambar/video → model context)
- **Apa**: upload sekarang masuk filesystem (agent baca via tool), bukan ke
  context model. deepagents sudah punya dukungan multimodal filesystem
  (`middleware/_video.py`).
- **Kenapa**: kebutuhan UI paling umum berikutnya setelah teks; AI SDK
  protocol sudah mendukung content parts.
- **Sentuh**: `schema/chat_schema.py` (content parts / image_url),
  parsing upload di chat endpoint, uji dengan model vision.
- **Estimasi**: sedang. Risiko: schema breaking change untuk klien lama —
  perlu backward-compatible `message` string.

### 4.5 Semantic memory di atas `memories/`
- **Apa**: sekarang memory = file AGENTS.md (memory middleware, per-user).
  Level-2: injeksi vector search atas `memories/**` pakai Weaviate yang sudah
  ada (infra KB reusable), atau summarization thread lama → memory entry.
- **Kenapa**: pattern memory terkuat di katalog agent; biaya tambahan kecil
  karena Weaviate sudah jalan.
- **Estimasi**: sedang.

---

## 5. Rekomendasi prioritas

| Prioritas | Fitur | Alasan |
|---|---|---|
| 1 | **Scheduled runs** (plan ada) | Blueprint lengkap, komponen sudah ada, nilai tinggi (proactive agents) |
| 2 | **Skill import** (plan ada) | Blueprint lengkap, ekosistem aktif, tanpa dependensi baru |
| 3 | **Permissions per agent config** (3.A) | Guardrail deepagents-native, 1-2 hari, langsung naikkan keamanan per-path |
| 4 | **Trace persistence + feedback** (4.2+4.3) | Fondasi observability & eval; memakai arus event yang sudah ada |
| 5 | **Chat rate limit & budget** (4.1) | Proteksi produksi; pasangan scheduled runs (autonomous = boros) |
| 6 | **Structured output** (3.B) | Melengkapi scheduled runs + agent API |
| 7 | Multimodal, semantic memory | Nilai UX, effort lebih besar; tunggu permintaan konkret |

Prinsip: dahulukan fitur yang memakai arsitektur yang sudah ada (event stream,
RunManager, NotificationHub, store), bukan yang menarik dependensi baru.

---

## Lampiran: sumber riset

- Deep Agents v0.7 — leaner harness, 65% lebih hemat base tokens (langchain.com/blog/deep-agents-v0-7)
- Deep Agents v0.6 — code interpreter, harness profiles, streaming v3, DeltaChannel, ContextHubBackend (langchain.com/blog/deep-agents-0-6)
- Scheduled Agent — Agent Patterns Catalog (agentpatternscatalog.org/patterns/scheduled-agent)
- Proactive agents: cron-driven patterns, overlap & catch-up (tianpan.co, solana.garden, calegix.com)
- MCP spec 2025-11-25 — OAuth/OIDC discovery, elicitation, tool icons (modelcontextprotocol.io)
- OpenAI AgentKit; Bedrock AgentCore Evaluations & policy controls (aws.amazon.com/blogs/aws)
- Observability/evals: LangSmith, Guild AI traces, MLflow AI Monitoring, Fiddler AI lifecycle evals
- Eval code-first: OpenAI mengakhiri Evals product → Agent SDK + code (openai.com/index/introducing-agentkit)