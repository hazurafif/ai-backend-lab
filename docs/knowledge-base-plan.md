# Rencana Fitur: Upload File/Folder + Knowledge Base (RAG)

Status: **PLAN — belum diimplementasikan.** Dokumen ini adalah blueprint backend
untuk fitur knowledge base di atas stack yang sudah ada (FastAPI + Deep Agents +
Postgres + MCP + SearXNG).

---

## 1. Hasil Review Codebase (temuan)

Yang **sudah ada** dan bisa dipakai ulang:

| Komponen | Lokasi | Catatan |
|---|---|---|
| Agent Deep Agents + filesystem tools (`ls`, `read_file`, `grep`, ...) | `services/agent.py` | Workspace per-user via `StoreBackend` (Postgres store) |
| CRUD skills + bundled files (admin-only, JSON body) | `api/v1/endpoints/agent.py`, `services/resources.py` | Hanya teks JSON, bukan upload file mentah |
| MCP servers + web_search (SearXNG) | `services/mcp.py`, `services/searxng.py` | Pola "tool tambahan" (`extra_tools`) sudah ada → bisa ditiru untuk tool KB |
| Persistence Postgres (checkpointer, store, chat history, users) | `core/database.py`, `migrations/*.sql` | Pool psycopg + runner migrasi otomatis |
| Auth (JWT, role admin/user) | `core/security.py`, `core/dependencies.py` | KB bersifat per-user → pakai `get_current_user` |
| SSE contract | `services/chat.py` | Tidak perlu diubah; tool calls sudah otomatis emit `tool_start`/`tool_end` |

**Gap yang ditemukan (perlu dikerjakan):**

1. **Tidak ada endpoint upload file** — multipart tidak dipakai di mana pun;
   skills API hanya menerima konten JSON, admin-only.
2. **Tidak ada vector store / RAG** — tidak ada parsing dokumen, chunking,
   embedding, atau semantic search.
3. **Bug/kekosongan isolasi user**: `services/agent.py` sudah punya
   `_user_namespace_factory` yang membaca `rt.context.user_id`, tapi
   `services/chat.py` **tidak pernah meneruskan** `user_id` ke
   `astream_events(..., context=...)`. Akibatnya semua workspace jatuh ke
   `"anonymous"`. Sudah diverifikasi bahwa `astream_events` (langgraph
   `Pregel`) mendukung parameter `context=` (lihat `langgraph/pregel/main.py`),
   jadi perbaikannya murah dan merupakan prasyarat untuk tool KB yang sadar-user.
4. **Belum ada folder `services/kb/`** untuk bisnis logic domain baru.

---

## 2. Keputusan Vector DB: Weaviate vs Alternatif

| Kriteria | **Weaviate** (rekomendasi) | pgvector (di Postgres yang ada) | Chroma / LanceDB |
|---|---|---|---|
| Infra tambahan | 1 service docker-compose + volume | 0 (ganti image `postgres:16` → `pgvector/pgvector:pg16`) | 0 (embedded / 1 service kecil) |
| Hybrid search (BM25F + vector) | **Bawaan** (`hybrid` query + alpha) | Manual (FTS Postgres + vector, fusion dikerjain sendiri) | Tidak |
| Multi-tenant / filter | Property filter + multi-tenancy resmi | Mudah via kolom relasional | Terbatas |
| Integrasi LangChain | `langchain-weaviate` (v1) | `langchain-postgres` | `langchain-chroma` |
| Backup/ops | Terpisah dari Postgres | Satu titik (backup Postgres saja) | Minim |
| Cocok untuk | RAG produksi, prioritas kualitas retrieval | Tim yang mau zero-infra, skala kecil | POC / local |

**Rekomendasi: Weaviate**, sesuai arah Anda — alasan utamanya hybrid search
(bobot teks + semantik) yang jauh lebih baik untuk RAG dokumen campuran
(kode, markdown, PDF), plus multi-tenancy resmi. Project ini sudah multi-service
(postgres + searxng di compose), jadi satu service lagi bukan beban baru.

**Catatan penting:** embedding **tidak** lewat modul Weaviate (mis.
`text2vec-transformers` = image besar + GPU/RAM), tapi di **app code** lewat
abstraksi `Embeddings` LangChain → default `OpenAIEmbeddings`
(`text-embedding-3-small`, provider OpenAI sudah dipakai), bisa ditukar ke
provider lain. Abstraksi `VectorStore` dipakai supaya Weaviate bisa
ditukar pgvector di kemudian hari tanpa mengubah service layer.

**Fallback (jika menolak service baru):** pgvector — plan ini tetap berlaku,
hanya `services/kb/vectorstore.py` yang beda implementasinya.

---

## 3. Arsitektur Usulan

```
Frontend ──multipart──▶ POST /kb/{id}/files (UploadFile + rel path)
                            │
                            ▼
                  api/v1/endpoints/kb.py   (router tipis: auth + validasi)
                            │
                            ▼
                  services/kb/ingest.py    (pipeline: simpan blob → parse → chunk → embed → upsert)
                            │
          ┌─────────────────┼──────────────────────┐
          ▼                 ▼                      ▼
  Postgres (migrasi      services/kb/parse.py   services/kb/vectorstore.py
  0004: kb + kb_docs)    pypdf/python-docx/     Weaviate collection
  blob bytea + status    bs4/plaintext          "KnowledgeChunk"
          │                 │                      │
          └─────────────────┴──────────┬───────────┘
                                       ▼
                            services/kb/search.py  (hybrid, filter owner+kb)
                                       │
                                       ▼
                       Tool agent: `search_knowledge_base`
                       (extra_tools di build_agent, sadar-user via
                       get_runtime().context.user_id — prasyarat: fix
                       context plumbing di services/chat.py)
```

Alur data:

1. **Upload** — multipart `file` + `path` (relatif, untuk folder) per file;
   validasi ekstensi + ukuran (default cap 25 MB/file); blob disimpan di
   Postgres (`kb_documents.content bytea`) supaya backup satu titik; status
   dokumen `pending`.
2. **Parse** — ekstraksi teks via `run_in_threadpool` (blocking CPU):
   `.pdf`→pypdf, `.docx`→python-docx, `.md`→markdown (dipisah header),
   `.csv`→csv module, `.html`→BeautifulSoup, `.txt`/kode→decode charset.
3. **Chunk** — `langchain-text-splitters`:
   `MarkdownHeaderTextSplitter` untuk .md, `RecursiveCharacterTextSplitter`
   default 1000 char / overlap 200 (configurable via env).
4. **Embed** — abstraksi `Embeddings`; prod `OpenAIEmbeddings`; test pakai
   `FakeEmbeddings` deterministik (tanpa API key, sesuai aturan offline test).
   Chunk diembat per dokumen, di-cache per hash teks (hemat biaya re-index).
5. **Simpan** — upsert ke Weaviate collection `KnowledgeChunk` dengan
   properties: `owner`, `kb_id`, `doc_id`, `path`, `chunk_index`, `content`;
   status dokumen → `ready` (atau `failed` + error).
6. **Search** — `hybrid` query Weaviate, filter `owner == user` (wajib) +
   `kb_id` (opsional), return top-k chunk + skor + path sumber.
7. **Agent tool** — `search_knowledge_base(query, kb_id?, top_k?)` didaftarkan
   via `extra_tools` di `build_agent`; user di-resolve dari runtime context;
   hasil diformat markdown dengan path sumber supaya agent bisa mengutip.

**Perubahan di luar modul baru (wajib, kecil):**

- `services/chat.py` — teruskan `context={"user_id": username}` ke
  `astream_events` (dan juga di resume path). Ini sekaligus mengaktifkan
  isolasi workspace yang sudah dirancang tapi mati.
- `services/agent.py` — daftarkan tool KB di `extra_tools` (dibangun di
  `build_agent`, stateful via singleton vector store service).
- `core/config.py` — section baru `Knowledge base` (lihat §6).
- `docker-compose.yml` — service `weaviate`.
- `migrations/0004_create_kb_tables.sql` — tabel baru.
- `core/database.py` — akses tabel KB via pool yang sudah ada
  (pola sama seperti `UserStore`/`ChatHistoryStore`).

---

## 4. Rancangan API (REST, auth `get_current_user` — bukan admin)

Prefix: `/kb` (router baru `api/v1/endpoints/kb.py`, daftar di `routes.py`).

| Method | Path | Fungsi |
|---|---|---|
| POST | `/kb` | Buat KB `{name, description?}` → `KBOut` |
| GET | `/kb` | List KB milik user (dengan stats dokumen/chunk) |
| GET | `/kb/{id}` | Detail KB + status |
| PATCH | `/kb/{id}` | Rename / ganti deskripsi |
| DELETE | `/kb/{id}` | Hapus KB: dokumen + semua vektor Weaviate (cascade) |
| POST | `/kb/{id}/files` | **Upload multipart**: pasangan `file` + `path` (relatif, mendukung folder), bisa multi-file sekali request |
| GET | `/kb/{id}/files` | List dokumen + status ingest (`pending`/`processing`/`ready`/`failed`) |
| GET | `/kb/{id}/files/{doc_id}` | Detail dokumen (nama, path, ukuran, status, chunk_count) |
| DELETE | `/kb/{id}/files/{doc_id}` | Hapus dokumen + vektornya |
| POST | `/kb/{id}/reindex` | Re-embed semua dokumen (mis. ganti model embedding) |
| GET | `/kb/{id}/search?q=&limit=` | Endpoint uji/coba search hybrid (buat debugging frontend) |

**Semantics folder upload:**
- Single file: satu pasangan `(file, path="nama.md")`.
- Folder: client (HTML5 `webkitdirectory`) mengirim semua file masing-masing
  dengan `path` relatif (`subfolder/nama.md`); backend menyimpan struktur.
- Opsional fase 2: `.zip` di-extract server-side (guard: path traversal,
  batas jumlah entri + total ukuran).

**Skema baru** (`schema/kb_schema.py`, Pydantic v2):
`KBIn`, `KBOut`, `DocumentOut`, `UploadResult` (per file: ok/error),
`SearchHit {text, path, score, chunk_index}`, `SearchOut {hits}`.
Validasi nama KB (regex, seperti `SKILL_NAME_RE`), path relatif aman
(tolak `../`, absolut, karakter berbahaya).

---

## 5. Skema Database (migrasi `0004_create_kb_tables.sql`)

```sql
CREATE TABLE IF NOT EXISTS kb (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner       TEXT NOT NULL,              -- username, owner-only access
    name        TEXT NOT NULL,
    description TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (owner, name)
);

CREATE TABLE IF NOT EXISTS kb_documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    kb_id       UUID NOT NULL REFERENCES kb(id) ON DELETE CASCADE,
    owner       TEXT NOT NULL,
    path        TEXT NOT NULL,              -- relatif, mendukung folder
    mime_type   TEXT,
    size_bytes  BIGINT NOT NULL,
    content     BYTEA NOT NULL,             -- blob mentah (backup satu titik)
    status      TEXT NOT NULL DEFAULT 'pending',  -- pending|processing|ready|failed
    error       TEXT,
    chunk_count INT NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (kb_id, path)
);

CREATE INDEX IF NOT EXISTS ix_kb_documents_kb ON kb_documents (kb_id);
CREATE INDEX IF NOT EXISTS ix_kb_documents_owner ON kb_documents (owner);
```

Catatan: teks hasil ekstraksi **tidak** disimpan di Postgres (vektor Weaviate
sudah menyimpan konten chunk); re-index cukup dari blob via parse ulang.
`gen_random_uuid()` tersedia di Postgres 13+ (`pgcrypto` bawaan sejak 13).

---

## 6. Konfigurasi Baru (`.env` / `core/config.py`, section `Knowledge base`)

```env
# --- Knowledge base (RAG) ---
WEAVIATE_URL=http://localhost:8093        # service baru di compose
# WEAVIATE_API_KEY=...                    # opsional, auth
EMBEDDINGS_MODEL=text-embedding-3-small   # dipakai via OpenAIEmbeddings
# EMBEDDINGS_BASE_URL=...                 # custom OpenAI-compatible endpoint
KB_MAX_FILE_SIZE_MB=25                    # cap per file
KB_ALLOWED_EXTENSIONS=.md,.txt,.pdf,.docx,.csv,.html,.json,.py,.js,.ts,.go,.rs,.java,.sql
KB_CHUNK_SIZE=1000                        # chars
KB_CHUNK_OVERLAP=200
KB_MAX_UPLOAD_BATCH=100                   # file per request
```

---

## 7. Strategi Testing (tetap offline, tanpa API key)

- **Vector store**: `services/kb/vectorstore.py` menerima `VectorStore`
  injected → test pakai `InMemoryVectorStore` (langchain-core) +
  `FakeEmbeddings` (hash → vektor deterministik). Integration test Weaviate
  hanya di `scripts/` (live, di-skip pytest).
- **Upload endpoint**: `httpx.AsyncClient` + `ASGITransport`, multipart
  sungguhan, override auth via `app.dependency_overrides` (aturan AGENTS.md).
- **Parser**: file fixture kecil (pdf/docx/txt/md) di `tests/fixtures/`.
- **Tool agent**: extend `tests/test_smoke.py` dengan scripted model —
  panggil tool KB, verifikasi tool call + jawaban.
- **Isolasi user**: tes bahwa user A tidak bisa melihat/query KB user B
  (403/404) dan `context.user_id` benar sampai ke tool.

---

## 8. Fase Implementasi (urutan commit)

| Fase | Isi | Deliverable |
|---|---|---|
| **0. Infra** | `weaviate` di docker-compose, config baru, migrasi 0004, `services/kb/__init__`, schema kb | App tetap boot tanpa Weaviate (degradasi: KB API 503/skip) |
| **1. CRUD KB** | Tabel + `KbStore` (pola `UserStore`), endpoint `/kb` CRUD | KB bisa dibuat/dilihat/dihapus |
| **2. Upload & ingest** | Multipart + blob store + parser + chunker + status dokumen | File/folder masuk, teks terekstrak, status terpantau |
| **3. Embed & search** | `vectorstore.py` (Weaviate + fake), embedding, endpoint `/kb/{id}/search` | Semantic/hybrid search jalan (bisa diuji via REST) |
| **4. Agent tool** | Fix `context.user_id` di chat.py, tool `search_knowledge_base`, daftar di `build_agent` | Agent bisa jawab dari KB dalam chat (tool events otomatis masuk SSE) |
| **5. Hardening** | Reindex, hapus cascade, limit/quota, zip folder (opsional), README + .env.example | Fitur lengkap + terdokumentasi |

---

## 9. Risiko & Mitigasi

| Risiko | Mitigasi |
|---|---|
| Biaya embedding API (rate limit/cost) | Batch embed per dokumen; cache per hash teks; model murah (text-embedding-3-small) |
| File besar memblok event loop | Parsing via `run_in_threadpool`; cap 25 MB; status `processing` async (fase 5 bisa pindah ke worker task) |
| Zip bomb / path traversal (upload zip) | Guard: ekstrak aman, batas entri + ukuran, validasi path relatif |
| Test offline rusak karena butuh Weaviate | `InMemoryVectorStore` + `FakeEmbeddings`; Weaviate hanya live script |
| Weaviate mati saat runtime | Search tool return error informatif; app tetap jalan (degradasi, bukan crash) |
| `rt.context` tidak sampai ke tool (ketergantungan pada runtime langgraph) | Verifikasi di fase 4 via scripted test; fallback: argumen tool wajib `kb_id` + filter owner di service layer (defense-in-depth, bukan trust tool args saja) |
| Multi-replica (blob di Postgres, bukan disk lokal) | Sudah aman: semua state di Postgres + Weaviate |

---

## 10. Yang TIDAK termasuk (scope v1)

- OCR untuk PDF hasil scan / gambar (fase lanjutan: `pytesseract`).
- Preview/download file mentah dari browser (blob bisa di-serve nanti via
  `GET /kb/{id}/files/{doc_id}/content`).
- Permissions sharing KB antar-user (owner-only dulu; admin bisa manage semua).
- Background job queue (Celery/arq) — ingest sinkron per request dulu, cukup
  untuk ukuran cap 25 MB; pindah ke task queue jika kebutuhan naik.
