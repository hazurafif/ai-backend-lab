# Plan: Scheduled Agent Runs (cron-triggered proactive agents)

Status: **PLAN — belum diimplementasikan.** Blueprint backend untuk fitur
jadwal eksekusi agent: cron expression + agent config + prompt template,
dieksekusi otomatis tanpa permintaan chat, hasilnya masuk ke thread dan
memakai notifikasi run lifecycle yang sudah ada. Format blueprint mengikuti
`docs/knowledge-base-plan.md` dan `docs/skill-import-plan.md`.

---

## 1. Hasil Review Codebase (temuan)

Semua komponen runtime untuk run *background* sudah ada — yang hilang hanya
**pemicu** (trigger). Yang bisa dipakai ulang:

| Komponen | Lokasi | Catatan |
|---|---|---|
| Background run body (`_run_agent`: stream → persist → notify, never raises) | `services/chat.py` | Run hidup lebih lama dari HTTP stream; hanya `POST /threads/{id}/cancel` yang membatalkannya |
| `RunManager` (registry run aktif per thread, `Conflict` kalau thread sudah punya run) | `core/run_manager.py` | Memberi overlap policy **gratis**: schedule yang masih berjalan → `Conflict` |
| `NotificationHub` (`run_started/completed/interrupted/cancelled/failed` + replay) | `core/notification_hub.py` | Schedule run cukup menambah `schedule_id` ke payload event |
| Agent configs (model + prompt + skills + tools; resolve: own → global → `default`) | `services/agent_configs.py` | Target eksekusi: `agent` name |
| Resolusi agent + build graph (cached) | `api/v1/endpoints/chat.py::_resolve_agent`, `services/agent.py` | Scheduler memanggil path yang sama |
| Thread metadata + chat history persistence | `services/chat.py::_record_thread_metadata`, `_save_history` | Setiap scheduled run = satu thread (atau thread tetap) |
| Postgres + in-memory fallback, migrasi otomatis | `core/database.py`, `migrations/` | Tabel baru: `migrations/0008_create_schedules.sql` |
| Pola admin-per-user (`?username=`) | `/agent/skills?username=` | Dipakai ulang untuk manajemen schedule |
| Settings env-driven | `core/config.py` | `SCHEDULER_ENABLED`, `SCHEDULER_POLL_SECONDS` |

**Gap yang ditemukan (perlu dikerjakan):**

1. **Tidak ada tabel schedule / riwayat run** — belum ada migrasi `0008`.
2. **Tidak ada loop pemicu** — tidak ada proses yang menghitung "kapan
   jadwal berikutnya" dan memulai run. `run_manager`/`hub` bersifat
   in-process → scheduler juga harus in-process (single worker; lihat §4.4).
3. **Tidak ada dependency cron** — butuh `croniter` (pure Python, kecil)
   untuk parse cron + hitung `next_fire`.
4. **Tidak ada API surface** — belum ada router/schema schedule.
5. **Payload event belum membawa `schedule_id`** — frontend butuh pembeda
   run manual vs terjadwal (perubahan kecil di `services/chat.py`).

---

## 2. Keputusan Desain

### 2.1 Mekanisme trigger: loop in-process + `croniter`, bukan APScheduler

| Opsi | Keputusan |
|---|---|
| **APScheduler** (AsyncIOScheduler, misfire policy bawaan) | Heavy: proses lifetime terpisah, job store ganda, konflik dengan `run_manager` yang sudah jadi source of truth run aktif |
| **Loop asyncio kecil + `croniter`** (rekomendasi) | `croniter` = parse + `get_next()` saja; loop kita sendiri yang tidur sampai `next_run_at` (cap poll interval), lalu memanggil `tick(now)` — **mudah ditest tanpa tidur** |

`croniter` menambah 1 dependency kecil ke `pyproject.toml` (pure Python,
zero deps). Pola `Scheduler.tick(now: datetime) -> list[FiredRun]` membuat
inti logika murni-sinkron → unit test langsung, tanpa fake clock.

### 2.2 Model data: dua tabel, bukan metadata

- `schedules` — definisi jadwal (cron, agent, prompt, timezone, policy).
- `schedule_runs` — riwayat eksekusi (audit trail: status, thread, error).

Alternatif ditolak: menyimpan riwayat di `thread_metadata_ns` (store)
karena tidak ada endpoint filter "threads milik schedule X", dan riwayat
yang gagal sebelum thread dibuat (mis. model belum dikonfigurasi) tidak
akan pernah tercatat.

### 2.3 Policy per schedule

| Policy | Default | Perilaku |
|---|---|---|
| `overlap` | `skip` | Fire berikutnya **dilewati** kalau run jadwal ini masih aktif (gunakan `Conflict` dari `run_manager.start`) |
| `thread_mode` | `new` | `new` = thread baru per run; `fixed` = thread_id tetap (percakapan berlanjut, seperti resume) |
| `catchup` | `false` | App down saat fire → jadwal **tidak** dikejar; `next_run_at` dihitung ulang dari now |
| `hitl` | `pause` | Run terjadwal yang kena interrupt (mis. `INTERRUPT_ON_JSON`) menunggu approval seperti biasa; event `run_interrupted` tetap terkirim. (v1 tidak punya `abort` — dokumentasikan saja) |

### 2.4 Multi-worker (fly.toml scale > 1)

`run_manager` dan `hub` sudah in-process-only (tertulis di docstring-nya).
Scheduler mengikuti batasan yang sama, dengan satu pengaman: **Postgres
advisory lock** (`pg_try_advisory_lock`) dengan key konstan — worker yang
dapat lock menjalankan loop; lainnya idle. Mode in-memory (tanpa `DATABASE_URI`)
= asumsi single process, sama seperti status quo. Ini didokumentasikan,
bukan diselesaikan di v1.

---

## 3. Data Model (`migrations/0008_create_schedules.sql`)

```sql
CREATE TABLE IF NOT EXISTS schedules (
    id            UUID PRIMARY KEY,
    username      TEXT NOT NULL,              -- pemilik (per-user, seperti skills)
    name          TEXT NOT NULL,              -- label, mis. "daily briefing"
    agent         TEXT,                       -- NULL = agent default
    cron          TEXT NOT NULL,              -- 5-field POSIX cron
    prompt        TEXT NOT NULL,              -- template run input (lihat §5.2)
    timezone      TEXT NOT NULL DEFAULT 'UTC',
    thread_mode   TEXT NOT NULL DEFAULT 'new' CHECK (thread_mode IN ('new','fixed')),
    fixed_thread_id TEXT,                     -- wajib saat thread_mode='fixed'
    overlap_policy TEXT NOT NULL DEFAULT 'skip' CHECK (overlap_policy IN ('skip')),
    enabled       BOOLEAN NOT NULL DEFAULT TRUE,
    next_run_at   TIMESTAMPTZ,                -- dihitung ulang tiap fire/update
    last_run_at   TIMESTAMPTZ,
    last_status   TEXT,                       -- ok | failed | interrupted | skipped
    run_count     INTEGER NOT NULL DEFAULT 0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_schedules_due ON schedules (enabled, next_run_at);

CREATE TABLE IF NOT EXISTS schedule_runs (
    id          UUID PRIMARY KEY,
    schedule_id UUID NOT NULL REFERENCES schedules(id) ON DELETE CASCADE,
    thread_id   TEXT,                          -- NULL kalau gagal sebelum run
    status      TEXT NOT NULL,                 -- started | ok | failed | interrupted | skipped
    error       TEXT,
    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_schedule_runs_schedule ON schedule_runs (schedule_id, started_at DESC);
```

---

## 4. API (`schema/schedule_schema.py`, `api/v1/endpoints/schedules.py`)

Didaftarkan di `api/v1/routes.py` (`include_router(schedules_router, prefix="/schedules")`).
Semua endpoint user-scoped via `get_current_user`; admin dapat
`?username=` (pola `/agent/skills`).

| Endpoint | Deskripsi |
|---|---|
| `GET /schedules` | Daftar jadwal milik user (termasuk `next_run_at`, `last_status`) |
| `POST /schedules` | Buat; validasi cron + timezone + agent di **schema** (422) |
| `GET /schedules/{id}` | Detail (404 kalau bukan milik user) |
| `PATCH /schedules/{id}` | Update sebagian (cron/prompt/agent/enabled/...); recalc `next_run_at` |
| `DELETE /schedules/{id}` | Hapus; cascade ke `schedule_runs` |
| `POST /schedules/{id}/run` | **Manual trigger** ("run now") — masuk `schedule_runs` seperti fire normal |
| `GET /schedules/{id}/runs` | Riwayat eksekusi (limit/offset) |

Schema inti:

```python
class ScheduleIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)
    agent: str | None = Field(default=None, max_length=64)   # None = default
    cron: str = Field(..., max_length=32)                     # divalidasi croniter di schema
    prompt: str = Field(..., min_length=1)                    # template, lihat §5.2
    timezone: str = Field(default="UTC")                      # IANA, divalidasi zoneinfo
    thread_mode: Literal["new", "fixed"] = "new"
    fixed_thread_id: str | None = Field(default=None, max_length=128)
    overlap_policy: Literal["skip"] = "skip"
    enabled: bool = True
```

---

## 5. Service Design (`services/scheduling.py`, `services/cron.py`)

### 5.1 Loop (`services/scheduling.py::Scheduler`)

```
startup (lifespan, setelah persistence.start):
  if not settings.scheduler_enabled: skip
  lock = pg_try_advisory_lock(KEY) bila Postgres   # §2.4
  task = asyncio.create_task(_loop())

_loop:
  while True:
    fired = await tick(now_utc())                  # murni, sync, testable
    for f in fired: asyncio.create_task(_fire(f))  # tidak memblokir tick berikutnya
    sleep(min(60, detik sampai next_run_at terdekat + 1))

tick(now) -> [FiredSchedule]:
  SELECT * FROM schedules WHERE enabled AND next_run_at <= now
  untuk tiap row: mark next_run_at = croniter(now).get_next()  (kalau < now, hitung dari now — catchup=false)
  return rows

_fire(schedule):
  thread_id = baru (uuid) atau fixed_thread_id
  try:
    agent = await resolve_agent(agent_name, username)   # path chat yang sama; 503 kalau model belum dikonfigurasi
    active = run_manager.start(thread_id, username, agent_name)   # Conflict → skipped
    active.task = asyncio.create_task(_run_agent(active, agent, username,
        message=templated_prompt, thread_id=thread_id, resume=None, agent_name=agent_name))
    # catat schedule_runs started; event hub otomatis membawa schedule_id (lihat §5.3)
  except Exception as e:
    catat schedule_runs failed + last_status; hub.publish(failed, schedule_id)
```

Kunci desain: **`_fire` memakai `_run_agent` persis seperti `agent_stream`
dipakai sekarang** (tanpa SSE subscribers — run tetap berjalan, history +
notification tetap ditulis; `GET /threads/{id}/stream` bisa attach belakangan).
Tidak ada duplikasi logika streaming/persist.

### 5.2 Templating prompt

`prompt` mendukung placeholder sederhana, dirender per fire:

| Placeholder | Contoh hasil |
|---|---|
| `{{username}}` | `rafif` |
| `{{date}}` | `2026-06-15` (UTC) |
| `{{datetime}}` | `2026-06-15T08:00:00Z` |
| `{{threads_today}}` | jumlah thread user hari ini (opsional, v1.1) |

Implementasi: `str.replace` loop sederhana — **bukan** engine template
(Jinja = dependency + injection risk; prompt adalah input user sendiri).
`{{date}}` merujuk waktu fire, bukan waktu render.

### 5.3 Notifikasi & integrasi

- `_run_event` di `services/chat.py` ditambah field `schedule_id` (optional,
  di-pass lewat `agent_stream`/`_run_agent` kwarg baru) → frontend memakai
  badge "scheduled" dan membedakan run terjadwal.
- `GET /notifications` + `/notifications/stream` bekerja tanpa perubahan —
  schedule run hanyalah run biasa.
- `GET /threads/{id}/usage` bekerja — biaya per run terjadwal terlihat di
  thread-nya.

### 5.4 Settings (`core/config.py`)

```python
scheduler_enabled: bool = field(default_factory=lambda: _bool_env("SCHEDULER_ENABLED", True))
scheduler_poll_seconds: int = field(default_factory=lambda: int(os.environ.get("SCHEDULER_POLL_SECONDS", "60")))
```

---

## 6. Edge Cases & Keputusan

| Kasus | Keputusan |
|---|---|
| Cron tidak valid | 422 di schema (validasi `croniter.is_valid` + `zoneinfo.ZoneInfo`) |
| Timezone tidak valid | 422 (langsung gagal di schema, bukan di loop) |
| App down saat fire | `catchup=false`: `next_run_at` dihitung dari now; tercatat `skipped` di `schedule_runs` bila mau audit (opsional) |
| Run masih aktif saat fire berikutnya | `Conflict` dari `run_manager` → `skipped`, tanpa tumpukan run |
| Model belum dikonfigurasi | `_fire` gagal → `schedule_runs.failed` + event `run_failed`; jadwal tetap hidup, coba lagi fire berikutnya |
| Run kena HITL interrupt | Menunggu approval seperti run biasa; `run_interrupted` terkirim |
| Schedule dihapus saat run aktif | Run selesai normal; hanya definisi yang hilang (run tidak dibatalkan — konsisten dengan `DELETE /threads`) |
| DST | `croniter` + `zoneinfo` menghitung `next_run_at` di timezone jadwal; simpan `TIMESTAMPTZ` |
| Multi-worker | Advisory lock (§2.4); di luar scope v1 untuk leader failover |

---

## 7. Rencana Test (`tests/test_scheduling.py`, offline)

1. **cron**: `next_fire` untuk cron harian/mingguan, validasi input invalid (422).
2. **tick**: schedule `next_run_at <= now` terpilih; `next_run_at` maju;
   schedule disabled/`next_run_at > now` terlewat.
3. **fire**: scripted model (`tests/test_smoke.py::Scripted`) → thread dibuat,
   history terisi, `run_completed` dengan `schedule_id` di hub.
4. **overlap**: schedule yang run-nya masih aktif → `skipped` (pakai
   `slow_tool` dari `test_chat_features.py`).
5. **gagal**: model belum dikonfigurasi → `schedule_runs.failed` + event.
6. **CRUD**: create/list/patch/delete, 404 untuk milik user lain, cascade runs.
7. **manual trigger**: `POST /schedules/{id}/run` = fire tanpa ubah `next_run_at`.
8. **template**: `{{date}}`/`{{username}}` dirender benar.

Batas: loop `asyncio.sleep` tidak ditest langsung — `tick()` dan `_fire()`
diuji sebagai unit; loop hanya memanggil keduanya (pola yang sama dengan
`test_background_runs.py` yang menguji generator langsung).

---

## 8. Langkah Implementasi

1. `uv add croniter`; tambah `SCHEDULER_*` ke `core/config.py`.
2. `migrations/0008_create_schedules.sql` (dua tabel + index).
3. `services/cron.py` (wrapper `next_fire(cron, tz, now)`) + `schema/schedule_schema.py`.
4. `services/scheduling.py`: `Scheduler` (tick + fire) + hook `_run_event(schedule_id=...)` di `services/chat.py`.
5. `api/v1/endpoints/schedules.py` + register di `routes.py`.
6. Lifespan `main.py`: start/stop scheduler (advisory lock bila Postgres).
7. `tests/test_scheduling.py` + `ruff` + `pytest -q` hijau.
8. Update `docs/api-reference.md` (tabel endpoint) + README fitur + `.env.example`.

Estimasi: ~4-6 commit (migration, service+loop, API, integration, tests, docs),
mengikuti Conventional Commits (`feat(scheduling): ...`).
