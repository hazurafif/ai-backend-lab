# Riset: Pendekatan Baru untuk Authoring & Persisten Skill

Status: **TERIMPLEMENTASI (pendekatan B).** Riset yang melahirkan pendekatan B; implementasinya di `services/workspace.py` (`sync_skills_to_store` + snapshot generasi/hash di `materialize_skills`), dipanggil `services/chat.py` di akhir run, dengan `tests/test_workspace_sync.py` dan E2E `test_agent_writes_skill_directly_into_skills_dir`. Menjawab pertanyaan operasional
"kenapa agent membuat skill di `tmp/` bukan `skills/`?", lalu memetakan
pilihan pendekatan baru untuk alur authoring skill. Bacaan pendamping:
`docs/skill-import-plan.md` (import eksternal — gap #2/#3 relevan di sini),
`docs/api-reference.md`.

---

## 1. Masalah pada desain saat ini (terverifikasi dari kode)

Pipeline saat ini **store-first**: skill tinggal di LangGraph store (Postgres
di prod), dan `skills/` di disk hanyalah **cermin sekali-jalan**.

```
tinggal di store         /skills CRUD (frontend)+ publish_skill (agent)
        │  materialize_skills (setiap mulai run: BERSIHKAN + tulis ulang dari store)
        ▼
     skills/ di disk  ──►  SkillsMiddleware baca per run (hanya SKILL.md masuk konteks)
        │
        ▼ run selesai → git_commit(workspace) — tapi TIDAK ADA writeback ke store
```

Fakta yang terverifikasi (`src/app/services/workspace.py`, `agent.py`,
`api/v1/endpoints/skills.py`):

1. **Tidak ada read-only yang di-enforce.** `UserShellBackend._resolve_path`
   hanya memblokir traversal (`..`, `~`, keluar root). `write_file`/
   `edit_file` pada `/skills/...` berfungsi normal; `execute` juga bisa
   `cd skills` (deepagents: `virtual_mode` "does NOT restrict shell
   commands"). Yang "read-only" hanyalah instruksi di
   `DEFAULT_SYSTEM_PROMPT` (`core/constants.py`: "your skills/ ... read-only
   — Do NOT write new skills into skills/ directly").
2. **Tulisan langsung ke `skills/` dibuang diam-diam.** `materialize_skills`
   → `_replace_skill_files` menghapus isi `skills/` lalu menulis ulang dari
   store di awal setiap run. Skill yang ditulis agent langsung di `skills/`
   **hilang di run berikutnya** tanpa error. Ini alasan sebenarnya prompt
   melarangnya.
3. **Alur yang diminta prompt**: drafting di `tmp/<nama>/` (atau `scripts/`)
   → panggil tool `publish_skill` (membaca SKILL.md + bundled files dari
   folder draft, menulis ulang ke store via `resources.create_skill`).
   Draft di `tmp/` hanya bertahan lewat git workspace, bukan sebagai skill.
4. **Ketidakcocokan mental model.** Prompt menyebut `skills/` "read-only"
   tapi tool membiarkan menulis; hasilnya agent yang mengedit langsung di
   sana kehilangan kerjanya. Draft yang ditinggalkan di `tmp/` sering tidak
   di-`publish_skill` (diamati di workspace: `admin/scripts/deep-research/`
   ada, sementara versi ter-publish ada di `admin/skills/deep-research/`).

Esensi masalah: **dua kebenaran** — store (kebenaran, untuk frontend) dan
disk (yang agent lihat dan bisa tulis) — dengan sinkronisasi satu arah
(store→disk) dan kontrak *implisit* ("jangan tulis di sini") yang tidak
di-enforce.

---

## 2. Temuan ekosistem (riset web)

### a. Files-as-source-of-truth (Claude Code, LangChain Agent Builder)

- Claude Code: skill = `SKILL.md` di `.claude/skills/<nama>/`, dimuat dari
  filesystem; authoring-nya **manusia** (IDE, git) — agent hanya memakai.
- LangChain Agent Builder (blog resmi): memilih **file sebagai interface**
  memori/skill *justru agar agent bisa baca-modifikasi tanpa tool khusus —
  hanya akses filesystem*. Standar: `AGENTS.md` + agent skills.
  → Mendukung intuisi pengguna: kalau agent sudah punya file tools, jadikan
  file (bukan tool khusus) permukaan authoring skill.

### b. DB-as-truth + disk materialization dengan *sync contract* (Ledgenter)

- Desain hibrida: **DB tetap source of truth** (RBAC/RLS, versi, atribusi,
  search), disk hanyalah materialisasi untuk host (Claude Code) yang hanya
  bisa load dari disk.
- **Sync manifest ber-hash** mendeteksi drift — staleness "honest, tidak
  silent". Satu tool `skill_upsert` (create+update) keyed by slug/scope.
  - Di project ini arahnya terbalik: agent MENULIS di disk (bukan host yang
    load), tapi prinsipnya sama: *cermin disk boleh tertinggal asalkan
    drift-nya terdeteksi, dan DB tetap kebenaran*.

### c. Substrat DB-native untuk katalog skill (YantrikDB)

- Tesis: format `SKILL.md` dioptimalkan untuk authoring **manusia**; saat
  yang menulis adalah **agent massal**, failure mode naik: biaya token
  (YAML frontmatter ~36 token/skill — ablasi menunjukkan 1.49×→1.000× saat
  frontmatter dibuang), latency, dan yang terbesar — **admission skill
  invalid**: katalog filesystem menerima 68/70 (97%) skill malformed vs 0%
  untuk database dengan schema-validation saat write.
- Relevansi di sini: project **sudah** validasi ketat saat write
  (`SKILL_NAME_RE`, parse frontmatter wajib `name`+`description` di
  `resources.py`/`publish_skill`). Jadi pelajaran YantrikDB (validasi di
  boundary write) **sudah terpenuhi** — tidak perlu pindah ke DB-native.

### d. Skill yang berevolusi (riset self-improvement)

SkillHone/AutoSkill/MUSE: *unit of change = folder skill, tiap keputusan =
artifact git*, tapi dengan lifecycle validasi/evaluasi. Berguna sebagai arah
jauh (skim); bukan kebutuhan sekarang.

---

## 3. Pendekatan kandidat

| # | Pendekatan | Inti | Agent menulis di `skills/`? |
|---|---|---|---|
| **A** | Enforce read-only sungguhan | Backend menolak write/edit/delete/execute pada `/skills/*`; alur tmp→`publish_skill` satu-satunya | ❌ (error) |
| **B** | **Sync dua arah (rekomendasi)** | Store tetap kebenaran; `skills/` jadi permukaan authoring agent; run-end **writeback** disk→store (hanya yang berubah) | ✅ — dan persist |
| **C** | Disk = satu-satunya kebenaran | Lepas store untuk skill; frontend baca disk/git; agent tulis langsung jadi kebenaran | ✅ — jadi kebenaran |
| **D** | DB-native (YantrikDB penuh) | Agent author lewat tool `skill_upsert` ke DB; tanpa folder `skills/` untuk authoring | ❌ (pakai tool khusus) |

---

## 4. Perbandingan & rekomendasi

| Kriteria | A (enforce ro) | **B (sync 2 arah)** | C (disk=truth) | D (DB-native) |
|---|---|---|---|---|
| Mental model agent = perilaku | Ya (tool tolak tulis) | **Ya (tulis = persist)** | Ya | Tidak (butuh tool khusus) |
| Durabilitas lintas redeploy (Fly.io) | via store | **via store (Postgres)** | ❓ bertumpu bind-mount/git (push opsional — rapuh) | via store |
| Frontend skills list | store | **store (tak berubah)** | refactor → baca disk | store |
| Skill invalid tertangkap saat write | validasi di `publish_skill` | **validasi via `resources.create/update` (0%-admission ala YantrikDB)** | validasi saat agent pakai (terlambat) | validasi saat write |
| Ukuran perubahan | kecil | **sedang (satu hook sync)** | besar | besar |
| Menghapus draft-nyangkut | ❌ tetap ada | **✅ tulis langsung = ter-hubungi** | ✅ | ❌ |

**Rekomendasi: B — sync dua arah dengan store tetap sebagai kebenaran.**
Ini menjawab langsung pertanyaan awal ("kenapa tmp bukan skills?") dengan
cara yang membuat file tools berperilaku seperti yang user harapkan: agent
menulis skill di `skills/` → persist → muncul di frontend list → tersedia
di run berikutnya. Menahan semua keuntungan desain saat ini (durabilitas
Postgres, list frontend store-backed, validasi write-time), tanpa memaksa
dance `tmp`→`publish_skill` — karena tulisan langsung kini dipersist.

Mengapa bukan yang lain:
- **A** mempertahankan dance dan "menutup" fitur yang user sadari bisa
  dipakai; perbaikan gejala, bukan akar.
- **C** mengorbankan durabilitas lintas redeploy (jenis rapuh untuk
  bind-mount/git yang push-nya opsional).
- **D** menukar folder `skills/` (sudah materialisasi + dibaca middleware)
  dengan tool khusus — bertentangan dengan arah LangChain (file = interface)
  dan overkill untuk skala belasan skill. Satu pelajaran yang dipertahankan:
  validasi di boundary write *(sudah ada)*.

---

## 5. Implementasi (pendekatan B — SELESAI)

Lokasi: `src/app/services/workspace.py` (di samping `materialize_skills` /
`git_commit`), dipanggil dari `services/chat.py` di hook run-end yang sama
dengan `git_commit` (blok `finally`).

1. **Snapshot saat materialize (store→disk).** `materialize_skills` mencatat
   hash tiap file (`<relpath> → sha256`) per user di dalam
   `_workspace_lock(username)` yang sudah ada, beserta **generasi**, dan
   mengembalikan generasi itu ke `chat.py`.
2. **Writeback di run-end (disk→store).** `sync_skills_to_store` me-diff
   `skills/` vs snapshot: file baru/berubah → `resources.create_skill` /
   `update_skill` dengan `raw_markdown` (SKILL.md disimpan verbatim —
   frontmatter `license`/`metadata` tidak hilang; menutup gap #2 di
   `skill-import-plan.md`); folder skill dihapus → `resources.delete_skill`;
   bundled file dihapus → `resources.delete_skill_file`. Validasi frontmatter
   terjadi di boundary write (pelajaran YantrikDB).
3. **Reuse resources.\*** yang sudah ada → store write + `agents.invalidate()`
   otomatis (cache graph dibuang, run berikutnya baca fresh).
4. **`publish_skill` dipertahankan** sebagai fallback untuk draft di luar
   `skills/` (`tmp/`, `scripts/`); deskripsi tool + `DEFAULT_SYSTEM_PROMPT`
   diperbarui: folder authoring utama adalah `skills/`.
5. **Frontmatter mentah** — poin 2 (`raw_markdown`). Provenance (gap #3 di
   `skill-import-plan.md`) tetap pekerjaan terpisah.
6. **Race dua run user sama** — writeback memakai `_workspace_lock(username)`
   yang sama; guard generasi men-skip writeback yang materialisasinya sudah
   digantikan run lain (run itulah yang mempersist diff-nya sendiri).

Keamanan: tidak ada jalur eksekusi baru; writeback hanya memanggil CRUD yang

## 6. Sumber

- LangChain — How we built Agent Builder's memory: <https://www.langchain.com/blog/how-we-built-agent-builders-memory>
- Claude Code — Extend Claude with skills: <https://code.claude.com/docs/en/skills>
- Ledgenter — Skills (hybrid sync contract): <https://docs.ledgenter.com/docs/skills.html>
- YantrikDB — Skill as Memory, Not Document: <https://yantrikdb.com/papers/skill-substrate/> (Zenodo DOI 10.5281/zenodo.20128887)
- Agent Skills spec: <https://agentskills.io/specification>
- SkillHone (self-improving skills): <https://github.com/Tencent/SkillHone>
