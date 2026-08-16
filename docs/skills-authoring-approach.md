# Riset: Pendekatan Baru untuk Authoring & Persisten Skill

Status: **RISET — belum diimplementasikan.** Menjawab pertanyaan operasional
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

## 5. Catatan implementasi (untuk B — bila lanjut)

Lokasi: `src/app/services/workspace.py` (di samping `materialize_skills` /
`git_commit`), dipanggil dari `services/chat.py` di hook run-end yang sama
dengan `git_commit` (baris ~733).

1. **Snapshot saat materialize (store→disk).** Saat `materialize_skills`
   menulis cermin, catat hash tiap file (`<path> → sha256`) per user
   (mis. cache modul, bersamaan dengan `_workspace_lock(username)` yang
   sudah ada).
2. **Writeback di run-end (disk→store).** Setelah run selesai, diff
   `skills/` vs snapshot:
   - file baru/berubah → `resources.create_skill` / `update_skill` (validasi
     frontmatter = boundary validation ala YantrikDB; `name`/`description`
     wajib; path/name regex),
   - folder skill dihapus → `resources.delete_skill` (hanya jika memang ada
     di store — bedakan "dihapus" vs "belum termaterialisasi").
   - hash update hanya untuk yang berubah → clobber-edits antarrun minimal.
3. **Reuse resoures.* yang sudah ada** → store write + `agents.invalidate()`
   otomatis (cache graph dibuang, run berikutnya baca fresh). Tanpa rebuild.
4. **Retire/alias `publish_skill`.** Bisa dihapus (tulis langsung = persist),
   atau dipertahankan sebagai *fallback eksplisit* untuk draft di luar
   `skills/` (`tmp/`, `scripts/`). Rekomendasi: pertahankan dulu, dokumentasi
   ulang bahwa folder utamanya `skills/`.
5. **Frontmatter mentah.** Gap #2 di `skill-import-plan.md` (resources menulis
   ulang frontmatter dari name/description, field `license`/`metadata`
   hilang) ikut tersentuh: store value SKILL.md mentah, atau pertahankan apa
   adanya saat update. **Provenance** (gap #3) juga relevan: tandai
   `agent_edited_at`/`source: agent` pada writeback.
6. **Race dua run user sama** — pakai `_workspace_lock(username)` yang sama
   untuk snapshot+writeback, konsisten dengan materialize.

Keamanan: tidak ada jalur eksekusi baru; writeback hanya memanggil CRUD yang
ada. `allowed-tools` tetap metadata advisory (sesuai `skill-import-plan.md`).

---

## 6. Sumber

- LangChain — How we built Agent Builder's memory: <https://www.langchain.com/blog/how-we-built-agent-builders-memory>
- Claude Code — Extend Claude with skills: <https://code.claude.com/docs/en/skills>
- Ledgenter — Skills (hybrid sync contract): <https://docs.ledgenter.com/docs/skills.html>
- YantrikDB — Skill as Memory, Not Document: <https://yantrikdb.com/papers/skill-substrate/> (Zenodo DOI 10.5281/zenodo.20128887)
- Agent Skills spec: <https://agentskills.io/specification>
- SkillHone (self-improving skills): <https://github.com/Tencent/SkillHone>
