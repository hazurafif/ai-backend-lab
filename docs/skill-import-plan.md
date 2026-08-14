# Rencana Fitur: Import Skill Eksternal (skills.sh / Agent Skills)

Status: **PLAN — belum diimplementasikan.** Blueprint backend untuk fitur
import skill dari ekosistem open Agent Skills (skills.sh, repos GitHub, URL
langsung) ke store skill yang sudah ada, plus provenance untuk update.
Bacaan pendamping: `docs/knowledge-base-plan.md` (contoh blueprint lain),
`docs/api-reference.md` (kontrak endpoint saat ini).

---

## 1. Hasil Riset Ekosistem

### skills.sh (Vercel Labs)

skills.sh adalah direktori + leaderboard open source (MIT, dirilis Feb 2026)
untuk agent skills. Skill diindeks dari repos GitHub publik, diinstall dengan
satu perintah:

```bash
npx skills add vercel-labs/agent-skills
```

Fakta kunci dari CLI (`vercel-labs/skills`, repo terbuka):

| Aspek | Detail |
|---|---|
| **Format source** | `owner/repo` (shorthand GitHub), URL GitHub penuh, path `…/tree/<branch>/<path>` (skill langsung di dalam repo), URL GitLab, URL git apa pun (HTTPS/SSH), path lokal, dan **URL download langsung** (`SKILL.md` tunggal atau archive `.zip`/`.tar`/`.tar.gz`/`.tgz`, tanpa perlu ekstensi) |
| **Repo = koleksi skill** | Container discovery: `SKILL.md` bisa di root, `skills/`, `plugins/`, atau container bersarang; dipilih dengan `--skill <nama>` atau dilihat dengan `--list` |
| **Batas ukuran** | Download ≤ 10 MiB, hasil ekstrak ≤ 25 MiB, ≤ 1000 file (bisa di-override env) |
| **Auth private repo** | Git credential helper / `gh` CLI / SSH; `GITHUB_TOKEN` opsional untuk API |
| **Command lain** | `skills find` (pencarian interaktif/keyword), `skills list`, `skills update`, `skills remove`, `skills init` |

Registry lain yang setara (semua konvergen ke format Agent Skills yang sama):
**agentskills.io** (spesifikasi resmi), **Smithery** (sekarang bagian dari
Arcade.dev), **ClawHub**, **MCP Market**, dan meta-registry `c2s/agent-skills-hub`.

### Format Agent Skills (spesifikasi agentskills.io / anthropics/skills)

- Skill = direktori berisi `SKILL.md` (wajib) + folder opsional `scripts/`,
  `references/`, `assets/`.
- `SKILL.md` = YAML frontmatter + body markdown (instruksi).
- Frontmatter:
  - `name` — wajib, ≤ 64 char, lowercase alphanumeric + hyphen, **harus sama
    dengan nama direktori**.
  - `description` — wajib, ≤ 1024 char, menyebut apa yang skill lakukan dan
    kapan dipakai.
  - `license`, `compatibility` (≤ 500 char), `metadata` (map), `allowed-tools`
    (eksperimental) — opsional.
- Progressive disclosure: metadata dimuat saat startup, body saat skill
  diaktifkan, file pendukung on-demand. SKILL.md disarankan < 500 baris.
- Validasi resmi: library `skills-ref` (agentskills/agentskills).

### Postur keamanan (penting untuk review)

Skill = instruksi + file; agent **mengeksekusi `scripts/`** saat skill
dipanggil. Menginstall skill tak terpercaya adalah keputusan supply-chain
(model trust yang sama dengan npm/pip), ditambah permukaan prompt-injection
karena isi SKILL.md masuk ke konteks model. Registry seperti skills.sh tidak
melakukan kurasi keamanan (hanya statistik install + audit pihak ketiga).

---

## 2. Kesesuaian dengan Codebase

Yang **sudah ada** dan bisa dipakai ulang:

| Komponen | Lokasi | Catatan |
|---|---|---|
| Store skill global (admin): `/{name}/SKILL.md` + bundled files | `api/v1/endpoints/agent.py`, `services/resources.py` | Layout sudah persis spek Agent Skills |
| Store skill per-user | `api/v1/endpoints/skills.py`, `resources.py` (namespace `("user","skills",<user>)`) | Cocok untuk import "pribadi" seperti `npx skills add` lokal |
| Validasi nama/path | `schema/agent_schema.py` (`SKILL_NAME_PATTERN`, `SKILL_FILE_PATH_PATTERN`) | Regex identik dengan constraint spek |
| HTTP client async | `services/searxng.py` (pola `httpx.AsyncClient` + inject untuk test) | Pola ditiru untuk download URL langsung |
| Invalidation agent | `request.app.state.agents.invalidate()` | Dipakai di setiap mutasi skill — import tinggal ikut |
| Eksekusi blocking di threadpool | konvensi `run_in_threadpool` | `git clone`/extract zip = blocking → wajib threadpool |

**Gap yang ditemukan (perlu dikerjakan):**

1. **Tidak ada resolusi/akuisisi source** — belum ada kode yang bisa
   fetch skill dari luar (git clone, download URL, discovery container).
2. **`resources.py` menulis ulang frontmatter** dari `name/description/
   content` (`_skill_markdown`) — import butuh menyimpan `SKILL.md` mentah
   agar field `license`/`metadata`/`allowed-tools` tidak hilang.
3. **Tidak ada provenance** — store value hanya `content/encoding/timestamps`;
   belum ada jejak `source`, `commit_sha`, `imported_at` untuk update/audit.
4. **`allowed-tools` tidak di-enforce** oleh SkillsMiddleware project ini —
   perlu didokumentasikan sebagai metadata saja (advisory).

---

## 3. Desain API

Mirror CRUD yang ada, dua scope:

```
POST /skills/import            # user-scoped — user terautentikasi mana pun
POST /agent/skills/import      # global — admin only
POST /skills/import/preview    # dry-run: daftar skill yang ditemukan di source, tanpa install
```

Body `SkillImportRequest` (baru di `schema/agent_schema.py`):

```json
{
  "source": "vercel-labs/agent-skills",
  "skill": "web-design-guidelines",
  "mode": "create",
  "target_name": "nama-lain"
}
```

| Field | Wajib | Keterangan |
|---|---|---|
| `source` | ya | `owner/repo`, URL GitHub/GitLab/git, path tree, atau URL download langsung |
| `skill` | tidak | Filter: ambil satu skill dari repo koleksi (tanpa ini + repo koleksi → 409/daftar) |
| `mode` | tidak | `create` (default, 409 jika sudah ada) \| `replace` (update + cek provenance) |
| `target_name` | tidak | Rename saat import (default: nama dari frontmatter) |

Response: `SkillOut` yang sudah ada + field baru opsional `provenance`
(`source`, `resolved_url`, `commit_sha`, `imported_at`) — backward compatible,
default kosong untuk skill lama.

Preview response: `SkillImportPreview` — daftar `{name, description, license,
path_di_repo, total_files, total_bytes}` per skill yang ditemukan.

Update lanjutan (fase 3): `POST /skills/{name}/update-from-source` — fetch
ulang source, bandingkan `commit_sha`, replace jika berubah.

---

## 4. Desain Service: `services/skill_import.py`

Pipeline 6 langkah (pola: semua I/O blocking di threadpool, sesuai aturan
async project):

1. **Resolve** — parse string source: shorthand `owner/repo` → GitHub;
   deteksi host untuk URL (github.com, gitlab.com, host git lain); deteksi
   path tree; sisanya = URL download langsung.
2. **Acquire** — untuk source git: `git clone --depth 1` ke temp dir (seragam
   untuk GitHub/GitLab/SSH, sama dengan perilaku CLI skills.sh). Untuk URL
   langsung: `httpx` download dengan batas ukuran (10 MiB) + dukungan
   zip/tar. Temp dir dihapus di `finally`.
3. **Discover** — walk tree cari `SKILL.md` di kedalaman ≤ 2 (root, `skills/`,
   `agentskills/`, `plugins/`); terapkan filter `skill`; jika repo koleksi
   tanpa filter → kembalikan daftar (atau 409 minta preview dulu).
4. **Parse + validasi** — parse YAML frontmatter; validasi `name` (pattern
   yang ada, harus cocok dengan nama direktori), `description`; enforce caps
   (SKILL.md ≤ 100 KB, total ≤ 25 MB / 500 file) — guardrail terhadap input
   bermusuhan.
5. **Persist** — panggil `resources.create_skill` / `update_skill` yang ada
   (middleware, path, invalidation tetap satu pintu) dengan opsi baru
   "simpan SKILL.md mentah" agar frontmatter asli (license dkk.) utuh.
6. **Provenance** — tambahkan `source`/`resolved_url`/`commit_sha`/
   `imported_at` ke store value SKILL.md; expose via `SkillOut.provenance`.

---

## 5. Keamanan & Guardrails

- **Import tidak pernah mengeksekusi apa pun** — tidak ada script yang
  dijalankan saat import. Eksekusi `scripts/` hanya terjadi jika user/admin
  memakai skill itu nanti (keputusan trust pengguna; provenance jadi jejak
  audit).
- **SSRF guard** untuk URL non-git: hanya `https://`, tolak localhost/IP
  privat.
- **Batas ukuran** download/ekstrak/file-count (lihat §4 langkah 4).
- **`allowed-tools` tidak di-enforce** — disimpan sebagai metadata dan
  didokumentasikan advisory di docstring.
- **Scope**: import global = admin-only; import user = siapa pun (seperti
  `npx skills add` lokal, hanya memengaruhi namespace user sendiri).

---

## 6. Rencana Testing (offline, tanpa network/API key — sesuai konvensi repo)

- **Source git lokal**: fixture repo git lokal (`git init` + commit) lalu
  `git clone <path-lokal>` — jalan offline tanpa network.
- **URL langsung**: `httpx.MockTransport` untuk download + ekstrak
  zip/tar.gz.
- **Kasus**: SKILL.md di root; koleksi `skills/`; filter `--skill`; kolisi
  nama (409); file melebihi cap (tolak); round-trip frontmatter (license
  dipertahankan, nama direname via `target_name`); provenance tercatat di
  `replace`; cleanup temp dir walau gagal.
- Ekstensi `tests/test_smoke.py` tidak diperlukan — ini jalur baru yang
  terisolasi (service + endpoint test sendiri).

---

## 7. Roadmap (fase)

| Fase | Isi | Output |
|---|---|---|
| **1 — MVP** | Source git (`owner/repo`, URL, tree path) + discovery container (`skills/`) + preview endpoint + provenance | Menutup ~90% pemakaian skills.sh |
| **2** | URL download langsung + archive (zip/tar) | Source format lengkap |
| **3** | Update-from-source (bandingkan commit SHA) | Skill bisa di-refresh |
| **4** (stretch) | Discovery/browse (`GET /skills/browse?q=`) — skills.sh tidak punya public search API; proxy index agentskills.io atau GitHub code search | Eksplorasi di dalam app |

---

## 8. Keputusan yang Perlu Dikonfirmasi

1. **Scope import user** diperbolehkan (seperti `npx skills add` lokal) atau
   admin-only dulu?
2. **Default kolisi nama**: tolak (409, minta `target_name`) — rekomendasi —
   atau auto-rename?
3. Repo koleksi tanpa filter `skill`: kembalikan daftar (mirip preview) atau
   409 minta preview dulu?
4. Bahasa doc ini: ikut konvensi repo (Indonesia) — bisa dialihkan ke Inggris.

---

## Sumber riset

- skills.sh About: <https://www.skills.sh/about>
- Repo CLI: <https://github.com/vercel-labs/skills> (README — source formats,
  batas ukuran, container discovery)
- Spesifikasi Agent Skills: <https://agentskills.io/specification>
  (frontmatter, struktur direktori, progressive disclosure)
- Anthropic Agent Skills: <https://github.com/anthropics/skills>,
  <https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills>
- Registry lain: Smithery (<https://smithery.ai/skills>),
  c2s/agent-skills-hub (meta-registry)
