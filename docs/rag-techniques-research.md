# Riset: Teknik RAG Lanjutan & Roadmap Penerapan

Status: **Riset (sumber web, Feb 2026) — belum diimplementasikan.**
Bacaan pendamping: `docs/knowledge-base-plan.md` (blueprint v1 yang sudah
terbangun). Dokumen ini memetakan teknik RAG modern ke stack kita (FastAPI +
Deep Agents + Postgres + Weaviate) dengan bukti benchmark, lalu memberikan
roadmap penerapan terurut berdasarkan ROI.

---

## 1. Peta teknik RAG (per stage pipeline)

| Stage | Teknik | Apa yang diperbaiki | Bukti benchmark | Kompleksitas |
|---|---|---|---|---|
| **Ingestion** | Chunking sadar-struktur (markdown/HTML/kode) | Chunk terpotong di tengah ide | NVIDIA: page-level chunking tertinggi rata-rata (0.648); factoid → 256–512 token, analitis → 1024+ | Rendah |
| | Contextual retrieval (Anthropic) | Chunk terisolasi kehilangan konteks dokumen ("he then marched north" — siapa "he"?) | 67% lebih sedikit retrieval failure (dengan rerank); +2.2–2.8pp Recall@5 di benchmark finansial | Sedang (1 LLM call per chunk saat ingest) |
| | Late chunking (Jina) | Anaphora lintas paragraf ("this subsidiary") | Diklaim memperbaiki referensi jarak jauh; butuh model khusus | Tinggi |
| **Indexing** | Embedding model kuat | Semua hal di bawah terasa kurang jika embedder lemah | +18 recall@5 (studi 46k chunk) | Rendah (ganti model + reindex) |
| | Matryoshka/MRL dims | Biaya storage/search 3× | Truncate 3072→1024 dim dengan loss kecil | Rendah |
| | Metadata enrichment | Filter & boost hasil (sumber, tipe dok, path) | "Low-lift, high-impact" (Meilisearch) | Rendah |
| **Retrieval** | Hybrid search (BM25F + dense) | Exact-match (kode, akronim, ID) vs semantik | +8–12% recall@5 (NeuroLink); de facto standar produksi — TAPI: bisa ~0 jika embedder sudah kuat (studi 46k chunk: byte-identical) | Sudah kita punya |
| | Reranking (cross-encoder) | Urutan top-k; precision stage | **Komponen paling berdampak**: T2-RAGBench Recall@5 0.695→0.816, MRR@3 0.433→0.605 (+39.7%); overhead CPU FlashRank ~31ms vs ~10s LLM (0.3%) | Sedang |
| | Parent-child / small-to-big | Chunk kecil untuk recall, parent untuk konteks | Mixed: satu studi regresi completeness 3.22→2.67 | Sedang |
| **Query transform** | Query rewrite | Query vague/elliptical vs korpus teknis | Recall naik signifikan di sebagian besar tim; 100–500ms extra call → pakai model kecil + cache + selektif | Sedang |
| | HyDE | Vocabulary gap query↔dokumen | **Kontradiktif**: nDCG@10 61.3 vs 44.5 (DL-19) TAPI −9.7 recall@5 (studi 46k), drift di domain niche | Rendah–sedang |
| | Multi-query + RRF | Beberapa interpretasi query | +2.1 marginal; biaya N× | Sedang |
| | Step-back prompting | Query terlalu konkret | Menolong di sebagian kasus; risiko over-generalize → jangan menggantikan query asli | Rendah |
| **Post-retrieval** | Dedup + lost-in-the-middle | Duplikasi chunk; LLM bias posisi | Gratis, satu baris | Gratis |
| | CRAG (corrective) | Evaluasi hasil retrieval + fallback | Recall@5 0.658 vs BM25 0.644 — kalah dari hybrid saja (0.695) | Tinggi |
| | Self-RAG | Grounding + kritik output | Ceiling akurasi tertinggi tapi butuh fine-tune | Tinggi |
| **Arsitektur** | GraphRAG | Multi-hop, relationship-heavy | "Substantial" (MS Research) utk global sensemaking; retrieval time 44.87s — mahal | Tinggi |
| | RAPTOR | Long-doc, reasoning lintas seksi | +20% absolute di QuALITY; token cost 10jt | Tinggi |
| | Adaptive RAG (routing) | Campuran kompleksitas query | Berguna saat workload heterogen | Sedang–tinggi |
| **Evaluasi** | Golden set + RAGAS / Recall@k / MRR / nDCG | — | **Prasyarat semua keputusan** | Sedang |

---

## 2. Temuan penting (yang kontra-intuitif)

1. **"Satu teknik per stage, ukur dulu" (hukum binding constraint).** Studi
   terukur di 46k chunk (dev.to, Lev Riabov) menemukan: hybrid search = 0 gain,
   HyDE = −9.7, 4 dari 5 reranker = negatif/nol — bukan karena tekniknya buruk,
   tapi karena **embedder adalah constraint yang mengikat**; memperbaikinya
   membayar +18 poin. Setelah embedder kuat, semuanya jadi marginal.
   **Implikasi: keputusan teknik harus didasari evaluasi di korpus kita, bukan
   menyalin daftar menu.**

2. **Reranking adalah stage paling berdampak ketika kandidat pool sudah benar**
   (T2-RAGBench, 23k query finansial): hybrid + cross-encoder menang telak
   (+12.1pp Recall@5, +39.7% MRR@3). Tapi depth kandidat penting: rerank 20
   kandidat justru buruk (0.458) vs 50 (0.826) — **retrieve broad (50+), rerank
   fine (top 5–8)**. Latency CPU reranker (FlashRank MiniLM ~4MB) hanya ~31ms
   terhadap total ~10s — bisa diadopsi tanpa GPU.

3. **HyDE berbahaya di domain niche** (dokumen internal, jargon produk): teks
   hipotetis bisa melayang ke region embedding yang salah. Kalau dipakai,
   gating dengan confidence threshold (pakai query asli jika similarity sudah
   tinggi).

4. **"Rerank teks yang sama dengan yang di-embed."** Jika contextual retrieval
   diterapkan, chunk yang di-index berisi `context_note + teks`; reranker harus
   menilai representasi yang sama, bukan teks telanjang (menggagalkan gain
   contextual: 47.9% vs 51.6%).

5. **Weaviate 1.30 yang kita pakai:** default fusion `relativeScoreFusion`
   (sejak 1.24) — retain skor, bukan hanya ranking; server default `alpha=0.75`
   (lebih condong vektor), client kita eksplisit `alpha=0.5` via
   `KB_HYBRID_ALPHA`. Dukungan: `boost` (soft-ranking, misal bias ke path
   tertentu), `max vector distance` threshold, property weights BM25F (boost
   `path`/judul), tokenization (`word` vs `field`), dan reranker module
   (Cohere) di dalam Weaviate.

6. **Chunk size optimal tergantung tipe query** (NVIDIA: factoid 256–512 token,
   analitis 1024+/page-level) **dan model embedding** (decoder-based lebih suka
   chunk besar, encoder-based lebih suka kecil). Default kita (1000 char
   recursive + markdown-aware ≈ 250 token) sudah masuk rentang factoid yang
   wajar; evaluasi untuk menentukannya.

7. **Agent loop = query transformation gratis.** Deep Agents kita bisa
   memanggil tool berkali-kali dan mengubah query antar iterasi; ini men-subsum
   decomposition/step-back statis (sama argumennya dengan "JIT vs AOT context"
   di literatur).

---

## 3. Status implementasi kita (v1 yang sudah jalan)

| Komponen | Status | Catatan |
|---|---|---|
| Hybrid search (Weaviate BM25F + dense) | ✅ | `alpha` configurable (env + per-request `?alpha=`); filter `owner` wajib |
| BM25F property weights | ✅ | `KB_BM25_PROPERTY_WEIGHTS` (default `{"path": 2.0}`) |
| Chunking markdown-aware + recursive | ✅ | `services/kb/chunk.py` |
| Page-level chunking (PDF) | ✅ | NVIDIA: best average accuracy |
| Metadata (path, doc_id, kb_id, owner) | ✅ | Sudah jadi property + filter |
| Multi-tenant isolation | ✅ | Filter owner di semua query |
| Per-user quota, zip, download | ✅ | Hardening commit |
| Reindex (ganti embedding model) | ✅ | `POST /kb/{id}/reindex` |
| Embeddings swappable + Matryoshka dims | ✅ | `EMBEDDINGS_MODEL` / `EMBEDDINGS_DIMENSIONS`; ganti model = reindex |
| **Evaluasi (golden set + IR metrics)** | ✅ | `scripts/kb_eval.py` + `services/kb/eval.py` (R1) |
| Reranking | ❌ | Fase R3 berikutnya |
| Query rewrite | ❌ | Belum (sebagian ditutup agent loop) |
| Contextual retrieval | ❌ | Belum |
| Dedup + lost-in-middle reorder | ✅ | Tool agent (gratis) |

---

## 4. Roadmap penerapan (urut berdasarkan ROI, tiap langkah = 1 commit terukur)

### Fase R1 — Evaluasi (dasar semua keputusan)
- Golden set: 30–60 pasang (query, chunk relevan, dokumen) dari KB sungguhan,
  dikategorikan (factoid / prosedural / konseptual / multi-hop).
- Script evaluasi: `scripts/kb_eval.py` — hitung `Recall@k`, `MRR@3`, `nDCG@5`
  untuk kombinasi (alpha, limit, chunk config) terhadap vector store yang sama.
- Opsional RAGAS (faithfulness, answer relevancy, context precision/recall)
  untuk jawaban end-to-end lewat tool agent.
- **Output: baseline angka.** Setiap fase berikutnya wajib membandingkan ke
  baseline ini; teknik yang tidak naik = di-reject (disiplin dari temuan #1).

### Fase R2 — Perbaikan murah (tanpa LLM call baru)
- **Dedup + lost-in-the-middle**: tool agent dedup chunk id; urutkan konteks
  relevansi tertinggi di tengah output tool.
- **BM25F property weights**: di Weaviate, property `path` diberi bobot lebih
  tinggi dari `content` (judul > isi), `alpha` diekspos per-request di
  `/kb/{id}/search` (`alpha` query param) untuk tuning live.
- **Chunking per-ekstensi**: PDF → page-level chunks (temuan NVIDIA), tetap
  markdown-aware untuk .md; expose `KB_CHUNK_SIZE` per-KB opsional.
- **Embedding model upgrade** (jika evaluasi menunjukkan recall rendah):
  `text-embedding-3-small` → `text-embedding-3-large`/`voyage-3`, Matryoshka
  dims 1024, lalu reindex. **Jangan pernah mencampur model lama & baru.**

### Fase R3 — Reranking (impact terbesar setelah baseline)
- Retrieve hybrid top 20–50 (recall stage) → rerank cross-encoder → top 5–8.
- Opsi: (a) lokal CPU `flashrank`/`bge-reranker-v2-m3` via
  `langchain`/`rerankers` lib (~31ms, tanpa GPU); (b) API Cohere/Voyage
  (akurasi terbaik per benchmark, tapi biaya per query).
- Abstraksi `Reranker` di `services/kb/rerank.py` (sama polanya dengan
  vector store: in-memory/fake untuk test offline, config-driven).
- **Gerbang keputusan**: jika R2 (embedder kuat) sudah menaikkan baseline,
  ukur reranker di golden set — kecil kemungkinan menang (temuan studi 46k);
  jika baseline stagnan, rerank hampir pasti menang (+12pp).
- Jangan lupa hukum #4: rerank teks yang sama dengan yang di-index.

### Fase R4 — Query transform (selektif)
- **Query rewrite untuk query vague** (LLM kecil, mis. model yang sama dgn
  `DEEPAGENTS_MODEL` atau lebih kecil): 1 call sebelum search, cache
  content-addressed per hash query, hanya untuk query yang lolos classifier
  sederhana (panjang < N kata / tanpa proper noun korpus).
- **HyDE: skip dulu** (bukti kontradiktif, risiko drift di dokumen internal).
  Jika recall faktoid rendah dan ingin dicoba: gating confidence + A/B di
  golden set.

### Fase R5 — Contextual retrieval (jika evaluasi menunjukkan context starvation)
- Saat ingest: LLM kecil menulis 1–2 kalimat konteks per chunk
  (`context_note`), disimpan sebagai property + dipakai sebagai prefix saat
  embedding DAN saat rerank (hukum #4).
- Biaya: 1 LLM call per chunk (sekali saja saat upload/reindex). Implementasi
  murah di pipeline kita karena `ingest_document` sudah modular.
- Hanya diaktifkan jika golden set menunjukkan chunk telanjang kehilangan
  konteks (query yang butuh referensi lintas paragraf).

### Fase R6 — Lanjutan (hanya jika kebutuhan muncul)
- **GraphRAG / RAPTOR**: hanya untuk query multi-hop / sensemaking global
  (mis. "ringkas semua keputusan arsitektur di semua dokumen"). Biaya offline
  konstruksi tinggi; pertimbangkan setelah R1–R5 dievaluasi.
- **Parent-child / sentence-window**: alternatif murah dari contextual
  retrieval jika R5 terlalu berat.
- **Sparse encoders (SPLADE/bge-m3)**: menggantikan BM25 untuk recall keyword
  yang lebih baik; bisa jadi pengganti/penambah hybrid saat ini.

---

## 5. Prinsip operasional (ringkas)

1. **Satu perubahan per commit, diukur di golden set.** Jangan "menu lengkap".
2. Evaluasi retriever dulu (`Recall@k`, `MRR`, `nDCG`), baru evaluasi generasi
   (RAGAS). Pipeline = retriever + generator; diagnosa per komponen
   (pola RAGCHECKER: claim recall, context precision, hallucination).
3. Latency budget: retrieval ~120ms + rerank ~31ms masih ≪ LLM generation
   (~10s) — ruang untuk R3 tanpa risiko UX.
4. Multi-tenant: semua teknik di atas harus tetap menghormati filter `owner`
   (sudah menjadi invariant di `search()`).
5. Test offline tetap wajib hijau: reranker & embedding punya fake
   deterministik untuk pytest (pola `LocalEmbeddings` + `InMemoryKbVectorStore`
   yang sudah ada).

---

## Sumber kunci

- Meilisearch — 9 advanced RAG techniques (chunking, rerank, hybrid, metadata)
- NeuroLink — 10 chunking strategies + hybrid RRF + rerank (recall +8–12%)
- Ragnight — production RAG: 256–512 token chunks, RRF k=60, funnel 50→5–8,
  checklist go-live
- Thread Transfer — hybrid search 2025: fusion RRF vs linear, tuning per tipe query
- NVIDIA — benchmark chunking: page-level terbaik rata-rata; factoid vs analitis
- arXiv 2505.21700 — chunk size vs embedding model interaction
- T2-RAGBench (arXiv 2604.01733) — hybrid + rerank dominan; BM25 > dense di
  domain finansial; HyDE & multi-query gagal di query numerik presisi
- Atlan — 12 advanced RAG techniques (contextual retrieval 67%, HyDE nDCG,
  RAPTOR +20% QuALITY, GraphRAG)
- dev.to (Lev Riabov) — pengukuran 46k chunk: embedder = binding constraint;
  hybrid 0 gain; rerank marginal; "rerank teks yang di-embed"
- Alex Chernysh / Jatin Bansal — query transformation taxonomy (rewrite,
  decomposition, step-back, HyDE, multi-query + RRF)
- RAGCHECKER (NeurIPS 2024) — metrik modular retriever/generator
- Weaviate docs — hybrid: alpha, relativeScoreFusion (default 1.24+), BM25F
  property weights, boost, max vector distance, reranker modules
