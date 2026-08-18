# DocuMind — Project Memory

Context for Claude Code. Loaded at the start of every session. Keep this concise; update the State section as work progresses.

## What we're building
DocuMind: a question-answering system over complex PDFs (financial reports, research papers) that contain **tables and charts**, not just text. You ask a question in plain English; it returns an answer **and cites the exact page** the answer came from.

## The core idea (mental model)
Do NOT try to parse tables/charts out of the PDF (that sinks most projects). Instead:
PDF -> render each page to an **image** -> a vision model (**ColPali**) embeds the page images -> a text query retrieves the most relevant page(s) via late-interaction scoring -> a **vision-language model (VLM)** reads the retrieved page image and answers, citing the page.
Retrieval sidesteps parsing entirely by "looking at" pages the way a person would.

## Constraints & key decisions
- **Compute:** developer is on a **laptop CPU only** (no local GPU). ColPali indexing is too slow on CPU for large corpora.
  - **Plan A (preferred, keeps ColPali):** run the *indexing* step on **Google Colab** (free browser GPU); everything else on the laptop.
  - **Plan B (fallback, zero GPU):** use a **hosted multimodal embedding API** (e.g. Voyage multimodal) so the laptop only makes API calls.
  - Decision deferred to Week 1 — Step 0 is identical either way.
- **Time budget:** ~5–10 hrs/week -> realistic MVP timeline is **~9–10 weeks**, not 6.
- **Generation:** use a **VLM via API** (no local GPU needed for answering).

## Build plan (dependency order; each phase ends with something demoable)
- **Step 0 — Setup:** venv, `pymupdf`, render `sample.pdf` -> `pages/*.png`. Demo: environment runs on one file.
- **Week 1 — Retrieval core:** ColPali indexes a small batch; question -> correct page. Demo: question -> right page image.
- **Week 2 — Answers:** retrieved page -> VLM -> written answer citing the page. Demo: thin end-to-end DocuMind.
- **Week 3 — Eval harness:** ~40 labeled Q&A with known source pages; measure retrieval accuracy (page hit@k) + answer quality.
- **Week 4 — Scale:** few thousand pages; handle memory (quantize / on-disk); measure latency; add exact caching.
- **Week 5 — Interface:** simple web app (FastAPI + light frontend) showing answer beside cited page; region highlight = stretch.
- **Week 6+ — Package:** Docker, README with eval chart + latency numbers, recorded demo, documented "Phase 2".

## Working style
- Move **one step at a time**. Finish a thin end-to-end version by Week 2, then improve.
- Give exact commands and code. When something errors, read the **full error** and fix before moving on.
- Prefer the smallest change that makes the current step work. Avoid over-scoping; push extras to "Phase 2".

## Tech (target stack)
Python + FastAPI; PyMuPDF for PDF->images; ColPali (Colab) or Voyage multimodal API for retrieval; a vector store (start simple, e.g. in-memory or Qdrant later); a VLM API for generation; Redis for caching (later); Prometheus + Grafana for metrics (later); Docker Compose to tie services together (later).

## Developer background (for calibration)
Comfortable with Python, full-stack (.NET, React, Node), Git, Docker. **Newer to ML / GPUs / vector search** — explain ML-specific concepts as we go; don't assume prior ML infra experience.

<!-- BEGIN STATE -->
## State
- Last updated: 2026-08-15
- **Repo: https://github.com/trungtin3007/documind (PUBLIC), branch `main`.** First commit `f54305b` covers Weeks 0-3. Pushed via `gh`, account `trungtin3007`.
- **Never commit:** `.env` (holds `VOYAGE_API_KEY`, `GEMINI_API_KEY`) and `sample.pdf` (third-party copyrighted paper, repo is public). Both are in `.gitignore`; verified absent from the remote. Also ignored: `venv/`, `venv_py39_old/`, `pages/`, `index/`, `results/`, `verify/`, `__pycache__/`. Tracked: code, `eval_set.json`, `eval_results.json`, `RESULTS.md`, `CLAUDE.md`.
- **Weeks 0-3 done.** Pipeline: `pdf_to_images.py` -> `build_index.py` (Voyage `voyage-multimodal-3`, 116 pages) -> `retrieval.py` (`search(question, k)`) -> `generate.py` (`answer()` reads top-3 page images with Gemini `gemini-3.5-flash-lite`, JSON-schema grounded, cites pages). CLIs: `search.py`, `generate.py`. Plan B confirmed: no ColPali/Colab needed.
- **Week 3 eval harness complete and run on fully verified labels.** `eval_set.json` (28 questions, **all `verified:true`, confirmed by the developer**), `eval.py` (hit@1/3/5 + MRR at k=5, generation top-3, breakdowns by scope AND by author, `--verified-only/--only/--no-judge/--resume`), `judge.py` (reference-based judge on a separate model `gemini-3.6-flash`, never sees page images), `find_collisions.py`, `verify_labels.py`, `api_retry.py`, `make_results.py`.
- **RESULTS.md is generated, never hand-written** — `make_results.py` reads every number from `eval_results.json`. Regenerate after any eval run.
- **Verified numbers (n=28):** overall hit@1 68%, hit@3 93%, hit@5 100%, MRR 0.81, correct 82%. By scope: main (13) hit@3 100% / correct 92%; appendix (10) hit@3 100% / correct 90%; **ambiguous (5) hit@3 60% / correct 40% / cites-gold 40%**. By author: claude-written (20) hit@3 100% / correct 95%; **developer-written (8) hit@3 75% / correct 50%**.
- **Key finding: ranking, not recall, is the failure.** hit@5 is 100% in every group — the right page is essentially always retrieved, just mis-ranked when a term appears in both the main paper and the appendix (the appendix is 98 of 116 pages, so it floods the top-3). Generation is not the weak link: it answers faithfully from what it is given and says "not found" rather than confabulating (e.g. `u02`).
- Caveats to keep repeating: ambiguous n=5 (one question = 20 points), single document, judge never validated against human grades, several questions accept multiple gold pages.
- **WEEK 4 COMPLETE — `hybrid-embedding` re-ranker ADOPTED as the default retrieval method.** `retrieval.DEFAULT_METHOD = "hybrid-embedding"`; `visual` and `hybrid-bm25` stay selectable via `--method` on `search.py` and `eval.py` for reproducibility. `search()` (visual cosine) is untouched. `generate.answer()` now retrieves via `search_by_method` with the default.
- Week 4 components: `extract_text.py` -> `pages_text.json` (text layer clean on all 116 pages, no OCR needed); `corpus.py` holds the shared page ranges (`MAIN_RANGE` 1-13, `NON_CONTENT_RANGE` 14-18, `APPENDIX_RANGE` 19+) imported by both `text_score.py` and `find_collisions.py` so they cannot drift; `text_score.py` offers `get_scorer("bm25")` and `get_scorer("embedding")`; `retrieval.search_hybrid()` fuses the visual top-15 with a text ranking via RRF (K=60, no weights); `compare_methods.py` does N-way comparison.
- Embedding scorer: **`sentence-transformers/all-MiniLM-L6-v2`**, CPU, free/local. Its max_seq_length is 256 tokens vs ~900-token pages, so pages are chunked (180 words, 40 overlap) and scored by **max cosine over chunks**; whole-page vectors would truncate most of each page away. Cache **`index/text_embeddings.npz`** (557 chunks / 111 content pages), fingerprinted on model+chunking+text so it self-invalidates.
- **Final numbers (n=28 verified). Retrieval hit@3 as counts: visual 26/28, bm25 26/28, embedding 27/28.** By scope embedding vs visual — main 13/13 (unchanged), appendix 10/10 (unchanged), ambiguous 3/5 -> 4/5. hit@1 overall unchanged at 19/28 (bm25 leads at 21/28 but breaks main hit@3 to 12/13). Answer quality for embedding: **correct 85% over 26 graded**, zero `wrong` (visual had one).
- **Honest framing, reflected in RESULTS.md:** the embedding win is **one question** at hit@3 — within noise at n=28, reported as *directional, not proven*. What justifies adoption is that the mechanism was predicted, tested, and confirmed: BM25 ranked m11's gold pages at the bottom of its candidates (answer page lowest of five, 8.91 vs p8's 23.91) because terse answer pages lose on term overlap; the embedding scorer ranks the same page 2nd (0.61 vs 0.62) and m11 returns to rank 2, with nothing else regressing.
- **Known complementarity (future work, not done):** BM25 still wins on distinctive rare tokens — u02 ("GLM-5.2") reaches rank 1 under bm25 vs rank 3 under embedding. Fusing all three rankings (visual + lexical + semantic) is the obvious next experiment.
- `eval.py` gained `--no-generate` (retrieval metrics with **zero** LLM calls; note `--no-judge` alone still generates) and `--summarize-only` (re-aggregate stored rows after a narrow `--only` run, which otherwise leaves `summary_over` describing just that batch).
- Results files: `eval_results.json` (visual baseline), `eval_results_hybrid.json` (bm25), `eval_results_hybrid_emb.json` (**adopted**, and the file `make_results.py` reads for RESULTS.md headline tables).
- u05 and u08 are `ungraded` in the embedding run — daily quota ran out mid-grading. Their retrieval metrics are complete; only their answer grades are missing, and grade rates are computed over graded rows only. Re-run `eval.py --method hybrid-embedding --only u05,u08 --resume` on a fresh quota day to finish them.
- **Week 5 (in progress): local web UI built and running.** `app.py` (FastAPI) + `static/index.html` (single page, plain HTML/CSS/vanilla JS, no build step, no framework). It is a **thin wrapper only** — `/ask` calls `generate.answer()`, which uses `retrieval.DEFAULT_METHOD` (hybrid-embedding). No retrieval/generation/eval code was modified.
  - `POST /ask {question}` -> `{question, answer, cited_pages, retrieved_pages:[{page,score}], found, method}`. `found:false` when the model reports the answer is not on the retrieved pages; the UI styles that as a warning rather than presenting it as an answer.
  - `GET /page/{filename}` serves page images, restricted to `^page_\d{1,4}\.png$` **and** a realpath check confirming the resolved file is still inside `pages/`. Verified: `../.env`, `..%2F.env`, `page_008.png/../../.env` all rejected; `sample.pdf` rejected 400.
  - Errors return clean JSON the frontend renders: 429 rate limit, 504 connection, 503 missing key, 400 empty question.
  - `GET /health` reports page count, retrieval method, answer model.
  - UI: question box (Enter submits), answer card, **cited page images shown large and prominently**, then a top-3 retrieved strip with scores where cited pages are outlined in blue. Loading spinner and visible error state.
  - Run: `./venv/bin/uvicorn app:app --reload --port 8000` -> http://127.0.0.1:8000
  - New deps: `fastapi`, `uvicorn`.
- Week 5 remaining: not deployed (local only), no polish pass yet, `static/` not yet reviewed by the developer.
- **Tier 1 step 1 DONE: 549-page, 6-document demo corpus built, embedded and serving locally.**
  - `corpora.py` — corpus registry. **"single"** = `pages/*.png` + `index/` + `pages_text.json` (the eval corpus, verified byte-identical); **"demo"** = `pages_demo/<docid>/*.png` + `index_demo/` (index, `pages_text.json`, `text_embeddings.npz`, `documents.json`). Switch with **`DOCUMIND_DATA=demo`**. A page id is its corpus-relative path, so document identity travels with the page through retrieval, generation, citations and the app.
  - **Separate index on purpose:** adding documents to `index/` would change retrieval for the eval questions and invalidate RESULTS.md.
  - Corpus — **all 8 approved documents ingested**, US federal, public domain under 17 USC 105: `usgs-mcs-2025` 216p, `fed-mpr-2025` 81p, `nist-sp800-63b` 80p, `nsf-nsb-2024` 63p, `census-income-2023` 59p, `eia-aeo-2023` 50p, `cbo-outlook-2025` 46p, `bls-ce-2023` 16p = **611 pages**. Source PDFs in `sources/`; filenames are the doc ids, so they are kept lowercase-hyphenated (the two dropped in by hand were renamed from spaced titles).
  - **Embedding cost paid: $0.693 for the first 549 pages + $0.100 for the 62 added later = $0.793 total.** Both matched the estimate exactly. Incremental caching confirmed working: the second run re-embedded only the 62 new pages.
  - Verified: images == index == text == **611 pages, same set**. Text layer: 6 empty pages (Fed full-page charts — visual-only, expected), 28 "garbled" (Census number-dense tables, same false positive as p46 of sample.pdf). No OCR run.
  - `text_score.py` is corpus-aware and caches scorers per corpus. **`corpus.NON_CONTENT_RANGE` (p14-18) is applied only to the single corpus** — it is a fact about sample.pdf, not a general rule.
  - **Bug found and fixed during testing: multi-doc citations came back empty.** Page numbers repeat across documents, so cite keys are prompt positions; the model was citing the number *printed on the page* instead, and every citation was filtered out. Fixed by labelling pages `PAGE <n> - from document: <name>` plus a multi-doc-only `MULTIDOC_INSTRUCTION`. **Single-corpus prompts remain byte-identical**, so cached eval answers stay valid (asserted in code by comparing `system_instruction()` output).
  - Run the demo app: `DOCUMIND_DATA=demo ./venv/bin/uvicorn app:app --port 8000` -> http://127.0.0.1:8000 (health reports corpus/pages/documents).
  - **Not committed:** `sources/` (22M), `pages_demo/` (154M), `index_demo/` (7M) are gitignored pending a decision — public domain so legally fine to publish, but 154M of page images is too big for a normal git repo. Decide before deploying.
- **Adding a document is a 3-command, incremental operation:** drop the PDF in `sources/`, then `pdf_to_images.py --sources sources --corpus demo`, `extract_text.py --corpus demo`, `build_index.py --corpus demo`. Only new pages render/embed; run with `--estimate` first to price it.
- **Answer-model finding: `gemini-3.5-flash-lite` is too weak for dense tables.** On the BLS food-expenditure question it returned a degenerate 22-character answer (`"According to Table D ("`) — valid JSON, `finish=completed`, not a truncation or token-budget problem. Same prompt, same page, same seed under `gemini-3.6-flash` gives the full table values. **`generate.MODEL` now reads `DOCUMIND_ANSWER_MODEL`, defaulting to flash-lite so every eval number and RESULTS.md figure stays reproducible.** Run the demo with `DOCUMIND_ANSWER_MODEL=gemini-3.6-flash` — a table-heavy corpus is exactly where flash-lite fails. Note 3.6-flash has a tighter free-tier quota (this is the judge model, ~20 req/min).
- Demo launch command: `DOCUMIND_DATA=demo DOCUMIND_ANSWER_MODEL=gemini-3.6-flash ./venv/bin/uvicorn app:app --port 8000`
- Rate limits: Gemini free tier caps requests per DAY. `--resume` caches generation and judging independently, `--only` batches across quota windows and merges into `eval_results.json`, `--no-judge` halves usage. A fully cached `--verified-only --resume` run costs 0 API calls.
- **TODO — write README.md.** Must cover: the arXiv source of `sample.pdf` (deliberately NOT committed — third-party copyrighted paper, repo is public) and the regeneration steps (`pdf_to_images.py` -> `pages/`, then `build_index.py` -> `index/`, which costs ~116 Voyage embedding calls), plus setup (`.env` keys `VOYAGE_API_KEY` + `GEMINI_API_KEY`), and the eval chart / latency numbers per the Week 6 plan.
- Blocked: none. (`venv_py39_old/` 162M can be deleted.)
<!-- END STATE -->
