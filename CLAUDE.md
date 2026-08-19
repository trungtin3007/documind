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
- **Week 5 / Tier 1 DONE: web UI + multi-document corpus, committed as `c3bc335`.**
  - `corpora.py` registry, selected with **`DOCUMIND_DATA`**: **"single"** = `pages/*.png` + `index/` (the eval corpus — verified byte-identical, never change it); **"demo"** = the **trimmed, committed** corpus (`pages_demo/` + `index_demo/`); **"demo-full"** = the full local corpus (`pages_demo_full/` + `index_demo_full/`, gitignored). A page id is its corpus-relative path, so document identity travels through retrieval, generation, citations and the app.
  - **Committed demo corpus: 194 pages, 57MB** — `fed-mpr-2025` 81p, `census-income-2023` 59p, `usgs-mcs-2025` 38p (a slice), `bls-ce-2023` 16p. All US federal, public domain (17 USC 105).
  - **Full corpus stays local: 611 pages, 8 documents, 182MB.** Rebuild with `pdf_to_images.py --sources sources --corpus demo-full`, `extract_text.py --corpus demo-full`, `build_index.py --corpus demo-full`. `sources/` (25MB) is gitignored. **README must document this as "run it yourself for the full corpus".**
  - `trim_corpus.py` builds the committed corpus by **slicing the full index's vectors** — trimming costs $0. Its `KEEP` dict is the selection; it keeps every page the tested demo questions land on.
  - **Total Voyage spend to date: $0.793** (549 pages $0.693 + 62 pages $0.100). Both matched `--estimate` exactly.
  - Launch the demo: `DOCUMIND_DATA=demo DOCUMIND_ANSWER_MODEL=gemini-3.6-flash ./venv/bin/uvicorn app:app --port 8000`
  - **`DOCUMIND_ANSWER_MODEL` is required for the demo.** `gemini-3.5-flash-lite` returns degenerate answers on dense tables (22 chars on a BLS table — valid JSON, `finish=completed`, not a token limit); `gemini-3.6-flash` answers correctly. The **default stays flash-lite** so every eval number and RESULTS.md figure remains reproducible.
  - Multi-doc citation fix: page numbers repeat across documents, so pages are labelled `PAGE <n> - from document: <name>` and cite keys are prompt positions. The instruction also forbids mentioning those labels in answer text (they leaked into user-visible output as "PAGE 1"). **Single-corpus prompts are byte-identical**, so cached eval answers stay valid.
  - Security: `/page/{id}` accepts only `page_NNN.png` or `<docid>/page_NNN.png`, re-checked after realpath resolution. Verified rejected: `../.env`, `..%2F.env`, `<doc>/../../.env`, `sources/*.pdf`, `sample.pdf`.
- **Verified demo questions (trimmed corpus, all four documents):** gallium production/price -> usgs p78-79 · US import reliance -> usgs p10-11 · food away from home -> bls p7 · median household income -> census p7 · unemployment rate -> fed p20-21.
- **Render deploy config committed; service NOT created (developer sets env vars).** `render.yaml` (free plan, oregon, branch main), `requirements.txt` (serving), `requirements-dev.txt` (corpus building + eval + embedding retrieval).
  - Build: `pip install --upgrade pip && pip install -r requirements.txt`. Start: `uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1`. Health check `/health`. No port is hardcoded in `app.py` — uvicorn takes it from `$PORT`.
  - **The deploy runs `DOCUMIND_METHOD=hybrid-bm25`, not the adopted hybrid-embedding.** New env override `DOCUMIND_METHOD` (validated against `retrieval.METHODS`). Reason: sentence-transformers pulls torch, ~500MB more to install and ~400MB more RSS against a 512MB instance. BM25 is 26/28 hit@3 vs 27/28 — one question, inside the noise. Also removes the ~90MB model download, so **there is no cold start** and no first-request latency penalty.
  - **Verified in a clean venv built only from `requirements.txt`** (not the dev venv): app imports, retrieval works, a real question answers with citations, page images serve, `torch` and `sentence_transformers` are never imported, **peak RSS ~116MB**, venv 166MB on disk. Comfortably inside the free tier.
  - **Correction worth remembering:** an earlier measurement showed `voyageai` costing ~403MB and pulling torch. That was an artifact of the dev venv having torch installed — `voyageai` imports it opportunistically but does not require it. Never measure deploy footprint in the dev venv.
  - `requirements.txt` deliberately omits `rank-bm25` (BM25 is hand-implemented in `text_score.py`), and `pymupdf`/`pillow` (corpus building only, never serving).
  - Env vars to set in the dashboard: `VOYAGE_API_KEY` and `GEMINI_API_KEY` (secret, `sync: false` in yaml), plus `DOCUMIND_DATA=demo`, `DOCUMIND_METHOD=hybrid-bm25`, `DOCUMIND_ANSWER_MODEL=gemini-3.6-flash`, `PYTHON_VERSION=3.13.5`, `PIP_NO_CACHE_DIR=1` (these five are already in `render.yaml`).
  - Known risk for the first deploy: Gemini free tier is a **daily** cap shared with the eval judge model, so a public demo can exhaust it; the UI shows a clean 429 message when it does.
- **Deploy bug: `index_demo/pages_text.json` never reached the server.** Root cause was `.gitignore`: the rule was `pages_text.json` with **no leading slash**, and git matches a bare filename at *any* depth, so it silently swallowed the demo corpus's text file. `index_demo/` had 5 files on disk but only 4 tracked. BM25 reads that file, so `/ask` failed while `/health` (which only touches `embeddings.npy` + `pages.json`) kept returning 194 pages.
  - Fix: anchored to `/pages_text.json` (repo root only). The full corpus's copy stays ignored via the `index_demo_full/` directory rule.
  - **Lesson: in .gitignore a bare filename matches at every depth — anchor with a leading slash when you mean one specific file.**
  - **Second lesson, on diagnosis:** `/health` OK + `/ask` failing looked like a CWD/relative-path bug, but the two endpoints simply read *different files*. **Path resolution was never broken** — every data path already flows through `corpora.py`'s `BASE_DIR = os.path.dirname(os.path.abspath(__file__))`, so `index_dir`, `pages_dir`, `text_path`, `manifest_path`, `text_cache_path` and `page_path` are all absolute and CWD-independent. `retrieval.py`, `text_score.py`, `generate.py` and `app.py` all go through that one helper, so `/health` and `/ask` cannot disagree.
  - The misleading part was our own error message: it printed `os.path.relpath(path)`, which made a missing *file* look like a relative-path problem. It now prints the absolute path and the CWD.
  - **Verified twice over:** a simulated fresh clone built from `git ls-files` only (229 files, 57MB), served by the clean deploy venv, **started with CWD=`/`** via `uvicorn --app-dir`. `/health` OK, `POST /ask` answered with the real Table D figures citing `bls-ce-2023.pdf — p7`, the cited page image returned 200, zero errors in the log.
- **Frontend updated for the multi-document demo (static only; no backend/retrieval change).** Tagline now reads "Ask across N pages from M U.S. federal reports — answers cite the document and page", with **N and M fetched from `/health`** rather than hardcoded, so it cannot drift from the served corpus. Example chips replaced with the four verified corpus questions.
- **"Documents in this demo" section added**, derived from `index_demo/documents.json` (4 docs, verified: HTML page counts [16, 38, 59, 81] == manifest exactly). Each card gives a friendly title, pages included, a one-line "what you can ask" summary, and a link to the original .gov PDF.
- **Source URLs for all 4 demo documents (recorded here; previously only in shell history):**
  - USGS `https://pubs.usgs.gov/periodicals/mcs2025/mcs2025.pdf` (200)
  - Fed `https://www.federalreserve.gov/monetarypolicy/files/20250620_mprfullreport.pdf` (200)
  - Census `https://www2.census.gov/library/publications/2024/demo/p60-282.pdf` (200)
  - BLS `https://www.bls.gov/opub/reports/consumer-expenditures/2023/` — supplied by the developer. It is the **report landing page, not a direct PDF**, so the card is labelled "Report page". Returns **403 to scripted requests** (BLS bot protection, same as when the download was attempted) but loads normally in a browser — do not treat that 403 as a broken link.
- **All 4 example questions re-verified end to end** (after the daily Gemini quota reset), each citing the expected document: food away from home -> `bls-ce-2023.pdf p7` · gallium -> `usgs-mcs-2025.pdf p78` · median household income -> `census-income-2023.pdf p7, p16` · unemployment rate -> `fed-mpr-2025.pdf p20, p62, p14`. Every cited page image returned 200 image/png.
- USGS is honestly labelled "38 of 216 pages included" — the demo ships a slice.
- **README.md written (Tier 1 complete).** Audience is recruiters/engineers skimming: one-line hook, live demo link (https://documind-yybc.onrender.com, with the free-tier wake warning and the 4 documents named), `docs/demo.gif` embed slot, "why not just paste a PDF into ChatGPT", ascii pipeline diagram, a prominent evaluation section, tech stack, run-it-yourself, and limitations.
  - **Every number in the README was cross-checked against RESULTS.md programmatically** (scope table n / hit@1 / hit@3 / hit@5 / correct all MATCH). Nothing hand-invented. It leads with the honest framing: hit@5 100% everywhere so the failure is ranking not recall; `ambiguous` n=5 hit@3 80% / correct 25%; directional not proven at n=28; and the provenance gap (claude-written 100% hit@3 vs developer-written 88% / 33% correct).
  - **`docs/demo.gif` does not exist yet** — the README embeds it, so GitHub shows a broken image until the developer drops the file in. `docs/README.md` placeholder notes the expected path.
  - Deliberately no source-PDF URLs invented for the eval paper: the README says source PDFs are not redistributed and points at the demo page's per-document links.
- **Live demo verified working after all deploys:** `GET /` 200 text/html, `/health` -> `{"corpus":"demo","pages":194,"documents":4,"method":"hybrid-bm25","answer_model":"gemini-3.6-flash"}`, and the served HTML contains the new tagline, "Documents in this demo", and the new example questions.- Rate limits: Gemini free tier caps requests per DAY. `--resume` caches generation and judging independently, `--only` batches across quota windows and merges into `eval_results.json`, `--no-judge` halves usage. A fully cached `--verified-only --resume` run costs 0 API calls.
- **TODO — write README.md.** Must cover: the arXiv source of `sample.pdf` (deliberately NOT committed — third-party copyrighted paper, repo is public) and the regeneration steps (`pdf_to_images.py` -> `pages/`, then `build_index.py` -> `index/`, which costs ~116 Voyage embedding calls), plus setup (`.env` keys `VOYAGE_API_KEY` + `GEMINI_API_KEY`), and the eval chart / latency numbers per the Week 6 plan.
- Blocked: none. (`venv_py39_old/` 162M can be deleted.)
<!-- END STATE -->
