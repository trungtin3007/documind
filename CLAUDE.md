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
- **Weeks 0-3 done.** Pipeline: `pdf_to_images.py` -> `build_index.py` (Voyage `voyage-multimodal-3`, 116 pages) -> `retrieval.py` (`search(question, k)`) -> `generate.py` (`answer()` reads top-3 page images with Gemini `gemini-3.5-flash-lite`, JSON-schema grounded, cites pages). CLIs: `search.py`, `generate.py`. Plan B confirmed: no ColPali/Colab needed.
- **Week 3 eval harness complete and run on fully verified labels.** `eval_set.json` (28 questions, **all `verified:true`, confirmed by the developer**), `eval.py` (hit@1/3/5 + MRR at k=5, generation top-3, breakdowns by scope AND by author, `--verified-only/--only/--no-judge/--resume`), `judge.py` (reference-based judge on a separate model `gemini-3.6-flash`, never sees page images), `find_collisions.py`, `verify_labels.py`, `api_retry.py`, `make_results.py`.
- **RESULTS.md is generated, never hand-written** — `make_results.py` reads every number from `eval_results.json`. Regenerate after any eval run.
- **Verified numbers (n=28):** overall hit@1 68%, hit@3 93%, hit@5 100%, MRR 0.81, correct 82%. By scope: main (13) hit@3 100% / correct 92%; appendix (10) hit@3 100% / correct 90%; **ambiguous (5) hit@3 60% / correct 40% / cites-gold 40%**. By author: claude-written (20) hit@3 100% / correct 95%; **developer-written (8) hit@3 75% / correct 50%**.
- **Key finding: ranking, not recall, is the failure.** hit@5 is 100% in every group — the right page is essentially always retrieved, just mis-ranked when a term appears in both the main paper and the appendix (the appendix is 98 of 116 pages, so it floods the top-3). Generation is not the weak link: it answers faithfully from what it is given and says "not found" rather than confabulating (e.g. `u02`).
- Caveats to keep repeating: ambiguous n=5 (one question = 20 points), single document, judge never validated against human grades, several questions accept multiple gold pages.
- **Next: Week 4 — hybrid text + visual retrieval to break scope ties.** Add a text index over extracted page text as a second signal so queries can match section context ("Section 5.5 Mitigations" vs "Appendix A"), then rerank. Since hit@5 is already 100%, reranking alone should recover most of the gap without rebuilding the index. Watch the `ambiguous` group. Also still on the list: scale to a few thousand pages, latency + caching, web UI (Week 5), Docker/README (Week 6).
- Rate limits: Gemini free tier caps requests per DAY. `--resume` caches generation and judging independently, `--only` batches across quota windows and merges into `eval_results.json`, `--no-judge` halves usage. A fully cached `--verified-only --resume` run costs 0 API calls.
- **TODO — write README.md.** Must cover: the arXiv source of `sample.pdf` (deliberately NOT committed — third-party copyrighted paper, repo is public) and the regeneration steps (`pdf_to_images.py` -> `pages/`, then `build_index.py` -> `index/`, which costs ~116 Voyage embedding calls), plus setup (`.env` keys `VOYAGE_API_KEY` + `GEMINI_API_KEY`), and the eval chart / latency numbers per the Week 6 plan.
- Blocked: none. (`venv_py39_old/` 162M can be deleted.)
<!-- END STATE -->
