# DocuMind

**Visual RAG over document images.** DocuMind answers questions about PDFs by *looking* at their pages instead of parsing them — so it reads the tables and charts that text extraction flattens or drops, finds the right page across a multi-document corpus, and cites the document and page it used.

### [→ Try the live demo](https://documind-yybc.onrender.com)

*Free tier — the first request may take 30–60s to wake the service.* The demo searches **194 pages across 4 U.S. federal reports**: USGS Mineral Commodity Summaries 2025, the Federal Reserve Monetary Policy Report (Jun 2025), Census *Income in the United States: 2023*, and BLS Consumer Expenditures 2023.

![DocuMind demo](docs/demo.gif)

---

## Why not just paste a PDF into ChatGPT?

Two reasons, both about scale and fidelity. **Retrieval:** pasting works for one document that fits in context; DocuMind searches hundreds of pages across many documents and returns the specific page an answer came from, which is what makes the answer checkable. **Visual reading:** the numbers in these reports often exist only as bar labels or table cells, so a text-extraction pipeline never sees them — DocuMind embeds the rendered page image, so a chart is first-class content.

It is not a general document assistant, and it does not beat a frontier model at reasoning. It is a retrieval-and-citation system with an honest evaluation attached.

## How it works

```
PDF ──▶ page images ──▶ visual embeddings ──▶ hybrid retrieval ──▶ VLM reads
        (PyMuPDF,       (voyage-multimodal-3)  (visual rank ⊕        top-3 pages
         150 DPI)                               text rank, RRF)      ──▶ answer
                                                                       + citation
```

1. **Render** every page to an image — no parsing, no layout heuristics.
2. **Embed** each page image with Voyage `voyage-multimodal-3` into one index.
3. **Retrieve** by fusing two rankings with Reciprocal Rank Fusion (no tuned weights): the visual cosine ranking, and a text ranking over the page's text layer. This exists because the visual index alone cannot break ties between pages that discuss the same topic in different sections.
4. **Answer** by sending the top-3 page images to a vision-language model under a JSON schema that forces a `found` flag, so "not on these pages" is a first-class answer rather than a hallucination.

Citations are filtered against the pages actually sent, so a page the model names but never saw is dropped rather than shown.

## Evaluation

The interesting part of this project is not the pipeline — it is what happens when you measure it honestly. Full detail in **[RESULTS.md](RESULTS.md)**.

**28 questions with hand-verified gold labels** over a 116-page research paper. Retrieval is scored at k=5; generation reads the top 3; answers are graded by a reference-based LLM judge running a *different* model from the one that answered, and which never sees the page images.

| scope | n | hit@1 | hit@3 | hit@5 | correct |
|---|---:|---:|---:|---:|---:|
| **overall** | 28 | 68% | 96% | 100% | 85% |
| main paper | 13 | 62% | 100% | 100% | 92% |
| appendix | 10 | 90% | 100% | 100% | 100% |
| **ambiguous** | 5 | 40% | **80%** | 100% | **25%** |

**The headline finding: `hit@5` is 100% in every scope and under every retrieval method.** The correct page is essentially always retrieved — it is just ranked too low to reach the model. The failure is *ranking, not recall*, which is what makes re-ranking the right lever and also bounds how much it can ever fix.

The gap is concentrated in one deliberately-constructed category. `ambiguous` questions are ones whose terms appear in **both** the main paper and the appendix, so retrieval must break a tie. There, hit@3 drops to 80% and correctness to 25%, and a gold page is cited only 40% of the time versus 100%/90% elsewhere.

**Read these numbers as directional, not proven.** At n=28 — and n=5 for `ambiguous` — a single question moves that group by 20 points, and no result here is statistically significant. What the eval is good for is *mechanism*: a specific failure was predicted, tested, and confirmed, and the fix removed it without regressing anything else.

One more result worth stating plainly: questions I wrote by reading the answer pages scored **100% hit@3**, while questions written independently scored **88% hit@3 and 33% correct**. An evaluation built only from page-derived questions would have overstated this system's quality — which is why the eval set records who wrote each question.

## Tech stack

- **Python 3.13**, **FastAPI** + **uvicorn**, vanilla HTML/CSS/JS frontend (no build step)
- **Voyage `voyage-multimodal-3`** — page-image embeddings
- **Google Gemini** — answer generation (`gemini-3.5-flash-lite` default, `gemini-3.6-flash` for the table-heavy demo) and a separate judge model for evaluation
- **sentence-transformers** `all-MiniLM-L6-v2` (CPU) for semantic page-text scoring; hand-written BM25 for the lexical variant
- **PyMuPDF** for rendering and text extraction, **NumPy** for the index
- **Render** for hosting

## Run it yourself

```bash
git clone https://github.com/trungtin3007/documind.git && cd documind
python3 -m venv venv && ./venv/bin/pip install -r requirements-dev.txt
```

Add two keys to `.env` (never committed):

```
VOYAGE_API_KEY=...
GEMINI_API_KEY=...
```

Serve the committed 194-page demo corpus straight away:

```bash
DOCUMIND_DATA=demo DOCUMIND_ANSWER_MODEL=gemini-3.6-flash \
  ./venv/bin/uvicorn app:app --port 8000
```

### Build the full corpus

**Source PDFs are not in this repo.** The demo documents are U.S. federal publications (public domain under [17 U.S.C. §105](https://www.law.cornell.edu/uscode/text/17/105)) and the evaluation paper is third-party copyrighted work, so none of the originals are redistributed here. Download the PDFs you want into `sources/` — the demo page links each original — then:

```bash
./venv/bin/python pdf_to_images.py --sources sources --corpus demo-full   # PDF -> page images
./venv/bin/python extract_text.py  --corpus demo-full                     # text layer
./venv/bin/python build_index.py   --corpus demo-full --estimate          # price it first
./venv/bin/python build_index.py   --corpus demo-full                     # embed
```

`build_index.py` is incremental — it embeds only pages missing from the index — and `--estimate` prints the exact page count and cost before any API call. Embedding is billed by pixels: at 150 DPI a page costs about **$0.0013**, so the full 611-page corpus came to **$0.79**.

The evaluation runs against its own separate corpus (`sample.pdf` + `index/`) so that adding demo documents cannot change published numbers. Reproduce it with:

```bash
./venv/bin/python eval.py --verified-only --resume    # metrics -> eval_results.json
./venv/bin/python make_results.py                     # regenerates RESULTS.md
```

`make_results.py` reads every figure from `eval_results.json`, so RESULTS.md cannot drift from the run that produced it.

## Limitations & future work

- **Cross-section ambiguity is the open problem.** When a term appears in two parts of a document, retrieval picks the wrong part often enough to matter (`ambiguous` hit@3 80%, correctness 25%). Since hit@5 is 100%, this is a re-ranking problem, not a recall one.
- **Small evaluation.** 28 questions over one document. Directionally useful, not statistically significant, and nothing here shows the effect generalises to other document types.
- **The judge is unvalidated.** Answer grades have never been checked against a human rater.
- **The deployed demo runs `hybrid-bm25`, not the adopted `hybrid-embedding`.** The semantic scorer needs sentence-transformers and torch — roughly 500MB more to install and 400MB more RSS than a free-tier instance allows. BM25 scores 26/28 hit@3 against embedding's 27/28, a one-question difference inside the noise. Locally, `DOCUMIND_METHOD=hybrid-embedding` uses the better one.
- **Next: upload your own PDF.** The corpus is currently fixed at build time; the natural next step is letting a visitor add a document and query it immediately.
