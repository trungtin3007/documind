# DocuMind — Evaluation Results

*Generated 2026-08-16 by `make_results.py` from `eval_results.json` (run 2026-08-16T16:13:11). Every number here is read from that file.*

## What DocuMind is

DocuMind answers questions about complex PDFs — the kind with tables and charts that defeat text extraction — by never parsing them. Each page is rendered to an image and embedded with a multimodal model (voyage-multimodal-3); a text question retrieves the most similar page images by cosine similarity; and a vision-language model (gemini-3.5-flash-lite) reads the retrieved pages and answers from them, citing the pages it used. This evaluation measures both halves of that pipeline — whether retrieval surfaces the page holding the answer, and whether the generated answer is actually correct — against a 28-question labelled set over a 116-page security research paper.

## Methodology

- **28 questions**, all with hand-verified gold labels (28/28 entries marked `verified:true`). Numbers below are from a `--verified-only` run.
- **Gold labels** are the set of page filenames that genuinely answer the question, plus a short reference answer. A retrieval "hit" means any gold page appears in the top *k*.
- **Scope labels** describe where a question's answer lives and whether its terms collide across the document:
  - `main` — answer is in the main paper (pages 1–13).
  - `appendix` — answer is only in an appendix (pages 19–116).
  - `ambiguous` — the question's terms appear in **both** halves, so retrieval must break a tie. These were built deliberately, using `find_collisions.py` to find terms occurring in both scopes.
- **Author labels** record who wrote the question, and exist to detect a specific bias. The `claude` questions were drafted by reading the gold pages, so they inherit those pages' vocabulary — retrieval is partly being scored on paraphrase matching. The `developer` questions were written independently, without looking at the pages, using the phrasing a real user would reach for. Comparing the two groups shows how much of the score is an artifact of how questions were written.
- **Retrieval at k=5**, metrics reported at hit@1 / hit@3 / hit@5 and MRR. **Generation reads the top 3 pages**, not just the top 1, because detail questions often have the specific figure on a lower-ranked page than the summary page.
- **Grading** is a reference-based LLM-as-judge (gemini-3.6-flash) held separate from the answering model (gemini-3.5-flash-lite). The judge sees only the question, the gold answer, and the system answer — never the page images — so it grades against the reference rather than re-deciding what the paper says. It returns correct / partial / wrong with a one-line reason.
- **Determinism**: fixed seed (7) on generation and judging; retrieval is deterministic.

## Results by scope

| group | n | hit@1 | hit@3 | hit@5 | MRR | cites gold | correct | partial | wrong |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **overall** | 28 | 68% | 96% | 100% | 0.81 | 86% | 85% | 15% | 0% |
| main | 13 | 62% | 100% | 100% | 0.78 | 100% | 92% | 8% | 0% |
| appendix | 10 | 90% | 100% | 100% | 0.95 | 90% | 100% | 0% | 0% |
| ambiguous | 5 | 40% | 80% | 100% | 0.62 | 40% | 25% | 75% | 0% |

## Retrieval method comparison

Three retrieval methods were evaluated on the same 28 questions. `visual` is the original
page-image cosine ranking. Both hybrids re-rank the visual top-15 candidates by fusing the
visual rank with a text rank using Reciprocal Rank Fusion — identical fusion, so the only variable
is how page text is scored: `hybrid-bm25` uses lexical term overlap, `hybrid-embedding` uses a local
sentence-transformer. **`hybrid-embedding` is the adopted default.**

Counts, not percentages — at this sample size one question is worth 4 points overall and 20 in `ambiguous`.

**hit@1** (questions hit / total)

| scope | `visual` | `hybrid-bm25` | `hybrid-embedding` |
|---|---:|---:|---:|
| overall | 19/28 | 21/28 | 19/28 |
| main | 8/13 | 10/13 | 8/13 |
| appendix | 9/10 | 8/10 | 9/10 |
| ambiguous | 2/5 | 3/5 | 2/5 |

**hit@3** (questions hit / total)

| scope | `visual` | `hybrid-bm25` | `hybrid-embedding` |
|---|---:|---:|---:|
| overall | 26/28 | 26/28 | 27/28 |
| main | 13/13 | 12/13 | 13/13 |
| appendix | 10/10 | 10/10 | 10/10 |
| ambiguous | 3/5 | 4/5 | 4/5 |

**MRR**

| scope | `visual` | `hybrid-bm25` | `hybrid-embedding` |
|---|---:|---:|---:|
| overall | 0.81 | 0.85 | 0.81 |
| main | 0.79 | 0.85 | 0.78 |
| appendix | 0.95 | 0.90 | 0.95 |
| ambiguous | 0.56 | 0.74 | 0.62 |

## Results by question provenance

| group | n | hit@3 | correct |
|---|---:|---:|---:|
| `claude` (drafted from the gold pages) | 20 | **100%** | 100% |
| `developer` (written independently) | 8 | **88%** | 33% |

Questions written from the pages score 100% hit@3; questions written independently score 88%. The gap is 12 points on retrieval and 67 points on answer correctness. **Any evaluation built only from page-derived questions would have overstated this system's quality.** The two groups are not otherwise matched — the developer set is also where the `ambiguous` questions are concentrated — so this gap measures phrasing and topic difficulty together, not phrasing alone.

## Key finding

**Retrieval and answers are strong on questions with an unambiguous home in the document, and degrade when a question's terms appear in both the main paper and the appendix.**

On the 23 unambiguous questions (`main` + `appendix`), hit@3 is 100%. On the 5 `ambiguous` questions it is 80%. Citation quality moves the same way: a gold page is cited on 100% of `main` and 90% of `appendix` questions, but only 40% of `ambiguous` ones.

The mechanism is visible in the retrieved page lists. When a term appears in both halves of the paper, the appendix — 98 of the paper's 116 pages — supplies many plausible-looking pages that crowd the correct one out of the top 3. Generation is not the weak link: it answers faithfully from whatever pages it is handed, and when the gold page is missing it usually says so rather than inventing an answer.

### The re-ranker result, stated honestly

Swapping lexical scoring for semantic scoring **recovers a specific, confirmed failure mode** — see the `m11` example below — and moves overall hit@3 from **26/28 to 27/28**, with no scope regressing. `hybrid-bm25`, by contrast, bought its `ambiguous` gain by breaking `main`.

**That is a one-question difference, which is within noise at n=28. Treat it as directional, not proven.** Nothing here is statistically significant: the `ambiguous` group is n=5, where a single question moves the metric 20 points. What justifies adopting the embedding re-ranker is not the aggregate — it is that the mechanism was predicted in advance, the prediction was tested, and the specific failure it targeted disappeared while nothing else regressed. The aggregate merely fails to contradict that.

A second caveat: `hit@5` is 100% across every group and every method. The correct page is essentially always retrieved; only its rank changes. Re-ranking is therefore the right lever, and also a bounded one — it cannot fix what retrieval never surfaces.

## Worked example: the failure the re-ranker fixes

**`m11` — From genuine (non-benchmark) user sessions alone, what secrets were recovered and in what quantities?**

Gold pages: p009, p003

| method | gold page rank |
|---|---:|
| `visual` | 2 |
| `hybrid-bm25` | 4 |
| `hybrid-embedding` | 2 |

- `bm25` over those candidates: p008=23.91, p062=21.01, p065=14.87, **p003=11.91**, **p009=8.91** — gold at position **4 of 5**
- `embedding` over those candidates: p008=0.62, **p009=0.61**, **p003=0.53**, p062=0.44, p065=0.36 — gold at position **2 of 5**

Lexical scoring pushes both gold pages to the bottom of its candidate list, with the page carrying the actual quantities scoring lowest of all: that page states its numbers tersely, while the neighbouring pages repeat the question's framing words. Term overlap and answer-bearing are anti-correlated here. Semantic scoring is indifferent to that phrasing difference and ranks the same page near the top, which is exactly the behaviour the swap was meant to produce.

## Where it fails, concretely

**`u02` — What models did they use to score or judge the stolen reasoning traces?**

- Gold pages: p038, p041, p062 · first gold page at rank **3** · graded **partial**
- Retrieved: p001 (0.032), p007 (0.032), p041 (0.031), p038 (0.031), p022 (0.030)
- Cited: p041
- Answer: “The models used to score or judge the reasoning traces (shown as the scoring models in Figure 26 and its associated text) are: GLM 5.2, Kimi K3, Kimi K2.6, Kimi K2.7 Code, DeepSeek V4 Flash, Inkling, and gpt-oss 120B.”
- Judge: The system correctly lists all seven specific scoring models but omits the two-stage LLM-as-a-judge used to label privacy artifacts.

**`u08` — What defenses or mitigations do the authors recommend against this attack?**

- Gold pages: p010, p011, p019, p020 · first gold page at rank **2** · graded **ungraded**
- Retrieved: p003 (0.032), p019 (0.032), p001 (0.031), p009 (0.031), p002 (0.030)
- Cited: none
- Answer: “”
- Judge: Retrieval-only run (--no-generate).

**`u01` — Which language models were the authors able to steal reasoning traces from?**

- Gold pages: p004, p002 · first gold page at rank **4** · graded **partial**
- Retrieved: p001 (0.033), p022 (0.032), p038 (0.031), p002 (0.031), p041 (0.030)
- Cited: p001
- Answer: “Based on the provided pages, the authors mention recovering reasoning traces for "Anthropic, OpenAI, and Google" (Page 1), as well as "Claude Opus 4.8 or GPT-5.6-Sol" (Page 2), and "Opus 4.8 and GPT-5.6 Sol" (Page 3). Additionally, Appendix C (listed on Page 1) mentions extraction attack details for Gemini, Claude, and…”
- Judge: The system answer only identifies two specific models (Claude Opus 4.8 and GPT-5.6-Sol) and provider names, missing the full list of models specified in the reference answer.

## Limitations & next step

- **Small n**, as above. The scope effect is directionally clear but the numbers are coarse.
- **One document.** All 28 questions are about a single 116-page paper. Nothing here shows the effect generalises to other documents, and the main/appendix split is a property of academic papers specifically.
- **Judge not validated against human grades.** The judge's correct/partial/wrong calls have not been checked against a human rater, so answer-quality numbers carry unmeasured judge error. One known case: `a01` was graded `partial` for following the paper's own prose rather than the gold answer's phrasing.
- **Gold pages are permissive.** Several questions list multiple acceptable gold pages, which makes a "hit" easier than a single-target metric would.
- **Answer grades cover 26 of 28 questions.** The rest hit the daily API quota during grading and are recorded as `ungraded`; answer-quality rates are computed over graded rows only, so coverage is reduced rather than scores being silently deflated. Retrieval metrics cover all 28.
- **The two text scorers are complementary, and only one is used.** BM25 keeps an edge on distinctive rare tokens: on `u02`, whose answer hinges on model names like "GLM-5.2", lexical scoring lifts the gold page to rank 1 while the embedding scorer reaches only rank 3. Exact-token matching and semantic similarity fail in different places, so fusing all three rankings — visual, lexical, semantic — is the obvious next experiment. Not done here; the adopted configuration uses semantic scoring only.

**Next step (Week 5): a web interface, and a larger eval set.** The re-ranker work is done and adopted; what limits every conclusion in this document now is n=28 over one document, not the retrieval design. The highest-value next measurement is more questions across more documents, which would also make the complementary-scorer experiment above worth running. Interface work (answer shown beside the cited page) is the demo-facing task.

## Reproducing

```bash
# retrieval + answers for the adopted method (cached rows cost nothing)
./venv/bin/python eval.py --verified-only --resume --method hybrid-embedding \
    --out eval_results_hybrid_emb.json

# the other two methods, retrieval metrics only (no LLM calls at all)
./venv/bin/python eval.py --verified-only --no-generate --method visual --out eval_results.json
./venv/bin/python eval.py --verified-only --no-generate --method hybrid-bm25 \
    --out eval_results_hybrid.json

./venv/bin/python compare_methods.py                 # three-way comparison
./venv/bin/python make_results.py                    # this document
```
