# DocuMind — Evaluation Results

*Generated 2026-08-15 by `make_results.py` from `eval_results.json` (run 2026-08-15T23:21:33). Every number here is read from that file.*

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
| **overall** | 28 | 68% | 93% | 100% | 0.81 | 89% | 82% | 14% | 4% |
| main | 13 | 62% | 100% | 100% | 0.79 | 100% | 92% | 8% | 0% |
| appendix | 10 | 90% | 100% | 100% | 0.95 | 100% | 90% | 10% | 0% |
| ambiguous | 5 | 40% | 60% | 100% | 0.56 | 40% | 40% | 40% | 20% |

## Results by question provenance

| group | n | hit@3 | correct |
|---|---:|---:|---:|
| `claude` (drafted from the gold pages) | 20 | **100%** | 95% |
| `developer` (written independently) | 8 | **75%** | 50% |

Questions written from the pages score 100% hit@3; questions written independently score 75%. The gap is 25 points on retrieval and 45 points on answer correctness. **Any evaluation built only from page-derived questions would have overstated this system's quality.** The two groups are not otherwise matched — the developer set is also where the `ambiguous` questions are concentrated — so this gap measures phrasing and topic difficulty together, not phrasing alone.

## Key finding

**Retrieval and answers are strong on questions with an unambiguous home in the document, and degrade sharply when a question's terms appear in both the main paper and the appendix.**

On the 23 unambiguous questions (`main` + `appendix`), hit@3 is 100% and 91% of answers are graded correct. On the 5 `ambiguous` questions, hit@3 falls to 60% and correctness to 40%. Citation quality moves the same way: the system cites a gold page on 100% of `main` and 100% of `appendix` questions, but only 40% of `ambiguous` ones.

The mechanism is visible in the retrieved page lists. When a term like a model name or "mitigations" appears in both halves of the paper, the appendix — which is 98 of the paper's 116 pages — supplies many plausible-looking pages that crowd the correct one out of the top 3. Generation is not the weak link: it answers faithfully from whatever pages it is handed, and when the gold page is missing it usually says so rather than inventing an answer.

**Sample-size caveats, which are severe here.** The `ambiguous` group is **n=5**. One question moving from miss to hit shifts its hit@3 by 20 percentage points, so the figure should be read as "roughly half" and not as 60%. The `appendix` group is n=10 and the whole set is n=28. No confidence intervals are reported because none would be meaningful at these sizes. The direction of the effect is consistent across three independent signals — retrieval rank, citation correctness, and judge grade — which is what makes it worth acting on; the exact magnitudes are not yet trustworthy.

A second caveat: `hit@5` is 100% across every group. The correct page is nearly always *somewhere* in the top 5 — the failure is ranking, not recall. That is what makes the fix below plausible.

## Where it fails, concretely

**`u02` — What models did they use to score or judge the stolen reasoning traces?**

- Gold pages: p038, p041, p062 · first gold page at rank **5** · graded **wrong**
- Retrieved: p002 (0.493), p001 (0.487), p007 (0.482), p015 (0.472), p038 (0.462)
- Cited: none
- Answer: “The provided pages do not contain information about what models were used to score or judge the stolen reasoning traces.”
- Judge: The system incorrectly states that the information is not present, whereas the reference answer lists several specific models used for scoring and judging.

**`u08` — What defenses or mitigations do the authors recommend against this attack?**

- Gold pages: p010, p011, p019, p020 · first gold page at rank **3** · graded **partial**
- Retrieved: p001 (0.451), p003 (0.447), p019 (0.438), p009 (0.422), p010 (0.420)
- Cited: p019
- Answer: “The authors propose a "Context-Bound Envelope Defense" that re-binds every envelope to its originating user, session, and conversational position, along with operational measures for legacy data, backwards compatibility, and training-time hardening (Page 19). Table 2 (Page 19) outlines specific mitigations for various…”
- Judge: The system answer detailed the context-bound envelope defense and post-training mitigations, but omitted the architectural revisions (keeping traces server-side with opaque lookup IDs) and infrastructure guardrails from Section 5.5.

**`u01` — Which language models were the authors able to steal reasoning traces from?**

- Gold pages: p004, p002 · first gold page at rank **4** · graded **partial**
- Retrieved: p001 (0.546), p022 (0.512), p038 (0.511), p002 (0.508), p039 (0.483)
- Cited: p001
- Answer: “Based on the provided pages, the authors mention recovering reasoning traces for "Anthropic, OpenAI, and Google" (Page 1), as well as "Claude Opus 4.8 or GPT-5.6-Sol" (Page 2), and "Opus 4.8 and GPT-5.6 Sol" (Page 3). Additionally, Appendix C (listed on Page 1) mentions extraction attack details for Gemini, Claude, and…”
- Judge: The system answer only identifies two specific models (Claude Opus 4.8 and GPT-5.6-Sol) and provider names, missing the full list of models specified in the reference answer.

## Limitations & next step

- **Small n**, as above. The scope effect is directionally clear but the numbers are coarse.
- **One document.** All 28 questions are about a single 116-page paper. Nothing here shows the effect generalises to other documents, and the main/appendix split is a property of academic papers specifically.
- **Judge not validated against human grades.** The judge's correct/partial/wrong calls have not been checked against a human rater, so answer-quality numbers carry unmeasured judge error. One known case: `a01` was graded `partial` for following the paper's own prose rather than the gold answer's phrasing.
- **Gold pages are permissive.** Several questions list multiple acceptable gold pages, which makes a "hit" easier than a single-target metric would.

**Next step (Week 4): hybrid text + visual retrieval to break scope ties.** The current index has exactly one signal — the visual embedding of a page image — so two pages discussing the same term look equally good regardless of which is the real answer. Adding a text index over the extracted page text gives a second, independent signal, and lets a query match on section context ("Section 5.5 Mitigations" vs "Appendix A") rather than on topic alone. Since hit@5 is already 100%, the correct page is nearly always retrieved and merely mis-ranked, so a reranking fix should be able to convert most of the gap without changing the index. The `ambiguous` group is the metric to watch.

## Reproducing

```bash
./venv/bin/python eval.py --verified-only --resume   # metrics -> eval_results.json
./venv/bin/python make_results.py                    # this document
```
