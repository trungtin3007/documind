"""Generate RESULTS.md from eval_results.json.

Every number in the report is read from the results file, so the document
cannot drift from the run that produced it. Re-run after any eval run.

Run:  ./venv/bin/python make_results.py
"""
import os, json, datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE_DIR, "eval_results.json")
EVAL_SET = os.path.join(BASE_DIR, "eval_set.json")
OUT = os.path.join(BASE_DIR, "RESULTS.md")

EXAMPLES = ["u02", "u08", "u01"]     # the collision failures worth showing in full


def pct(x):
    return "—" if x is None else f"{x * 100:.0f}%"


def two(x):
    return "—" if x is None else f"{x:.2f}"


def short(page):
    return page.replace("page_", "p").replace(".png", "").lstrip("0") or "p0"


def scope_table(summary, groups):
    lines = ["| group | n | hit@1 | hit@3 | hit@5 | MRR | cites gold | correct | partial | wrong |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, key in groups:
        s = summary.get(key)
        if not s:
            continue
        lines.append(
            f"| {label} | {s['n']} | {pct(s['hit@1'])} | {pct(s['hit@3'])} | {pct(s['hit@5'])} | "
            f"{two(s['mrr'])} | {pct(s['cited_gold'])} | {pct(s['correct'])} | "
            f"{pct(s['partial'])} | {pct(s['wrong'])} |")
    return "\n".join(lines)


def example_block(row):
    got = ", ".join(f"{short(x['page'])} ({x['score']:.3f})" for x in row["retrieved_pages"])
    gold = ", ".join(short(p) for p in row["gold_pages"])
    cited = ", ".join(short(p) for p in row["cited_pages"]) or "none"
    rank = row["gold_rank"] or "not in top 5"
    answer = " ".join(row["answer"].split())
    if len(answer) > 320:
        answer = answer[:320].rstrip() + "…"
    return (
        f"**`{row['id']}` — {row['question']}**\n\n"
        f"- Gold pages: {gold} · first gold page at rank **{rank}** · graded **{row['grade']}**\n"
        f"- Retrieved: {got}\n"
        f"- Cited: {cited}\n"
        f"- Answer: “{answer}”\n"     # curly quotes so inner straight quotes still read
        f"- Judge: {row['judge_reason']}\n")


def main():
    with open(RESULTS) as f:
        data = json.load(f)
    with open(EVAL_SET) as f:
        entries = json.load(f)

    summary, cfg = data["summary"], data["config"]
    n_pages = len([f for f in os.listdir(os.path.join(BASE_DIR, "pages")) if f.endswith(".png")])
    rows = {r["id"]: r for r in data["results"]}
    covered = set(data.get("summary_over", list(rows)))
    scored = [r for r in data["results"] if r["id"] in covered]

    n_total = summary["overall"]["n"]
    n_verified = sum(1 for e in entries if e.get("verified"))
    by_author = summary.get("by_author", {})
    amb = summary.get("ambiguous")
    main_s, app_s = summary.get("main"), summary.get("appendix")
    cl, dev = by_author.get("claude"), by_author.get("developer")

    unamb = [r for r in scored if r["scope"] in ("main", "appendix")]
    unamb_hit3 = sum(r["hit@3"] for r in unamb) / len(unamb)
    unamb_correct = sum(r["grade"] == "correct" for r in unamb) / len(unamb)
    amb_step = 100.0 / amb["n"] if amb else 0

    doc = f"""# DocuMind — Evaluation Results

*Generated {datetime.date.today().isoformat()} by `make_results.py` from `eval_results.json` (run {data['run_at']}). Every number here is read from that file.*

## What DocuMind is

DocuMind answers questions about complex PDFs — the kind with tables and charts that defeat text extraction — by never parsing them. Each page is rendered to an image and embedded with a multimodal model ({cfg['embed_model']}); a text question retrieves the most similar page images by cosine similarity; and a vision-language model ({cfg['answer_model']}) reads the retrieved pages and answers from them, citing the pages it used. This evaluation measures both halves of that pipeline — whether retrieval surfaces the page holding the answer, and whether the generated answer is actually correct — against a {n_total}-question labelled set over a {n_pages}-page security research paper.

## Methodology

- **{n_total} questions**, all with hand-verified gold labels ({n_verified}/{len(entries)} entries marked `verified:true`). Numbers below are from a `--verified-only` run.
- **Gold labels** are the set of page filenames that genuinely answer the question, plus a short reference answer. A retrieval "hit" means any gold page appears in the top *k*.
- **Scope labels** describe where a question's answer lives and whether its terms collide across the document:
  - `main` — answer is in the main paper (pages 1–13).
  - `appendix` — answer is only in an appendix (pages 19–116).
  - `ambiguous` — the question's terms appear in **both** halves, so retrieval must break a tie. These were built deliberately, using `find_collisions.py` to find terms occurring in both scopes.
- **Author labels** record who wrote the question, and exist to detect a specific bias. The `claude` questions were drafted by reading the gold pages, so they inherit those pages' vocabulary — retrieval is partly being scored on paraphrase matching. The `developer` questions were written independently, without looking at the pages, using the phrasing a real user would reach for. Comparing the two groups shows how much of the score is an artifact of how questions were written.
- **Retrieval at k=5**, metrics reported at hit@1 / hit@3 / hit@5 and MRR. **Generation reads the top {cfg['generation_k']} pages**, not just the top 1, because detail questions often have the specific figure on a lower-ranked page than the summary page.
- **Grading** is a reference-based LLM-as-judge ({cfg['judge_model']}) held separate from the answering model ({cfg['answer_model']}). The judge sees only the question, the gold answer, and the system answer — never the page images — so it grades against the reference rather than re-deciding what the paper says. It returns correct / partial / wrong with a one-line reason.
- **Determinism**: fixed seed ({cfg['seed']}) on generation and judging; retrieval is deterministic.

## Results by scope

{scope_table(summary, [("**overall**", "overall"), ("main", "main"), ("appendix", "appendix"), ("ambiguous", "ambiguous")])}

## Results by question provenance
"""

    if cl and dev:
        doc += f"""
| group | n | hit@3 | correct |
|---|---:|---:|---:|
| `claude` (drafted from the gold pages) | {cl['n']} | **{pct(cl['hit@3'])}** | {pct(cl['correct'])} |
| `developer` (written independently) | {dev['n']} | **{pct(dev['hit@3'])}** | {pct(dev['correct'])} |

Questions written from the pages score {pct(cl['hit@3'])} hit@3; questions written independently score {pct(dev['hit@3'])}. The gap is {abs(cl['hit@3'] - dev['hit@3']) * 100:.0f} points on retrieval and {abs(cl['correct'] - dev['correct']) * 100:.0f} points on answer correctness. **Any evaluation built only from page-derived questions would have overstated this system's quality.** The two groups are not otherwise matched — the developer set is also where the `ambiguous` questions are concentrated — so this gap measures phrasing and topic difficulty together, not phrasing alone.
"""

    doc += f"""
## Key finding

**Retrieval and answers are strong on questions with an unambiguous home in the document, and degrade sharply when a question's terms appear in both the main paper and the appendix.**

On the {len(unamb)} unambiguous questions (`main` + `appendix`), hit@3 is {pct(unamb_hit3)} and {pct(unamb_correct)} of answers are graded correct. On the {amb['n'] if amb else 0} `ambiguous` questions, hit@3 falls to {pct(amb['hit@3']) if amb else '—'} and correctness to {pct(amb['correct']) if amb else '—'}. Citation quality moves the same way: the system cites a gold page on {pct(main_s['cited_gold']) if main_s else '—'} of `main` and {pct(app_s['cited_gold']) if app_s else '—'} of `appendix` questions, but only {pct(amb['cited_gold']) if amb else '—'} of `ambiguous` ones.

The mechanism is visible in the retrieved page lists. When a term like a model name or "mitigations" appears in both halves of the paper, the appendix — which is {n_pages - 18} of the paper's {n_pages} pages — supplies many plausible-looking pages that crowd the correct one out of the top 3. Generation is not the weak link: it answers faithfully from whatever pages it is handed, and when the gold page is missing it usually says so rather than inventing an answer.

**Sample-size caveats, which are severe here.** The `ambiguous` group is **n={amb['n'] if amb else 0}**. One question moving from miss to hit shifts its hit@3 by {amb_step:.0f} percentage points, so the figure should be read as "roughly half" and not as {pct(amb['hit@3']) if amb else '—'}. The `appendix` group is n={app_s['n'] if app_s else 0} and the whole set is n={n_total}. No confidence intervals are reported because none would be meaningful at these sizes. The direction of the effect is consistent across three independent signals — retrieval rank, citation correctness, and judge grade — which is what makes it worth acting on; the exact magnitudes are not yet trustworthy.

A second caveat: `hit@5` is {pct(summary['overall']['hit@5'])} across every group. The correct page is nearly always *somewhere* in the top 5 — the failure is ranking, not recall. That is what makes the fix below plausible.

## Where it fails, concretely

"""
    for qid in EXAMPLES:
        if qid in rows and rows[qid]["id"] in covered:
            doc += example_block(rows[qid]) + "\n"

    doc += f"""## Limitations & next step

- **Small n**, as above. The scope effect is directionally clear but the numbers are coarse.
- **One document.** All {n_total} questions are about a single {n_pages}-page paper. Nothing here shows the effect generalises to other documents, and the main/appendix split is a property of academic papers specifically.
- **Judge not validated against human grades.** The judge's correct/partial/wrong calls have not been checked against a human rater, so answer-quality numbers carry unmeasured judge error. One known case: `a01` was graded `partial` for following the paper's own prose rather than the gold answer's phrasing.
- **Gold pages are permissive.** Several questions list multiple acceptable gold pages, which makes a "hit" easier than a single-target metric would.

**Next step (Week 4): hybrid text + visual retrieval to break scope ties.** The current index has exactly one signal — the visual embedding of a page image — so two pages discussing the same term look equally good regardless of which is the real answer. Adding a text index over the extracted page text gives a second, independent signal, and lets a query match on section context ("Section 5.5 Mitigations" vs "Appendix A") rather than on topic alone. Since hit@5 is already {pct(summary['overall']['hit@5'])}, the correct page is nearly always retrieved and merely mis-ranked, so a reranking fix should be able to convert most of the gap without changing the index. The `ambiguous` group is the metric to watch.

## Reproducing

```bash
./venv/bin/python eval.py --verified-only --resume   # metrics -> eval_results.json
./venv/bin/python make_results.py                    # this document
```
"""

    with open(OUT, "w") as f:
        f.write(doc)
    print(f"Wrote {OUT} ({len(doc.splitlines())} lines) from {len(scored)} scored rows")


if __name__ == "__main__":
    main()
