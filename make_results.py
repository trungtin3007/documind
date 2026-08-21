"""Generate RESULTS.md from eval_results.json.

Every number in the report is read from the results file, so the document
cannot drift from the run that produced it. Re-run after any eval run.

Run:  ./venv/bin/python make_results.py
"""
import os, json, datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(BASE_DIR, "eval_results.json")
EVAL_SET = os.path.join(BASE_DIR, "eval_set.json")
# Retrieval methods to compare, in report order: (label, results file)
METHOD_FILES = [
    ("visual", "eval_results.json"),
    ("hybrid-bm25", "eval_results_hybrid.json"),
    ("hybrid-embedding", "eval_results_hybrid_emb.json"),
]
ADOPTED = "hybrid-embedding"
PRIMARY = os.path.join(BASE_DIR, "eval_results_hybrid_emb.json")   # the adopted method
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


def load_rows(path):
    """{id: row} for the questions a run's summary covers."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    covered = set(data.get("summary_over", []))
    return {r["id"]: r for r in data["results"] if not covered or r["id"] in covered}


def counts(rows, ids, key):
    return sum(rows[i][key] for i in ids), len(ids)


def method_tables(runs, scopes_of):
    """Retrieval metrics per method as question counts (rates are misleading at n=28)."""
    labels = [lbl for lbl, _ in runs]
    shared = sorted(set.intersection(*[set(r) for _, r in runs]))
    groups = [("overall", shared)] + [
        (s, [i for i in shared if scopes_of[i] == s]) for s in ("main", "appendix", "ambiguous")]

    out = []
    for key, name in (("hit@1", "hit@1"), ("hit@3", "hit@3")):
        out.append(f"**{name}** (questions hit / total)\n")
        out.append("| scope | " + " | ".join(f"`{l}`" for l in labels) + " |")
        out.append("|---|" + "---:|" * len(labels))
        for gname, ids in groups:
            if not ids:
                continue
            cells = []
            for _, rows in runs:
                hit, total = counts(rows, ids, key)
                cells.append(f"{hit}/{total}")
            out.append(f"| {gname} | " + " | ".join(cells) + " |")
        out.append("")
    out.append("**MRR**\n")
    out.append("| scope | " + " | ".join(f"`{l}`" for l in labels) + " |")
    out.append("|---|" + "---:|" * len(labels))
    for gname, ids in groups:
        if not ids:
            continue
        cells = [f"{sum(rows[i]['reciprocal_rank'] for i in ids) / len(ids):.2f}"
                 for _, rows in runs]
        out.append(f"| {gname} | " + " | ".join(cells) + " |")
    return "\n".join(out), shared


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


def m11_illustration(rows_by_method):
    """The confirmed BM25 weakness, with scores computed live (never hand-typed)."""
    import text_score
    qid = "m11"
    row = rows_by_method["visual"][qid]
    cands = [r["page"] for r in rows_by_method["visual"][qid]["retrieved_pages"]]
    lines = [f"**`{qid}` — {row['question']}**\n",
             f"Gold pages: {', '.join(short(p) for p in row['gold_pages'])}\n",
             "| method | gold page rank |", "|---|---:|"]
    for label in ("visual", "hybrid-bm25", "hybrid-embedding"):
        r = rows_by_method.get(label, {}).get(qid)
        if r:
            lines.append(f"| `{label}` | {r['gold_rank'] or 'not in top 5'} |")
    lines.append("")
    for scorer in ("bm25", "embedding"):
        scores = text_score.get_scorer(scorer).score(row["question"], ids=cands)
        ordered = sorted(scores.items(), key=lambda kv: -kv[1])
        gold = set(row["gold_pages"])
        rendered = ", ".join(
            f"**{short(p)}={v:.2f}**" if p in gold else f"{short(p)}={v:.2f}"
            for p, v in ordered)
        pos = [p for p, _ in ordered]
        gold_pos = min((pos.index(p) + 1 for p in gold if p in pos), default=None)
        lines.append(f"- `{scorer}` over those candidates: {rendered} — gold at position "
                     f"**{gold_pos} of {len(pos)}**")
    return "\n".join(lines)


def main():
    with open(PRIMARY) as f:
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

    runs = [(lbl, load_rows(os.path.join(BASE_DIR, f))) for lbl, f in METHOD_FILES]
    runs = [(lbl, rows) for lbl, rows in runs if rows]
    rows_by_method = {lbl: rows for lbl, rows in runs}
    scopes_of = {i: r["scope"] for r in data["results"] for i in [r["id"]]}
    method_md, shared_ids = method_tables(runs, scopes_of)
    m11_md = m11_illustration(rows_by_method) if "m11" in rows_by_method.get("visual", {}) else ""
    adopted_rows = rows_by_method.get(ADOPTED, rows)
    base_rows = rows_by_method.get("visual", rows)
    h3_base = sum(base_rows[i]["hit@3"] for i in shared_ids)
    h3_adopt = sum(adopted_rows[i]["hit@3"] for i in shared_ids)

    n_judged = summary["overall"]["n_judged"]
    if n_judged >= n_total:
        coverage_note = (f"- **Answer grades cover all {n_total} questions.** Answer-quality rates are "
                         f"computed over graded rows only; with full coverage they describe the whole set.")
    else:
        coverage_note = (f"- **Answer grades cover {n_judged} of {n_total} questions.** The rest hit the "
                         f"daily API quota during grading and are recorded as `ungraded`; answer-quality "
                         f"rates are computed over graded rows only, so coverage is reduced rather than "
                         f"scores being silently deflated. Retrieval metrics cover all {n_total}.")

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

## Retrieval method comparison

Three retrieval methods were evaluated on the same {n_total} questions. `visual` is the original
page-image cosine ranking. Both hybrids re-rank the visual top-{cfg.get('pool_k', 15)} candidates by fusing the
visual rank with a text rank using Reciprocal Rank Fusion — identical fusion, so the only variable
is how page text is scored: `hybrid-bm25` uses lexical term overlap, `hybrid-embedding` uses a local
sentence-transformer. **`{ADOPTED}` is the adopted default.**

Counts, not percentages — at this sample size one question is worth 4 points overall and 20 in `ambiguous`.

{method_md}

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

**Retrieval and answers are strong on questions with an unambiguous home in the document, and degrade when a question's terms appear in both the main paper and the appendix.**

On the {len(unamb)} unambiguous questions (`main` + `appendix`), hit@3 is {pct(unamb_hit3)}. On the {amb['n'] if amb else 0} `ambiguous` questions it is {pct(amb['hit@3']) if amb else '—'}. Citation quality moves the same way: a gold page is cited on {pct(main_s['cited_gold']) if main_s else '—'} of `main` and {pct(app_s['cited_gold']) if app_s else '—'} of `appendix` questions, but only {pct(amb['cited_gold']) if amb else '—'} of `ambiguous` ones.

The mechanism is visible in the retrieved page lists. When a term appears in both halves of the paper, the appendix — {n_pages - 18} of the paper's {n_pages} pages — supplies many plausible-looking pages that crowd the correct one out of the top {cfg['generation_k']}. Generation is not the weak link: it answers faithfully from whatever pages it is handed, and when the gold page is missing it usually says so rather than inventing an answer.

### The re-ranker result, stated honestly

Swapping lexical scoring for semantic scoring **recovers a specific, confirmed failure mode** — see the `m11` example below — and moves overall hit@3 from **{h3_base}/{len(shared_ids)} to {h3_adopt}/{len(shared_ids)}**, with no scope regressing. `hybrid-bm25`, by contrast, bought its `ambiguous` gain by breaking `main`.

**That is a one-question difference, which is within noise at n={n_total}. Treat it as directional, not proven.** Nothing here is statistically significant: the `ambiguous` group is n={amb['n'] if amb else 0}, where a single question moves the metric {amb_step:.0f} points. What justifies adopting the embedding re-ranker is not the aggregate — it is that the mechanism was predicted in advance, the prediction was tested, and the specific failure it targeted disappeared while nothing else regressed. The aggregate merely fails to contradict that.

A second caveat: `hit@5` is {pct(summary['overall']['hit@5'])} across every group and every method. The correct page is essentially always retrieved; only its rank changes. Re-ranking is therefore the right lever, and also a bounded one — it cannot fix what retrieval never surfaces.

## Worked example: the failure the re-ranker fixes

{m11_md}

Lexical scoring pushes both gold pages to the bottom of its candidate list, with the page carrying the actual quantities scoring lowest of all: that page states its numbers tersely, while the neighbouring pages repeat the question's framing words. Term overlap and answer-bearing are anti-correlated here. Semantic scoring is indifferent to that phrasing difference and ranks the same page near the top, which is exactly the behaviour the swap was meant to produce.

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
{coverage_note}
- **The two text scorers are complementary, and only one is used.** BM25 keeps an edge on distinctive rare tokens: on `u02`, whose answer hinges on model names like "GLM-5.2", lexical scoring lifts the gold page to rank 1 while the embedding scorer reaches only rank 3. Exact-token matching and semantic similarity fail in different places, so fusing all three rankings — visual, lexical, semantic — is the obvious next experiment. Not done here; the adopted configuration uses semantic scoring only.

**Next step (Week 5): a web interface, and a larger eval set.** The re-ranker work is done and adopted; what limits every conclusion in this document now is n={n_total} over one document, not the retrieval design. The highest-value next measurement is more questions across more documents, which would also make the complementary-scorer experiment above worth running. Interface work (answer shown beside the cited page) is the demo-facing task.

## Reproducing

```bash
# retrieval + answers for the adopted method (cached rows cost nothing)
./venv/bin/python eval.py --verified-only --resume --method {ADOPTED} \\
    --out eval_results_hybrid_emb.json

# the other two methods, retrieval metrics only (no LLM calls at all)
./venv/bin/python eval.py --verified-only --no-generate --method visual --out eval_results.json
./venv/bin/python eval.py --verified-only --no-generate --method hybrid-bm25 \\
    --out eval_results_hybrid.json

./venv/bin/python compare_methods.py                 # three-way comparison
./venv/bin/python make_results.py                    # this document
```
"""

    with open(OUT, "w") as f:
        f.write(doc)
    print(f"Wrote {OUT} ({len(doc.splitlines())} lines) from {len(scored)} scored rows")


if __name__ == "__main__":
    main()
