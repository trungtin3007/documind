"""Evaluate DocuMind retrieval and answer quality against eval_set.json.

Retrieval is scored at k=5 (hit@1 / hit@3 / hit@5 / MRR); the answer is graded
by a reference-based judge (judge.py). Every metric is broken down by scope,
so the main-paper vs appendix gap we saw in Weeks 1-2 is measurable.

Run:  ./venv/bin/python eval.py [--verified-only] [--limit N] [--out FILE]
"""
import os, sys, json, time, argparse, datetime

import retrieval
import generate
import judge

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_SET = os.path.join(BASE_DIR, "eval_set.json")
RESULTS = os.path.join(BASE_DIR, "eval_results.json")

RETRIEVAL_K = 5                 # metrics measured at k=5
GENERATION_K = generate.TOP_K   # generation still reads the top 3
SCOPES = ("main", "appendix", "ambiguous")   # ambiguous: topic appears in both scopes
GRADES = ("correct", "partial", "wrong", "error", "ungraded")
JUDGED_GRADES = ("correct", "partial", "wrong")   # a judge actually ran and returned these
PAUSE_SECONDS = 8.0             # free tier allows 20 requests/min; we make 2 per question


def rank_of_first_gold(retrieved, gold_pages):
    """1-based rank of the first retrieved page that is a gold page, else None."""
    gold = set(gold_pages)
    for rank, (page, _) in enumerate(retrieved, 1):
        if page in gold:
            return rank
    return None


def has_answer(row):
    """True if a stored row already holds a usable generation."""
    return bool(row and row.get("answer") and row.get("retrieved_pages"))


def is_judged(row):
    return bool(row and row.get("grade") in JUDGED_GRADES)


def is_complete(row, do_judge, method):
    """A stored row is reusable as-is only if it came from the same method."""
    return (has_answer(row) and row.get("method", "visual") == method
            and (is_judged(row) or not do_judge))


def top_pages(retrieved, n=None):
    pages = [p for p, _ in retrieved] if retrieved and isinstance(retrieved[0], tuple) \
        else [r["page"] for r in retrieved]
    return pages[:n] if n else pages


def evaluate_one(entry, prior=None, do_judge=True, method="visual", do_generate=True):
    """Run retrieval + generation + judging, reusing whatever `prior` already has.

    Generation is reused only when the pages handed to the model are identical,
    so switching retrieval method re-generates exactly the rows whose top-3
    actually changed and no others. Judging is cached on top of that.
    """
    question = entry["question"]
    retrieved = retrieval.search_by_method(question, k=RETRIEVAL_K, method=method)
    sent = top_pages(retrieved, GENERATION_K)

    # Generation sees only the top 3, matching the shipped behaviour.
    reuse_gen = has_answer(prior) and top_pages(prior["retrieved_pages"], GENERATION_K) == sent
    if reuse_gen:
        result = {"answer": prior["answer"], "cited_pages": prior["cited_pages"]}
    elif do_generate:
        result = generate.answer(question, retrieved=retrieved[:GENERATION_K])
    else:
        # Retrieval-only run: measure ranking without paying to answer.
        result = {"answer": "", "cited_pages": []}

    if reuse_gen and is_judged(prior):
        verdict, reason = prior["grade"], prior.get("judge_reason", "")
    elif not result["answer"]:
        verdict, reason = "ungraded", "Retrieval-only run (--no-generate)."
    elif do_judge:
        verdict, reason = judge.grade(question, entry["gold_answer"], result["answer"])
    else:
        verdict, reason = "ungraded", "Judging skipped (--no-judge)."

    rank = rank_of_first_gold(retrieved, entry["gold_pages"])
    return {
        "id": entry["id"],
        "question": question,
        "scope": entry["scope"],
        "method": method,
        "verified": entry.get("verified", False),
        "gold_pages": entry["gold_pages"],
        "retrieved_pages": [{"page": p, "score": round(s, 4)} for p, s in retrieved],
        "gold_rank": rank,
        "hit@1": rank is not None and rank <= 1,
        "hit@3": rank is not None and rank <= 3,
        "hit@5": rank is not None and rank <= 5,
        "reciprocal_rank": (1.0 / rank) if rank else 0.0,
        "answer": result["answer"],
        "cited_pages": result["cited_pages"],
        "cited_gold": bool(set(result["cited_pages"]) & set(entry["gold_pages"])),
        "grade": verdict,
        "judge_reason": reason,
    }


def _error_row(entry, exc, method="visual"):
    """Placeholder row for a question whose API calls failed outright."""
    return {
        "id": entry["id"], "question": entry["question"], "scope": entry["scope"],
        "method": method,
        "verified": entry.get("verified", False), "gold_pages": entry["gold_pages"],
        "retrieved_pages": [], "gold_rank": None,
        "hit@1": False, "hit@3": False, "hit@5": False, "reciprocal_rank": 0.0,
        "answer": "", "cited_pages": [], "cited_gold": False,
        "grade": "error", "judge_reason": f"{type(exc).__name__}: {exc}"[:200],
    }


def summarize(rows):
    """Aggregate metrics for one group of rows."""
    n = len(rows)
    if not n:
        return None
    out = {"n": n}
    for k in ("hit@1", "hit@3", "hit@5"):
        out[k] = sum(r[k] for r in rows) / n
    out["mrr"] = sum(r["reciprocal_rank"] for r in rows) / n
    out["cited_gold"] = sum(r["cited_gold"] for r in rows) / n
    # Answer quality is a rate over rows a judge actually graded, so skipping the
    # judge lowers coverage rather than silently deflating the scores.
    judged = [r for r in rows if r["grade"] in JUDGED_GRADES]
    out["n_judged"] = len(judged)
    for g in GRADES:
        out[g] = (sum(r["grade"] == g for r in judged) / len(judged)) if judged else None
    for g in ("error", "ungraded"):
        out[g] = sum(r["grade"] == g for r in rows) / n
    return out


def build_summary(rows):
    summary = {"overall": summarize(rows)}
    for scope in SCOPES:
        s = summarize([r for r in rows if r["scope"] == scope])
        if s:
            summary[scope] = s
    # Provenance split: 'claude' questions were drafted from the gold pages and
    # echo their wording; 'developer' ones were written independently.
    by_author = {}
    for author in sorted({r.get("author", "?") for r in rows}):
        s = summarize([r for r in rows if r.get("author", "?") == author])
        if s:
            by_author[author] = s
    summary["by_author"] = by_author
    return summary


def print_table(summary, rows, provisional):
    cols = [("hit@1", "hit@1"), ("hit@3", "hit@3"), ("hit@5", "hit@5"), ("mrr", "MRR"),
            ("cited_gold", "cite-gold")]
    any_judged = any(r["grade"] in JUDGED_GRADES for r in rows)
    if any_judged:
        cols += [("n_judged", "judged"), ("correct", "correct"),
                 ("partial", "partial"), ("wrong", "wrong")]
    if any(r["grade"] == "error" for r in rows):
        cols.append(("error", "error"))
    if any(r["grade"] == "ungraded" for r in rows):
        cols.append(("ungraded", "ungraded"))

    width = 16 + 11 * len(cols)
    print("\n" + "=" * width)
    if provisional:
        print("PROVISIONAL — gold labels are unverified (verified:false). Do not trust these numbers yet.")
        print("=" * width)
    if not any_judged:
        print("Retrieval only (--no-judge): answer quality not measured.")
    print(f"{'group':<12}{'n':>4}" + "".join(f"{label:>11}" for _, label in cols))
    print("-" * width)
    def row_for(label, s):
        cells = ""
        for key, _ in cols:
            v = s.get(key)
            if v is None:
                cells += f"{'-':>11}"
            elif key == "n_judged":
                cells += f"{v:>11d}"
            else:
                cells += f"{v:>11.2f}"
        print(f"{label:<12}{s['n']:>4}{cells}")

    for group in ("overall",) + SCOPES:
        s = summary.get(group)
        if s:
            row_for(group, s)

    by_author = summary.get("by_author") or {}
    if len(by_author) > 1:
        print("-" * width)
        for author, s in by_author.items():
            row_for(f"[{author}]", s)

    print("\nper question:")
    print(f"  {'id':<5}{'scope':<10}{'rank':>5}  {'grade':<9}{'cited':<7}reason")
    print("  " + "-" * 88)
    for r in sorted(rows, key=lambda r: r["id"]):
        rank = r["gold_rank"] if r["gold_rank"] else "-"
        cited = "gold" if r["cited_gold"] else ("none" if not r["cited_pages"] else "other")
        print(f"  {r['id']:<5}{r['scope']:<10}{str(rank):>5}  {r['grade']:<9}{cited:<7}{r['judge_reason'][:52]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verified-only", action="store_true",
                    help="evaluate only entries with verified:true")
    ap.add_argument("--limit", type=int, help="evaluate at most N entries (smoke test)")
    ap.add_argument("--only", help="comma-separated question ids to run, e.g. u04,u05,u06")
    ap.add_argument("--no-judge", action="store_true",
                    help="skip the judge call; generation still runs (halves API usage)")
    ap.add_argument("--no-generate", action="store_true",
                    help="retrieval metrics only: no generation and no judging (zero LLM calls)")
    ap.add_argument("--out", default=RESULTS, help="where to write results JSON")
    ap.add_argument("--pause", type=float, default=PAUSE_SECONDS,
                    help="seconds between questions, to stay under the rate limit")
    ap.add_argument("--method", choices=list(retrieval.METHODS) + list(retrieval.METHOD_ALIASES),
                    default=retrieval.DEFAULT_METHOD,
                    help="retrieval method: visual (baseline) or hybrid (BM25 re-rank)")
    ap.add_argument("--cache-from", help="seed generation/judge cache from another results file")
    ap.add_argument("--summarize-only", action="store_true",
                    help="re-aggregate the summary from stored rows; runs nothing, no API calls")
    ap.add_argument("--resume", action="store_true",
                    help="reuse stored rows; generation and judging are cached separately")
    args = ap.parse_args()
    do_generate = not args.no_generate
    do_judge = not (args.no_judge or args.no_generate)

    with open(EVAL_SET) as f:
        all_entries = json.load(f)
    entries = all_entries

    if args.verified_only:
        entries = [e for e in entries if e.get("verified")]
        if not entries:
            sys.exit("No entries with verified:true. Verify some gold labels first, "
                     "or drop --verified-only for a provisional run.")
    if args.only:
        wanted = [i.strip() for i in args.only.split(",") if i.strip()]
        known = {e["id"] for e in entries}
        missing = [i for i in wanted if i not in known]
        if missing:
            sys.exit(f"Unknown question id(s): {', '.join(missing)}")
        entries = [e for e in entries if e["id"] in wanted]
    if args.limit:
        entries = entries[:args.limit]

    # Always load existing results so batched runs can merge into them; --resume
    # additionally reuses their work. Generation and judging are cached
    # independently, so a row that generated but failed to grade only pays for
    # the judge call on the next run.
    prior = {}
    if args.cache_from and os.path.exists(args.cache_from):
        with open(args.cache_from) as f:
            prior = {r["id"]: r for r in json.load(f).get("results", [])}
    if os.path.exists(args.out):
        with open(args.out) as f:
            out_rows = json.load(f).get("results", [])
        # One results file holds one retrieval method. Writing a second method
        # into it silently corrupts the other method's rows, which is exactly
        # how the visual baseline once got embedding rows merged into it.
        clashing = sorted({r.get("method", "visual") for r in out_rows
                           if r.get("method", "visual") != args.method})
        if clashing and not args.summarize_only:
            sys.exit(f"{os.path.basename(args.out)} already holds rows from "
                     f"{', '.join(clashing)}, but this run uses {args.method}. "
                     f"Write to a different --out file.")
        with open(args.out) as f:
            for row in json.load(f).get("results", []):
                # Prefer whichever source actually carries a generation: a
                # retrieval-only run leaves blank rows here that must not
                # clobber reusable answers seeded from --cache-from.
                if has_answer(row) or row["id"] not in prior:
                    prior[row["id"]] = row

    rows, todo = [], []
    if args.summarize_only:
        # Re-aggregate what is already on disk. Used after a narrow --only run,
        # whose summary would otherwise describe just that batch.
        rows = [prior[e["id"]] for e in entries if e["id"] in prior]
        print(f"Summarising {len(rows)} stored rows; nothing to run.")
    else:
        for entry in entries:
            stored = prior.get(entry["id"]) if args.resume else None
            if is_complete(stored, do_judge, args.method) and not args.no_generate:
                rows.append(stored)
            else:
                todo.append((entry, stored))

    if args.resume:
        regen = sum(1 for _, s in todo if not has_answer(s))
        print(f"Resuming: {len(rows)} rows reused, {len(todo)} to run "
              f"({regen} need generation, {len(todo) - regen} need judging only)")

    print(f"Evaluating {len(todo)} questions (method={args.method}, retrieval k={RETRIEVAL_K}, "
          f"generation={'on' if do_generate else 'OFF'}, "
          f"judge={judge.JUDGE_MODEL if do_judge else 'off'})")

    for i, (entry, stored) in enumerate(todo, 1):
        if i > 1 and args.pause:
            time.sleep(args.pause)
        try:
            row = evaluate_one(entry, prior=stored, do_judge=do_judge,
                               method=args.method, do_generate=do_generate)
        except Exception as exc:
            # Keep the run alive so one failure does not discard the whole batch;
            # the error stays visible as its own grade rather than counting as wrong.
            print(f"  [{i}/{len(todo)}] {entry['id']}  ERROR: {type(exc).__name__}")
            row = _error_row(entry, exc, method=args.method)
        rows.append(row)
        if row["grade"] != "error":
            mark = "hit" if row["hit@3"] else "MISS"
            print(f"  [{i}/{len(todo)}] {row['id']}  rank={row['gold_rank'] or '-'} "
                  f"{mark:<4} grade={row['grade']}")

    # Merge over anything already on disk so batched runs (--only) accumulate
    # instead of discarding the questions this batch did not touch.
    merged = {r["id"]: r for r in prior.values() if r["id"] in {e["id"] for e in all_entries}}
    merged.update({r["id"]: r for r in rows})
    # Backfill provenance onto rows (cached rows predate the author field).
    author_of = {e["id"]: e.get("author", "?") for e in all_entries}
    for row in merged.values():
        row["author"] = author_of.get(row["id"], "?")

    stored_rows = [merged[e["id"]] for e in all_entries if e["id"] in merged]

    # The file keeps every row, but the summary must describe only the questions
    # this run selected — otherwise --verified-only would silently report
    # numbers that include unverified rows carried over from disk.
    rows = [merged[e["id"]] for e in entries if e["id"] in merged]

    provisional = any(not e.get("verified") for e in entries if e["id"] in merged)
    summary = build_summary(rows)
    payload = {
        "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "provisional": provisional,
        "summary_over": [r["id"] for r in rows],
        "config": {
            "retrieval_k": RETRIEVAL_K,
            "generation_k": GENERATION_K,
            "answer_model": generate.MODEL,
            "judge_model": judge.JUDGE_MODEL,
            "embed_model": retrieval.MODEL,
            "seed": generate.SEED,
            "method": args.method,
        },
        "summary": summary,
        "results": stored_rows,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print_table(summary, rows, provisional)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
