"""Compare two or more eval runs (visual vs hybrid variants).

Prints per-scope retrieval metrics as both rates and question counts, then the
per-question rank changes against the first (baseline) file.

Run:  ./venv/bin/python compare_methods.py base.json cand1.json [cand2.json ...]
"""
import os, sys, json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCOPES = ("main", "appendix", "ambiguous")
METRICS = (("hit@1", "hit@1"), ("hit@3", "hit@3"), ("mrr", "MRR"))

DEFAULTS = ["eval_results.json", "eval_results_hybrid.json", "eval_results_hybrid_emb.json"]


def load(path):
    with open(path) as f:
        data = json.load(f)
    covered = set(data.get("summary_over", []))
    rows = {r["id"]: r for r in data["results"] if not covered or r["id"] in covered}
    return data.get("config", {}).get("method", os.path.basename(path)), rows


def agg(rows):
    n = len(rows)
    if not n:
        return None
    return {
        "n": n,
        "hit@1": sum(r["hit@1"] for r in rows) / n,
        "hit@3": sum(r["hit@3"] for r in rows) / n,
        "mrr": sum(r["reciprocal_rank"] for r in rows) / n,
        "hit@1_ct": sum(r["hit@1"] for r in rows),
        "hit@3_ct": sum(r["hit@3"] for r in rows),
    }


def label(method):
    return method.replace("hybrid-", "hyb-")[:11]


def main():
    paths = sys.argv[1:] or [os.path.join(BASE_DIR, p) for p in DEFAULTS]
    paths = [p for p in paths if os.path.exists(p)]
    if len(paths) < 2:
        sys.exit("Need at least two existing results files.")

    runs = [load(p) for p in paths]
    methods = [m for m, _ in runs]
    shared = [i for i in runs[0][1] if all(i in rows for _, rows in runs)]

    for path, (method, _) in zip(paths, runs):
        print(f"  {method:<18} {os.path.basename(path)}")
    print(f"\n{len(shared)} shared questions. Baseline = {methods[0]}.\n")

    groups = [("overall", shared)] + [
        (s, [i for i in shared if runs[0][1][i]["scope"] == s]) for s in SCOPES]

    for key, name in METRICS:
        head = f"{name:<11}{'n':>4}" + "".join(f"{label(m):>14}" for m in methods)
        print(head)
        print("-" * len(head))
        for gname, ids in groups:
            if not ids:
                continue
            cells = ""
            for _, rows in runs:
                a = agg([rows[i] for i in ids])
                if key == "mrr":
                    cells += f"{a[key]:>14.2f}"
                else:
                    cells += f"{a[key + '_ct']:>9d}/{a['n']:<4d}"
            print(f"{gname:<11}{len(ids):>4}{cells}")
        print()

    base_rows = runs[0][1]
    for method, rows in runs[1:]:
        moved = [(i, base_rows[i]["gold_rank"], rows[i]["gold_rank"])
                 for i in shared if base_rows[i]["gold_rank"] != rows[i]["gold_rank"]]
        print(f"rank changes vs {methods[0]}: {method}")
        if not moved:
            print("  (none)")
        for qid, before, after in sorted(moved):
            verdict = "better" if (after or 99) < (before or 99) else "WORSE"
            print(f"  {qid:<5} {base_rows[qid]['scope']:<10} "
                  f"{str(before or '-'):>3} -> {str(after or '-'):<3}  {verdict}")
        print()


if __name__ == "__main__":
    main()
