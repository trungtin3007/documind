"""Retrieval sanity check: does the index actually discriminate between pages?

For each test question this prints the top-5 pages, the score spread, and where
the winner sits relative to the whole corpus. It also copies the top-3 page
images into results/qN/ so relevance can be judged by eye.

Run:  ./venv/bin/python diagnose_retrieval.py
"""
import os, shutil
import numpy as np
from retrieval import search, page_count, page_path

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
COPY_TOP_N = 3

# The paper: "Stealing Reasoning Traces from Proprietary LLM APIs".
QUESTIONS = [
    "How does the attack inject an encrypted reasoning trace into a weaker model to make it decode the trace verbatim?",
    "How many PII artifacts and credentials were recovered from the reasoning blocks scraped from public repositories?",
    "What cryptographic and system-level mitigations are proposed to secure client-side reasoning?",
    "How can an attacker hide an invisible prompt injection payload inside an encrypted reasoning block?",
    "Which Anthropic, OpenAI and Google models were tested, and what were the extraction success rates?",
]


def copy_top_pages(question_dir, ranked):
    """Copy the top-N page images into results/qN/, prefixed by rank."""
    os.makedirs(question_dir, exist_ok=True)
    for rank, (page, _) in enumerate(ranked[:COPY_TOP_N], 1):
        shutil.copy(page_path(page), os.path.join(question_dir, f"rank{rank}_{page}"))


def main():
    if os.path.isdir(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)                  # drop stale results from earlier runs

    n_pages = page_count()
    print(f"Index: {n_pages} pages | {len(QUESTIONS)} test questions\n")

    summary = []
    for qi, question in enumerate(QUESTIONS, 1):
        ranked = search(question, k=n_pages)        # full ranking, one embed call
        scores = np.array([s for _, s in ranked])
        top5 = ranked[:5]

        print(f"Q{qi}: {question}")
        for rank, (page, score) in enumerate(top5, 1):
            print(f"    {rank}. {page}   {score:.4f}")

        top1, top2, top5_score = scores[0], scores[1], scores[4]
        gap12 = top1 - top2
        spread = top1 - top5_score
        # How far the winner stands out from the corpus as a whole. Absolute
        # cosines sit in a narrow band, so this is the real signal.
        z = (top1 - scores.mean()) / scores.std()

        print(f"    spread : #1 {top1:.4f} | #5 {top5_score:.4f} | #1-#5 {spread:.4f} | gap #1->#2 {gap12:.4f}")
        print(f"    corpus : mean {scores.mean():.4f} | sd {scores.std():.4f} | "
              f"min {scores.min():.4f} -> #1 is {z:+.2f} sd above mean")

        qdir = os.path.join(RESULTS_DIR, f"q{qi}")
        copy_top_pages(qdir, ranked)
        print(f"    images : {os.path.relpath(qdir)}/ (top {COPY_TOP_N})\n")

        summary.append((qi, top5[0][0], top1, gap12, spread, z))

    print("=" * 78)
    print(f"{'Q':<3}{'top page':<16}{'#1 score':>10}{'gap 1->2':>10}{'#1-#5':>10}{'#1 z-score':>12}")
    for qi, page, top1, gap12, spread, z in summary:
        print(f"{qi:<3}{page:<16}{top1:>10.4f}{gap12:>10.4f}{spread:>10.4f}{z:>+12.2f}")


if __name__ == "__main__":
    main()
