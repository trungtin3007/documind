"""Find topics that appear in BOTH the main paper and the appendix.

These are the questions worth writing: when a term lives in both scopes,
retrieval has to pick a scope, and that is where DocuMind failed in Week 2
("which models were tested" pulled Appendix B.3's scorer models instead of
the main experiments).

Run:  ./venv/bin/python find_collisions.py [--top N] [--pdf sample.pdf]
"""
import re, argparse, collections

import pymupdf

# Boundaries checked against the PDF: main paper runs to the conclusion on p13,
# p14-18 are acknowledgments / references / appendix TOC (skipped -- reference
# lists name every model in the field and would produce fake collisions),
# and the appendix proper starts at p19.
MAIN_RANGE = (1, 13)
APPENDIX_RANGE = (19, None)      # None = to the end

MIN_TERM_LEN = 4
MAX_PAGE_SHARE = 0.35            # drop terms so common they are just the paper's vocabulary
TOP_DEFAULT = 12

# Terms worth reporting even if the automatic ranking buries them.
SEED_TERMS = [
    "success rate", "extraction rate", "queries", "query", "token count",
    "success", "prefill", "judge", "scorer", "summary", "temperature",
]

MODEL_RE = re.compile(
    r"\b(opus|sonnet|haiku|fable|gpt|gemini|kimi|glm|deepseek|inkling|claude|o4|gpt-oss)\b", re.I)
DATASET_RE = re.compile(
    r"\b(hle|aime|codeforces|clawbench|harmbench|math500|openthoughts|humanity)\b", re.I)

STOPWORDS = set("""
the a an and or but if then than that this these those with without within into onto from for
of to in on at by as is are was were be been being it its it's we our us they them their there
here what which who whom whose when where why how all any both each few more most other some such
no nor not only own same so too very can will just should now also however thus therefore
does do did doing have has had having would could may might must shall about above after again
against because before below between during further once out over under up down off
one two three four five six seven eight nine ten first second third
figure table section appendix page paper work model models used using use uses
show shows shown see e.g i.e et al fig eq
""".split())


def page_text(doc, page_range):
    """Yield (page_number, lowercased text) for a 1-based inclusive range."""
    start, end = page_range
    end = end or doc.page_count
    for n in range(start, min(end, doc.page_count) + 1):
        yield n, doc[n - 1].get_text().lower()


def terms_in(text):
    """Unigrams and bigrams worth considering as topics."""
    tokens = re.findall(r"[a-z][a-z0-9\-\.]{2,}", text)
    tokens = [t.strip(".-") for t in tokens]
    keep = [t for t in tokens if len(t) >= MIN_TERM_LEN and t not in STOPWORDS]

    out = set(keep)
    for a, b in zip(tokens, tokens[1:]):
        if a in STOPWORDS or b in STOPWORDS:
            continue
        if len(a) >= 3 and len(b) >= 3:
            out.add(f"{a} {b}")
    return out


def build_index(doc):
    """term -> {'main': set(pages), 'appendix': set(pages)}"""
    index = collections.defaultdict(lambda: {"main": set(), "appendix": set()})
    for scope, rng in (("main", MAIN_RANGE), ("appendix", APPENDIX_RANGE)):
        for page_no, text in page_text(doc, rng):
            for term in terms_in(text):
                index[term][scope].add(page_no)
    return index


def seed_hits(doc):
    """Literal substring search for the hand-picked terms."""
    hits = {t: {"main": set(), "appendix": set()} for t in SEED_TERMS}
    for scope, rng in (("main", MAIN_RANGE), ("appendix", APPENDIX_RANGE)):
        for page_no, text in page_text(doc, rng):
            for term in SEED_TERMS:
                if term in text:
                    hits[term][scope].add(page_no)
    return {t: v for t, v in hits.items() if v["main"] and v["appendix"]}


def collisions(index, n_pages):
    """Terms present in both scopes, excluding the paper's ubiquitous vocabulary."""
    out = []
    for term, scopes in index.items():
        main, app = scopes["main"], scopes["appendix"]
        if not main or not app:
            continue
        if (len(main) + len(app)) > MAX_PAGE_SHARE * n_pages:
            continue
        # Reward terms that are solidly present on both sides, not one stray mention.
        out.append((min(len(main), len(app)), len(main) + len(app), term, main, app))
    out.sort(key=lambda r: (-r[0], -r[1], r[2]))
    return out


def fmt(pages, limit=8):
    shown = sorted(pages)[:limit]
    s = ", ".join(f"p{p}" for p in shown)
    return s + (f" (+{len(pages) - limit})" if len(pages) > limit else "")


def report(title, rows, top):
    if not rows:
        return
    print(f"\n## {title}")
    print(f"{'term':<28}{'main pages':<34}appendix pages")
    print("-" * 96)
    for _, _, term, main, app in rows[:top]:
        print(f"{term:<28}{fmt(main, 5):<34}{fmt(app, 5)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="sample.pdf")
    ap.add_argument("--top", type=int, default=TOP_DEFAULT)
    args = ap.parse_args()

    doc = pymupdf.open(args.pdf)
    n_pages = doc.page_count
    print(f"Scanning {args.pdf}: main = p{MAIN_RANGE[0]}-{MAIN_RANGE[1]}, "
          f"appendix = p{APPENDIX_RANGE[0]}-{n_pages} "
          f"(p14-18 skipped: acknowledgments, references, appendix TOC)")

    rows = collisions(build_index(doc), n_pages)

    models = [r for r in rows if MODEL_RE.search(r[2])]
    datasets = [r for r in rows if DATASET_RE.search(r[2])]
    named = {id(r) for r in models + datasets}
    other = [r for r in rows if id(r) not in named]

    report("Model names (collide across scopes)", models, args.top)
    report("Datasets / benchmarks", datasets, args.top)
    report("Other topics", other, args.top)

    seeds = seed_hits(doc)
    if seeds:
        print("\n## Hand-picked terms")
        print(f"{'term':<28}{'main pages':<34}appendix pages")
        print("-" * 96)
        for term, v in sorted(seeds.items()):
            print(f"{term:<28}{fmt(v['main'], 5):<34}{fmt(v['appendix'], 5)}")

    print("\nA question is a good 'ambiguous' candidate when its answer is in one scope "
          "but its terms appear in both.")


if __name__ == "__main__":
    main()
