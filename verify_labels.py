"""Show unverified eval labels with evidence, so gold labels can be checked by hand.

For each entry still marked verified:false, prints the question, the proposed
gold answer, and — for every gold page — an excerpt of that page's text around
the terms the gold answer claims are there. A page whose text matches nothing
in the gold answer is flagged: either the fact is only in a figure image, or
the label is wrong.

Also copies each gold page image into verify/<id>/ for eyeballing.

Run:  ./venv/bin/python verify_labels.py [--all] [--only id1,id2] [--no-copy]
"""
import os, re, json, shutil, argparse

import pymupdf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_SET = os.path.join(BASE_DIR, "eval_set.json")
VERIFY_DIR = os.path.join(BASE_DIR, "verify")
PDF = os.path.join(BASE_DIR, "sample.pdf")

SCOPE_ORDER = ("main", "appendix", "ambiguous")
EXCERPT_PAD = 110

STOPWORDS = set("""
the a an and or but for of to in on at by as is are was were be been with from that this these those
it its their there here what which who when where why how all any both each few more most other some
such no not only own same so than too very can will just should now also however thus therefore does
do did have has had having would could may might must shall about above after again against because
before below between during further once out over under up down off page pages answer correct model
models used using use also note figure table section appendix
""".split())


def keywords(text):
    """Distinctive terms a gold answer claims: numbers, names, and content words."""
    nums = set(re.findall(r"\d[\d,\.]*%?", text))
    words = {w for w in re.findall(r"[A-Za-z][A-Za-z0-9\-\.]{3,}", text)
             if w.lower() not in STOPWORDS}
    return nums, words


def page_number(page):
    return int(os.path.splitext(page)[0].split("_")[-1])


def excerpt(page_text, terms):
    """Text around the first matching term, plus how many terms matched."""
    flat = re.sub(r"\s+", " ", page_text)
    hits = [t for t in terms if t and t.lower() in flat.lower()]
    if not hits:
        return None, []
    anchor = max(hits, key=len)
    i = flat.lower().find(anchor.lower())
    start = max(0, i - EXCERPT_PAD)
    return flat[start:i + EXCERPT_PAD].strip(), hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="include already-verified entries")
    ap.add_argument("--only", help="comma-separated ids")
    ap.add_argument("--no-copy", action="store_true", help="skip copying page images")
    args = ap.parse_args()

    with open(EVAL_SET) as f:
        entries = json.load(f)

    if not args.all:
        entries = [e for e in entries if not e.get("verified")]
    if args.only:
        wanted = {i.strip() for i in args.only.split(",") if i.strip()}
        entries = [e for e in entries if e["id"] in wanted]

    if not entries:
        print("Nothing to verify — every entry is already verified:true.")
        return

    doc = pymupdf.open(PDF)
    if not args.no_copy and os.path.isdir(VERIFY_DIR):
        shutil.rmtree(VERIFY_DIR)

    flagged = []
    print(f"{len(entries)} entr{'y' if len(entries) == 1 else 'ies'} to verify\n")

    for scope in SCOPE_ORDER:
        group = [e for e in entries if e["scope"] == scope]
        if not group:
            continue
        print("=" * 100)
        print(f"SCOPE: {scope}  ({len(group)})")
        print("=" * 100)

        for e in group:
            print(f"\n[{e['id']}]  author={e.get('author', '?')}")
            print(f"  Q: {e['question']}")
            print(f"  gold_pages: {', '.join(e['gold_pages'])}")
            print(f"  gold_answer: {e['gold_answer']}")

            nums, words = keywords(e["gold_answer"])
            for page in e["gold_pages"]:
                n = page_number(page)
                text = doc[n - 1].get_text()
                snippet, hits = excerpt(text, list(nums) + sorted(words, key=len, reverse=True))
                num_hits = [x for x in nums if x.lower() in text.lower()]
                if snippet is None:
                    print(f"    p{n}: NO TEXT MATCH — fact may be image-only, or label is wrong")
                    flagged.append((e["id"], n))
                else:
                    got = f"{len(hits)} term(s)"
                    if nums:
                        got += f", numbers {sorted(num_hits) or 'NONE FOUND'}"
                        if not num_hits:
                            flagged.append((e["id"], n))
                    print(f"    p{n}: {got}")
                    print(f"        ...{snippet}...")

            if not args.no_copy:
                dest = os.path.join(VERIFY_DIR, e["id"])
                os.makedirs(dest, exist_ok=True)
                for page in e["gold_pages"]:
                    src = os.path.join(BASE_DIR, "pages", page)
                    if os.path.exists(src):
                        shutil.copy(src, os.path.join(dest, page))

    if not args.no_copy:
        print(f"\nPage images copied to {os.path.relpath(VERIFY_DIR)}/<id>/")
    if flagged:
        print("\nNEEDS A CLOSER LOOK (no text match, or a claimed number absent from the page):")
        for qid, n in flagged:
            print(f"  {qid}  p{n}")
    else:
        print("\nEvery gold page contains text supporting its gold answer.")


if __name__ == "__main__":
    main()
