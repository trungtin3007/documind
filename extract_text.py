"""Extract the PDF text layer for each page into pages_text.json.

Free and local — no API calls. Produces {page_filename: text} keyed the same
way as the visual index, so the two can be fused later.

Reports pages whose text layer looks empty or garbled rather than silently
falling back to OCR: an empty text layer on a figure-only page is expected,
but a garbled one means the whole text signal is untrustworthy.

Run:  ./venv/bin/python extract_text.py [--pdf sample.pdf] [--show N]
"""
import os, re, json, glob, argparse

import pymupdf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PAGES_DIR = os.path.join(BASE_DIR, "pages")
OUT = os.path.join(BASE_DIR, "pages_text.json")

THIN_CHARS = 200          # below this, the page carries little usable text
GARBLED_ALPHA_RATIO = 0.55   # printable text should be mostly letters/spaces


def quality(text):
    """Cheap signals for 'is this text layer usable'."""
    stripped = text.strip()
    if not stripped:
        return {"chars": 0, "alpha_ratio": 0.0, "avg_word": 0.0, "replacement": 0}
    letters = sum(c.isalpha() or c.isspace() for c in stripped)
    words = re.findall(r"[A-Za-z]+", stripped)
    return {
        "chars": len(stripped),
        "alpha_ratio": letters / len(stripped),
        "avg_word": (sum(map(len, words)) / len(words)) if words else 0.0,
        "replacement": stripped.count("�"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=os.path.join(BASE_DIR, "sample.pdf"))
    ap.add_argument("--show", type=int, default=2, help="sample extractions to print")
    args = ap.parse_args()

    page_files = sorted(os.path.basename(p) for p in glob.glob(os.path.join(PAGES_DIR, "*.png")))
    doc = pymupdf.open(args.pdf)
    if len(page_files) != doc.page_count:
        raise SystemExit(f"Mismatch: {len(page_files)} page images vs {doc.page_count} PDF pages. "
                         "The text layer would be misaligned with the visual index.")

    texts, stats = {}, {}
    for i, name in enumerate(page_files):
        text = doc[i].get_text()
        texts[name] = text
        stats[name] = quality(text)

    with open(OUT, "w") as f:
        json.dump(texts, f)

    total = len(page_files)
    empty = [n for n, s in stats.items() if s["chars"] == 0]
    thin = [n for n, s in stats.items() if 0 < s["chars"] < THIN_CHARS]
    garbled = [n for n, s in stats.items()
               if s["chars"] >= THIN_CHARS and
               (s["alpha_ratio"] < GARBLED_ALPHA_RATIO or s["replacement"] > 0)]
    chars = [s["chars"] for s in stats.values()]

    print(f"Extracted text for {total} pages -> {os.path.relpath(OUT)}")
    print(f"  chars/page: min {min(chars)}  median {sorted(chars)[total // 2]}  max {max(chars)}")
    print(f"  empty   ({len(empty)}): {', '.join(empty[:10]) or '-'}")
    print(f"  thin    ({len(thin)}): {', '.join(thin[:10]) or '-'}   (<{THIN_CHARS} chars)")
    print(f"  garbled ({len(garbled)}): {', '.join(garbled[:10]) or '-'}")

    if empty or garbled:
        print("\n  NOTE: pages above have a weak text layer. Not OCRing anything — "
              "say the word if you want OCR for these.")

    for name in page_files[:args.show] + ([page_files[7]] if total > 7 else []):
        s = stats[name]
        snippet = " ".join(texts[name].split())[:300]
        print(f"\n--- {name}  ({s['chars']} chars, alpha {s['alpha_ratio']:.2f}, "
              f"avg word {s['avg_word']:.1f}) ---\n{snippet}...")


if __name__ == "__main__":
    main()
