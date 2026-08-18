"""Embed page images into a searchable index.

  ./venv/bin/python build_index.py                      # single corpus (sample.pdf)
  ./venv/bin/python build_index.py --corpus demo        # multi-document demo corpus

Incremental by default: pages already present in the index keep their existing
vector and are not re-embedded, so adding a document only costs the new pages.
Use --estimate to price a run without calling the API.
"""
import os, json, glob, argparse

import numpy as np
import voyageai
from PIL import Image
from dotenv import load_dotenv

import corpora

load_dotenv()

MODEL = "voyage-multimodal-3"
PIXEL_RATE = 0.60 / 1e9      # voyage-multimodal-3: $0.60 per billion pixels


def discover_pages(corpus):
    """Corpus-relative page ids, sorted, for whichever layout the corpus uses."""
    root = corpora.pages_dir(corpus)
    nested = corpora.config(corpus)["nested"]
    pattern = os.path.join(root, "*", "*.png") if nested else os.path.join(root, "*.png")
    ids = [os.path.relpath(p, root) for p in glob.glob(pattern)]
    return sorted(i for i in ids if corpora.is_valid_page_id(i))


def price(page_ids, corpus):
    """(total_pixels, dollars) for embedding these pages."""
    total = 0
    for pid in page_ids:
        with Image.open(corpora.page_path(pid, corpus)) as im:
            w, h = im.size
        total += w * h
    return total, total * PIXEL_RATE


def load_existing(index_dir):
    """({page_id: vector}, dim) from a previous build, or ({}, None)."""
    emb_path = os.path.join(index_dir, "embeddings.npy")
    pages_path = os.path.join(index_dir, "pages.json")
    if not (os.path.exists(emb_path) and os.path.exists(pages_path)):
        return {}, None
    vectors = np.load(emb_path)
    with open(pages_path) as f:
        pages = json.load(f)
    if len(pages) != len(vectors):
        return {}, None
    return {p: vectors[i] for i, p in enumerate(pages)}, vectors.shape[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=sorted(corpora.CORPORA), default=corpora.DEFAULT)
    ap.add_argument("--estimate", action="store_true",
                    help="report page count and cost, then exit without embedding")
    ap.add_argument("--force", action="store_true", help="re-embed every page")
    args = ap.parse_args()

    corpus = corpora.use(args.corpus)
    index_dir = corpora.index_dir()
    page_ids = discover_pages(corpus)
    if not page_ids:
        raise SystemExit(f"No page images under {os.path.relpath(corpora.pages_dir())}/ "
                         f"for corpus {corpus!r}. Run pdf_to_images.py first.")

    cached, _ = ({}, None) if args.force else load_existing(index_dir)
    todo = [p for p in page_ids if p not in cached]

    pixels, dollars = price(todo, corpus)
    docs = sorted({corpora.doc_of(p) or "sample.pdf" for p in page_ids})
    print(f"corpus {corpus!r}: {len(page_ids)} pages across {len(docs)} document(s)")
    print(f"  cached  : {len(page_ids) - len(todo)}")
    print(f"  to embed: {len(todo)}  ({pixels / 1e6:.1f}M pixels)")
    print(f"  estimated cost: ${dollars:.3f}  (voyage-multimodal-3 @ $0.60/Gpx)")

    if args.estimate:
        print("\n--estimate: nothing embedded.")
        return
    if not todo:
        print("\nNothing new to embed; index is up to date.")
        return

    vo = voyageai.Client()
    for n, pid in enumerate(todo, 1):
        with Image.open(corpora.page_path(pid, corpus)) as img:
            result = vo.multimodal_embed(inputs=[[img]], model=MODEL, input_type="document")
        cached[pid] = np.asarray(result.embeddings[0], dtype=np.float32)
        print(f"[{n}/{len(todo)}] embedded {pid}")

    os.makedirs(index_dir, exist_ok=True)
    ordered = [p for p in page_ids]
    matrix = np.stack([cached[p] for p in ordered]).astype(np.float32)
    np.save(os.path.join(index_dir, "embeddings.npy"), matrix)
    with open(os.path.join(index_dir, "pages.json"), "w") as f:
        json.dump(ordered, f)

    print(f"\nDone. {matrix.shape[0]} vectors of dim {matrix.shape[1]} "
          f"-> {os.path.relpath(index_dir)}/")


if __name__ == "__main__":
    main()
