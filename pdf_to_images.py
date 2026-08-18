"""Render PDF pages to images.

Two modes, and the default is unchanged from the original single-file script:

  ./venv/bin/python pdf_to_images.py
      sample.pdf -> pages/page_001.png ...            (the eval corpus)

  ./venv/bin/python pdf_to_images.py --sources sources/ --corpus demo
      sources/*.pdf -> pages/<docid>/page_001.png ...  (the demo corpus)

Page numbers restart per document, so <docid> is what keeps them apart.
Existing images are skipped unless --force, so re-running is cheap.
"""
import os, re, json, glob, argparse

import pymupdf

import corpora

DPI = 150          # 1241x1754 for A4; see --dpi to trade quality for embedding cost


def doc_id_from(path):
    """Filename -> a safe, stable document id."""
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._")
    return (stem or "doc")[:64]


def render(pdf_path, out_dir, dpi=DPI, force=False):
    """Render one PDF into out_dir. Returns (pages_written, pages_total)."""
    os.makedirs(out_dir, exist_ok=True)
    doc = pymupdf.open(pdf_path)
    written = 0
    for i, page in enumerate(doc):
        out_path = os.path.join(out_dir, f"page_{i + 1:03d}.png")
        if os.path.exists(out_path) and not force:
            continue
        page.get_pixmap(dpi=dpi).save(out_path)
        written += 1
    total = doc.page_count
    doc.close()
    return written, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?", default="sample.pdf",
                    help="single PDF to render (default: sample.pdf)")
    ap.add_argument("--sources", help="directory of PDFs to render, one subfolder each")
    ap.add_argument("--corpus", choices=sorted(corpora.CORPORA), default=None,
                    help="corpus to write into (default: single for a lone PDF, demo for --sources)")
    ap.add_argument("--dpi", type=int, default=DPI)
    ap.add_argument("--force", action="store_true", help="re-render pages that already exist")
    args = ap.parse_args()

    corpus = args.corpus or ("demo" if args.sources else "single")
    corpora.use(corpus)
    pages_root = corpora.pages_dir()

    if not args.sources:
        # Legacy single-document mode: flat pages/page_NNN.png.
        written, total = render(args.pdf, pages_root, args.dpi, args.force)
        print(f"{args.pdf}: {total} pages ({written} rendered, {total - written} already present) "
              f"-> {os.path.relpath(pages_root)}/")
        return

    pdfs = sorted(glob.glob(os.path.join(args.sources, "*.pdf")))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {args.sources}/")

    manifest, grand_total, grand_written = {}, 0, 0
    for pdf in pdfs:
        did = doc_id_from(pdf)
        if did in manifest:
            raise SystemExit(f"Duplicate document id {did!r} from {pdf}; rename the file.")
        written, total = render(pdf, os.path.join(pages_root, did), args.dpi, args.force)
        manifest[did] = {"title": os.path.basename(pdf), "source": os.path.basename(pdf),
                         "pages": total}
        grand_total += total
        grand_written += written
        print(f"  {did:<28} {total:>4} pages ({written} rendered)")

    # Record the document list next to the index it will be built into, so the
    # app can name documents without reading the PDFs again.
    os.makedirs(corpora.index_dir(), exist_ok=True)
    with open(corpora.manifest_path(), "w") as f:
        json.dump(manifest, f, indent=2)

    px = 0
    for did, meta in manifest.items():
        px += meta["pages"]
    print(f"\n{len(manifest)} documents, {grand_total} pages "
          f"({grand_written} newly rendered) -> {os.path.relpath(pages_root)}/<docid>/")
    print(f"manifest -> {os.path.relpath(corpora.manifest_path())}")
    print(f"Next: ./venv/bin/python build_index.py --corpus {corpus}")


if __name__ == "__main__":
    main()
