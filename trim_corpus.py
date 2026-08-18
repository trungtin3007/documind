"""Build the small, committed demo corpus by selecting pages from the full one.

The full corpus (611 pages, ~182MB of images) is too large for git, so the
repo ships a trimmed subset. This copies page images and *slices* the existing
vectors rather than re-embedding, so trimming costs nothing.

Selection lives in KEEP: whole documents, or explicit page ranges.

Run:  ./venv/bin/python trim_corpus.py [--dry-run]
"""
import os, json, glob, shutil, argparse

import numpy as np

import corpora

SOURCE_CORPUS = "demo-full"
TARGET_CORPUS = "demo"

# Chosen for the live demo: the table-heavy documents whole, plus one Fed report
# for chart variety, plus a USGS slice that must contain every page the tested
# demo questions land on (p9 summary, p78-79 gallium).
KEEP = {
    "bls-ce-2023": None,                       # None = every page
    "census-income-2023": None,
    "fed-mpr-2025": None,
    "usgs-mcs-2025": [(1, 12), (70, 95)],      # front matter + a commodity block
}


def selected_pages(source):
    """Page ids to keep, in index order."""
    root = corpora.pages_dir(source)
    out = []
    for did, ranges in KEEP.items():
        available = sorted(os.path.relpath(p, root)
                           for p in glob.glob(os.path.join(root, did, "*.png")))
        if not available:
            raise SystemExit(f"No pages found for {did!r} in {os.path.relpath(root)}/")
        if ranges is None:
            out.extend(available)
            continue
        wanted = {n for lo, hi in ranges for n in range(lo, hi + 1)}
        out.extend(p for p in available if corpora.page_number(p) in wanted)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src_pages_dir = corpora.pages_dir(SOURCE_CORPUS)
    src_index_dir = corpora.index_dir(SOURCE_CORPUS)
    dst_pages_dir = corpora.pages_dir(TARGET_CORPUS)
    dst_index_dir = corpora.index_dir(TARGET_CORPUS)

    keep = selected_pages(SOURCE_CORPUS)
    size = sum(os.path.getsize(os.path.join(src_pages_dir, p)) for p in keep)

    by_doc = {}
    for p in keep:
        by_doc.setdefault(corpora.doc_of(p), []).append(p)
    print(f"{'document':<24}{'pages':>6}{'MB':>9}")
    for did, pages in sorted(by_doc.items()):
        mb = sum(os.path.getsize(os.path.join(src_pages_dir, p)) for p in pages) / 1e6
        print(f"{did:<24}{len(pages):>6}{mb:>9.1f}")
    print(f"{'TOTAL':<24}{len(keep):>6}{size / 1e6:>9.1f}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    # Page images
    if os.path.isdir(dst_pages_dir):
        shutil.rmtree(dst_pages_dir)
    for p in keep:
        dst = os.path.join(dst_pages_dir, p)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(os.path.join(src_pages_dir, p), dst)

    # Vectors: slice the full index rather than paying to embed again.
    full_pages = json.load(open(os.path.join(src_index_dir, "pages.json")))
    full_vecs = np.load(os.path.join(src_index_dir, "embeddings.npy"))
    position = {p: i for i, p in enumerate(full_pages)}
    missing = [p for p in keep if p not in position]
    if missing:
        raise SystemExit(f"{len(missing)} selected pages are not in the full index, "
                         f"e.g. {missing[:3]}. Build the full corpus first.")
    os.makedirs(dst_index_dir, exist_ok=True)
    np.save(os.path.join(dst_index_dir, "embeddings.npy"),
            np.stack([full_vecs[position[p]] for p in keep]).astype(np.float32))
    with open(os.path.join(dst_index_dir, "pages.json"), "w") as f:
        json.dump(keep, f)

    # Page text, same page set
    full_text = json.load(open(corpora.text_path(SOURCE_CORPUS)))
    with open(corpora.text_path(TARGET_CORPUS), "w") as f:
        json.dump({p: full_text[p] for p in keep}, f)

    # Manifest, with the trimmed page counts
    full_manifest = corpora.load_manifest(SOURCE_CORPUS)
    manifest = {}
    for did, pages in by_doc.items():
        meta = dict(full_manifest.get(did, {}))
        meta["pages"] = len(pages)
        meta.setdefault("title", did)
        meta["full_pages"] = full_manifest.get(did, {}).get("pages", len(pages))
        manifest[did] = meta
    with open(corpora.manifest_path(TARGET_CORPUS), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nWrote {len(keep)} pages -> {os.path.relpath(dst_pages_dir)}/ "
          f"and {os.path.relpath(dst_index_dir)}/")
    print("Text embedding cache rebuilds locally on first query (free).")


if __name__ == "__main__":
    main()
