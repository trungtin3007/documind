"""Corpus registry — which page images and index a run should use.

Two corpora exist and are deliberately kept apart:

  "single"  the original one-paper setup: pages/*.png + index/. The eval set's
            gold labels ("page_008.png") and every number in RESULTS.md refer
            to this corpus, so it must never change. Adding documents to it
            would silently alter retrieval and invalidate the eval.
  "demo"      the trimmed, committed demo corpus that ships with the repo:
              pages_demo/<docid>/*.png + index_demo/. Kept small enough for git.
  "demo-full" the complete local corpus (all 8 documents, 611 pages):
              pages_demo_full/ + index_demo_full/. Not committed; rebuild with
              pdf_to_images.py --sources sources --corpus demo-full.

A page id is the corpus-relative path of its image: "page_008.png" in the
single corpus, "<docid>/page_008.png" in the demo corpus. Everything
downstream (retrieval, generation, citations, the web app) passes page ids
around, so doc identity travels with the page for free.

Select with the DOCUMIND_DATA env var, or corpora.use("demo").
"""
import os, re, json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT = "single"

CORPORA = {
    "single": {"pages": "pages", "index": "index", "nested": False,
               "text": "pages_text.json"},                 # historical location
    "demo": {"pages": "pages_demo", "index": "index_demo", "nested": True},
    "demo-full": {"pages": "pages_demo_full", "index": "index_demo_full", "nested": True},
}

PAGE_FILE = re.compile(r"^page_(\d{1,5})\.png$")
DOC_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

_active = os.environ.get("DOCUMIND_DATA") or os.environ.get("DOCUMIND_CORPUS") or DEFAULT
if _active not in CORPORA:
    raise SystemExit(f"DOCUMIND_DATA={_active!r} is not a known corpus; have {sorted(CORPORA)}")


def use(name):
    """Switch the active corpus for this process."""
    global _active
    if name not in CORPORA:
        raise ValueError(f"Unknown corpus {name!r}; have {sorted(CORPORA)}")
    _active = name
    return _active


def active():
    return _active


def config(name=None):
    return CORPORA[name or _active]


def pages_dir(name=None):
    return os.path.join(BASE_DIR, config(name)["pages"])


def index_dir(name=None):
    return os.path.join(BASE_DIR, config(name)["index"])


def is_valid_page_id(page_id):
    """True for 'page_003.png' or '<docid>/page_003.png'. No traversal, no nesting."""
    parts = page_id.split("/")
    if len(parts) == 1:
        return bool(PAGE_FILE.match(parts[0]))
    if len(parts) == 2:
        return bool(DOC_ID.match(parts[0]) and PAGE_FILE.match(parts[1]))
    return False


def doc_of(page_id):
    """Document id for a page id, or None for the flat single-doc corpus."""
    return page_id.split("/")[0] if "/" in page_id else None


def page_number(page_id):
    """1-based page number within its document."""
    m = PAGE_FILE.match(page_id.split("/")[-1])
    if not m:
        raise ValueError(f"Not a page id: {page_id!r}")
    return int(m.group(1))


def page_path(page_id, name=None):
    """Absolute path to a page image, guaranteed to stay inside pages/."""
    if not is_valid_page_id(page_id):
        raise ValueError(f"Invalid page id: {page_id!r}")
    root = os.path.realpath(pages_dir(name))
    path = os.path.realpath(os.path.join(root, page_id))
    if not (path == root or path.startswith(root + os.sep)):
        raise ValueError(f"Page id escapes the pages directory: {page_id!r}")
    return path


def text_path(name=None):
    """Where the extracted page text for this corpus lives.

    Each corpus states its own location. This used to be derived from `nested`,
    which meant the single corpus and every (non-nested) upload corpus resolved
    to the same root file — so indexing an upload overwrote the eval corpus's
    page text.
    """
    cfg = config(name)
    if cfg.get("text"):
        return os.path.join(BASE_DIR, cfg["text"])
    return os.path.join(index_dir(name), "pages_text.json")


def text_cache_path(name=None):
    """Where the local sentence-transformer chunk vectors are cached."""
    return os.path.join(index_dir(name), "text_embeddings.npz")


def manifest_path(name=None):
    return os.path.join(index_dir(name), "documents.json")


def load_manifest(name=None):
    """{docid: {title, source, pages}} for the corpus, or {} if none recorded."""
    path = manifest_path(name)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def doc_title(page_id, name=None):
    """Human-readable document name for display, falling back to the doc id."""
    doc = doc_of(page_id)
    if doc is None:
        return "sample.pdf"
    return load_manifest(name).get(doc, {}).get("title", doc)


def label(page_id, name=None):
    """Display label, e.g. 'Energy Outlook — p12' or 'Page 12'."""
    n = page_number(page_id)
    doc = doc_of(page_id)
    return f"{doc_title(page_id, name)} — p{n}" if doc else f"Page {n}"
