"""User-uploaded PDFs: validate, index, store durably, and query.

Durability (Stage 2b): the authoritative copy of an upload lives in object
storage (Cloudflare R2 via storage.py), so an upload survives a Render sleep or
redeploy. Local disk is only a cache — `.upload_cache/<id>/` — hydrated from
storage on demand and safe to delete at any time.

The demo corpus and the evaluation corpus are untouched by all of this: they
stay on committed local files and never go through storage.py.

Each upload becomes its own corpus `upload:<id>` so retrieval, generation and
the app treat it exactly like any other corpus. Nothing here re-implements the
pipeline: rendering comes from pdf_to_images, text from extract_text, embedding
from build_index.
"""
import os, re, json, uuid, time, shutil, threading

import pymupdf

import corpora
import pdf_to_images
import extract_text
import build_index
import retrieval
import text_score
import storage

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_ROOT = os.path.join(BASE_DIR, ".upload_cache")

# --- guardrails -------------------------------------------------------------
# Embedding is billed per page and answering is billed per page image, so both
# the size of an upload and how many a visitor may make are capped.
MAX_PAGES = 30                 # per PDF; checked before any rendering or API call
MAX_BYTES = 20 * 1024 * 1024   # per PDF
TOP_K = 5                      # pages read by the VLM for uploads (demo uses 3)

MAX_UPLOADS_PER_SESSION = 3    # a visitor may index at most 3 PDFs...
MAX_PAGES_PER_SESSION = 60     # ...and at most 60 pages in total
TTL_DAYS = 7                   # uploads older than this are pruned
PRUNE_INTERVAL = 3600          # at most one bucket sweep per hour

MAX_CACHED_UPLOADS = 20        # local hydration cache bound (disk, not memory)

PDF_MAGIC = b"%PDF-"
ALLOWED_CONTENT_TYPES = {"application/pdf", "application/x-pdf", "application/octet-stream"}
UPLOAD_ID = re.compile(r"^[0-9a-f]{12}$")

# Session budgets are held in this process's memory, which has two consequences
# worth being explicit about rather than papering over:
#   1. They reset when the process restarts. On Render's free tier the service
#      sleeps when idle, so a visitor can clear their budget by waiting.
#   2. They are per-instance. Scale to more than one instance and each gets its
#      own counters, multiplying the effective limit by the instance count.
# For a single free-tier instance this is an acceptable cost ceiling, not a
# security control. A durable limiter (counters in R2 or Redis, keyed by IP or
# session) is the fix if this ever needs to hold against a determined visitor.
_sessions = {}                 # session id -> {"uploads": n, "pages": n}
_lock = threading.Lock()
_last_prune = 0.0


class UploadError(Exception):
    """A rejection the user should see, with an HTTP status and a hint."""

    def __init__(self, status, message, hint=None):
        super().__init__(message)
        self.status, self.message, self.hint = status, message, hint


# --- naming -----------------------------------------------------------------
def corpus_name(upload_id):
    return f"upload:{upload_id}"


def key(upload_id, *parts):
    return "/".join(("uploads", upload_id) + parts)


def cache_dir(upload_id):
    return os.path.join(CACHE_ROOT, upload_id)


def valid_id(upload_id):
    return isinstance(upload_id, str) and bool(UPLOAD_ID.match(upload_id))


def register(upload_id):
    """Make this upload addressable as a corpus backed by its cache dir."""
    name = corpus_name(upload_id)
    if name not in corpora.CORPORA:
        rel = os.path.relpath(cache_dir(upload_id), corpora.BASE_DIR)
        corpora.CORPORA[name] = {"pages": os.path.join(rel, "pages"),
                                 "index": os.path.join(rel, "index"),
                                 "nested": False}
    return name


# --- validation -------------------------------------------------------------
def inspect(data, content_type=None, filename=None):
    """Validate raw bytes as an acceptable PDF; return its page count.

    Ordered cheapest-first, so an oversized or non-PDF file costs no work.
    """
    if not data:
        raise UploadError(400, "That file was empty.")
    if len(data) > MAX_BYTES:
        raise UploadError(413, f"That file is {len(data) / 1e6:.1f} MB. "
                               f"The limit is {MAX_BYTES // 1024 // 1024} MB.")
    if content_type and content_type.split(";")[0].strip().lower() not in ALLOWED_CONTENT_TYPES:
        raise UploadError(415, f"Expected a PDF, got content type {content_type!r}.")
    if not data.startswith(PDF_MAGIC):
        raise UploadError(415, "That file is not a PDF.",
                          "Its contents do not start with the PDF marker, whatever it is named.")
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
        pages = doc.page_count
        doc.close()
    except Exception:
        raise UploadError(400, "That PDF could not be opened.", "It may be corrupt or encrypted.")
    if pages == 0:
        raise UploadError(400, "That PDF has no pages.")
    if pages > MAX_PAGES:
        raise UploadError(413, f"That PDF has {pages} pages. The limit is {MAX_PAGES}.",
                          "Indexing is billed per page, so larger uploads are capped. "
                          "Try a shorter document or an extract.")
    return pages


def check_session(session_id, pages):
    """Enforce the per-visitor budget before spending anything on embedding."""
    with _lock:
        s = _sessions.get(session_id, {"uploads": 0, "pages": 0})
        if s["uploads"] >= MAX_UPLOADS_PER_SESSION:
            raise UploadError(429, f"You have reached the limit of {MAX_UPLOADS_PER_SESSION} "
                                   f"uploads for this session.",
                              "This demo caps indexing cost per visitor.")
        if s["pages"] + pages > MAX_PAGES_PER_SESSION:
            raise UploadError(429, f"That would exceed the limit of {MAX_PAGES_PER_SESSION} "
                                   f"indexed pages per session "
                                   f"(you have used {s['pages']}).",
                              "This demo caps indexing cost per visitor.")


def record_session(session_id, pages):
    with _lock:
        s = _sessions.setdefault(session_id, {"uploads": 0, "pages": 0})
        s["uploads"] += 1
        s["pages"] += pages


def session_usage(session_id):
    with _lock:
        s = _sessions.get(session_id, {"uploads": 0, "pages": 0})
        return {"uploads_used": s["uploads"], "uploads_left": MAX_UPLOADS_PER_SESSION - s["uploads"],
                "pages_used": s["pages"], "pages_left": MAX_PAGES_PER_SESSION - s["pages"]}


# --- build ------------------------------------------------------------------
def build(data, filename=None, progress=None):
    """Render, extract text and embed one PDF, then persist it to storage."""
    upload_id = uuid.uuid4().hex[:12]
    corpus = register(upload_id)
    root = cache_dir(upload_id)
    os.makedirs(root, exist_ok=True)

    pdf_path = os.path.join(root, "source.pdf")
    with open(pdf_path, "wb") as f:
        f.write(data)

    _, total = pdf_to_images.render(pdf_path, corpora.pages_dir(corpus), dpi=pdf_to_images.DPI)
    page_ids = build_index.discover_pages(corpus)
    if len(page_ids) != total:
        raise UploadError(500, "Page rendering did not match the PDF's page count.")

    texts = extract_text.page_texts(pdf_path)
    text_path = corpora.text_path(corpus)
    os.makedirs(os.path.dirname(text_path), exist_ok=True)   # index/ may not exist yet
    with open(text_path, "w") as f:
        json.dump({pid: texts[corpora.page_number(pid) - 1] for pid in page_ids}, f)

    matrix = build_index.embed_page_ids(
        page_ids, corpus,
        progress=(lambda n, t, pid: progress("embedded", n, t)) if progress else None)
    build_index.write_index(corpora.index_dir(corpus), page_ids, matrix)

    meta = {"upload_id": upload_id,
            "filename": os.path.basename(filename or "uploaded.pdf"),
            "pages": total,
            "uploaded_at": time.time()}
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(meta, f)

    _push(upload_id, page_ids)
    retrieval.forget(corpus)
    text_score.forget(corpus)
    return meta


def _push(upload_id, page_ids):
    """Copy this upload's artifacts from the cache into durable storage."""
    root = cache_dir(upload_id)
    storage.put(key(upload_id, "meta.json"), open(os.path.join(root, "meta.json"), "rb").read())
    storage.put(key(upload_id, "source.pdf"), open(os.path.join(root, "source.pdf"), "rb").read())
    for name in ("pages.json", "embeddings.npy", "pages_text.json"):
        path = os.path.join(root, "index", name)
        if os.path.exists(path):
            storage.put(key(upload_id, "index", name), open(path, "rb").read())
    for pid in page_ids:
        storage.put(key(upload_id, "pages", pid),
                    open(os.path.join(root, "pages", pid), "rb").read())


# --- read -------------------------------------------------------------------
def exists(upload_id):
    """True if this id is well-formed and its index is in durable storage."""
    return valid_id(upload_id) and storage.exists(key(upload_id, "index", "pages.json"))


def hydrate_index(upload_id):
    """Ensure the index and page text are on local disk. Returns the corpus name."""
    corpus = register(upload_id)
    root = cache_dir(upload_id)
    os.makedirs(os.path.join(root, "index"), exist_ok=True)
    for name in ("pages.json", "embeddings.npy", "pages_text.json"):
        dest = os.path.join(root, "index", name)
        if not os.path.exists(dest):
            with open(dest, "wb") as f:
                f.write(storage.get(key(upload_id, "index", name)))
    _touch(upload_id)
    return corpus


def hydrate_pages(upload_id, page_ids):
    """Fetch just the page images needed for this request."""
    root = os.path.join(cache_dir(upload_id), "pages")
    os.makedirs(root, exist_ok=True)
    for pid in page_ids:
        dest = os.path.join(root, pid)
        if not os.path.exists(dest):
            with open(dest, "wb") as f:
                f.write(storage.get(key(upload_id, "pages", pid)))


def meta(upload_id):
    try:
        return json.loads(storage.get(key(upload_id, "meta.json")))
    except Exception:
        return {"upload_id": upload_id}


# --- lifecycle --------------------------------------------------------------
def _touch(upload_id):
    """Mark a cache entry as recently used, then bound the cache."""
    path = cache_dir(upload_id)
    if os.path.isdir(path):
        os.utime(path, None)
    _evict_cache()


def _evict_cache():
    """Keep the local hydration cache bounded; storage remains authoritative."""
    if not os.path.isdir(CACHE_ROOT):
        return
    entries = [(os.path.getmtime(os.path.join(CACHE_ROOT, d)), d)
               for d in os.listdir(CACHE_ROOT)
               if os.path.isdir(os.path.join(CACHE_ROOT, d))]
    for _, name in sorted(entries)[:max(0, len(entries) - MAX_CACHED_UPLOADS)]:
        shutil.rmtree(os.path.join(CACHE_ROOT, name), ignore_errors=True)
        corpora.CORPORA.pop(corpus_name(name), None)
        retrieval.forget(corpus_name(name))
        text_score.forget(corpus_name(name))


def maybe_prune(force=False):
    """Prune expired uploads, at most once per PRUNE_INTERVAL.

    Called on startup and at the top of every upload, so expired uploads are
    cleaned without a cron job. Never raises: a storage hiccup must not take
    down an upload or the whole app.
    """
    global _last_prune
    now = time.time()
    with _lock:
        if not force and now - _last_prune < PRUNE_INTERVAL:
            return None
        _last_prune = now
    try:
        return prune()
    except Exception:
        return None


def prune(ttl_days=TTL_DAYS):
    """Delete uploads older than the TTL. Returns the ids removed."""
    cutoff = time.time() - ttl_days * 86400
    removed = []
    for k in storage.list_keys("uploads/"):
        if not k.endswith("/meta.json"):
            continue
        upload_id = k.split("/")[1]
        try:
            uploaded_at = json.loads(storage.get(k)).get("uploaded_at", 0)
        except Exception:
            continue
        if uploaded_at and uploaded_at < cutoff:
            remove(upload_id)
            removed.append(upload_id)
    return removed


def remove(upload_id):
    """Delete an upload from storage and drop every trace of it locally."""
    corpus = corpus_name(upload_id)
    retrieval.forget(corpus)
    text_score.forget(corpus)
    corpora.CORPORA.pop(corpus, None)
    shutil.rmtree(cache_dir(upload_id), ignore_errors=True)
    storage.delete_prefix(f"uploads/{upload_id}/")
