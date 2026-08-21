"""DocuMind web app — a thin wrapper over the existing engine.

Retrieval and generation are not reimplemented here: /ask calls
generate.answer(), which uses retrieval.DEFAULT_METHOD (hybrid-embedding).

Run:  ./venv/bin/uvicorn app:app --reload --port 8000
"""
import os, re, uuid, contextlib

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

import generate
import retrieval
import corpora
import uploads
import storage

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

@contextlib.asynccontextmanager
async def lifespan(_app):
    # Sweep expired uploads once at boot. Wrapped because a storage
    # misconfiguration must not stop the app from serving the demo corpus.
    try:
        removed = uploads.maybe_prune(force=True)
        if removed:
            print(f"startup prune: removed {len(removed)} expired upload(s)")
    except Exception as exc:
        print(f"startup prune skipped: {type(exc).__name__}")
    yield


app = FastAPI(title="DocuMind", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    # When present, answer from that upload's index instead of the demo corpus.
    upload_id: str | None = Field(default=None, max_length=64)


SESSION_COOKIE = "documind_sid"


def error(status, message, hint=None):
    return JSONResponse(status_code=status, content={"error": message, "hint": hint})


def session_id(request: Request, response: Response = None):
    """Stable per-visitor id, used only to meter uploads. Not authentication."""
    sid = request.cookies.get(SESSION_COOKIE)
    if not sid:
        sid = uuid.uuid4().hex
        if response is not None:
            response.set_cookie(SESSION_COOKIE, sid, max_age=7 * 86400,
                                httponly=True, samesite="lax")
    return sid


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/page/{page_id:path}")
def page_image(page_id: str, upload_id: str | None = None):
    """Serve one page image.

    Accepts 'page_003.png' and '<docid>/page_003.png' only. corpora.page_path
    validates the shape and re-checks, after resolving symlinks, that the file
    is still inside pages/ — so no traversal reaches .env or sample.pdf.
    """
    corpus = None
    if upload_id is not None:
        if not uploads.exists(upload_id):
            return error(404, "That upload is no longer available.")
        corpus = uploads.hydrate_index(upload_id)
        if corpora.is_valid_page_id(page_id):
            try:
                uploads.hydrate_pages(upload_id, [page_id])
            except Exception:
                return error(404, f"No such page: {page_id}")
    try:
        path = corpora.page_path(page_id, corpus)
    except ValueError:
        return error(400, f"Invalid page id: {page_id!r}")
    if not os.path.isfile(path):
        return error(404, f"No such page: {page_id}")
    return FileResponse(path, media_type="image/png")


@app.post("/ask")
def ask(req: AskRequest):
    """Answer a question from the indexed pages. Sync on purpose: FastAPI runs
    this in a threadpool, and the engine call is blocking."""
    question = req.question.strip()
    if not question:
        return error(400, "Please enter a question.")

    corpus = None
    if req.upload_id:
        if not uploads.exists(req.upload_id):
            return error(404, "That upload is no longer available.",
                         "Uploads expire after a while. Upload the PDF again.")
        try:
            corpus = uploads.hydrate_index(req.upload_id)
        except Exception:
            return error(503, "Could not load that upload from storage.")

    # Uploads read more pages than the demo; see uploads.TOP_K.
    top_k = uploads.TOP_K if corpus else generate.TOP_K
    try:
        retrieved = None
        if corpus:
            # Retrieve first, then fetch only those page images from storage —
            # never the whole document.
            retrieved = retrieval.search_by_method(question, k=top_k, corpus=corpus)
            uploads.hydrate_pages(req.upload_id, [p for p, _ in retrieved])
        result = generate.answer(question, k=top_k, retrieved=retrieved, corpus=corpus)
    except SystemExit as exc:                      # missing API key
        return error(503, str(exc), "Add GEMINI_API_KEY to .env and restart.")
    except Exception as exc:
        name = type(exc).__name__
        if "ratelimit" in name.lower() or "429" in str(exc):
            return error(429, "The model API is rate limited right now.",
                         "Free tier quota. Wait a moment and try again.")
        if "connection" in name.lower() or "timeout" in name.lower():
            return error(504, "Could not reach the model API.", "Check your connection.")
        return error(500, f"{name}: {exc}"[:300])

    upload_name = uploads.meta(req.upload_id).get("filename") if req.upload_id else None

    def describe(page_id):
        n = corpora.page_number(page_id)
        return {
            "page": page_id,
            "doc": corpora.doc_of(page_id),
            "doc_title": upload_name or corpora.doc_title(page_id, corpus),
            "page_number": n,
            "label": f"{upload_name} — p{n}" if upload_name else corpora.label(page_id, corpus),
        }

    return {
        "question": question,
        "answer": result["answer"],
        "cited_pages": result["cited_pages"],
        "cited": [describe(p) for p in result["cited_pages"]],
        "retrieved_pages": [dict(describe(p), score=round(s, 4))
                            for p, s in result["retrieved_pages"]],
        # No citation means the model reported the answer is not on these pages.
        "found": bool(result["cited_pages"]),
        "method": retrieval.DEFAULT_METHOD,
        "upload_id": req.upload_id,
        "pages_read": len(result["retrieved_pages"]),
    }


@app.post("/upload")
def upload_pdf(request: Request, response: Response, file: UploadFile = File(...)):
    """Accept a PDF, index it, and return an id to ask questions against.

    Every rejection happens before any rendering or embedding, so an oversized
    or non-PDF file costs nothing.
    """
    try:
        data = file.file.read(uploads.MAX_BYTES + 1)
    except Exception:
        return error(400, "Could not read the uploaded file.")

    uploads.maybe_prune()          # lazy TTL sweep, throttled to once an hour
    sid = session_id(request, response)
    try:
        pages = uploads.inspect(data, file.content_type, file.filename)
        uploads.check_session(sid, pages)
    except uploads.UploadError as exc:
        return error(exc.status, exc.message, exc.hint)

    try:
        meta = uploads.build(data, file.filename)
    except uploads.UploadError as exc:
        return error(exc.status, exc.message, exc.hint)
    except Exception as exc:
        name = type(exc).__name__
        if "ratelimit" in name.lower() or "429" in str(exc):
            return error(429, "The embedding API is rate limited right now.",
                         "Free tier quota. Wait a moment and try again.")
        return error(500, f"Indexing failed: {name}"[:200])

    uploads.record_session(sid, pages)
    payload = {"upload_id": meta["upload_id"], "filename": meta.get("filename"),
               "pages": pages, "max_pages": uploads.MAX_PAGES,
               "session": uploads.session_usage(sid)}
    out = JSONResponse(content=payload)
    for k, v in response.raw_headers:
        out.raw_headers.append((k, v))       # carry the session cookie through
    return out


@app.get("/limits")
def limits():
    return {"max_pages": uploads.MAX_PAGES,
            "max_mb": uploads.MAX_BYTES // 1024 // 1024,
            "upload_top_k": uploads.TOP_K,
            "demo_top_k": generate.TOP_K,
            "max_uploads_per_session": uploads.MAX_UPLOADS_PER_SESSION,
            "max_pages_per_session": uploads.MAX_PAGES_PER_SESSION,
            "upload_ttl_days": uploads.TTL_DAYS,
            "storage": storage.describe()["backend"]}


@app.get("/health")
def health():
    return {
        "ok": True,
        "corpus": corpora.active(),
        "pages": retrieval.page_count(),
        "documents": len(corpora.load_manifest()) or 1,
        "method": retrieval.DEFAULT_METHOD,
        "answer_model": generate.MODEL,
        "upload_storage": storage.describe()["backend"],
    }
