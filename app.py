"""DocuMind web app — a thin wrapper over the existing engine.

Retrieval and generation are not reimplemented here: /ask calls
generate.answer(), which uses retrieval.DEFAULT_METHOD (hybrid-embedding).

Run:  ./venv/bin/uvicorn app:app --reload --port 8000
"""
import os, re

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

import generate
import retrieval
import corpora

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="DocuMind")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


def error(status, message, hint=None):
    return JSONResponse(status_code=status, content={"error": message, "hint": hint})


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/page/{page_id:path}")
def page_image(page_id: str):
    """Serve one page image.

    Accepts 'page_003.png' and '<docid>/page_003.png' only. corpora.page_path
    validates the shape and re-checks, after resolving symlinks, that the file
    is still inside pages/ — so no traversal reaches .env or sample.pdf.
    """
    try:
        path = corpora.page_path(page_id)
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

    try:
        result = generate.answer(question)
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

    def describe(page_id):
        return {
            "page": page_id,
            "doc": corpora.doc_of(page_id),
            "doc_title": corpora.doc_title(page_id),
            "page_number": corpora.page_number(page_id),
            "label": corpora.label(page_id),
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
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "corpus": corpora.active(),
        "pages": retrieval.page_count(),
        "documents": len(corpora.load_manifest()) or 1,
        "method": retrieval.DEFAULT_METHOD,
        "answer_model": generate.MODEL,
    }
