"""Page retrieval over the Voyage multimodal index.

Importable core shared by search.py (CLI) and diagnose_retrieval.py.
"""
import os, json
import numpy as np
import voyageai
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_DIR = os.path.join(BASE_DIR, "index")
PAGES_DIR = os.path.join(BASE_DIR, "pages")
MODEL = "voyage-multimodal-3"
TOP_K = 5

_state = {}


def _load():
    """Load client + normalized index once, on first use."""
    if not _state:
        embeddings = np.load(os.path.join(INDEX_DIR, "embeddings.npy"))
        with open(os.path.join(INDEX_DIR, "pages.json")) as f:
            _state["pages"] = json.load(f)
        _state["emb_norm"] = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        _state["vo"] = voyageai.Client()
    return _state["vo"], _state["emb_norm"], _state["pages"]


def page_count():
    """Number of pages in the index."""
    _, _, pages = _load()
    return len(pages)


def page_path(page):
    """Full path to a page image given its filename from a search result."""
    return os.path.join(PAGES_DIR, page)


def search(question, k=TOP_K):
    """Return the k best-matching pages as [(page_filename, cosine_score), ...]."""
    vo, emb_norm, pages = _load()
    q = vo.multimodal_embed(
        inputs=[[question]], model=MODEL, input_type="query"
    ).embeddings[0]
    q = np.array(q, dtype=np.float32)
    q = q / np.linalg.norm(q)
    scores = emb_norm @ q                       # cosine similarity
    top = np.argsort(scores)[::-1][:k]
    return [(pages[i], float(scores[i])) for i in top]
