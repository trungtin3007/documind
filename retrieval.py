"""Page retrieval over the Voyage multimodal index.

Two methods:
  search()        visual only — cosine over page-image embeddings (unchanged).
  search_hybrid() re-ranks the visual candidates by fusing them with a BM25
                  ranking over the same pages' text layer.

The hybrid exists because the visual index alone cannot break ties between
pages that discuss the same topic in different sections: hit@5 was 100% while
hit@1 was 68%, so the right page was retrieved but mis-ranked.

Importable core shared by search.py (CLI), diagnose_retrieval.py and eval.py.
"""
import os, json
import numpy as np
import voyageai
from dotenv import load_dotenv

import text_score
import corpora

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Paths come from the active corpus (corpora.py). The defaults are the original
# single-paper layout, so eval and RESULTS.md are unaffected.
INDEX_DIR = corpora.index_dir()
PAGES_DIR = corpora.pages_dir()
MODEL = "voyage-multimodal-3"
TOP_K = 5
POOL_K = 15       # visual candidates the re-ranker is allowed to reorder
RRF_K = 60        # standard Reciprocal Rank Fusion constant; no weights to tune
# "hybrid" is kept as an alias for the BM25 variant so result files written
# before the embedding scorer existed still match on method.
METHODS = ("visual", "hybrid-bm25", "hybrid-embedding")
METHOD_ALIASES = {"hybrid": "hybrid-bm25"}
# Adopted Week 4. DOCUMIND_METHOD overrides it per-process: the Render free tier
# cannot hold sentence-transformers on top of what voyageai already loads, so
# the deploy runs hybrid-bm25 (26/28 hit@3 vs 27/28 — one question, within noise).
DEFAULT_METHOD = os.environ.get("DOCUMIND_METHOD", "hybrid-embedding")
if DEFAULT_METHOD not in METHODS and DEFAULT_METHOD not in METHOD_ALIASES:
    raise SystemExit(f"DOCUMIND_METHOD={DEFAULT_METHOD!r} is not a known method; have {list(METHODS)}")

_state = {}     # corpus name -> {pages, emb_norm}
_clients = {}


def _load(corpus=None):
    """Load client + normalized index for a corpus, cached per corpus name.

    `corpus` is explicit so a web request can select one without mutating the
    process-wide active corpus, which would race other in-flight requests.
    """
    corpus = corpus or corpora.active()
    entry = _state.get(corpus)
    if entry is None:
        index_dir = corpora.index_dir(corpus)
        embeddings = np.load(os.path.join(index_dir, "embeddings.npy"))
        with open(os.path.join(index_dir, "pages.json")) as f:
            pages = json.load(f)
        entry = {
            "pages": pages,
            "emb_norm": embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True),
        }
        _state[corpus] = entry
    return _client(), entry["emb_norm"], entry["pages"]


def _client():
    if "voyage_client" not in _clients:
        _clients["voyage_client"] = voyageai.Client()
    return _clients["voyage_client"]


def forget(corpus):
    """Drop a cached index (used when a temporary upload corpus is removed)."""
    _state.pop(corpus, None)


def page_count(corpus=None):
    """Number of pages in the index."""
    _, _, pages = _load(corpus)
    return len(pages)


def page_path(page, corpus=None):
    """Full path to a page image given its page id from a search result."""
    return corpora.page_path(page, corpus)


def search(question, k=TOP_K, corpus=None):
    """Return the k best-matching pages as [(page_filename, cosine_score), ...]."""
    vo, emb_norm, pages = _load(corpus)
    q = vo.multimodal_embed(
        inputs=[[question]], model=MODEL, input_type="query"
    ).embeddings[0]
    q = np.array(q, dtype=np.float32)
    q = q / np.linalg.norm(q)
    scores = emb_norm @ q                       # cosine similarity
    top = np.argsort(scores)[::-1][:k]
    return [(pages[i], float(scores[i])) for i in top]


def search_hybrid(question, k=TOP_K, pool=POOL_K, text_scorer="bm25", corpus=None):
    """Re-rank the top `pool` visual candidates by fusing visual and text ranks.

    Returns [(page, rrf_score)]. The score is a Reciprocal Rank Fusion value,
    not a cosine — comparable within a result list, not against search().
    """
    visual = search(question, k=pool, corpus=corpus)
    candidates = [page for page, _ in visual]

    text_scores = text_score.get_scorer(text_scorer, corpus=corpus).score(question, ids=candidates)
    text_ranked = sorted(candidates, key=lambda p: (-text_scores[p], p))

    # RRF: each list contributes 1/(RRF_K + rank). A page ranked well by either
    # signal survives; a page ranked well by both wins. No weights to tune.
    fused = {}
    for rank, page in enumerate(candidates, 1):
        fused[page] = fused.get(page, 0.0) + 1.0 / (RRF_K + rank)
    for rank, page in enumerate(text_ranked, 1):
        fused[page] = fused.get(page, 0.0) + 1.0 / (RRF_K + rank)

    ranked = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(page, float(score)) for page, score in ranked[:k]]


def search_by_method(question, k=TOP_K, method=DEFAULT_METHOD, corpus=None):
    """Dispatch for the --method flag. Only the text scorer varies between hybrids."""
    method = METHOD_ALIASES.get(method, method)
    if method == "visual":
        return search(question, k=k, corpus=corpus)
    if method.startswith("hybrid-"):
        return search_hybrid(question, k=k, text_scorer=method.split("-", 1)[1], corpus=corpus)
    raise ValueError(f"Unknown method {method!r}; have {list(METHODS)}")
