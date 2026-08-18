"""Text relevance scoring for page text — free and local, no API calls.

Two scorers behind get_scorer(name):
  "bm25"       lexical term overlap. No model, no download.
  "embedding"  semantic similarity from a small local sentence-transformer.
               Downloads the model once (~90MB), then runs on CPU.

Both exclude corpus.NON_CONTENT_RANGE (references etc.) from the corpus.

Run:  ./venv/bin/python text_score.py "your question" [--scorer bm25|embedding]
"""
import os, re, json, math, sys, argparse, hashlib
from collections import Counter

import numpy as np

import corpus
import corpora

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# The model truncates at 256 tokens but pages run ~900, so a whole-page vector
# would silently drop most of the page. Chunk instead and score a page by its
# best-matching chunk — that is also what lets a terse fact buried mid-page win.
CHUNK_WORDS = 180
CHUNK_OVERLAP = 40

K1 = 1.5      # term-frequency saturation
B = 0.75      # length normalisation


def tokenize(text):
    return re.findall(r"[a-z0-9][a-z0-9\-\.]*", text.lower())


class BM25:
    """Standard BM25 over a fixed set of documents."""

    def __init__(self, docs):
        self.ids = list(docs)
        self.tokens = {k: tokenize(v) for k, v in docs.items()}
        self.freqs = {k: Counter(t) for k, t in self.tokens.items()}
        self.lengths = {k: len(t) for k, t in self.tokens.items()}
        n = len(self.ids) or 1
        self.avg_len = sum(self.lengths.values()) / n

        df = Counter()
        for toks in self.tokens.values():
            df.update(set(toks))
        self.idf = {
            term: math.log(1 + (n - d + 0.5) / (d + 0.5))
            for term, d in df.items()
        }

    def score(self, query, ids=None):
        """Return {doc_id: score} for the given ids (default: all)."""
        terms = tokenize(query)
        out = {}
        for doc_id in (ids if ids is not None else self.ids):
            freq = self.freqs.get(doc_id)
            if freq is None:
                out[doc_id] = 0.0
                continue
            norm = K1 * (1 - B + B * self.lengths[doc_id] / (self.avg_len or 1))
            total = 0.0
            for term in terms:
                f = freq.get(term, 0)
                if f:
                    total += self.idf.get(term, 0.0) * f * (K1 + 1) / (f + norm)
            out[doc_id] = total
        return out


_cache = {}


def load_pages_text(path=None):
    path = path or corpora.text_path()
    if path not in _cache:
        if not os.path.exists(path):
            # Report the absolute path: relpath() made a missing *file* look like
            # a relative-path/CWD bug and sent us chasing the wrong cause once.
            raise SystemExit(f"Page text not found at {path} (cwd={os.getcwd()}) — run "
                             f"extract_text.py --corpus {corpora.active()} first.")
        with open(path) as f:
            _cache[path] = json.load(f)
    return _cache[path]


def content_pages_text(path=None):
    """Page text with front/back matter dropped.

    corpus.NON_CONTENT_RANGE (pages 14-18: references, acknowledgments, TOC) is
    a fact about sample.pdf specifically, so it is applied only to the single
    corpus. Other documents have their own structure and are used whole.

    Excluded pages simply have no entry, so BM25.score returns 0.0 for them —
    they keep whatever support the visual ranking gives and gain none here.
    """
    texts = load_pages_text(path)
    if corpora.config()["nested"]:
        return dict(texts)
    return {page: text for page, text in texts.items()
            if not corpus.is_non_content(corpus.page_number(page))}


def chunks_of(text, size=CHUNK_WORDS, overlap=CHUNK_OVERLAP):
    """Overlapping word windows, so a fact never falls across a chunk boundary."""
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    out = [" ".join(words[i:i + size]) for i in range(0, len(words), step)]
    return [c for c in out if c.strip()]


class EmbeddingScorer:
    """Semantic similarity: page score = max cosine over that page's chunks."""

    def __init__(self, docs, model_name=EMB_MODEL, cache_path=None):
        cache_path = cache_path or corpora.text_cache_path()
        self.model_name = model_name
        self.ids, self.chunk_page, texts = [], [], []
        for page in sorted(docs):
            pieces = chunks_of(docs[page])
            self.ids.append(page)
            for piece in pieces:
                self.chunk_page.append(page)
                texts.append(piece)
        self.chunk_page = np.array(self.chunk_page)

        fingerprint = hashlib.sha256(
            (model_name + "|" + "|".join(f"{p}:{len(docs[p])}" for p in sorted(docs))
             + f"|{CHUNK_WORDS}:{CHUNK_OVERLAP}").encode()).hexdigest()[:16]

        cached = self._load_cache(cache_path, fingerprint)
        if cached is None:
            self.vectors = self._encode(texts)
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez(cache_path, vectors=self.vectors,
                     chunk_page=self.chunk_page, fingerprint=fingerprint)
            print(f"  embedded {len(texts)} chunks from {len(self.ids)} pages "
                  f"-> {os.path.relpath(cache_path)}", file=sys.stderr)
        else:
            self.vectors, self.chunk_page = cached

    @staticmethod
    def _load_cache(path, fingerprint):
        if not os.path.exists(path):
            return None
        data = np.load(path, allow_pickle=True)
        if str(data.get("fingerprint")) != fingerprint:
            return None       # model, chunking, or page text changed
        return data["vectors"], data["chunk_page"]

    def _model(self):
        from sentence_transformers import SentenceTransformer
        if "st_model" not in _cache:
            _cache["st_model"] = SentenceTransformer(self.model_name, device="cpu")
        return _cache["st_model"]

    def _encode(self, texts):
        return np.asarray(self._model().encode(
            texts, normalize_embeddings=True, batch_size=64,
            show_progress_bar=False), dtype=np.float32)

    def score(self, query, ids=None):
        q = self._encode([query])[0]
        sims = self.vectors @ q
        wanted = set(ids) if ids is not None else set(self.ids)
        best = {page: 0.0 for page in wanted}
        for page, sim in zip(self.chunk_page, sims):
            page = str(page)
            if page in best and sim > best[page]:
                best[page] = float(sim)
        return best


def _bm25_scorer():
    key = ("bm25", corpora.active())
    if key not in _cache:
        _cache[key] = BM25(content_pages_text())
    return _cache[key]


def _embedding_scorer():
    key = ("embedding", corpora.active())
    if key not in _cache:
        _cache[key] = EmbeddingScorer(content_pages_text(),
                                      cache_path=corpora.text_cache_path())
    return _cache[key]


SCORERS = {"bm25": _bm25_scorer, "embedding": _embedding_scorer}


def get_scorer(name="bm25"):
    """Return an object with .score(query, ids) -> {page: score}."""
    if name not in SCORERS:
        raise ValueError(f"Unknown text scorer {name!r}; have {sorted(SCORERS)}")
    return SCORERS[name]()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*")
    ap.add_argument("--scorer", choices=sorted(SCORERS), default="bm25")
    args = ap.parse_args()

    question = " ".join(args.question) or "What models were used to score the reasoning traces?"
    scores = get_scorer(args.scorer).score(question)
    top = sorted(scores.items(), key=lambda kv: -kv[1])[:8]
    print(f"{args.scorer} top pages for {question!r}\n")
    for rank, (page, s) in enumerate(top, 1):
        print(f"{rank}. {page}   {s:.4f}")
