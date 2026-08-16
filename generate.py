"""Answer questions about the indexed PDF from its page images.

Retrieval finds the most relevant pages; a vision model reads those page
images and answers from them alone, citing the pages it used.

Run:  ./venv/bin/python generate.py "your question"
"""
import os, sys, json, base64
from dotenv import load_dotenv
from google import genai

import retrieval
import api_retry

load_dotenv()

MODEL = "gemini-3.5-flash-lite"
TOP_K = 3                       # detail questions often need a lower-ranked page too
SEED = 7                        # fixed seed so eval runs are reproducible

SYSTEM_INSTRUCTION = """You answer questions about a research paper using ONLY the page images provided.

Rules:
- Use only what is visible in the page images. Never use outside knowledge, and never guess or infer beyond what the pages show.
- Every claim in your answer must come from a page you cite. Quote exact figures and numbers as printed.
- Cite the page numbers you actually used, using the PAGE numbers labelled in the input.
- If the provided pages do not contain the answer, set found to false, leave cited_pages empty, and say plainly in answer what is missing. Do not answer from memory in that case.
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "found": {
            "type": "boolean",
            "description": "True only if the provided pages contain the answer.",
        },
        "answer": {
            "type": "string",
            "description": "The answer drawn from the pages, or what is missing if found is false.",
        },
        "cited_pages": {
            "type": "array",
            "items": {"type": "integer"},
            "description": "PAGE numbers actually used. Empty if found is false.",
        },
    },
    "required": ["found", "answer", "cited_pages"],
}

_client = None


def _get_client():
    global _client
    if _client is None:
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            sys.exit("No Gemini key found. Add GEMINI_API_KEY=... to .env")
        _client = genai.Client()
    return _client


def page_number(page):
    """'page_054.png' -> 54"""
    return int(os.path.splitext(page)[0].split("_")[-1])


def _build_input(question, retrieved):
    """Question plus one labelled image block per retrieved page."""
    parts = [{
        "type": "text",
        "text": f"Question: {question}\n\nAnswer using only the {len(retrieved)} page images below.",
    }]
    for page, _ in retrieved:
        with open(retrieval.page_path(page), "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        parts.append({"type": "text", "text": f"PAGE {page_number(page)}"})
        parts.append({"type": "image", "data": data, "mime_type": "image/png"})
    return parts


def answer(question, k=TOP_K, retrieved=None, method=None):
    """Retrieve the k best pages, read them with the VLM, and answer from them.

    Pass `retrieved` (a list of (page, score)) to reuse pages already fetched
    by a caller, so eval does not pay for a second query embedding.
    """
    if retrieved is None:
        retrieved = retrieval.search_by_method(
            question, k=k, method=method or retrieval.DEFAULT_METHOD)

    result = api_retry.call_with_retry(
        _get_client().interactions.create,
        model=MODEL,
        system_instruction=SYSTEM_INSTRUCTION,
        input=_build_input(question, retrieved),
        generation_config={"seed": SEED},
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": RESPONSE_SCHEMA,
        },
    )

    raw = (result.output_text or "").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # Schema is enforced server-side, so this should not happen; degrade to raw text.
        return {
            "answer": raw or "(empty response from the model)",
            "cited_pages": [],
            "retrieved_pages": retrieved,
        }

    # Only trust citations pointing at pages we actually sent.
    sent = {page_number(page): page for page, _ in retrieved}
    cited = [sent[n] for n in parsed.get("cited_pages", []) if n in sent]
    if not parsed.get("found", False):
        cited = []

    return {
        "answer": parsed.get("answer", "").strip(),
        "cited_pages": cited,
        "retrieved_pages": retrieved,
    }


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or input("Question: ")
    result = answer(question)

    print(f"\nQ: {question}\n")
    print(result["answer"])

    if result["cited_pages"]:
        print("\nCited: " + ", ".join(result["cited_pages"]))
    else:
        print("\nCited: (none — answer not found in the retrieved pages)")

    print(f"\nLooked at {len(result['retrieved_pages'])} pages:")
    for rank, (page, score) in enumerate(result["retrieved_pages"], 1):
        print(f"  {rank}. {page}   (score {score:.3f})")
