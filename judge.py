"""Reference-based LLM-as-a-judge for grading DocuMind answers.

Deliberately isolated from generate.py: the judge has its own model, its own
prompt, and no access to the page images. It sees only the question, the gold
answer, and the system answer, so it grades against the reference rather than
re-deciding what the paper says. Swap JUDGE_MODEL to change judges.
"""
import os, sys, json
from dotenv import load_dotenv
from google import genai

import api_retry

load_dotenv()

JUDGE_MODEL = "gemini-3.6-flash"     # stronger than the answering model on purpose
SEED = 7

VERDICTS = ("correct", "partial", "wrong")

SYSTEM_INSTRUCTION = """You grade answers from a document question-answering system against a reference answer.

You are given a question, the reference (gold) answer, and the system's answer. Judge ONLY whether the system answer matches the reference on the facts the question asks for.

Grade:
- "correct": every fact the question asks for is present and agrees with the reference. Extra correct detail is fine. Different wording is fine.
- "partial": some of the asked-for facts are right but others are missing or wrong. Also use this when the answer is right but hedged into uselessness, or covers only part of a multi-part question.
- "wrong": the answer contradicts the reference, answers a different question, or states it could not find the answer.

Rules:
- Judge against the reference, not your own knowledge of the subject.
- Numbers matter: a wrong or missing figure the question asked for is at best "partial".
- Do not reward fluency, length, or confidence. Do not penalise terse answers that are correct.
- Give one short sentence of reason naming the specific fact that decided the grade.
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(VERDICTS)},
        "reason": {"type": "string", "description": "One sentence naming the deciding fact."},
    },
    "required": ["verdict", "reason"],
}

_client = None


def _get_client():
    global _client
    if _client is None:
        if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            sys.exit("No Gemini key found. Add GEMINI_API_KEY=... to .env")
        _client = genai.Client()
    return _client


def grade(question, gold_answer, system_answer):
    """Return (verdict, reason) with verdict in correct / partial / wrong."""
    if not (system_answer or "").strip():
        return "wrong", "System produced an empty answer."

    prompt = (
        f"QUESTION:\n{question}\n\n"
        f"REFERENCE ANSWER:\n{gold_answer}\n\n"
        f"SYSTEM ANSWER:\n{system_answer}"
    )

    result = api_retry.call_with_retry(
        _get_client().interactions.create,
        model=JUDGE_MODEL,
        system_instruction=SYSTEM_INSTRUCTION,
        input=[{"type": "text", "text": prompt}],
        generation_config={"seed": SEED},
        response_format={
            "type": "text",
            "mime_type": "application/json",
            "schema": RESPONSE_SCHEMA,
        },
    )

    try:
        parsed = json.loads((result.output_text or "").strip())
    except json.JSONDecodeError:
        return "wrong", "Judge returned unparseable output."

    verdict = parsed.get("verdict", "wrong")
    if verdict not in VERDICTS:
        verdict = "wrong"
    return verdict, parsed.get("reason", "").strip()
