"""Retry helper for Gemini calls.

Two failure modes show up in a long eval run, and both are transient:
rate limits (free tier allows 20 requests/minute) and dropped connections.
The API tells us how long to wait for the former, so honour that when present.
"""
import re, time, random

MAX_ATTEMPTS = 8
BASE_DELAY = 5.0


def is_rate_limit(exc):
    name = type(exc).__name__
    text = str(exc).lower()
    return "ratelimit" in name.lower() or "429" in text or "too_many_requests" in text


def is_connection_error(exc):
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return ("connection" in name or "timeout" in name
            or "connection reset" in text or "connection error" in text)


def is_transient(exc):
    return is_rate_limit(exc) or is_connection_error(exc)


def _retry_after(exc):
    """The API suggests a wait in its message; prefer it over our own backoff."""
    m = re.search(r"retry in ([\d.]+)\s*s", str(exc), re.I)
    return float(m.group(1)) if m else None


def call_with_retry(fn, *args, **kwargs):
    """Call fn, backing off and retrying on rate limits."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if not is_transient(exc) or attempt == MAX_ATTEMPTS:
                raise
            delay = (_retry_after(exc) or BASE_DELAY * 2 ** (attempt - 1)) + random.uniform(0.5, 1.5)
            why = "rate limited" if is_rate_limit(exc) else type(exc).__name__
            print(f"      {why}, waiting {delay:.0f}s "
                  f"(attempt {attempt}/{MAX_ATTEMPTS})", flush=True)
            time.sleep(delay)
