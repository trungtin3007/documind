"""Document structure for sample.pdf, shared so the ranges cannot drift apart.

Pages 14-18 are acknowledgments, the reference list, and the appendix table of
contents. They are excluded wherever text is scored: a bibliography is dense in
exactly the terms a question uses (model names, "reasoning", "language models")
while containing no answer, so lexical scorers rank it highly for the wrong
reason. This is why hybrid retrieval lost u01's gold page to p17.

Both find_collisions.py and text_score.py import from here.
"""
import os

MAIN_RANGE = (1, 13)              # title through conclusion
NON_CONTENT_RANGE = (14, 18)      # acknowledgments, references, appendix TOC
APPENDIX_RANGE = (19, None)       # None = to the end of the document


def page_number(page):
    """'page_054.png' -> 54"""
    return int(os.path.splitext(page)[0].split("_")[-1])


def in_range(n, rng):
    lo, hi = rng
    return lo <= n and (hi is None or n <= hi)


def is_non_content(n):
    """True for front/back matter that should never be scored as an answer."""
    return in_range(n, NON_CONTENT_RANGE)


def scope_of(n):
    """'main' | 'appendix' | 'non-content' for a 1-based page number."""
    if in_range(n, MAIN_RANGE):
        return "main"
    if is_non_content(n):
        return "non-content"
    return "appendix"
