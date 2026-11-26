"""Snap a (possibly misaligned) labeled span to the category's regex match.

Used in two places:
  * offline - repairing NER training data whose spans don't line up with
    what the rule-based patterns expect (annotator included/excluded a
    currency symbol, a ថ្ងៃទី prefix, a trailing word, ...)
  * runtime - aligning a model-predicted span before verbalizing it with
    the category's expand(), so a slightly-off prediction still works

Strategy: search a padded window around the span with every pattern of
that category; among matches that overlap the span, take the one with the
largest overlap (ties: the longer match). Returns None when nothing in
that category matches anywhere near the span.
"""

from .categories import __dict__ as _cat_ns  # noqa: F401  (import side effect)
from .normalizer import DEFAULT_CATEGORIES

_PAD = 12  # chars of context searched on each side of the labeled span


def build_category_map(categories=None):
    cats = categories if categories is not None else [cls() for cls in DEFAULT_CATEGORIES]
    return {c.category: c for c in cats}


def snap_span(text: str, start: int, end: int, category: str, cat_map=None, pad: int = _PAD):
    """Return (new_start, new_end, replacement) or None if unalignable."""
    cat_map = cat_map or build_category_map()
    norm = cat_map.get(category)
    if norm is None:
        return None

    lo = max(0, start - pad)
    hi = min(len(text), end + pad)
    window = text[lo:hi]

    best = None  # (overlap, length, abs_start, abs_end, expand_fn, match)
    for pattern, expand in norm.rules():
        for m in pattern.finditer(window):
            s, e = m.start() + lo, m.end() + lo
            overlap = min(e, end) - max(s, start)
            if overlap <= 0:
                continue
            key = (overlap, e - s)
            if best is None or key > best[0]:
                best = (key, s, e, expand, m)

    if best is None:
        return None
    _, s, e, expand, m = best
    try:
        rep = expand(m)
    except Exception:
        return None
    if rep == m.group(0):        # category declined (e.g. ID heuristic)
        return None
    return s, e, rep
