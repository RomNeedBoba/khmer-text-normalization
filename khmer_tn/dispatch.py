"""Hybrid runtime: spaCy NER decides the category, the regex normalizers
parse structure and verbalize.

    doc = nlp(text)                 # your trained spaCy model
    entities = [(e.start_char, e.end_char, e.label_) for e in doc.ents]
    spoken = normalize_with_ner(text, entities)

Each spaCy-predicted (start, end, label) is matched back to the SAME regex
that would have produced it, re-run against the full original `text` (not
the sliced substring) so lookbehind/lookahead assertions still see the real
surrounding characters. The regex's own groups are then used for expansion,
exactly as in the priority-based tag()/verbalize() flow - the only thing
that changed is who decides the label.

Anything spaCy doesn't tag (bare numbers, mostly) still needs to be spoken
for TTS, so a final CardinalNormalizer.normalize() pass runs regex-only
over whatever's left, same as a NORMAL pass would have done as the lowest
priority fallback.
"""

from .categories import (
    EmailNormalizer, TelephoneNormalizer, DateNormalizer, TimeNormalizer,
    DurationNormalizer, MoneyNormalizer, PercentNormalizer, MeasureNormalizer,
    FractionNormalizer, RangeNormalizer, DecimalNormalizer, OrdinalNormalizer,
    YearNormalizer, IdNormalizer, CardinalNormalizer,
)

_LABELED_CATEGORIES = [
    EmailNormalizer(), TelephoneNormalizer(), DateNormalizer(), TimeNormalizer(),
    DurationNormalizer(), MoneyNormalizer(), PercentNormalizer(), MeasureNormalizer(),
    FractionNormalizer(), RangeNormalizer(), DecimalNormalizer(), OrdinalNormalizer(),
    YearNormalizer(), IdNormalizer(),
]
BY_LABEL = {n.category: n for n in _LABELED_CATEGORIES}
_FALLBACK = CardinalNormalizer()  # never a NER label, always runs at the end


def verbalize_span(text: str, start: int, end: int, label: str):
    """Find the regex match for `label`'s normalizer whose span COVERS
    [start, end) in `text`, and expand using its own capture groups.
    Returns (replacement, actual_start, actual_end) or None if the model's
    label has no known normalizer, or the regex can't confirm the span."""
    norm = BY_LABEL.get(label)
    if norm is None:
        return None
    for pattern, expand in norm.rules():
        for m in pattern.finditer(text):
            s, e = m.span()
            if s <= start and e >= end:
                try:
                    return expand(m), s, e
                except Exception:
                    continue
    return None


def normalize_with_ner(text: str, entities) -> str:
    """entities: [(start, end, label), ...] as predicted by spaCy.
    Returns the fully-verbalized text."""
    out, last = [], 0
    for start, end, label in sorted(entities):
        if start < last:          # overlapping predictions - skip, keep first
            continue
        result = verbalize_span(text, start, end, label)
        if result is None:
            out.append(text[last:end])   # leave the model's span untouched
            last = end
            continue
        rep, s, e = result
        out.append(text[last:s])
        out.append(rep)
        last = e
    out.append(text[last:])
    normalized = "".join(out)
    return _FALLBACK.normalize(normalized)


def normalize_with_spacy_doc(nlp, text: str) -> str:
    """Convenience wrapper: run your trained spaCy pipeline and verbalize."""
    doc = nlp(text)
    entities = [(e.start_char, e.end_char, e.label_) for e in doc.ents]
    return normalize_with_ner(text, entities)