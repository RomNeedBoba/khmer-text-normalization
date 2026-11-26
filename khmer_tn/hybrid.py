"""Hybrid normalizer: a spaCy NER model decides WHERE and WHICH CATEGORY,
the existing rule-based regexes decide the SPOKEN FORM.

    nlp = spacy.load("path/to/your_km_model")   # in-process, no server
    n = HybridNormalizer(nlp)
    n.normalize("តម្លៃ $25.50")

Every model prediction is snapped to its category's regex match
(khmer_tn.align.snap_span), so slightly-off boundaries still verbalize.
Predictions that can't be aligned to any rule of their category are
skipped — and, with rule_fallback=True (default), the plain rule-based
tagger then fills in anything the model missed entirely. That makes the
hybrid strictly safer than the model alone: the rules are the floor.
"""

from .align import snap_span, build_category_map
from .normalizer import KhmerNormalizer


class HybridNormalizer(KhmerNormalizer):
    def __init__(self, nlp, rule_fallback: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.nlp = nlp
        self.rule_fallback = rule_fallback
        self._cat_map = build_category_map(self.categories)

    def tag(self, text: str):
        claimed = []

        def overlaps(s, e):
            return any(s < ce and cs < e for cs, ce, _, _ in claimed)

        # 1) model proposes spans + categories; rules verbalize
        for ent in self.nlp(text).ents:
            out = snap_span(text, ent.start_char, ent.end_char, ent.label_, self._cat_map)
            if out is None:
                continue
            s, e, rep = out
            if overlaps(s, e):
                continue
            claimed.append((s, e, rep, ent.label_))

        # 2) optional: plain rule pass catches what the model missed
        if self.rule_fallback:
            for s, e, rep, cat in super().tag(text):
                if not overlaps(s, e):
                    claimed.append((s, e, rep, cat))

        return sorted(claimed)
