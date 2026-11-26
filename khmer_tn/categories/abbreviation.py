"""ABBREVIATION

Two distinct patterns under one category:

  1. Known Latin titles/honorifics (Mr., Dr., Prof., ...) -> looked up and
     expanded to their spoken Khmer form.
  2. Generic Khmer dot-separated acronyms (ក.ស.ម) -> dots stripped, each
     letter spoken as its own syllable, space-separated. This is a
     GENERALIZATION of the pattern YearNormalizer already uses for the
     ព.ស./គ.ស. era marker specifically -- so this category must run AFTER
     YearNormalizer in priority order, or it will steal that match.

Add new entries to TITLES as you find them; the pattern rebuilds
automatically from the dict keys.
"""

import re

from .base import BaseNormalizer

TITLES = {
    "Mr.": "លោក",
    "Mrs.": "លោកស្រី",
    "Ms.": "កញ្ញា",
    "Dr.": "វេជ្ជបណ្ឌិត",
    "Prof.": "សាស្ត្រាចារ្យ",
}
# longest first so "Mrs." doesn't get shadowed by a shorter partial match
_TITLE_ALT = "|".join(re.escape(t) for t in sorted(TITLES, key=len, reverse=True))

# one bare Khmer consonant or independent vowel, no diacritics --
# same character class base.py/repeat.py already use for a single letter
_KH_LETTER = "[\u1780-\u17a2\u17a5-\u17b3]"


class AbbreviationNormalizer(BaseNormalizer):
    category = "ABBREVIATION"

    def __init__(self):
        self.titles = re.compile(rf"(?<![A-Za-z]){_TITLE_ALT}(?![A-Za-z])")
        # first letter, then 1+ more ".letter" segments, optional trailing dot
        self.acronym = re.compile(rf"{_KH_LETTER}(?:\.{_KH_LETTER}){{1,}}\.?")
        self.pattern = self.titles  # satisfies the base contract

    def _expand_title(self, m):
        return TITLES[m.group(0)]

    def _expand_acronym(self, m):
        letters = [c for c in m.group(0) if c != "."]
        return " ".join(letters)

    def expand(self, match):
        return self._expand_title(match)

    @staticmethod
    def _safe(fn):
        def wrap(m):
            try:
                return fn(m)
            except Exception:
                return m.group(0)
        return wrap

    def rules(self):
        return [
            (self.titles, self._safe(self._expand_title)),
            (self.acronym, self._safe(self._expand_acronym)),
        ]

    def normalize(self, text: str) -> str:
        text = self.titles.sub(self._safe(self._expand_title), text)
        text = self.acronym.sub(self._safe(self._expand_acronym), text)
        return text