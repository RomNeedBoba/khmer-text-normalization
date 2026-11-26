# Khmer Text Normalization for Text-to-Speech

Code, data, and evaluation material for the paper

> **A Hybrid Tag-and-Verbalize Framework for Khmer Text Normalization in Text-to-Speech**
> Pichphyrom Rin, Sokkong Lean, Rathana Pin, Soklong Him, Sotheara Leang
> Cambodia Academy of Digital Technology (CADT). ACET 2026, Student Track.

Text normalization (TN) converts non-standard words such as numerals, dates, and
currency into their spoken form so a TTS system can read them. This repository
holds a hybrid Khmer normalizer: a learned sequence-labeling classifier decides
where the non-standard words are and which category they belong to, a
deterministic rule engine verbalizes each span, and the same rules act as a
fallback for spans the classifier misses.

| Raw | Normalized |
| --- | --- |
| `ថ្ងៃទី 15/07/2026` | `ថ្ងៃទីដប់ប្រាំ ខែកក្កដា ឆ្នាំពីរពាន់ម្ភៃប្រាំមួយ` |
| `$25.50` | `ម្ភៃប្រាំដុល្លារហាសិបសេន` |
| `ម៉ោង 8:30` | `ម៉ោងប្រាំបី សាមសិបនាទី` |
| `012 345 678` | `សូន្យ មួយ ពីរ បី បួន ប្រាំ ប្រាំមួយ ប្រាំពីរ ប្រាំបី` |

## Repository layout

```
khmer_tn/              Normalizer package (no required dependencies)
  normalizer.py        Rule pipeline: CLEAN -> TAG+VERBALIZE -> REPEAT -> PAUSE -> SEGMENT
  hybrid.py            HybridNormalizer: classifier spans, rule verbalization, rule fallback
  align.py             Snapping of a predicted span to its category's pattern
  categories/          One normalizer per semiotic category, plus pause and repeat
  utils/               Number reader, digit conversion, letter names
tests/                 Test suite for the verbalizer
data/
  dataset.jsonl        The corpus: 103,000 annotated records
  split/               Template-stratified train / val / test split (seed 42)
  audit/               Gold-form audit: flagged sentences and summary
configs/config.cfg     spaCy training configuration (Experiment B)
experiments/
  make_split.py        Builds data/split from data/dataset.jsonl
  audit_gold.py        Builds data/audit from data/dataset.jsonl
  evaluate.py          Span P/R/F1, exact match, CER, Khmer-digit test
  run_spacy_experiments.py       Experiment B with spaCy's tokenizer
  run_spacy_regex_tokenizer.py   Experiment B with the regex tokenizer
  results_*.json       Full results behind the tables of the paper
  spacy_run*/model/    The two trained spaCy models (model-best)
tools/                 Data-preparation helpers (format conversion, span repair)
```

## Installation

Python 3.9 or later. The rule-based normalizer has no required dependencies.

```bash
git clone https://github.com/RomNeedBoba/khmer-text-normalization.git
cd khmer-text-normalization
pip install -e .                  # rules only
pip install -r requirements.txt   # adds spaCy, for the hybrid and the experiments
```

Optional: `pip install khmercut` enables dictionary-based output segmentation.

## Usage

Rules only:

```python
from khmer_tn import KhmerNormalizer

normalizer = KhmerNormalizer()
normalizer.normalize("តម្លៃ $25.50។ សូមទូរស័ព្ទ 012 345 678!")
# 'តម្លៃ ម្ភៃប្រាំដុល្លារហាសិបសេន <break time="500ms"/> សូមទូរស័ព្ទ សូន្យ មួយ ពីរ បី បួន ប្រាំ ប្រាំមួយ ប្រាំពីរ ប្រាំបី <break time="500ms"/>'

normalizer.tag("ម៉ោង 8:30")       # spans without rewriting the text
# [(0, 9, 'ម៉ោងប្រាំបី សាមសិបនាទី', 'TIME')]
```

Hybrid (the classifier decides where and which category, the rules produce the
spoken form, and the rules fill in what the classifier misses):

```python
import spacy
from khmer_tn.hybrid import HybridNormalizer

nlp = spacy.load("experiments/spacy_run/model/model-best")
hybrid = HybridNormalizer(nlp)                 # rule_fallback=True by default
hybrid.normalize("តម្លៃ 199-483 ដុល្លារ")
```

The evaluation in the paper runs the tag-and-verbalize stage alone, so that
character offsets match the gold annotations:

```python
kw = dict(cleaner=False, repeat=False, pause=False, segmenter=False)
rules_only = KhmerNormalizer(**kw)
classifier_only = HybridNormalizer(nlp, rule_fallback=False, **kw)
hybrid = HybridNormalizer(nlp, rule_fallback=True, **kw)
```

Other options: `decimal_convention="official"` reads `,` as the decimal
separator, and `PauseNormalizer(mode="preserve")` keeps punctuation instead of
emitting SSML breaks.

## Categories

All categories propose spans on the same text. Overlaps are resolved by this
priority order (most specific first) and the winners are replaced in one pass,
which avoids cascades such as an ordinal rule taking the day out of a date.

| # | Category | Raw | Verbalized |
| --- | --- | --- | --- |
| 1 | EMAIL | `champa@example.com` | `ស៊ី អេច អេ អឹម ភី អេ អាត់ អ៊ី អែក្ស អេ អឹម ភី អែល អ៊ី ចុច ស៊ី អូ អឹម` |
| 2 | TELEPHONE | `012 345 678` | `សូន្យ មួយ ពីរ បី បួន ប្រាំ ប្រាំមួយ ប្រាំពីរ ប្រាំបី` |
| 3 | DATE | `ថ្ងៃទី 15/07/2026` | `ថ្ងៃទីដប់ប្រាំ ខែកក្កដា ឆ្នាំពីរពាន់ម្ភៃប្រាំមួយ` |
| 4 | TIME | `ម៉ោង 8:30` | `ម៉ោងប្រាំបី សាមសិបនាទី` |
| 5 | DURATION | `45 mins` | `សែសិបប្រាំនាទី` |
| 6 | MONEY | `$25.50` | `ម្ភៃប្រាំដុល្លារហាសិបសេន` |
| 7 | PERCENT | `25%` | `ម្ភៃប្រាំភាគរយ` |
| 8 | MEASURE | `5km` | `ប្រាំគីឡូម៉ែត្រ` |
| 9 | FRACTION | `3/4` | `បីភាគបួន` |
| 10 | RANGE | `2020-2026` | `ពីរពាន់ម្ភៃដល់ពីរពាន់ម្ភៃប្រាំមួយ` |
| 11 | DECIMAL | `3.14` | `បីក្បៀសដប់បួន` |
| 12 | ORDINAL | `ទី 3` | `ទីបី` |
| 13 | YEAR | `ឆ្នាំ 2026 គ.ស.` | `ឆ្នាំ ពីរពាន់ម្ភៃប្រាំមួយ គ្រិស្តសករាជ` |
| 14 | ID | `AB1234` | `អេ ប៊ី មួយ ពីរ បី បួន` |
| 15 | CARDINAL (fallback) | `150` | `មួយរយហាសិប` |

The package also contains an `AbbreviationNormalizer` (titles such as `Dr.` and
dotted Khmer acronyms). It is outside the 15 categories of the paper and has no
label in the corpus.

## Corpus

The corpus is generated from 125 templates, each a short carrier phrase with one
typed slot, filled with generated values in varied formats.

- `data/dataset.jsonl`: 103,000 annotated records, 90,304 unique sentences, 15 categories
- Each record: `sentence`, `span`, `start`, `end`, `label`, `normalized` (the gold spoken form)
- One non-standard word per sentence, median length 16 characters, Arabic digits only

**Template-stratified split** (seed 42). Each sentence is reduced to its
template, and whole template groups are assigned per label, so no template
appears in more than one partition. Each line of `data/split/*.jsonl` holds
`text`, `entities` (`[start, end, label]`) and `gold` (the fully verbalized
sentence).

| Partition | Sentences | Templates |
| --- | ---: | ---: |
| Train | 56,661 | 78 |
| Validation | 11,030 | 15 |
| Test | 22,613 | 32 |

**Gold-form audit.** 4,897 sentences (5.42%) have a gold spoken form that leaves
a unit unspoken: whenever the value contains a decimal point, the number is
verbalized but its unit is copied as written (for example `83.90°C` becomes
`ប៉ែតសិបបីក្បៀសប្រាំបួនសូន្យ°C`). This affects decimal MEASURE, PERCENT, and
MONEY values (818, 1,795 and 2,283 sentences), plus one TELEPHONE number. The
forms are kept unchanged so results stay comparable. The flagged sentences are
listed in `data/audit/gold_audit_flagged.jsonl`.

## Reproducing the results

All commands run from the repository root. The numbers in the paper were
produced with Python 3.12 and spaCy 3.8.14.

```bash
python tests/test_pipeline.py                                          # 15 span cases + 6 stage checks
python experiments/make_split.py data/dataset.jsonl data/split --seed 42
python experiments/audit_gold.py data/dataset.jsonl data/audit
python experiments/evaluate.py --code . --test data/split/test.jsonl --systems rules
python experiments/run_spacy_experiments.py                            # spaCy tokenizer
python experiments/run_spacy_regex_tokenizer.py                        # regex tokenizer
```

`make_split.py` and `audit_gold.py` regenerate the files already in `data/`.
The two `run_spacy_*` scripts evaluate the trained models shipped in
`experiments/spacy_run*/model/model-best` and write `experiments/results_*.json`.
To retrain from scratch (30 to 60 minutes on a laptop CPU), delete the
corresponding `experiments/spacy_run*/model` folder first; a retrained model may
differ slightly from the shipped one.

## Results

End-to-end on the released test set (22,613 sentences, every template unseen in
training). Acc is sentence-level exact match; CER is character error rate.

| System | Acc (Arabic digits) | CER | Acc (Khmer digits) | CER |
| --- | ---: | ---: | ---: | ---: |
| Rules only | 90.00 | **1.89** | **90.00** | **1.89** |
| Classifier only, spaCy tokenizer | 53.54 | 29.13 | 35.80 | 46.56 |
| Hybrid, spaCy tokenizer | **90.51** | 2.30 | 86.57 | 3.28 |
| Classifier only, regex tokenizer | 57.36 | 25.07 | 54.42 | 29.00 |
| Hybrid, regex tokenizer | 86.16 | 4.15 | 87.33 | 4.28 |

Span tagging on the same test set (strict match, macro-F1 over 13 categories):
rules 82.9, spaCy with its own tokenizer 64.0, spaCy with the regex tokenizer 59.0.
Per-class scores and error breakdowns are in `experiments/results_*.json`.

The original architecture comparison (Experiment A, earlier corpus version,
80/20 split) gave macro-F1 of 94.2 for spaCy, 92.4 for a CRF, and 82.2 for
XLM-RoBERTa. Those scores were not reproduced under the conditions of
Experiment B, which is the reference result. Experiment A is not part of this
repository.

## Limitations

- The corpus is template-generated, so all scores are in-domain. The system has
  not yet been evaluated on naturally occurring Khmer text.
- On unseen templates the classifier generalizes poorly, and the hybrid matches
  rather than beats the rules-only baseline.
- 5.42% of the gold forms leave a unit unspoken, which lowers measured accuracy
  for systems that read units correctly.
- The Khmer-digit test is a transliteration of the test set, not natural usage.
- Classifier spans take unconditional precedence over rules; a confidence-gated
  combination was not evaluated.

## License

The corpus and everything under `data/` are released under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/); see
`data/LICENSE.md`.

## Citation

```bibtex
@inproceedings{rin2026khmertn,
  title     = {A Hybrid Tag-and-Verbalize Framework for Khmer Text Normalization in Text-to-Speech},
  author    = {Rin, Pichphyrom and Lean, Sokkong and Pin, Rathana and Him, Soklong and Leang, Sotheara},
  booktitle = {ACET 2026, Student Track},
  year      = {2026}
}
```

## Acknowledgments

We thank Sabay Digital Corporation for providing text from its Khmer TTS corpus,
and the Cambodia Academy of Digital Technology (CADT) for supporting this
research. The released corpus consists of template-generated sentences built
from 125 short carrier phrases and contains no complete text from Sabay's
archives.
