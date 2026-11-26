# Khmer Text Normalization

Text normalization for Khmer TTS: turning numbers, dates, money, phone numbers
and so on into the words you would actually say.

| Input | Output |
| --- | --- |
| `ថ្ងៃទី 15/07/2026` | `ថ្ងៃទីដប់ប្រាំ ខែកក្កដា ឆ្នាំពីរពាន់ម្ភៃប្រាំមួយ` |
| `$25.50` | `ម្ភៃប្រាំដុល្លារហាសិបសេន` |
| `ម៉ោង 8:30` | `ម៉ោងប្រាំបី សាមសិបនាទី` |
| `012 345 678` | `សូន្យ មួយ ពីរ បី បួន ប្រាំ ប្រាំមួយ ប្រាំពីរ ប្រាំបី` |

This is the code and data for our paper *A Hybrid Tag-and-Verbalize Framework
for Khmer Text Normalization in Text-to-Speech*. There are two ways to run it:

- **Rules only.** Regex patterns find each number-like span and a small rule
  engine reads it out. No dependencies.
- **Hybrid.** A spaCy NER model decides where the spans are and what type they
  are, the same rules read them out, and the rules also catch anything the
  model missed.

## Install

Needs Python 3.9+.

```bash
git clone https://github.com/RomNeedBoba/khmer-text-normalization.git
cd khmer-text-normalization
pip install -e .
pip install -r requirements.txt   # only if you want the hybrid / experiments (spaCy)
```

`pip install khmercut` is optional, it gives better word segmentation of the output.

## Quick start

```python
from khmer_tn import KhmerNormalizer

n = KhmerNormalizer()
n.normalize("តម្លៃ $25.50។ សូមទូរស័ព្ទ 012 345 678!")
# 'តម្លៃ ម្ភៃប្រាំដុល្លារហាសិបសេន <break time="500ms"/> សូមទូរស័ព្ទ សូន្យ មួយ ពីរ បី បួន ប្រាំ ប្រាំមួយ ប្រាំពីរ ប្រាំបី <break time="500ms"/>'

n.tag("ម៉ោង 8:30")
# [(0, 9, 'ម៉ោងប្រាំបី សាមសិបនាទី', 'TIME')]
```

With the model:

```python
import spacy
from khmer_tn.hybrid import HybridNormalizer

nlp = spacy.load("experiments/spacy_run/model/model-best")
n = HybridNormalizer(nlp)
n.normalize("តម្លៃ 199-483 ដុល្លារ")
```

A few options:

- `KhmerNormalizer(decimal_convention="official")` treats `,` as the decimal mark
- `PauseNormalizer(mode="preserve")` keeps punctuation instead of SSML breaks
- `cleaner=False, repeat=False, pause=False, segmenter=False` turns off every
  stage except tagging and reading out (this is what the evaluation uses, so
  character offsets stay the same as in the dataset)

## What it handles

When two patterns want the same piece of text, the one higher in this list wins.
That is why `15/07/2026` is read as a date and not as fractions.

| # | Type | Example | Read as |
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
| 15 | CARDINAL | `150` | `មួយរយហាសិប` |

There is also an abbreviation normalizer (`Dr.`, `Mr.`, dotted Khmer acronyms).
We added it after building the dataset, so it has no label there.

## Data

Everything is in `data/`.

`dataset.jsonl` has 103,000 records (90,304 different sentences) over the 15
types above. The sentences are generated from 125 short templates with one slot
each, so every sentence has exactly one span to normalize. Digits are Arabic
only. A record looks like this:

```json
{"label": "MONEY", "span": "USD8593", "sentence": "ប្រាក់ឧបត្ថម្ភ USD8593", "start": 15, "end": 22, "normalized": "ប្រាំបីពាន់ប្រាំរយកៅសិបបីដុល្លារ"}
```

`split/` is the train / val / test split we used. It is split by template, not
by sentence, so the test set only has templates the model never saw in training.

| | Sentences | Templates |
| --- | ---: | ---: |
| train | 56,661 | 78 |
| val | 11,030 | 15 |
| test | 22,613 | 32 |

`audit/` lists the sentences where the `normalized` text is not fully spoken.
We found 4,897 of them (5.42%). Almost all are values with a decimal point where
the number is read out but the unit is left as written, e.g. `83.90°C` becomes
`ប៉ែតសិបបីក្បៀសប្រាំបួនសូន្យ°C`. We have not fixed these yet, so keep it in mind
when you look at accuracy numbers: a system that reads the unit correctly gets
marked wrong on those sentences.

## Running the experiments

Run everything from the repo root. We used Python 3.12 and spaCy 3.8.14.

```bash
python tests/test_pipeline.py

python experiments/make_split.py data/dataset.jsonl data/split --seed 42
python experiments/audit_gold.py data/dataset.jsonl data/audit

python experiments/evaluate.py --code . --test data/split/test.jsonl --systems rules
python experiments/run_spacy_experiments.py        # spaCy's own tokenizer
python experiments/run_spacy_regex_tokenizer.py    # our regex tokenizer
```

The two `run_spacy_*` scripts use the trained models that are already in
`experiments/spacy_run*/model/model-best`, so they only take a few minutes. If
you want to train again, delete that `model` folder and rerun (30-60 minutes on
a laptop CPU). Numbers from a retrained model can move a little.

## Results

On the test set (22,613 sentences). Acc is the share of sentences where the
whole output matches, CER is character error rate. "Khmer digits" is the same
test set with the digits converted to Khmer numerals.

| | Acc | CER | Acc (Khmer digits) | CER (Khmer digits) |
| --- | ---: | ---: | ---: | ---: |
| Rules only | 90.00 | 1.89 | 90.00 | 1.89 |
| Model only, spaCy tokenizer | 53.54 | 29.13 | 35.80 | 46.56 |
| Hybrid, spaCy tokenizer | 90.51 | 2.30 | 86.57 | 3.28 |
| Model only, regex tokenizer | 57.36 | 25.07 | 54.42 | 29.00 |
| Hybrid, regex tokenizer | 86.16 | 4.15 | 87.33 | 4.28 |

So on templates it has not seen, the model adds very little on top of the rules.
It fixes some cases the rules get wrong (for example a range in front of a
currency word, `199-483 ដុល្លារ`), but it also overrides rules that were right.
Per-type scores and error examples are in `experiments/results_*.json`.

## Known issues

- The data is generated from templates. We have not tested on real-world Khmer
  text yet.
- The 5.42% of unfinished `normalized` forms mentioned above.
- In the hybrid, the model always wins over the rules. Only letting it win when
  it is confident would probably help, but we have not tried it.

## Folders

```
khmer_tn/       the normalizer (rules in categories/, hybrid in hybrid.py)
data/           dataset, split, audit
experiments/    scripts, result files, trained models
configs/        spaCy training config
tools/          helper scripts for preparing NER data
tests/
```

## License

The data in `data/` is CC BY-NC 4.0, see `data/LICENSE.md`.

## Citation

```bibtex
@misc{rin2026khmertn,
  title  = {A Hybrid Tag-and-Verbalize Framework for Khmer Text Normalization in Text-to-Speech},
  author = {Rin, Pichphyrom and Lean, Sokkong and Pin, Rathana and Him, Soklong and Leang, Sotheara},
  year   = {2026}
}
```

## Thanks

Thanks to Sabay Digital Corporation for the source text and to CADT for
supporting this work. The dataset only contains generated sentences, not text
from Sabay's archives.
