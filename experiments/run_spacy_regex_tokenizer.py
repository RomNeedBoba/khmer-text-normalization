"""Experiment B, second run: spaCy with the paper's offset-preserving regex tokenizer.

Same as run_spacy_experiments.py, except that spaCy's own tokenizer is
bypassed (as described in the paper and in the internship report): tokens are
maximal runs of Khmer letters, of digits (Arabic or Khmer), of Latin letters,
or single other visible characters. Training uses these tokens directly
(gold_preproc), and the same tokenizer is installed at inference.

Writes results_spacy_regex.json next to this file.
"""
import json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CODE = ROOT                                        # repository root (contains khmer_tn/)
CONFIG = os.path.join(ROOT, "configs", "config.cfg")
SPLIT = os.path.join(ROOT, "data", "split")
WORK = os.path.join(HERE, "spacy_run_regex")
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)

import spacy                                       # noqa: E402
from spacy.tokens import Doc, DocBin               # noqa: E402
from spacy.cli.train import train as spacy_train   # noqa: E402
import evaluate as ev                              # noqa: E402
from run_spacy_experiments import load, classifier_spans  # noqa: E402

# Khmer letters/marks (U+1780-U+17DF, U+19E0-U+19FF, minus the Khmer digits U+17E0-U+17E9),
# digits (Arabic or Khmer), Latin letters, any other single visible character.
TOKEN_RE = re.compile(r"[ក-៟៰-៿᧠-᧿​‌‍]+"
                      r"|[0-9០-៩]+|[A-Za-z]+|\S")


def regex_tokens(text):
    """Return (words, spaces) that reproduce `text` exactly."""
    words, spaces, pos = [], [], 0
    for m in TOKEN_RE.finditer(text):
        gap = text[pos:m.start()]
        if gap:                                    # whitespace before this token
            if words and gap == " " and not spaces[-1]:
                spaces[-1] = True
            else:
                words.append(gap); spaces.append(False)
        words.append(m.group()); spaces.append(False)
        pos = m.end()
    tail = text[pos:]
    if tail:
        if words and tail == " " and not spaces[-1]:
            spaces[-1] = True
        else:
            words.append(tail); spaces.append(False)
    return words, spaces


class RegexTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words, spaces = regex_tokens(text)
        doc = Doc(self.vocab, words=words, spaces=spaces)
        assert doc.text == text
        return doc


def to_docbin(exs, nlp, path):
    db, skipped = DocBin(), 0
    for ex in exs:
        doc = nlp.make_doc(ex["text"])
        ents = []
        for s, e, lab in ex["entities"]:
            span = doc.char_span(s, e, label=lab, alignment_mode="contract")
            if span is None:
                skipped += 1
            else:
                ents.append(span)
        doc.ents = ents
        db.add(doc)
    db.to_disk(path)
    return skipped


def main():
    t0 = time.time()
    os.makedirs(WORK, exist_ok=True)
    train, val, test = load("train.jsonl"), load("val.jsonl"), load("test.jsonl")
    print(f"train={len(train)} val={len(val)} test={len(test)}", flush=True)

    blank = spacy.blank("xx")
    blank.tokenizer = RegexTokenizer(blank.vocab)
    sk_tr = to_docbin(train, blank, os.path.join(WORK, "train.spacy"))
    sk_va = to_docbin(val, blank, os.path.join(WORK, "val.spacy"))
    print(f"docbins written (spans not alignable to tokens: train={sk_tr}, val={sk_va})", flush=True)

    out_dir = os.path.join(WORK, "model")
    if not os.path.isdir(os.path.join(out_dir, "model-best")):
        print("training spaCy with the regex tokenizer (slow part)...", flush=True)
        spacy_train(CONFIG, out_dir, use_gpu=-1,
                    overrides={"paths.train": os.path.join(WORK, "train.spacy"),
                               "paths.dev": os.path.join(WORK, "val.spacy"),
                               "corpora.train.gold_preproc": True,
                               "corpora.dev.gold_preproc": True,
                               "system.seed": 42, "training.seed": 42})
    nlp = spacy.load(os.path.join(out_dir, "model-best"))
    nlp.tokenizer = RegexTokenizer(nlp.vocab)

    print("evaluating...", flush=True)
    res = {"tokenizer": "regex (Khmer letters | digits | Latin | other)",
           "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
           "token_alignment_skips": {"train": sk_tr, "val": sk_va},
           "spacy_classifier": classifier_spans(nlp, test)}
    from khmer_tn.hybrid import HybridNormalizer
    kw = dict(cleaner=False, pause=False, repeat=False, segmenter=False)
    systems = {"spacy_only": HybridNormalizer(nlp, rule_fallback=False, **kw),
               "hybrid": HybridNormalizer(nlp, rule_fallback=True, **kw)}
    for name, s in systems.items():
        res[name] = {"arabic": ev.run(s, test), "khmer_digits": ev.run(s, test, khmer_digits=True)}
        a, k = res[name]["arabic"], res[name]["khmer_digits"]
        print(f"{name:10s} exact={100*a['exact_match']:.2f} CER={100*a['cer']:.2f} | "
              f"khmer-digit exact={100*k['exact_match']:.2f} CER={100*k['cer']:.2f}", flush=True)
    # classifier spans on the Khmer-digit version too
    kd = [dict(x, text=x["text"].translate(ev.A2K) if x["entities"][0][2] != "EMAIL" else x["text"]) for x in test]
    res["spacy_classifier_khmer_digits"] = classifier_spans(nlp, kd)
    m = res["spacy_classifier"]["macro13"]
    print(f"spaCy classifier macro (13 classes): P={100*m['P']:.2f} R={100*m['R']:.2f} F1={100*m['F1']:.2f}")
    res["minutes"] = (time.time() - t0) / 60
    with open(os.path.join(HERE, "results_spacy_regex.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\nDONE in {res['minutes']:.0f} min -> results_spacy_regex.json")


if __name__ == "__main__":
    main()
