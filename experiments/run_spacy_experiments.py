"""One-command spaCy experiments (Experiment B, spaCy tokenizer).

What it does (no input needed, about 30-60 min on a normal laptop CPU):
  1. builds spaCy training files from data/split/train.jsonl and val.jsonl
     (template-stratified split, made by make_split.py with seed 42)
  2. retrains the spaCy NER with configs/config.cfg, choosing the
     best checkpoint on the VALIDATION set (the test set is never used for this)
  3. evaluates on data/split/test.jsonl:
       - spaCy classifier spans (strict per-class P/R/F1)       -> Table 5
       - rules only / spaCy only (no fallback) / hybrid          -> Table 6
         (exact match, CER, Arabic digits and Khmer digits)
       - error breakdown of the hybrid                          -> Sec. 5.4
  4. writes everything to experiments/results_spacy.json

Training is skipped when experiments/spacy_run/model/model-best already exists
(the trained model is shipped with the repository); delete that folder to retrain.

Run:  python experiments/run_spacy_experiments.py
"""
import json, os, sys, time, collections

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CODE = ROOT                                        # repository root (contains khmer_tn/)
CONFIG = os.path.join(ROOT, "configs", "config.cfg")
SPLIT = os.path.join(ROOT, "data", "split")
WORK = os.path.join(HERE, "spacy_run")
sys.path.insert(0, CODE)
sys.path.insert(0, HERE)

import spacy                                   # noqa: E402
from spacy.tokens import DocBin               # noqa: E402
from spacy.cli.train import train as spacy_train  # noqa: E402
import evaluate as ev                          # noqa: E402


def load(name):
    with open(os.path.join(SPLIT, name), encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


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


def classifier_spans(nlp, exs):
    """Strict span P/R/F1 of the raw spaCy predictions (before any snapping)."""
    tp, fp, fn = collections.Counter(), collections.Counter(), collections.Counter()
    for doc, ex in zip(nlp.pipe([x["text"] for x in exs], batch_size=256), exs):
        gold = {tuple(x) for x in ex["entities"]}
        pred = {(e.start_char, e.end_char, e.label_) for e in doc.ents}
        for p in pred:
            (tp if p in gold else fp)[p[2]] += 1
        for g in gold - pred:
            fn[g[2]] += 1
    per = {}
    for k in ev.ALL15:
        p = tp[k] / (tp[k] + fp[k]) if tp[k] + fp[k] else 0.0
        r = tp[k] / (tp[k] + fn[k]) if tp[k] + fn[k] else 0.0
        per[k] = {"P": p, "R": r, "F1": 2 * p * r / (p + r) if p + r else 0.0, "support": tp[k] + fn[k]}
    macro = lambda ks, m: sum(per[k][m] for k in ks) / len(ks)
    return {"per_class": per,
            "macro13": {m: macro(ev.BENCH13, m) for m in ("P", "R", "F1")},
            "macro15": {m: macro(ev.ALL15, m) for m in ("P", "R", "F1")}}


def main():
    t0 = time.time()
    os.makedirs(WORK, exist_ok=True)
    train, val, test = load("train.jsonl"), load("val.jsonl"), load("test.jsonl")
    print(f"train={len(train)} val={len(val)} test={len(test)}", flush=True)

    nlp_blank = spacy.blank("xx")             # same language as config.cfg
    sk_tr = to_docbin(train, nlp_blank, os.path.join(WORK, "train.spacy"))
    sk_va = to_docbin(val, nlp_blank, os.path.join(WORK, "val.spacy"))
    print(f"docbins written (spans not alignable to tokens: train={sk_tr}, val={sk_va})", flush=True)

    out_dir = os.path.join(WORK, "model")
    if not os.path.isdir(os.path.join(out_dir, "model-best")):
        print("training spaCy (this is the slow part)...", flush=True)
        spacy_train(CONFIG, out_dir, use_gpu=-1,
                    overrides={"paths.train": os.path.join(WORK, "train.spacy"),
                               "paths.dev": os.path.join(WORK, "val.spacy"),
                               "system.seed": 42, "training.seed": 42})
    nlp = spacy.load(os.path.join(out_dir, "model-best"))

    print("evaluating...", flush=True)
    res = {"split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
           "token_alignment_skips": {"train": sk_tr, "val": sk_va},
           "spacy_classifier": classifier_spans(nlp, test)}
    from khmer_tn.normalizer import KhmerNormalizer
    from khmer_tn.hybrid import HybridNormalizer
    kw = dict(cleaner=False, pause=False, repeat=False, segmenter=False)
    systems = {"rules": KhmerNormalizer(**kw),
               "spacy_only": HybridNormalizer(nlp, rule_fallback=False, **kw),
               "hybrid": HybridNormalizer(nlp, rule_fallback=True, **kw)}
    for name, s in systems.items():
        res[name] = {"arabic": ev.run(s, test), "khmer_digits": ev.run(s, test, khmer_digits=True)}
        a, k = res[name]["arabic"], res[name]["khmer_digits"]
        print(f"{name:10s} exact={100*a['exact_match']:.2f} CER={100*a['cer']:.2f} | "
              f"khmer-digit exact={100*k['exact_match']:.2f} CER={100*k['cer']:.2f}", flush=True)
    m = res["spacy_classifier"]["macro13"]
    print(f"spaCy classifier macro (13 classes): P={100*m['P']:.2f} R={100*m['R']:.2f} F1={100*m['F1']:.2f}")
    res["minutes"] = (time.time() - t0) / 60
    with open(os.path.join(HERE, "results_spacy.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\nDONE in {res['minutes']:.0f} min -> results_spacy.json")


if __name__ == "__main__":
    main()
