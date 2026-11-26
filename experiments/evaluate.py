"""Evaluation: span scores and end-to-end accuracy.

Systems:
  rules        - the khmer_tn rule-based normalizer alone (no classifier)
  spacy        - classifier only: spaCy spans, verbalized by the rules, NO fallback
  hybrid       - spaCy spans + regex fallback (khmer_tn.hybrid.HybridNormalizer)
(the spacy/hybrid systems need spaCy installed and --model)

Metrics:
  span P/R/F1  strict: predicted (start, end, label) must equal gold exactly;
               reported per class and macro-averaged over the 13 benchmark
               classes (EMAIL, ID excluded) and over all 15.
  exact match  output == gold sentence after NFC + removing all whitespace
  CER          sum of Levenshtein distances / sum of reference lengths,
               on the same normalized strings
  khmer-digit  the same test sentences with Arabic digits transliterated to
               Khmer digits (gold spoken forms unchanged)

Usage: python experiments/evaluate.py --code . --test data/split/test.jsonl
                         --systems rules [spacy hybrid --model model-best] --out results.json
"""
import argparse, collections, json, sys, unicodedata

BENCH13 = ["CARDINAL", "DATE", "DECIMAL", "DURATION", "FRACTION", "MEASURE", "MONEY",
           "ORDINAL", "PERCENT", "RANGE", "TELEPHONE", "TIME", "YEAR"]
ALL15 = sorted(BENCH13 + ["EMAIL", "ID"])
A2K = str.maketrans("0123456789", "០១២៣៤៥៦៧៨៩")


def norm(s):
    return "".join(unicodedata.normalize("NFC", s).split())


def lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def build_systems(names, code, model):
    sys.path.insert(0, code)
    from khmer_tn.normalizer import KhmerNormalizer
    kw = dict(cleaner=False, pause=False, repeat=False, segmenter=False)
    out = {}
    if "rules" in names:
        out["rules"] = KhmerNormalizer(**kw)
    if {"spacy", "hybrid"} & set(names):
        import spacy
        from khmer_tn.hybrid import HybridNormalizer
        nlp = spacy.load(model)
        if "spacy" in names:
            out["spacy"] = HybridNormalizer(nlp, rule_fallback=False, **kw)
        if "hybrid" in names:
            out["hybrid"] = HybridNormalizer(nlp, rule_fallback=True, **kw)
    return out


def run(system, exs, khmer_digits=False):
    tp, fp, fn = collections.Counter(), collections.Counter(), collections.Counter()
    exact = edits = ref_len = 0
    errors = collections.Counter()           # gold label of failed sentences
    err_type = collections.Counter()
    samples = []
    for ex in exs:
        text = ex["text"]
        s, e, lab = ex["entities"][0]
        if khmer_digits and lab != "EMAIL":
            text = text.translate(A2K)
        gold = {(s, e, lab)}
        spans = system.tag(text)
        pred = {(a, b, c) for a, b, _, c in spans}
        for p in pred:
            if p in gold: tp[p[2]] += 1
            else: fp[p[2]] += 1
        for g in gold - pred:
            fn[g[2]] += 1
        out = []; last = 0
        for a, b, rep, _ in spans:
            out.append(text[last:a]); out.append(rep); last = b
        out.append(text[last:])
        hyp, ref = norm("".join(out)), norm(ex["gold"])
        d = lev(hyp, ref); edits += d; ref_len += len(ref)
        if d == 0:
            exact += 1
        else:
            errors[lab] += 1
            same_extent = [p for p in pred if (p[0], p[1]) == (s, e)]
            overlap = [p for p in pred if p[0] < e and s < p[1]]
            if (s, e, lab) in pred:   t = "span+label correct, spoken form differs from gold"
            elif same_extent:         t = "right extent, wrong category"
            elif overlap:             t = "boundary error"
            elif not pred:            t = "missed span"
            else:                     t = "missed span + spurious span elsewhere"
            err_type[t] += 1
            if len(samples) < 40:
                samples.append({"text": text, "label": lab, "hyp": "".join(out), "gold": ex["gold"], "type": t})
    per = {}
    for k in ALL15:
        p = tp[k] / (tp[k] + fp[k]) if tp[k] + fp[k] else 0.0
        r = tp[k] / (tp[k] + fn[k]) if tp[k] + fn[k] else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per[k] = {"P": p, "R": r, "F1": f, "support": tp[k] + fn[k]}
    macro = lambda ks, m: sum(per[k][m] for k in ks) / len(ks)
    return {
        "n": len(exs), "exact_match": exact / len(exs), "cer": edits / ref_len,
        "macro13": {m: macro(BENCH13, m) for m in ("P", "R", "F1")},
        "macro15": {m: macro(ALL15, m) for m in ("P", "R", "F1")},
        "per_class": per, "failed_sentences_by_gold_label": dict(errors.most_common()),
        "error_types": dict(err_type.most_common()), "other_label_fp": {k: v for k, v in fp.items() if k not in ALL15},
        "error_samples": samples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True); ap.add_argument("--test", required=True)
    ap.add_argument("--systems", nargs="+", default=["rules"]); ap.add_argument("--model")
    ap.add_argument("--out", default="results.json")
    a = ap.parse_args()
    exs = [json.loads(l) for l in open(a.test, encoding="utf-8") if l.strip()]
    res = {}
    for name, sysm in build_systems(a.systems, a.code, a.model).items():
        res[name] = {"arabic": run(sysm, exs), "khmer_digits": run(sysm, exs, khmer_digits=True)}
        r = res[name]
        print(f"{name:7s} exact={100*r['arabic']['exact_match']:.2f} CER={100*r['arabic']['cer']:.2f} "
              f"macroF1(13)={100*r['arabic']['macro13']['F1']:.2f} | khmer-digit exact={100*r['khmer_digits']['exact_match']:.2f} "
              f"CER={100*r['khmer_digits']['cer']:.2f}", flush=True)
    json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
