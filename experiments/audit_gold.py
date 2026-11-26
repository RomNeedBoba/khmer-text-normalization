"""Audit the gold spoken forms for written-form characters left unspoken.

A gold spoken form should contain only Khmer words. This script flags every
unique sentence whose gold form still contains a digit (Arabic or Khmer), a
currency, percent or degree symbol, or a Latin letter (Section 3.3 of the
paper). EMAIL is excluded.

Input : dataset.jsonl (one record per entity: label, span, sentence, start,
        end, normalized). Records are deduplicated by sentence, as in
        make_split.py.
Output: <outdir>/gold_audit_flagged.jsonl  one flagged sentence per line
        <outdir>/gold_audit_summary.json   counts per label and per reason

Usage: python experiments/audit_gold.py data/dataset.jsonl data/audit
"""
import argparse, collections, json, os, re

CHECKS = [
    ("digit", re.compile(r"[0-9០-៩]")),
    ("currency_symbol", re.compile(r"[$៛€£¥]")),
    ("percent_symbol", re.compile(r"[%‰]")),               # percent and per-mille
    ("degree_symbol", re.compile(r"°")),
    ("latin_letter", re.compile(r"[A-Za-z]")),
]
DECIMAL_VALUE = re.compile(r"[0-9០-៩]\.[0-9០-៩]")      # value written with a decimal point


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("outdir")
    a = ap.parse_args()

    seen, rows = set(), []
    for line in open(a.src, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["sentence"] in seen:
            continue
        seen.add(r["sentence"]); rows.append(r)

    flagged = []
    total = collections.Counter(); decimal_total = collections.Counter()
    by_label = collections.Counter(); by_reason = collections.Counter()
    decimal_flagged = collections.Counter()
    for r in rows:
        lab = r["label"]; total[lab] += 1
        is_decimal = bool(DECIMAL_VALUE.search(r["span"]))
        decimal_total[lab] += is_decimal
        if lab == "EMAIL":
            continue
        reasons = [name for name, rx in CHECKS if rx.search(r["normalized"])]
        if not reasons:
            continue
        by_label[lab] += 1; decimal_flagged[lab] += is_decimal
        for name in reasons:
            by_reason[name] += 1
        flagged.append({"label": lab, "sentence": r["sentence"], "span": r["span"],
                        "normalized": r["normalized"], "reasons": reasons,
                        "decimal_value": is_decimal})

    summary = {
        "unique_sentences": len(rows),
        "flagged": len(flagged),
        "flagged_percent": round(100 * len(flagged) / len(rows), 2),
        "flagged_by_label": dict(by_label.most_common()),
        "flagged_by_reason": dict(by_reason.most_common()),
        "decimal_values_by_label": {k: {"flagged": decimal_flagged[k], "total": decimal_total[k]}
                                    for k in sorted(by_label)},
        "sentences_by_label": dict(sorted(total.items())),
    }
    os.makedirs(a.outdir, exist_ok=True)
    with open(os.path.join(a.outdir, "gold_audit_flagged.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for x in flagged:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    with open(os.path.join(a.outdir, "gold_audit_summary.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
