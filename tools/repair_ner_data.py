"""Repair misaligned NER spans in labeled data before spaCy training.

Input : JSONL, one example per line:
        {"text": "...", "entities": [[start, end, "CATEGORY"], ...]}
Output: same format, with every span snapped to its category's regex
        match. Unalignable spans are dropped and reported to stderr so
        you can review them (usually mislabels, or categories the rules
        don't cover yet). Overlapping snapped spans are resolved by
        keeping the first (input order), same as the pipeline's
        first-claim rule.

Usage : python tools/repair_ner_data.py in.jsonl out.jsonl
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from khmer_tn.align import snap_span, build_category_map  # noqa: E402


def repair_line(obj, cat_map):
    text = obj["text"]
    kept, dropped = [], []
    claimed = []

    def overlaps(s, e):
        return any(s < ce and cs < e for cs, ce in claimed)

    for s, e, cat in obj["entities"]:
        out = snap_span(text, s, e, cat, cat_map)
        if out is None:
            dropped.append((s, e, cat, text[s:e]))
            continue
        ns, ne, _ = out
        if overlaps(ns, ne):
            dropped.append((s, e, cat, text[s:e] + " (overlap after snap)"))
            continue
        claimed.append((ns, ne))
        kept.append([ns, ne, cat])
    return {"text": text, "entities": sorted(kept)}, dropped


def main(src, dst):
    cat_map = build_category_map()
    total = fixed = dropped_n = skipped = 0
    with open(src, encoding="utf-8") as f_in, open(dst, "w", encoding="utf-8") as f_out:
        for i, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"line {i}: SKIPPED invalid JSON ({e})", file=sys.stderr)
                skipped += 1
                continue
            if "text" not in obj or "entities" not in obj:
                print(
                    f"line {i}: SKIPPED missing 'text'/'entities' "
                    f"(got keys: {sorted(obj)}) -- is this file in spaCy format?",
                    file=sys.stderr,
                )
                skipped += 1
                continue
            before = {(s, e, c) for s, e, c in map(tuple, obj["entities"])}
            repaired, dropped = repair_line(obj, cat_map)
            after = {(s, e, c) for s, e, c in map(tuple, repaired["entities"])}
            total += len(before)
            fixed += len(after - before)
            dropped_n += len(dropped)
            for s, e, cat, frag in dropped:
                print(f"line {i}: DROPPED [{cat}] {s}:{e} {frag!r}", file=sys.stderr)
            f_out.write(json.dumps(repaired, ensure_ascii=False) + "\n")
    summary = f"spans: {total} total, {fixed} boundary-adjusted, {dropped_n} dropped"
    if skipped:
        summary += f", {skipped} lines skipped"
    print(summary, file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python tools/repair_ner_data.py in.jsonl out.jsonl")
    main(sys.argv[1], sys.argv[2])

#     python tools/spacy_export.py dataset.jsonl converted.jsonl
# python tools/repair_ner_data.py converted.jsonl out.jsonl