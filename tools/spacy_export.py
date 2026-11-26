"""
Convert flat per-entity NER records into the spaCy-style {"text", "entities"}
format expected by repair_ner_data.py.

Input : JSONL, one entity per line:
        {"label": "CATEGORY", "span": "...", "sentence": "...",
         "start": int, "end": int, "normalized": "..."}
        (start/end are offsets into "sentence"; sentence[start:end] == span)

Output: JSONL, one example per line:
        {"text": "...", "entities": [[start, end, "CATEGORY"], ...]}
        Records that share the exact same "sentence" text are merged into
        one example with multiple entities - repair_ner_data.py's overlap
        check only looks within a single example's entity list, so entities
        belonging to the same sentence need to already be grouped together
        before it runs.

Usage : python tools/spacy_export.py dataset.jsonl converted.jsonl
"""

import json
import sys
from collections import OrderedDict


def convert(src, dst):
    by_text = OrderedDict()
    skipped = 0

    with open(src, encoding="utf-8") as f_in:
        for i, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"line {i}: SKIPPED invalid JSON ({e})", file=sys.stderr)
                skipped += 1
                continue
            try:
                text = rec["sentence"]
                entity = [rec["start"], rec["end"], rec["label"]]
            except KeyError as e:
                print(f"line {i}: SKIPPED missing field {e}", file=sys.stderr)
                skipped += 1
                continue
            entry = by_text.setdefault(text, {"text": text, "entities": []})
            entry["entities"].append(entity)

    with open(dst, "w", encoding="utf-8") as f_out:
        for entry in by_text.values():
            entry["entities"].sort()
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(
        f"wrote {len(by_text)} examples ({skipped} lines skipped) to {dst}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: python tools/spacy_export.py dataset.jsonl converted.jsonl")
    convert(sys.argv[1], sys.argv[2])