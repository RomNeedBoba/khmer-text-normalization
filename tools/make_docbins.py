"""Split a repaired JSONL dataset into train/dev and write .spacy DocBins.

Usage:
    python make_docbins.py out.jsonl --train train.spacy --dev dev.spacy --split 0.9
"""

import argparse
import json
import random

import spacy
from spacy.tokens import DocBin


def load_rows(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def make_docbin(rows, nlp):
    db = DocBin()
    skipped = 0
    for row in rows:
        doc = nlp.make_doc(row["text"])
        ents = []
        for start, end, label in row["entities"]:
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                skipped += 1
                continue
            ents.append(span)
        doc.ents = ents
        db.add(doc)
    return db, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--train", default="train.spacy")
    ap.add_argument("--dev", default="dev.spacy")
    ap.add_argument("--split", type=float, default=0.9)
    ap.add_argument("--lang", default="km")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = list(load_rows(args.src))
    random.Random(args.seed).shuffle(rows)
    cut = int(len(rows) * args.split)
    train_rows, dev_rows = rows[:cut], rows[cut:]

    nlp = spacy.blank(args.lang)

    train_db, train_skipped = make_docbin(train_rows, nlp)
    dev_db, dev_skipped = make_docbin(dev_rows, nlp)

    train_db.to_disk(args.train)
    dev_db.to_disk(args.dev)

    print(f"train: {len(train_rows)} examples -> {args.train} ({train_skipped} spans skipped)")
    print(f"dev:   {len(dev_rows)} examples -> {args.dev} ({dev_skipped} spans skipped)")


if __name__ == "__main__":
    main()