"""Build a deduplicated, template-stratified train/validation/test split.

Input : dataset.jsonl from github.com/RomNeedBoba/dataset (one record per
        entity: label, span, sentence, start, end, normalized).
Output: train.jsonl / val.jsonl / test.jsonl, one sentence per line:
        {"text", "entities": [[start, end, label]], "gold": <fully verbalized sentence>}

Template = the sentence with its NSW span replaced by <LABEL>. All sentences
that share a template go to the same partition, so no template is seen in
both training and test. The split is done per label (stratified), 80/20 by
template group, and 10% of the training groups are held out as validation.

Usage: python experiments/make_split.py data/dataset.jsonl data/split --seed 42
"""
import argparse, collections, json, os, random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("outdir")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test", type=float, default=0.2)
    ap.add_argument("--val", type=float, default=0.1)
    a = ap.parse_args()

    rows = {}
    n_records = 0
    for line in open(a.src, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line); n_records += 1
        s = r["sentence"]
        assert s[r["start"]:r["end"]] == r["span"], r
        ex = rows.setdefault(s, {"text": s, "entities": [], "gold": None})
        ent = [r["start"], r["end"], r["label"]]
        if ent not in ex["entities"]:
            ex["entities"].append(ent)
            ex["gold"] = s[:r["start"]] + r["normalized"] + s[r["end"]:]
    assert all(len(ex["entities"]) == 1 for ex in rows.values()), "expected one NSW per sentence"

    groups = collections.defaultdict(list)          # (label, template) -> sentences
    for ex in rows.values():
        s, e, lab = ex["entities"][0]
        groups[(lab, ex["text"][:s] + f"<{lab}>" + ex["text"][e:])].append(ex)

    rng = random.Random(a.seed)
    split = {"train": [], "val": [], "test": []}
    by_label = collections.defaultdict(list)
    for key in sorted(groups):
        by_label[key[0]].append(key)
    for lab in sorted(by_label):
        keys = by_label[lab]; rng.shuffle(keys)
        total = sum(len(groups[k]) for k in keys)
        n_test = 0; train_keys = []
        for k in keys:                                   # fill test to ~20% of this label
            if n_test < a.test * total:
                split["test"] += groups[k]; n_test += len(groups[k])
            else:
                train_keys.append(k)
        n_train = sum(len(groups[k]) for k in train_keys); n_val = 0
        for k in train_keys:                             # ~10% of training groups -> validation
            if n_val < a.val * n_train:
                split["val"] += groups[k]; n_val += len(groups[k])
            else:
                split["train"] += groups[k]

    os.makedirs(a.outdir, exist_ok=True)
    for name, exs in split.items():
        with open(os.path.join(a.outdir, f"{name}.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for ex in exs:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    stats = {
        "records": n_records, "unique_sentences": len(rows), "templates": len(groups),
        **{f"{k}_sentences": len(v) for k, v in split.items()},
        **{f"{k}_templates": len({(x['entities'][0][2], x['text'][:x['entities'][0][0]] + '<>' + x['text'][x['entities'][0][1]:]) for x in v}) for k, v in split.items()},
    }
    json.dump(stats, open(os.path.join(a.outdir, "split_stats.json"), "w"), indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
