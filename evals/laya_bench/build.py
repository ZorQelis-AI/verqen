"""Laya's own public benchmarks as cases for the head-to-head harness (evals/jev_parity/run.py).

    python evals/laya_bench/build.py --out evals/laya_bench/cases.jsonl --per-dataset 2000

The seven public datasets Laya reports on (laya.convaiinnovations.com; HF model cards), test or
validation splits only, each capped at --per-dataset with a fixed seed. Every case is one
`choice` question; both models receive the identical state, question and option list:

  dataset        split       options  Laya's published number (its own setup)
  ag_news        test          4      0.950
  emotion        test          6      0.595
  banking77      test         77      0.425 (Jev 0.870)
  boolq          validation    2      reported, value not restated here
  sst5           test          5      held-out ordinal task
  xnli_en        test          3      0.860
  massive_en     test         60      0.783

Training exposure, so results are read correctly: VerQen-T2's decision data includes the TRAIN
splits of ag_news, boolq, emotion, banking77, sst5 and massive_en (the last four added 25 Sep 2026)
plus MNLI/SNLI/WANLI (XNLI's English train split is MNLI). Test splits are never trained on, and
train rows that exactly match a case here were removed. Laya's training data is not published. The question wording is ours, and Laya's
exact prompts are not public, so its published numbers will not reproduce exactly.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

from datasets import load_dataset

SPECS = {
    # name: (hf id, config, split, text fields, label field, question)
    "ag_news": ("fancyzhx/ag_news", None, "test", ("text",), "label", "What is the topic of this news article?"),
    "emotion": ("dair-ai/emotion", "split", "test", ("text",), "label", "Which emotion does the author express?"),
    "banking77": ("mteb/banking77", None, "test", ("text",), "label", "What does the customer want to do?"),  # parquet copy of PolyAI/banking77 test
    "boolq": ("google/boolq", None, "validation", ("passage", "question"), "answer", None),
    "sst5": ("SetFit/sst5", None, "test", ("text",), "label", "What is the sentiment of this review?"),
    "xnli_en": ("facebook/xnli", "en", "test", ("premise", "hypothesis"), "label", "How does the hypothesis relate to the premise?"),
    "massive_en": ("mteb/amazon_massive_intent", "en", "test", ("text",), "label", "What is the user's intent?"),
}
SST5 = ["very negative", "negative", "neutral", "positive", "very positive"]
XNLI = ["entailment", "neutral", "contradiction"]


def label_names(ds, field):
    feat = ds.features[field]
    return [n.replace("_", " ") for n in feat.names] if hasattr(feat, "names") else None


def build(name, per, seed):
    hf, cfg, split, fields, lab, question = SPECS[name]
    ds = load_dataset(hf, cfg, split=split)
    names = label_names(ds, lab)
    if name == "sst5":
        names = SST5
    if name == "xnli_en":
        names = XNLI
    if names is None and isinstance(ds[0][lab], str):  # string intents (MASSIVE): the label IS the name
        names = sorted({v.replace("_", " ") for v in ds[lab]})
    elif names is None and "label_text" in ds.column_names:  # mteb copies: int label + label_text
        m = {int(l): t.replace("_", " ") for l, t in zip(ds[lab], ds["label_text"])}
        names = [m[k] for k in sorted(m)]
        assert names == [m[i] for i in range(len(names))], f"{name}: labels are not 0..K-1"
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    out = []
    for i in idx[:per]:
        r = ds[i]
        if name == "boolq":
            state, q, opts, ans = r["passage"], r["question"].rstrip("?") + "?", ["yes", "no"], "yes" if r["answer"] else "no"
        elif name == "xnli_en":
            state = f"Premise: {r['premise']}\nHypothesis: {r['hypothesis']}"
            q, opts, ans = question, names, names[r[lab]]
        else:
            state, q, opts = r[fields[0]], question, names
            ans = r[lab].replace("_", " ") if isinstance(r[lab], str) else names[r[lab]]
        out.append({"id": f"{name}/{i}", "capability": name, "domain": "laya_bench", "state": state, "question": q,
                    "options": list(opts), "answer": ans, "subdomain": name})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evals/laya_bench/cases.jsonl")
    ap.add_argument("--per-dataset", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    rows = []
    for name in SPECS:
        got = build(name, a.per_dataset, a.seed)
        assert got and all(r["answer"] in r["options"] for r in got), name
        rows += got
        print(f"{name:12s} {len(got):5d} cases, {len(got[0]['options'])} options", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:  # LF on every OS, so SHA256SUMS matches
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} cases -> {a.out}", dict(collections.Counter(r["capability"] for r in rows)))


if __name__ == "__main__":
    main()
