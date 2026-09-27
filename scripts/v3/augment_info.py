"""Paraphrase augmentation for the conflicting-fact and unknown-fact decisions.

    python scripts/v3/augment_info.py --src data/decision/v3 --out data/decision/info_para \
        --conflict 30000 --unknown 20000

Why (25 Sep 2026, T2 release): on Jev-parity, which renders these cases with the engine's HELD-OUT
frame 0 ("Two sources disagree on ...", "... is not stated anywhere in the file."), T2 scored
conflicting 2.4% and unknown-phrased missing info 1.4%, against 100% on the training-frame val rows
and 97.8% when the fact is simply absent. It learned three sentences, not the concept "one fact, two
values" / "the fact is unknown". Real inputs phrase both in endless ways.

What it does: takes training rows of variant info_conflict / info_unknown (frames 1-3 only; training
never contains frame 0), finds the one conflict/unknown sentence by the engine's own templates, and
rewrites it with a template from a NEW pool. The label is unchanged: the case is the same case, only
the sentence reporting the fact is worded differently.

Leakage discipline: pool A (training) contains neither engine frame 0 nor any wording built from it
("two sources", "disagree", "not stated anywhere"). Pool B is held out from training altogether:
val_B.jsonl measures transfer to unseen paraphrases without consulting Jev-parity. The Jev-parity
number after this fix is still reported, with the note that the fix was chosen after seeing it.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

# the engine's TRAINING frames (verqen/data/v3/engine.py CONFLICT[1:], MISSING[1:]); frame 0 is held out
SRC_CONFLICT = [
    re.compile(r"^The (?P<phrase>.+?) is (?P<a>.+?) according to the form, but (?P<b>.+?) according to the system\.$"),
    re.compile(r"^One record gives the (?P<phrase>.+?) as (?P<a>.+?); a second record gives (?P<b>.+?)\.$"),
    re.compile(r"^The (?P<phrase>.+?) was reported as (?P<a>.+?) and later as (?P<b>.+?), and nobody has resolved which is right\.$"),
]
SRC_UNKNOWN = [
    re.compile(r"^The (?P<phrase>.+?) has not been recorded\.$"),
    re.compile(r"^Nobody has confirmed the (?P<phrase>.+?) yet\.$"),
    re.compile(r"^The (?P<phrase>.+?) field was left blank\.$"),
]

CONFLICT_A = [
    "Records conflict on the {phrase}: {a} in one, {b} in the other.",
    "The {phrase} is listed as {a} on the intake sheet and as {b} in the database.",
    "The {phrase}: {a} per the first report, {b} per the second.",
    "One document states the {phrase} is {a}, while another states it is {b}.",
    "There is a discrepancy in the {phrase} ({a} versus {b}).",
    "The {phrase} reads {a} in the ticket but {b} in the audit trail.",
    "Conflicting values were recorded for the {phrase}: {a} and {b}.",
    "The {phrase} was entered twice, once as {a} and once as {b}.",
    "Our notes show the {phrase} as {a}, but the customer's copy shows {b}.",
    "The {phrase} is either {a} or {b}; the reports do not match.",
    "The {phrase} does not match across systems: {a} vs. {b}.",
    "Report A lists the {phrase} as {a}; report B lists it as {b}.",
    "The {phrase} was logged as {a}, yet a later entry says {b}.",
    "It is unclear whether the {phrase} is {a} or {b}; the records contradict each other.",
    "The phone agent recorded the {phrase} as {a}, while the online form says {b}.",
    "Depending on which file you trust, the {phrase} is {a} or {b}.",
    "The {phrase} has two inconsistent entries: {a} and {b}.",
    "The system of record says the {phrase} is {a}; the paper copy says {b}.",
]
CONFLICT_B = [  # held out from training: transfer check only
    "Mismatch on the {phrase}: {a} (source 1) and {b} (source 2).",
    "The {phrase} appears as {a} in one place and {b} in another.",
    "According to one entry the {phrase} is {a}, but another entry puts it at {b}.",
    "The {phrase} is contested: {a} against {b}.",
]
UNKNOWN_A = [
    "The {phrase} is unknown.",
    "No value was provided for the {phrase}.",
    "The {phrase} is missing from the record.",
    "We do not know the {phrase}.",
    "The {phrase} could not be determined.",
    "Information about the {phrase} is unavailable.",
    "The {phrase} was never captured.",
    "The {phrase} is still pending confirmation.",
    "The {phrase}: not provided.",
    "There is no information on the {phrase}.",
    "The {phrase} remains unverified.",
    "The {phrase} is marked as unknown in the file.",
    "Nobody has been able to find out the {phrase}.",
    "The form has no entry for the {phrase}.",
]
UNKNOWN_B = [
    "The {phrase} has yet to be established.",
    "At this point nobody knows the {phrase}.",
    "The {phrase} is not available.",
    "The record gives no {phrase}.",
]
FORBIDDEN = ("two sources", "disagree", "not stated anywhere")  # engine frame 0 wording
for _t in CONFLICT_A + UNKNOWN_A:
    assert not any(f in _t.lower() for f in FORBIDDEN), _t


def rewrite(row: dict, pool_c: list[str], pool_u: list[str], rng: random.Random) -> dict | None:
    kind = row.get("variant")
    src, pool = (SRC_CONFLICT, pool_c) if kind == "info_conflict" else (SRC_UNKNOWN, pool_u)
    lines = row["state"].split("\n")
    for i, line in enumerate(lines):
        for rx in src:
            m = rx.match(line.strip())
            if m:
                g = m.groupdict()
                new = rng.choice(pool).format(**g)
                out = dict(row)
                out["state"] = "\n".join(lines[:i] + [new] + lines[i + 1:])
                out["source"] = row["source"] + "+para"
                out["meta"] = dict(row.get("meta") or {}, paraphrased=True)
                return out
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/decision/v3")
    ap.add_argument("--out", default="data/decision/info_para")
    ap.add_argument("--conflict", type=int, default=30000)
    ap.add_argument("--unknown", type=int, default=20000)
    ap.add_argument("--val", type=int, default=1500, help="per kind, from the val split")
    a = ap.parse_args()
    rng = random.Random(1234)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for split, pools, caps in (("train", (CONFLICT_A, UNKNOWN_A), {"info_conflict": a.conflict, "info_unknown": a.unknown}),
                               ("val_A", (CONFLICT_A, UNKNOWN_A), {"info_conflict": a.val, "info_unknown": a.val}),
                               ("val_B", (CONFLICT_B, UNKNOWN_B), {"info_conflict": a.val, "info_unknown": a.val})):
        srcf = os.path.join(a.src, "train.jsonl" if split == "train" else "val.jsonl")
        cand = {"info_conflict": [], "info_unknown": []}
        with open(srcf, encoding="utf-8") as f:
            for line in f:
                if '"info_conflict"' not in line and '"info_unknown"' not in line:
                    continue
                r = json.loads(line)
                if r.get("variant") in cand:
                    cand[r["variant"]].append(r)
        rows, miss = [], {}
        for kind, pool in cand.items():
            rng.shuffle(pool)
            made, failed = 0, 0
            for r in pool:
                if made >= caps[kind]:
                    break
                n = rewrite(r, *pools, rng)
                if n is None:
                    failed += 1
                    continue
                rows.append(n)
                made += 1
            miss[kind] = {"available": len(pool), "made": made, "unmatched": failed}
        rng.shuffle(rows)
        tmp = out / f"{split}.jsonl.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        os.replace(tmp, out / f"{split}.jsonl")
        report[split] = miss
    json.dump(report, open(out / "report.json", "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
