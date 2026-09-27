"""Long-state robustness: bury training decisions in unrelated prose, at 2048-token context.

    python scripts/v3/augment_long.py --src data/decision/v3 --out data/decision/long_noise --n 30000

Why (26 Sep 2026, T2 release on Jev-parity long_state, 41.0%): the case is buried in ~45
sentences (~1,300-1,800 tokens) of unrelated prose.
  - filler FIRST 51.4%: at max_seq_len 1024 the packer keeps the head + the last 64 tokens, so the
    policy itself was cut off;
  - filler LAST 24.2%: the whole case fits, yet trailing noise flips the decision. The training
    data only ever had 1-6 distractor sentences ("irrelevant" variant).
The top-up therefore runs at 2048 tokens (pretrain_ext trained that length) and adds these rows.

Leakage discipline: Jev-parity's filler comes from the pinned probe shard 002_00000; this noise
comes from shard 000_00000 via build_data.first_shard_prose, the pool training distractors were
already drawn from. Only procedural rows (v3:/v2:/fix:, whose labels are decided by the policy
text) are padded; a real-data row's label can depend on its whole text, so those are left alone.
Labels are unchanged: the added prose is irrelevant by construction.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "v3"))
os.chdir(ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/decision/v3")
    ap.add_argument("--out", default="data/decision/long_noise")
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--val", type=int, default=1500)
    ap.add_argument("--min-sent", type=int, default=15)
    ap.add_argument("--max-sent", type=int, default=50)
    ap.add_argument("--max-chars", type=int, default=7200, help="stay inside 2048 tokens with the options")
    a = ap.parse_args()
    from build_data import first_shard_prose
    prose = first_shard_prose(20000)
    if len(prose) < 5000:
        sys.exit(f"[long] only {len(prose)} prose sentences: shard 000_00000 missing?")
    rng = random.Random(4242)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"prose_pool": len(prose)}
    for split, src, n in (("train", "train.jsonl", a.n), ("val", "val.jsonl", a.val)):
        cand = []
        with open(os.path.join(a.src, src), encoding="utf-8") as f:
            for line in f:
                if '"source": "v3:' in line or '"source": "v2:' in line or '"source": "fix:' in line:
                    cand.append(line)
        rng.shuffle(cand)
        rows, pos_count = [], {"before": 0, "after": 0, "both": 0}
        for line in cand:
            if len(rows) >= n:
                break
            r = json.loads(line)
            if r.get("family") == "agentic" or len(r["state"]) > 2500:
                continue
            k = rng.randint(a.min_sent, a.max_sent)
            filler = rng.sample(prose, k)
            pos = rng.choices(("before", "after", "both"), weights=(0.4, 0.4, 0.2))[0]
            if pos == "before":
                state = " ".join(filler) + "\n\n" + r["state"]
            elif pos == "after":
                state = r["state"] + "\n\n" + " ".join(filler)
            else:
                h = k // 2
                state = " ".join(filler[:h]) + "\n\n" + r["state"] + "\n\n" + " ".join(filler[h:])
            while len(state) > a.max_chars and k > a.min_sent:  # trim filler, never the case
                k -= 5
                filler = filler[:k]
                state = (" ".join(filler) + "\n\n" + r["state"]) if pos == "before" else (
                    (r["state"] + "\n\n" + " ".join(filler)) if pos == "after" else
                    (" ".join(filler[:k // 2]) + "\n\n" + r["state"] + "\n\n" + " ".join(filler[k // 2:])))
            r = dict(r, state=state, source=r["source"] + "+long", variant="long_noise",
                     meta=dict(r.get("meta") or {}, long_noise={"sentences": k, "position": pos}))
            rows.append(json.dumps(r, ensure_ascii=False) + "\n")
            pos_count[pos] += 1
        tmp = out / f"{split}.jsonl.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(rows)
        os.replace(tmp, out / f"{split}.jsonl")
        rep[split] = {"rows": len(rows), "positions": pos_count}
    json.dump(rep, open(out / "report.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
