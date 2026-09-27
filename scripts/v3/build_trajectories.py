"""Multi-step (agentic) decision data from the procedural environments, mixed with replay.

    python scripts/v3/build_trajectories.py --out data/decision/agentic_mix --replay data/decision/v3 \
        --episodes 6000 --eps 0.25 --replay-frac 0.5 \
        --extra-train data/decision/info_para/train.jsonl --extra-val data/decision/info_para/val_A.jsonl

Why (docs/production/NEXT_RUN.md item 1): before RLCD, T2 trailed T1 on the multi-step
environments (grid_nav 0.105 vs 0.47), because only ~13% of its 1.5M decisions were
environment-like. RLCD already imitates the oracle on the states its own rollouts visit; this
adds OFFLINE coverage: perfect-play episodes with deliberate detours, so the model also sees the
off-path states it must recover from (DAgger-style state coverage).

Each step is emitted exactly as `EnvRunner._pack` presents it at play time: the environment's
`observe()` state/question/options, one `choice` question, target uniform over the oracle-optimal
actions. Observations are Markov (the page, board or map carries everything the decision
needs), so no history text is added: training and play inputs are identical.

Leakage: evaluation episodes use seeds 999..1038 (`evaluate_envs`); train episodes use seeds
from 1,000,000 up and validation from 500,000 up. Replay rows come from the main decision train
split (never its val/eval), so single-step skills are rehearsed while the new data is learned
(50% replay by default: training only on environments makes a model forget).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from verqen.environments.procedural import ENVIRONMENTS  # noqa: E402

TRAIN_SEED0, VAL_SEED0 = 1_000_000, 500_000
# Tetris is ~500x slower to roll out (oracle searches placements: 0.1 s/episode vs 0.0002) and
# yields 15 rows/episode, so it hits the --max-share cap with a quarter of the episodes anyway
EPISODE_SCALE = {"tetris": 0.25}


def episode(name: str, seed: int, eps: float, rng: random.Random) -> list[dict]:
    env = ENVIRONMENTS[name]()
    env.reset(random.Random(seed))
    rows, done, t = [], False, 0
    while not done and t < 60:
        o = env.observe()
        n = len(o["options"])
        if n < 2:
            break
        opt = env.oracle()
        if not opt:  # no labelled optimum from this state (terminal / unreachable): stop the episode
            break
        rows.append({"id": f"agentic_{name}_{seed}_{t}", "source": f"agentic:{name}", "family": "agentic",
                     "variant": "trajectory", "state": o["state"],
                     "questions": [{"type": "choice", "text": o["question"], "options": list(o["options"]),
                                    "target": [1.0 / len(opt) if k in opt else 0.0 for k in range(n)]}],
                     "meta": {"env": name, "step": t, "detour": False}})
        wrong = [k for k in range(n) if k not in opt]
        if wrong and rng.random() < eps:  # deliberate detour: the next states need recovery
            a = rng.choice(wrong)
            rows[-1]["meta"]["detour_taken_after"] = True
        else:
            a = rng.choice(opt)
        _, done = env.step(a)
        t += 1
    return rows


def build(split: str, episodes: int, eps: float, seed0: int) -> list[dict]:
    rng = random.Random(seed0 + 7)
    out, seen = [], set()
    for name in sorted(ENVIRONMENTS):
        for k in range(max(1, int(episodes * EPISODE_SCALE.get(name, 1.0)))):
            for r in episode(name, seed0 + k, eps, rng):
                q = r["questions"][0]
                key = (r["state"], tuple(q["options"]))
                if key in seen:  # identical states recur (openings, empty boards): keep one
                    continue
                seen.add(key)
                out.append(r)
    return out


def cap_share(rows: list[dict], max_share: float) -> list[dict]:
    """Iteratively cap each environment at max_share of the total (seeded subsample)."""
    rng = random.Random(3)
    by = collections.defaultdict(list)
    for r in rows:
        by[r["meta"]["env"]].append(r)
    for _ in range(10):
        total = sum(len(v) for v in by.values())
        cap = int(max_share * total)
        over = {k: v for k, v in by.items() if len(v) > cap}
        if not over:
            break
        for k, v in over.items():
            rng.shuffle(v)
            by[k] = v[:cap]
    return [r for v in by.values() for r in v]


def reservoir(path: str, k: int, seed: int) -> list[str]:
    rng, buf = random.Random(seed), []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if len(buf) < k:
                buf.append(line)
            else:
                j = rng.randint(0, i)
                if j < k:
                    buf[j] = line
    return buf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/decision/agentic_mix")
    ap.add_argument("--replay", default="data/decision/v3", help="main decision data dir (train/val used for replay)")
    ap.add_argument("--episodes", type=int, default=6000, help="train episodes per environment")
    ap.add_argument("--val-episodes", type=int, default=300)
    ap.add_argument("--eps", type=float, default=0.25, help="probability of a deliberate wrong action per step")
    ap.add_argument("--replay-frac", type=float, default=0.5)
    ap.add_argument("--max-share", type=float, default=0.3,
                    help="cap any one environment's share of trajectory rows (Tetris episodes are long: 49% uncapped)")
    ap.add_argument("--extra-train", nargs="*", default=[], help="extra rows appended to train (e.g. info_para/train.jsonl)")
    ap.add_argument("--extra-val", nargs="*", default=[], help="extra rows appended to val (e.g. info_para/val_A.jsonl)")
    ap.add_argument("--gap-sources", default="",
                    help="comma list of real TRAIN-split sources to rehearse extra (where a competitor leads)")
    ap.add_argument("--gap-per-source", type=int, default=7000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tr = cap_share(build("train", a.episodes, a.eps, TRAIN_SEED0), a.max_share)
    va = build("val", a.val_episodes, 0.0, VAL_SEED0)
    by = collections.Counter(r["meta"]["env"] for r in tr)
    n_rep = int(len(tr) * a.replay_frac / (1 - a.replay_frac))
    rep_tr = reservoir(os.path.join(a.replay, "train.jsonl"), n_rep, 11)
    rep_va = reservoir(os.path.join(a.replay, "val.jsonl"), len(va), 12)
    rng = random.Random(5)
    lines_tr = [json.dumps(r, ensure_ascii=False) + "\n" for r in tr] + rep_tr
    lines_va = [json.dumps(r, ensure_ascii=False) + "\n" for r in va] + rep_va
    extra = {}
    for key, paths, lines in (("train", a.extra_train, lines_tr), ("val", a.extra_val, lines_va)):
        for path in paths:
            add = open(path, encoding="utf-8").readlines()
            lines += add
            extra.setdefault(key, []).append({"path": path, "rows": len(add)})
    gap = [s for s in a.gap_sources.split(",") if s]
    if gap:  # per-source reservoir over the main TRAIN split only (never its val/eval)
        grng, bufs, seen = random.Random(21), {s: [] for s in gap}, {s: 0 for s in gap}
        with open(os.path.join(a.replay, "train.jsonl"), encoding="utf-8") as f:
            for line in f:
                i = line.find('"source": "')
                if i < 0:
                    continue
                src = line[i + 11:line.find('"', i + 11)]
                if src in bufs:
                    seen[src] += 1
                    if len(bufs[src]) < a.gap_per_source:
                        bufs[src].append(line)
                    else:
                        j = grng.randrange(seen[src])
                        if j < a.gap_per_source:
                            bufs[src][j] = line
        for s in gap:
            lines_tr += bufs[s]
        extra["gap"] = {s: {"available": seen[s], "added": len(bufs[s])} for s in gap}
    rng.shuffle(lines_tr)
    rng.shuffle(lines_va)
    for name, lines in (("train.jsonl", lines_tr), ("val.jsonl", lines_va)):
        with open(out / (name + ".tmp"), "w", encoding="utf-8") as f:
            f.writelines(lines)
        os.replace(out / (name + ".tmp"), out / name)
    rep = {"trajectory_train": len(tr), "trajectory_val": len(va), "replay_train": len(rep_tr), "replay_val": len(rep_va),
           "by_env": dict(by), "extra": extra, "replay_frac": a.replay_frac, "eps": a.eps, "episodes_per_env": a.episodes,
           "detour_steps": sum(1 for r in tr if r["meta"].get("detour_taken_after")),
           "mean_oracle_actions": round(sum(sum(1 for x in r["questions"][0]["target"] if x > 0) for r in tr) / max(1, len(tr)), 3)}
    json.dump(rep, open(out / "build_report.json", "w"), indent=2)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
