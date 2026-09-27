"""Build the v3 hybrid decision dataset (24 domains; synthetic + curated real) with quality gates.

    python scripts/v3/build_data.py --out data/decision/v3 [--scale 1.0] [--commercial-only] [--workers 8]

Layout (compatible with the existing store/suite code):
    <out>/train.jsonl, val.jsonl, calib.jsonl
    <out>/eval/<name>.jsonl      real held-out splits, proc_v3_test (IID phrasing),
                                 proc_v3_unseen (held-out phrasing frames), proc_v2_test
    <out>/build_report.json      counts, gates, stats; a failed gate exits non-zero

Every example carries `family` = its domain (the group used for worst-group risk control)
and `source` = its generator or dataset.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import random
import zlib
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "audit")]
os.chdir(ROOT)

from verqen.data.audit import content_key, ngrams  # noqa: E402
from verqen.data.v3.engine import Engine, verify  # noqa: E402
from verqen.data.v3.specs import SPECS  # noqa: E402

V2_FAMILY_DOMAIN = {
    "grid_nav": "games", "graph_route": "games", "tictactoe": "games", "nim": "games", "tetris": "games",
    "browser": "browser", "tool_routing": "tools", "refund_policy": "customer_support", "incident_triage": "devops",
    "phishing": "security", "scheduling": "hr", "arithmetic": "quantitative", "shopping": "quantitative",
    "comparison_shopping": "quantitative", "rates": "quantitative", "temporal": "quantitative", "counting": "quantitative",
    "logic": "reasoning", "spatial": "robotics", "entity_tracking": "reasoning", "code_exec": "software",
    "sentiment": "labeling", "intent": "labeling",
}
# examples per split for the whole synthetic v3 library at scale 1.0
SYN = {"train": 640_000, "val": 8_000, "calib": 12_000, "test": 12_000, "test_unseen": 12_000}
V2 = {"train": 200_000, "val": 3_000, "calib": 4_000, "test": 5_000}
FIX = {"train": 60_000}
REAL_CALIB_FRAC = 0.03


def write_jsonl(path, rows):
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def first_shard_prose(n=20000):
    import pyarrow.parquet as pq
    # both layouts: the local PC's data/pretrain/raw/... and the box's prepare_pretrain.py output
    # data/pretrain/v3/raw/fineweb_edu/... (T2, 25 Sep: only the first was searched, the box had only
    # the second, and all 1.5M decisions were built WITHOUT distractor prose, silently)
    paths = sorted(glob.glob("data/pretrain/raw/sample/10BT/*.parquet")) or         sorted(glob.glob("data/pretrain/v3/raw/fineweb_edu/sample/10BT/*.parquet"))
    out = []
    if not paths:
        sys.exit("[build_data] no FineWeb-Edu shard for distractor prose (looked in data/pretrain/raw/sample/10BT "
                 "and data/pretrain/v3/raw/fineweb_edu/sample/10BT): refusing to build prose-free data")
    for batch in pq.ParquetFile(paths[0]).iter_batches(batch_size=512, columns=["text"]):
        for t in batch.column(0).to_pylist():
            out += [s for s in re.split(r"(?<=[.!?])\s+", t) if 60 <= len(s) <= 200 and not re.search(r"\d{3,}", s)]
        if len(out) >= 2 * n:
            break
    random.Random(0).shuffle(out)
    return out[:n]


def synthetic(split: str, n_total: int, prose, seed: int) -> list[dict]:
    """Equal budget per domain, split evenly across that domain's specs; whole families kept together."""
    by_dom = collections.defaultdict(list)
    for s in SPECS:
        by_dom[s.domain].append(s)
    per_dom = n_total // len(by_dom)
    rows = []
    for d_i, (dom, specs) in enumerate(sorted(by_dom.items())):
        per_spec = per_dom // len(specs)
        for s_i, sp in enumerate(specs):
            eng = Engine(sp, split, prose=prose, seed=seed * 7919 + d_i * 131 + s_i)
            got = 0
            while got < per_spec:
                fam = eng.family()
                for ex in fam:
                    ex["family"] = dom
                    ex["source"] = f"v3:{dom}/{sp.subdomain}"
                rows += fam
                got += len(fam)
    return rows


def v2_procedural(split: str, n: int, seed: int) -> list[dict]:
    from verqen.data.procedural.engine import generate
    rows = generate(n, seed=seed, split={"calib": "calib", "val": "val", "test": "test"}.get(split, "train"))
    for r in rows:
        fam = r.get("family", "")
        r["source"] = f"v2:{fam}"
        r["family"] = V2_FAMILY_DOMAIN.get(fam, "reasoning")
    return rows


def fix_families(n: int, prose, seed: int) -> list[dict]:
    from verqen.data.procedural.discourse import discourse_families
    from verqen.data.procedural.rules import rule_families
    rng, out = random.Random(seed), []
    while len(out) < n:
        fam = rule_families(rng, 1) if rng.random() < 0.5 else discourse_families(rng, 1, prose=prose, mention_all=True)
        for ex in fam:
            ex["source"] = f"fix:{ex['family']}"
            ex["family"] = "safety" if ex["family"] == "discourse_role" else "rules_generic"
        out += fam
    return out


def eval_blocklist(eval_rows: list[dict]):
    """13-grams + option ids of everything that must never leak into training."""
    import probes as P
    texts, opts = [], set()
    bench = Path("verqen_benchmark/dataset/benchmark_cases.jsonl")
    if bench.exists():
        for line in open(bench, encoding="utf-8"):
            c = json.loads(line)
            texts.append(c["input"]["state"])
            for q in c["input"]["questions"]:
                texts.append(q["text"])
                opts |= set(q.get("options") or [])
            opts |= set(c.get("allowed_decisions") or [])
    prose = P.fineweb_sentences(3000)  # pinned shard, downloaded if absent: never silently skip the distractor probes
    for pr in P.distractor_probes(prose) + P.boundary_probes() + P.missing_probes() + P.missing_probes_v2():
        texts += [pr["state"], pr["question"]]
        opts |= set(pr["options"])
    for pr in P.counterfactual_probes():
        texts += [pr["a"][0], pr["b"][0]]
        opts |= set(pr["options"])
    opts |= {a[1] for a in P.ACTIONS}
    grams = set()
    for t in texts:
        grams |= ngrams(t, 13)
    for r in eval_rows:
        grams |= ngrams(r["state"][:2000], 13)
        for q in r["questions"]:
            grams |= ngrams(q["text"], 13)
    return grams, opts


def load_real(scale: float, commercial_only: bool, workers: int, log, evals_only: bool = False):
    from verqen.data.v3.real_catalog import catalog
    srcs = [s for s in catalog() if s.commercial or not commercial_only]

    def one(s):
        t0 = time.time()
        rng = random.Random(zlib.crc32(s.name.encode()))  # stable across runs (hash() is salted per process)
        ev_rng = random.Random(zlib.crc32(s.name.encode()) + 1)  # own stream: --evals-only picks the same eval rows
        tr = [r for r in (s.train() if s.train and not evals_only else []) if r] if s.train else []
        rng.shuffle(tr)
        cap = int(s.train_cap * scale) if s.train_cap else int(len(tr) * scale)
        tr = tr[:cap]
        ev = [r for r in (s.eval() if s.eval else []) if r] if s.eval else []
        ev_rng.shuffle(ev)
        ev = ev[: s.eval_cap or len(ev)]
        for r in tr + ev:
            r["family"], r["source"], r["license"] = s.domain, s.name, s.license
        return s, tr, ev, time.time() - t0

    out = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:  # parallel downloads: never let I/O serialize (LESSONS #8/#11)
        futs = {ex.submit(one, s): s.name for s in srcs}
        for f in as_completed(futs):
            s, tr, ev, dt_ = f.result()
            out[s.name] = (s, tr, ev)
            log(f"  real {s.name:16s} train={len(tr):7d} eval={len(ev):5d} ({dt_:.0f}s) [{s.license}]")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/decision/v3")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--commercial-only", action="store_true")
    ap.add_argument("--no-real", action="store_true", help="synthetic only (fast local check)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--evals-only", action="store_true",
                    help="write only the real held-out eval splits (minutes); the pretraining 13-gram blocklist "
                         "needs them BEFORE tokenization, while the full build is still running")
    ap.add_argument("--evals-ref", default="data/decision/v3_evals",
                    help="an --evals-only output; the full build checks its eval files are identical")
    a = ap.parse_args()
    out = Path(a.out)
    if a.evals_only:
        (out / "eval").mkdir(parents=True, exist_ok=True)
        for f in (out / "eval").glob("*.jsonl"):
            f.unlink()  # write_jsonl appends
        real = load_real(1.0, a.commercial_only, a.workers, print, evals_only=True)
        empty = [n for n, (s, _, ev) in real.items() if s.eval and not ev]
        if empty:
            sys.exit(f"GATE FAIL declared eval splits with zero rows: {empty}")
        for name, (s, _, ev) in real.items():
            if ev:
                write_jsonl(out / "eval" / f"{name}.jsonl", ev)
        (out / "EVALS_OK").write_text(json.dumps({n: len(ev) for n, (_, _, ev) in real.items()}))
        print(f"[evals-only] {sum(1 for *_, ev in real.values() if ev)} eval splits -> {out / 'eval'}", flush=True)
        return
    (out / "eval").mkdir(parents=True, exist_ok=True)
    for f in list(out.glob("*.jsonl")) + list((out / "eval").glob("*.jsonl")):
        f.unlink()
    report: dict = {"scale": a.scale, "commercial_only": a.commercial_only, "gates": {}, "counts": {}}
    log = lambda m: print(m, flush=True)
    t0 = time.time()

    # gate 0: every spec verifies (labels, visibility, determinism, family relations)
    bad = {}
    for sp in SPECS:
        r = verify(Engine(sp, "train", seed=3), n=100)
        if r["n_errors"]:
            bad[f"{sp.domain}/{sp.subdomain}"] = r["errors"][:3]
    report["gates"]["spec_verification"] = {"specs": len(SPECS), "failing": bad}
    if bad:
        log(f"GATE FAIL spec verification: {bad}")
        sys.exit(2)

    prose = first_shard_prose()
    S = lambda n: max(1, int(n * a.scale))
    splits: dict[str, list] = {k: [] for k in ("train", "val", "calib")}
    evals: dict[str, list] = {}
    for sp_name, n in SYN.items():
        rows = synthetic(sp_name, S(n), prose, seed={"train": 1, "val": 2, "calib": 3, "test": 4, "test_unseen": 5}[sp_name])
        (splits[sp_name] if sp_name in splits else evals.setdefault({"test": "proc_v3_test", "test_unseen": "proc_v3_unseen"}[sp_name], [])).extend(rows)
    for sp_name, n in V2.items():
        rows = v2_procedural(sp_name, S(n), seed=21)
        (splits[sp_name] if sp_name in splits else evals.setdefault("proc_v2_test", [])).extend(rows)
    splits["train"] += fix_families(S(FIX["train"]), prose, seed=31)
    log(f"synthetic built in {time.time() - t0:.0f}s")

    if not a.no_real:
        real = load_real(a.scale, a.commercial_only, a.workers, log)
        for name, (s, tr, ev) in real.items():
            k = int(len(tr) * REAL_CALIB_FRAC)
            splits["calib"] += tr[:k]
            splits["val"] += tr[k: k + max(1, k // 3)]
            splits["train"] += tr[k + max(1, k // 3):]
            if ev:
                evals[name] = ev
            # counted per DECLARED role: MMLU is eval-only and QASC train-only by design, but a
            # declared loader that yields nothing (a renamed split, a gated repo) must fail the build
            if s.train:
                report["counts"][f"real:{name}"] = len(tr)
            if s.eval:
                report["counts"][f"real_eval:{name}"] = len(ev)

    # decontamination. External evals (Artifact-G, audit probes, real held-out splits): no shared
    # 13-gram, no shared option id. Synthetic test splits share templates BY DESIGN (they are
    # separated by seed, and proc_v3_unseen by held-out phrasing frames), so against them only
    # exact duplicates are removed; 13-gram filtering them deleted whole domains.
    real_eval_rows = [r for name, rs in evals.items() if not name.startswith("proc_") for r in rs]
    grams, opts = eval_blocklist(real_eval_rows)
    synth_eval_keys = {content_key(r) for name, rs in evals.items() if name.startswith("proc_") for r in rs}
    kept, dropped = [], collections.Counter()
    seen = set()
    for r in splits["train"]:
        k = content_key(r)
        if k in seen:
            dropped["duplicate"] += 1
            continue
        if k in synth_eval_keys:
            dropped["equals_synthetic_eval_example"] += 1
            continue
        seen.add(k)
        g = ngrams(r["state"][:2000], 13)
        for q in r["questions"]:
            g |= ngrams(q["text"], 13)
        if g & grams:
            dropped["13gram_overlap"] += 1
            continue
        if r["source"].startswith(("v3:", "fix:")) and any(set(q["options"]) & opts for q in r["questions"] if q["type"] != "noul"):
            log(f"GATE FAIL option id overlaps eval vocabulary in {r['source']}: {sorted(set(r['questions'][0]['options']) & opts)}")
            sys.exit(2)
        kept.append(r)
    splits["train"] = kept
    report["gates"]["decontamination"] = dict(dropped)

    # gate: option mentions must not predict the answer. Measured PER OPTION (does a given option
    # appear in the text more often when it is correct than when it is wrong?) and averaged by
    # count, so option identity cannot masquerade as a mention effect (Simpson's paradox: an
    # option never named in any policy text is 0/0 either way, which is not a shortcut).
    def mentioned(opt, st):
        w = [x for x in re.split(r"[\s_\-]+", str(opt).lower()) if len(x) > 3]
        return bool(w) and all(x in st.lower() for x in w)
    per = collections.defaultdict(lambda: [0, 0, 0, 0])  # (domain, option) -> m|correct, n|correct, m|wrong, n|wrong
    for r in splits["train"]:
        if not r["source"].startswith(("v3:", "fix:")):
            continue
        q = r["questions"][0]
        best = q["target"].index(max(q["target"]))
        for i, o in enumerate(q["options"]):
            m = per[(r["family"], o)]
            j = 0 if i == best else 2
            m[j] += mentioned(o, r["state"]); m[j + 1] += 1
    dom = collections.defaultdict(lambda: [0.0, 0])
    for (d, _), (mc, nc, mw, nw) in per.items():
        if nc and nw:
            w = min(nc, nw)
            dom[d][0] += w * (mc / nc - mw / nw)
            dom[d][1] += w
    balance = {d: round(v / max(1, n), 4) for d, (v, n) in dom.items()}
    report["gates"]["mention_balance_per_option"] = balance
    worst = max(balance.values(), default=0.0)  # one-sided: mentions predicting CORRECTNESS is the shortcut
    if worst > 0.10:
        log(f"GATE FAIL mention predicts correctness by {worst:.3f} > 0.10: {balance}")
        sys.exit(2)

    # gate: noul answers balanced; no zero-row sources
    yes = [q["target"][1] for r in splits["train"] for q in r["questions"] if q["type"] == "noul"]
    report["gates"]["noul_yes_rate"] = round(sum(yes) / max(1, len(yes)), 4)
    counts = collections.Counter(r["source"].split(":")[0] if ":" in r["source"] else r["source"] for r in splits["train"])
    fam_counts = collections.Counter(r["family"] for r in splits["train"])
    report["counts"].update({"train_by_origin": dict(counts), "train_by_domain": dict(fam_counts)})
    zero = [n for n, c in report["counts"].items() if n.startswith(("real:", "real_eval:")) and c == 0]
    if zero:
        log(f"GATE FAIL sources with zero rows: {zero}")
        sys.exit(2)

    rng = random.Random(7)
    for name, rows in splits.items():
        rng.shuffle(rows)
        write_jsonl(out / f"{name}.jsonl", rows)
        report["counts"][name] = len(rows)
    for name, rows in evals.items():
        write_jsonl(out / "eval" / f"{name}.jsonl", rows)
        report["counts"][f"eval:{name}"] = len(rows)
    # the pretraining corpus was decontaminated against --evals-ref: it must be these exact rows
    ref = Path(a.evals_ref) / "eval"
    if ref.is_dir() and not a.no_real:
        real = [n for n in evals if not n.startswith("proc_")]
        # sources added after pretraining was tokenized are NOT in its blocklist: report, don't fake
        report["gates"]["evals_not_in_pretrain_blocklist"] = [n for n in real if not (ref / f"{n}.jsonl").exists()]
        diff = [n for n in real if (ref / f"{n}.jsonl").exists()
                and (ref / f"{n}.jsonl").read_bytes() != (out / "eval" / f"{n}.jsonl").read_bytes()]
        report["gates"]["evals_match_pretrain_blocklist"] = diff or "identical"
        if diff:
            log(f"GATE FAIL eval splits differ from the pretraining blocklist source {ref}: {diff}")
            sys.exit(2)
    report["seconds"] = round(time.time() - t0)
    json.dump(report, open(out / "build_report.json", "w"), indent=2)
    log(json.dumps({k: v for k, v in report.items() if k != "counts"}, indent=1))
    log(json.dumps({k: v for k, v in report["counts"].items() if not isinstance(v, dict)}, indent=1))
    log(json.dumps(report["counts"]["train_by_domain"], indent=1))


if __name__ == "__main__":
    main()
