"""T2 pretraining corpus: download (pull, parallel) -> digit-split tokenizer -> decontaminated uint16 token files.

    python scripts/v3/prepare_pretrain.py download  --out data/pretrain/v3 --plan full
    python scripts/v3/prepare_pretrain.py tokenizer --out data/pretrain/v3 --vocab 32768
    python scripts/v3/prepare_pretrain.py tokenize  --out data/pretrain/v3 --workers 32
    python scripts/v3/prepare_pretrain.py synthetic --out data/pretrain/v3 --tokens 400000000

Mix (docs/production/T2_PLAN.md P6): FineWeb-Edu 60%, Cosmopedia-v2 15%, FineMath-4+ 8%,
Wikipedia-en 8%, structured program state 5%, decision text 4%. Weights live in the run config
(`pretrain.sources`); this script only produces one token file per source plus a manifest.

Decontamination: any document containing a 13-gram from an evaluation text (Artifact-G,
audit probes, real held-out splits in data/decision/v3/eval) is dropped, and the drop count is
recorded per source.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shutil
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "audit")]
os.chdir(ROOT)

# name -> (repo, file glob, text column, target tokens at plan "full")
PLAN = {
    "fineweb_edu": ("HuggingFaceFW/fineweb-edu", "sample/10BT/*.parquet", "text", 2_600_000_000),
    "cosmopedia": ("HuggingFaceTB/smollm-corpus", "cosmopedia-v2/*.parquet", "text", 650_000_000),
    "finemath": ("HuggingFaceTB/finemath", "finemath-4plus/*.parquet", "text", 350_000_000),
    "wikipedia": ("wikimedia/wikipedia", "20231101.en/*.parquet", "text", 350_000_000),
}
BYTES_PER_TOKEN = 4.2  # rough: used only to decide how many parquet files to fetch (x1.3 margin)


def _files(repo, pattern):
    from huggingface_hub import HfApi
    import fnmatch
    files = [f for f in HfApi().list_repo_files(repo, repo_type="dataset") if fnmatch.fnmatch(f, pattern)]
    return sorted(files)


def cmd_download(a):
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    plan = {k: v for k, v in PLAN.items() if not a.only or k in a.only.split(",")}
    jobs = []
    for name, (repo, pattern, _, target) in plan.items():
        target = int(target * (0.02 if a.plan == "smoke" else a.scale))
        info = api.dataset_info(repo, files_metadata=True)
        sizes = {s.rfilename: s.size or 0 for s in info.siblings}
        need, have = target * BYTES_PER_TOKEN * 1.3 * 0.45, 0  # parquet compresses text ~2.2x
        for f in _files(repo, pattern):
            if have >= need:
                break
            jobs.append((name, repo, f))
            have += sizes.get(f, 0)
        print(f"[download] {name}: {sum(1 for j in jobs if j[0] == name)} files (~{have / 1e9:.1f} GB)", flush=True)

    def fetch(job):
        name, repo, f = job
        t0 = time.time()
        p = hf_hub_download(repo, f, repo_type="dataset", local_dir=os.path.join(a.out, "raw", name))
        return name, p, os.path.getsize(p), time.time() - t0

    with ThreadPoolExecutor(max_workers=a.workers) as ex:  # parallel: never let I/O serialize (LESSONS #8/#11)
        for name, p, sz, dt_ in ex.map(fetch, jobs):
            print(f"[download] {name} {os.path.basename(p)} {sz / 1e6:.0f} MB in {dt_:.0f}s", flush=True)


def _texts(a, name, limit_docs=None):
    import pyarrow.parquet as pq
    col = PLAN[name][2] if name in PLAN else "text"
    n = 0
    for p in sorted(glob.glob(os.path.join(a.out, "raw", name, "**", "*.parquet"), recursive=True)):
        for batch in pq.ParquetFile(p).iter_batches(batch_size=1024, columns=[col]):
            for t in batch.column(0).to_pylist():
                yield t
                n += 1
                if limit_docs and n >= limit_docs:
                    return


def cmd_tokenizer(a):
    """Digit-split byte-level BPE trained on a balanced sample of the actual mix (+ decision text)."""
    from verqen.tokenizer import VerqenTokenizer

    def sample():
        for name, docs in (("fineweb_edu", 400_000), ("cosmopedia", 120_000), ("finemath", 80_000), ("wikipedia", 80_000)):
            yield from _texts(a, name, max(1000, int(docs * a.sample_scale)))
        syn = os.path.join(a.out, "synthetic_text.jsonl")
        if os.path.exists(syn):
            for i, line in enumerate(open(syn, encoding="utf-8")):
                if i >= 100_000:
                    break
                yield json.loads(line)["text"]

    t0 = time.time()
    tok = VerqenTokenizer.train(sample(), vocab_size=a.vocab, split_digits=True)
    path = os.path.join(a.out, f"tokenizer_v3_{a.vocab // 1024}k.json")
    tok.save(path)
    ids = tok.encode(" The invoice total is $1,249.50 due 2026-03-14.")
    print(f"[tokenizer] vocab={tok.vocab_size} -> {path} ({time.time() - t0:.0f}s) sample={[tok.decode([i]) for i in ids]}",
          flush=True)


# ── decontamination blocklist (13-grams of every eval text) ──────────────────────────────
def _blocklist(a) -> set:
    from verqen.data.audit import ngrams
    texts = []
    bench = Path("verqen_benchmark/dataset/benchmark_cases.jsonl")
    if bench.exists():
        for line in open(bench, encoding="utf-8"):
            c = json.loads(line)
            texts += [c["input"]["state"]] + [q["text"] for q in c["input"]["questions"]]
    for f in glob.glob(os.path.join(a.decision_dir, "eval", "*.jsonl")):
        if os.path.basename(f).startswith("proc_"):
            continue
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            texts += [r["state"][:3000]] + [q["text"] for q in r["questions"]]
    n_real = len([f for f in glob.glob(os.path.join(a.decision_dir, "eval", "*.jsonl")) if not os.path.basename(f).startswith("proc_")])
    if n_real < 20:  # silently decontaminating against Artifact-G alone is how a leak ships
        raise SystemExit(f"[tokenize] only {n_real} real eval splits in {a.decision_dir}/eval; "
                         "run scripts/v3/build_data.py --evals-only --out data/decision/v3_evals first")
    grams = set()
    for t in texts:
        grams |= ngrams(t, 13)
    return grams


_G: dict = {}


def _init_worker(tok_path, grams):
    from verqen.tokenizer import VerqenTokenizer
    _G["tok"] = VerqenTokenizer.from_pretrained(tok_path)
    _G["grams"] = grams


def _work(texts):
    from verqen.data.audit import ngrams
    tok, grams = _G["tok"], _G["grams"]
    keep = [t for t in texts if t and not (ngrams(t, 13) & grams)]
    bos = tok.special_token_ids["bos"]
    enc = tok.encode_batch(keep)
    arr = np.fromiter((x for ids in enc for x in ([bos] + ids)), dtype=np.uint16)
    return arr, len(texts) - len(keep)


def _bounded_map(ex, fn, items, window: int):
    """Ordered map with at most `window` chunks in flight. Executor.map submits the WHOLE input
    up front, i.e. reads the entire corpus into RAM (MemoryError locally at 86M tokens)."""
    import collections
    q = collections.deque()
    for it in items:
        q.append(ex.submit(fn, it))
        if len(q) >= window:
            yield q.popleft().result()
    while q:
        yield q.popleft().result()


def cmd_tokenize(a):
    tok_path = a.tokenizer or sorted(glob.glob(os.path.join(a.out, "tokenizer_v3_*.json")))[-1]
    grams = _blocklist(a)
    print(f"[tokenize] tokenizer={tok_path} blocklist={len(grams):,} 13-grams", flush=True)
    manifest = {}
    names = [n for n in list(PLAN) + ["synthetic"] if not a.only or n in a.only.split(",")]
    for name in names:
        target = int(PLAN[name][3] * (0.02 if a.plan == "smoke" else a.scale)) if name in PLAN else 10**12
        out = os.path.join(a.out, f"{name}.bin")
        if os.path.exists(out) and not a.force:
            n = os.path.getsize(out) // 2
            manifest[name] = {"path": out, "tokens": n, "skipped": True}
            print(f"[tokenize] {name}: exists ({n / 1e6:.0f}M tokens)", flush=True)
            continue
        src = (_texts(a, name) if name in PLAN else
               (json.loads(l)["text"] for l in open(os.path.join(a.out, "synthetic_text.jsonl"), encoding="utf-8")))
        total, dropped, t0 = 0, 0, time.time()

        def chunks():
            buf = []
            for t in src:
                buf.append(t)
                if len(buf) == 512:
                    yield buf
                    buf = []
            if buf:
                yield buf

        # write-then-rename: a crash mid-source must not leave a partial .bin that the resume above
        # would then accept as finished
        with open(out + ".tmp", "wb") as f, ProcessPoolExecutor(a.workers, initializer=_init_worker, initargs=(tok_path, grams)) as ex:
            for arr, d in _bounded_map(ex, _work, chunks(), window=4 * a.workers):
                f.write(arr.tobytes())
                total += arr.size
                dropped += d
                if total >= target:
                    break
        os.replace(out + ".tmp", out)
        manifest[name] = {"path": out, "tokens": int(total), "docs_dropped_decontam": dropped,
                          "seconds": round(time.time() - t0)}
        print(f"[tokenize] {name}: {total / 1e6:,.0f}M tokens, {dropped} docs dropped (13-gram), {time.time() - t0:.0f}s",
              flush=True)
    mpath = os.path.join(a.out, "manifest.json")
    old = json.load(open(mpath)) if os.path.exists(mpath) else {}
    old.update(manifest)
    json.dump(old, open(mpath, "w"), indent=2)


def _synthetic_shard(args):
    """One deterministic shard of the synthetic text (seeded by shard index)."""
    k, tokens, path = args
    from verqen.data.procedural.engine import generate
    from verqen.data.v3.engine import Engine
    from verqen.data.v3.specs import SPECS
    rng = random.Random(99 + 7919 * k)
    engines = [Engine(s, "train", seed=1000 + i + 100_003 * k) for i, s in enumerate(SPECS)]
    n_tok, n = 0, 0
    with open(path, "w", encoding="utf-8") as f:
        while n_tok < tokens:
            docs = rng.choice(engines).family() if rng.random() < 0.55 else generate(8, seed=rng.randrange(10**9), split="train")
            for d in docs:
                q = d["questions"][0]
                ans = q["options"][q["target"].index(max(q["target"]))]
                style = rng.randrange(3)
                if style == 0:  # decision text
                    text = f"{d['state']}\n\nQuestion: {q['text']}\nOptions: {', '.join(q['options'])}\nDecision: {ans}"
                elif style == 1:  # structured state as JSON
                    text = json.dumps({"state": d["state"], "question": q["text"], "options": q["options"], "decision": ans},
                                      ensure_ascii=False, indent=rng.choice([None, 2]))
                else:  # key: value log style
                    text = "\n".join(f"{k_}: {v}" for k_, v in (("case", d["state"]), ("query", q["text"]),
                                                                  ("choices", " | ".join(q["options"])), ("result", ans)))
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                n_tok += len(text) // 4
                n += 1
    return n, n_tok


def cmd_synthetic(a):
    """Structured program state + decision text as plain documents (5% + 4% of the mix).

    Sharded over a.workers processes (single-threaded it ran ~150K tok/s: 400M tokens = 45 min of
    box time); shard k is seeded by k and shards are concatenated in order, so the output is
    deterministic for a given worker count."""
    out = os.path.join(a.out, "synthetic_text.jsonl")
    shards = max(1, a.workers)
    paths = [f"{out}.part{k:03d}" for k in range(shards)]
    t0 = time.time()
    with ProcessPoolExecutor(shards) as ex:
        res = list(ex.map(_synthetic_shard, [(k, a.tokens // shards + 1, p) for k, p in enumerate(paths)]))
    with open(out + ".tmp", "wb") as f:
        for p in paths:
            with open(p, "rb") as g:
                shutil.copyfileobj(g, f)
            os.remove(p)
    os.replace(out + ".tmp", out)
    n, n_tok = sum(r[0] for r in res), sum(r[1] for r in res)
    print(f"[synthetic] {n:,} docs, ~{n_tok / 1e6:.0f}M tokens, {shards} shards, {time.time() - t0:.0f}s -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["download", "tokenizer", "tokenize", "synthetic"])
    ap.add_argument("--out", default="data/pretrain/v3")
    ap.add_argument("--plan", default="full", choices=["full", "smoke"])
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply every PLAN target (preflight writes pretrain.data_scale from the measured throughput)")
    ap.add_argument("--only", default="")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--tokens", type=int, default=400_000_000)
    ap.add_argument("--decision-dir", dest="decision_dir", default="data/decision/v3")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sample-scale", dest="sample_scale", type=float, default=1.0,
                    help="tokenizer: fraction of the default training sample (RAM-limited machines)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"download": cmd_download, "tokenizer": cmd_tokenizer, "tokenize": cmd_tokenize, "synthetic": cmd_synthetic}[a.cmd](a)


if __name__ == "__main__":
    main()
