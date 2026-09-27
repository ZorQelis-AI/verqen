"""Pre-flight on the rented box: NCCL, measured throughput / memory, and an auto-sized config.

    .venv/bin/python production/preflight.py --config configs/v3/t2.yaml

Writes <run>/preflight.json and configs/v3/t2.box.yaml (the base config with micro batch sizes
chosen from MEASURED peak memory, never estimates; LESSONS #13, #26, #27, and the
"allocator thrash is silent" note). The run uses t2.box.yaml.

Throughput = median of >= 5 timed steps after 2 untimed warmup steps, spread printed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch  # noqa: E402
import yaml  # noqa: E402

from verqen.config import VerqenConfig  # noqa: E402

HEADROOM_GB = 28.0  # of 31.4 GB usable per 5090: fragmentation and eval spikes need the rest


def nccl_check(py: str) -> dict:
    code = ("import os,time,torch,torch.distributed as d\n"
            "d.init_process_group('nccl'); r=d.get_rank(); torch.cuda.set_device(r)\n"
            "x=torch.ones(64*1024*1024,device='cuda'); d.all_reduce(x); torch.cuda.synchronize()\n"
            "t=time.time()\nfor _ in range(10): d.all_reduce(x)\ntorch.cuda.synchronize(); dt=(time.time()-t)/10\n"
            "assert abs(x[0].item()-2**11)<1e-3\n"
            "print(f'NCCL_OK {x.numel()*4/dt/1e9:.1f} GB/s') if r==0 else None\nd.destroy_process_group()\n")
    Path("/tmp/nccl_check.py").write_text(code)
    torchrun = str(Path(py).parent / "torchrun")
    out = subprocess.run([torchrun, "--standalone", "--nproc_per_node", "2", "/tmp/nccl_check.py"],
                         capture_output=True, text=True, timeout=300)
    line = next((l for l in out.stdout.splitlines() if l.startswith("NCCL_OK")), None)
    return {"ok": line is not None, "detail": line or (out.stderr[-800:])}


def probe_pretrain(cfg: VerqenConfig, seq: int, micro: int, compile_: bool, steps: int = 5) -> dict:
    from verqen.model.attention_mask import SEG_BOS, SEG_STATE
    from verqen.training.common import autocast, build_optimizer, maybe_compile
    from verqen.training.pretrain_v3 import build_core
    dev = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    cfg.model.vocab_size = 32768
    core = build_core(cfg).to(dev)
    if compile_:
        for mod in (getattr(core, "gen", None), getattr(core, "model", None)):
            if mod is not None:
                mod.transformer = maybe_compile(mod.transformer, True)
        if hasattr(core, "disc"):
            core.disc = maybe_compile(core.disc, True)
    opt = build_optimizer(core, 1e-4, 0.1)
    ids = torch.randint(20, 32000, (micro, seq), device=dev)
    seg = torch.full((micro, seq), SEG_STATE, device=dev)
    seg[:, 0] = SEG_BOS
    pos = torch.arange(seq, device=dev)[None].expand(micro, seq).contiguous()
    gen = torch.Generator(device=dev).manual_seed(0)
    sp = {"mask": 13, "bos": 1}
    times = []
    try:
        for i in range(2 + steps):
            torch.cuda.synchronize()
            t = time.time()
            with autocast(dev):
                loss, _ = core(ids, seg, pos, sp, cfg.pretrain.mask_prob, 32768, 16, gen)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if i >= 2:
                times.append(time.time() - t)
    except torch.OutOfMemoryError:
        return {"micro": micro, "seq": seq, "compile": compile_, "oom": True}
    finally:
        del core, opt
    peak = torch.cuda.max_memory_allocated(dev) / 2**30
    med = statistics.median(times)
    return {"micro": micro, "seq": seq, "compile": compile_, "oom": False, "peak_gb": round(peak, 2),
            "step_s_median": round(med, 4), "step_s_spread": round(max(times) - min(times), 4),
            "tok_per_s_per_gpu": int(micro * seq / med)}


def _decision_store(cfg: VerqenConfig, n_families: int = 400):
    """Real packed v3 decision rows at data.max_seq_len (the decision data is not built yet at
    preflight time, so a few hundred engine families stand in; rows are packed full either way)."""
    import random
    import tempfile
    from verqen.data.store import load_or_build
    from verqen.data.v3.engine import Engine
    from verqen.data.v3.specs import SPECS
    rng = random.Random(0)
    path = os.path.join(tempfile.mkdtemp(prefix="preflight_"), "probe.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n_families):
            for ex in Engine(SPECS[i % len(SPECS)], "train", seed=rng.randrange(10**6)).family():
                f.write(json.dumps(ex) + "\n")
    return load_or_build(path, cfg.data.tokenizer_path, cfg.data.max_seq_len)


def probe_decisions(cfg: VerqenConfig, store, micro: int, compile_: bool, steps: int = 5) -> dict:
    """The decisions stage at data.max_seq_len: full model, decision head, DecisionCore, loss, AdamW."""
    from verqen.losses.decision import decision_loss
    from verqen.model.verqen import VerqenModel
    from verqen.training.common import autocast, maybe_compile
    from verqen.training.decisions import token_budget_batches
    from verqen.training.decisions_v3 import DecisionCore
    from verqen.data.collator import collate_rows, pack_rows
    dev = torch.device("cuda:0")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(dev)
    L = cfg.data.max_seq_len
    cfg.model.vocab_size = 32768
    model = VerqenModel(cfg.model).to(dev)
    if compile_:
        model.transformer = maybe_compile(model.transformer, True)
    core = DecisionCore(model)
    opt = torch.optim.AdamW(core.parameters(), lr=1e-4, fused=True)
    it = iter(token_budget_batches(store, micro, L, 0, epochs=None))
    times = []
    try:
        for i in range(2 + steps):
            packed, _ = pack_rows([store[j] for j in next(it)], L, micro)
            b = {k: (v.to(dev) if isinstance(v, torch.Tensor) else v) for k, v in collate_rows(packed, L, 0, micro).items()}
            torch.cuda.synchronize()
            t = time.time()
            with autocast(dev):
                out = core(b)
            decision_loss(out, b["target"], b["qtype"], cfg.loss)["loss"].backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if i >= 2:
                times.append(time.time() - t)
    except torch.OutOfMemoryError:
        return {"micro": micro, "seq": L, "compile": compile_, "oom": True}
    finally:
        del core, opt, model
    peak = torch.cuda.max_memory_allocated(dev) / 2**30
    med = statistics.median(times)
    return {"micro": micro, "seq": L, "compile": compile_, "oom": False, "peak_gb": round(peak, 2),
            "step_s_median": round(med, 4), "tok_per_s_per_gpu": int(micro * L / med)}


def _non_embedding_params(cfg: VerqenConfig, dims: dict) -> int:
    from verqen.model.verqen import VerqenModel
    for k, v in dims.items():
        setattr(cfg.model, k, v)
    cfg.model.vocab_size = 32768
    with torch.device("meta"):
        m = VerqenModel(cfg.model)
    return sum(p.numel() for n, p in m.named_parameters() if "embed" not in n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/v3/t2.yaml")
    ap.add_argument("--skip-nccl", action="store_true")
    a = ap.parse_args()
    cfg = VerqenConfig.from_yaml(a.config)
    raw = yaml.safe_load(open(a.config))
    out = {"gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
           "usable_gb": [round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2) for i in range(torch.cuda.device_count())]}
    if not a.skip_nccl:
        out["nccl"] = nccl_check(sys.executable)
        print("[preflight] NCCL:", out["nccl"], flush=True)
        if not out["nccl"]["ok"]:
            raise SystemExit("[preflight] NCCL all-reduce failed; do not start a multi-GPU run")
    torch.backends.cuda.matmul.allow_tf32 = True
    headroom = min(HEADROOM_GB, 0.89 * min(out["usable_gb"]))  # 28 of 31.4 GB on a 5090
    out["headroom_gb"] = round(headroom, 2)
    # make "does not fit" a clean OOM: without a cap the Windows driver spills into system RAM
    # (hung the local dry run at 100% GPU for 20+ min and starved other jobs of RAM)
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, headroom / total), i)
    choices = {}
    world = 2
    hours = raw.get("stages", {}).get("pretrain", 13.0)

    # ── model size: measured, then chosen by a rule fixed before the run ──────────────────
    # For each candidate: largest fitting micro (compiled; eager if compile fails), measured
    # tokens/s, projected pretraining tokens D in the stage budget, non-embedding params N.
    # size_rule "auto" picks the candidate closest (in log) to D = 20 N, the compute-optimal
    # ratio; naming a candidate forces it. VRAM decides what fits; only compute decides D.
    cands_cfg = raw.get("size_candidates") or [{"name": "default", **{k: raw["model"][k] for k in ("hidden_size", "num_heads", "ffn_size", "num_layers")}}]
    sized = []
    for c in cands_cfg:
        dims = {k: c[k] for k in ("hidden_size", "num_heads", "ffn_size", "num_layers")}
        best = None
        for compile_ in (True, False):
            for micro in [48, 32, 24, 16, 12, 8]:
                ccfg = VerqenConfig.from_yaml(a.config)
                for k, v in dims.items():
                    setattr(ccfg.model, k, v)
                try:
                    r = probe_pretrain(ccfg, ccfg.pretrain.seq_len, micro, compile_)
                except Exception as e:
                    r = {"micro": micro, "compile": compile_, "error": f"{type(e).__name__}: {str(e)[:200]}"}
                print(f"[preflight] size {c['name']} pretrain {r}", flush=True)
                if r.get("oom") or "error" in r or r["peak_gb"] > headroom:
                    continue
                best = r
                break
            if best is not None:
                break
        if best is None:
            print(f"[preflight] size {c['name']}: nothing fits, dropped", flush=True)
            continue
        n = _non_embedding_params(VerqenConfig.from_yaml(a.config), dims)
        d = best["tok_per_s_per_gpu"] * world * hours * 3600
        sized.append({"name": c["name"], "dims": dims, "lr": c.get("learning_rate"), "probe": best,
                      "non_embedding_params_M": round(n / 1e6, 1), "projected_tokens_B": round(d / 1e9, 2),
                      "tokens_per_param": round(d / n, 1), "distance_from_20": round(abs(math.log(d / (20 * n))), 3)})
    if not sized:
        raise SystemExit("[preflight] no model size fits")
    rule = raw.get("size_rule", "auto")
    forced = [x for x in sized if x["name"] == rule]
    pick = forced[0] if forced else min(sized, key=lambda x: x["distance_from_20"])
    out["size_selection"] = {"rule": rule if forced else "auto: closest to 20 tokens/param", "candidates": sized, "chosen": pick["name"]}
    print(f"[preflight] size candidates {json.dumps(sized)}", flush=True)
    print(f"[preflight] chosen size {pick['name']}", flush=True)
    raw["model"].update(pick["dims"])
    if pick["lr"]:
        raw.setdefault("training", {})["learning_rate"] = pick["lr"]
    choices["pretrain"] = pick["probe"]
    # the corpus must cover the projected tokens (<= 2.5 epochs is cheap; more wastes compute)
    planned = raw.get("pretrain", {}).get("planned_corpus_tokens", 3.95e9)
    raw.setdefault("pretrain", {})["data_scale"] = round(min(2.5, max(1.0, pick["projected_tokens_B"] * 1e9 / planned)), 2)
    tmp_cfg = Path(a.config).with_suffix(".sized.tmp.yaml")
    tmp_cfg.write_text(yaml.safe_dump(raw, sort_keys=False))
    a.config = str(tmp_cfg)  # every later probe uses the chosen size
    cfg = VerqenConfig.from_yaml(a.config)

    for stage, seq, cands in (("pretrain_ext", cfg.pretrain.ext_seq_len, [12, 8, 6, 4, 2]),):
        best = None
        for compile_ in (True, False):
            for micro in cands:
                try:
                    r = probe_pretrain(VerqenConfig.from_yaml(a.config), seq, micro, compile_)
                except Exception as e:  # compile failures fall back to eager, never block the run
                    r = {"micro": micro, "seq": seq, "compile": compile_, "error": f"{type(e).__name__}: {str(e)[:200]}"}
                print(f"[preflight] {stage} {r}", flush=True)
                out.setdefault(stage, []).append(r)
                if r.get("oom") or "error" in r or r["peak_gb"] > headroom:
                    continue
                if best is None or r["tok_per_s_per_gpu"] > best["tok_per_s_per_gpu"]:
                    best = r
                break  # largest fitting micro for this compile setting found
        if best is None:
            raise SystemExit(f"[preflight] nothing fits for {stage}")
        choices[stage] = best
    # decisions (and RLCD, which is smaller) run at data.max_seq_len: measure them, never reuse the
    # pretraining micro size (at 1024 tokens a 512-token micro needs ~1.6x the activation memory)
    store = _decision_store(VerqenConfig.from_yaml(a.config))
    comp = bool(choices["pretrain"]["compile"])
    for micro in [32, 24, 16, 12, 8, 6, 4]:
        try:
            r = probe_decisions(VerqenConfig.from_yaml(a.config), store, micro, comp)
        except Exception as e:
            r = {"micro": micro, "compile": comp, "error": f"{type(e).__name__}: {str(e)[:200]}"}
        print(f"[preflight] decisions {r}", flush=True)
        out.setdefault("decisions", []).append(r)
        if not (r.get("oom") or "error" in r or r["peak_gb"] > headroom):
            choices["decisions"] = r
            break
    if "decisions" not in choices:
        raise SystemExit("[preflight] nothing fits for the decisions stage")
    raw.setdefault("training", {})["micro_rows"] = choices["pretrain"]["micro"]
    raw["training"]["compile"] = bool(choices["pretrain"]["compile"])
    raw["training"]["decision_micro_rows"] = choices["decisions"]["micro"]
    tmp_cfg.unlink(missing_ok=True)
    raw.setdefault("pretrain", {})["ext_micro_rows"] = choices["pretrain_ext"]["micro"]
    per_step = choices["pretrain"]["micro"] * world  # a whole number of accumulation steps, >= the planned batch
    raw["training"]["batch_rows"] = per_step * math.ceil(raw["training"].get("batch_rows", 128) / per_step)
    tok_s = choices["pretrain"]["tok_per_s_per_gpu"] * world
    out["choices"] = choices
    out["projected_pretrain_tokens_B"] = round(tok_s * hours * 3600 / 1e9, 2)
    Path("configs/v3/t2.box.yaml").write_text(yaml.safe_dump(raw, sort_keys=False))
    run = Path(raw["logging"]["output_dir"])
    run.mkdir(parents=True, exist_ok=True)
    (run / "preflight.json").write_text(json.dumps(out, indent=2))
    print(f"[preflight] chosen {json.dumps(choices)}\n[preflight] projected pretraining tokens: "
          f"{out['projected_pretrain_tokens_B']}B in {hours} h -> configs/v3/t2.box.yaml", flush=True)


if __name__ == "__main__":
    main()
