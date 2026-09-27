"""Calibrate, evaluate, export and (optionally) upload a trained T2 checkpoint.

    python production/finalize.py --config configs/v3/t2.box.yaml --checkpoint runs/verqen-t2/rlcd/final \
        --out runs/verqen-t2/release --upload

1. Calibration on data/decision/v3/calib.jsonl (never trained on):
   * marginal: temperatures, Platt p_correct, conformal, risk thresholds (T1 behaviour);
   * per-domain worst-group risk control (Fix C). Groups are the 24 domains, so a deployment
     can pass `risk_domain=<its domain>` and get a certified threshold for that domain.
2. Evaluation, each tool in its own process (one process per GPU stage; T1 OOM lesson):
   held-out suite (real benchmarks + proc_v3_test + proc_v3_unseen + environments + speed),
   audit probes, Artifact-G through its unchanged harness plus the corrected analysis.
3. `<out>/model_card.json` with every headline number and where it came from; FINAL_OK marker.
4. --upload: push <out> and the run's metrics/logs to the private HF repo $VERQEN_HF_REPO.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from verqen.config import VerqenConfig  # noqa: E402


def calibrate_into(cfg: VerqenConfig, ckpt: str, out: str) -> dict:
    from verqen.calibration.fit import calibrate, collect_outputs, worst_group_risk_control
    from verqen.data.store import load_or_build
    from verqen.utils.checkpoint import load_model, save_checkpoint
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(ckpt, dev)
    store = load_or_build(os.path.join(cfg.data.decision_dir, "calib.jsonl"), cfg.data.tokenizer_path, cfg.data.max_seq_len)
    outs = collect_outputs(model, store, dev, cfg.data.max_seq_len)
    cal = calibrate(outs)
    groups = np.array([store.meta(int(i))["family"] for i in outs["example"]])
    cal["risk_control_worst_group"] = worst_group_risk_control(outs, groups, cal, min_group=150)
    cal["source_checkpoint"] = ckpt
    save_checkpoint(model, out, cfg.data.tokenizer_path, calibration=cal)
    return cal


def sh(cmd: list[str], log: str) -> int:
    with open(log, "w") as f:
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode


STAGE_CHECKPOINTS = ("pretrain", "pretrain_ext", "decisions", "rlcd", "agentic_sft", "agentic_rlcd", "format_sft")  # each stage's exported `final` model


def upload_stages(cfg: VerqenConfig) -> list[str]:
    """Push every finished stage's final model (weights + config + tokenizer; no optimizer state) to
    <repo>/checkpoints/<stage>/, once each (markers in <run>/uploads/). Called by the watchdog every
    tick, so each stage leaves the box as soon as it finishes (LESSONS checklist #9)."""
    from huggingface_hub import HfApi
    run_dir = Path(cfg.logging.output_dir)
    todo = [st for st in STAGE_CHECKPOINTS
            if (run_dir / st / "final" / "model.safetensors").exists() and not (run_dir / "uploads" / f"{st}.done").exists()]
    if not todo:
        return []
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    try:
        user = api.whoami()["name"]
    except Exception as e:
        print(f"[upload] no usable Hugging Face token ({type(e).__name__}): {todo} wait for `hf auth login`", flush=True)
        return []
    repo = os.environ.get("VERQEN_HF_REPO") or f"{user}/verqen-t2-results"
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    done = []
    for st in todo:
        api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(run_dir / st / "final"), path_in_repo=f"checkpoints/{st}",
                          ignore_patterns=["optimizer*", "*.pt"])  # weights only; optimizer state stays on the box
        (run_dir / "uploads").mkdir(exist_ok=True)
        (run_dir / "uploads" / f"{st}.done").write_text(repo)
        done.append(st)
        print(f"[upload] {st} -> {repo}/checkpoints/{st}", flush=True)
    return done


def upload(out: Path, cfg: VerqenConfig) -> bool:
    """Push <out> and the run's metrics to a private HF model repo; True once UPLOADED is written.

    The token may be added AFTER launch (`hf auth login` on the box writes ~/.cache/huggingface/token),
    and a detached run_all.sh cannot see env vars exported later: HfApi() with no token reads that file.
    """
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    try:
        user = api.whoami()["name"]
    except Exception as e:
        print(f"[finalize] no usable Hugging Face token ({type(e).__name__}): upload skipped; "
              "run `hf auth login` on the box (the watchdog retries every 5 min)", flush=True)
        return False
    repo = os.environ.get("VERQEN_HF_REPO") or f"{user}/verqen-t2-results"
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(out), path_in_repo=f"release/{out.name}")
    run_dir = Path(cfg.logging.output_dir)
    for f in list(run_dir.glob("*/metrics.jsonl")) + [run_dir / "progress.json", run_dir / "preflight.json",
                                                      run_dir / "release_choice.json", run_dir / "agentic_choice.json"]:
        if f.exists():
            api.upload_file(path_or_fileobj=str(f), path_in_repo=f"run/{f.relative_to(run_dir)}", repo_id=repo, repo_type="model")
    (out / "UPLOADED").write_text(repo)
    print(f"[finalize] uploaded to {repo}", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", default=None, help="required unless --upload-only")
    ap.add_argument("--out", default=None, help="release directory (required except with --upload-stages)")
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--calibrate", action="store_true", help="(always on; kept for CLI compatibility)")
    ap.add_argument("--upload-only", action="store_true", help="only push an already finalized <out> (watchdog retries)")
    ap.add_argument("--upload-stages", action="store_true", help="only push finished stage checkpoints (watchdog, every tick)")
    a = ap.parse_args()
    cfg = VerqenConfig.from_yaml(a.config)
    if a.upload_stages:
        upload_stages(cfg)
        sys.exit(0)
    if not a.out:
        ap.error("--out is required")
    out = Path(a.out)
    if a.upload_only:
        sys.exit(0 if upload(out, cfg) else 1)
    if not a.checkpoint:
        ap.error("--checkpoint is required")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    card: dict = {"checkpoint": a.checkpoint, "config": a.config, "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    cal = calibrate_into(cfg, a.checkpoint, str(out))
    card["calibration"] = {"report": cal["calibration_report"], "marginal_risk": cal["risk_control"]["by_risk"],
                           "worst_group": {r: {"threshold": v["threshold"], "coverage": v["coverage"],
                                               "certified_domains": sorted(g for g, t in v["per_group_threshold"].items() if t <= 1.0)}
                                           for r, v in cal["risk_control_worst_group"]["by_risk"].items()}}
    torch.cuda.empty_cache()
    py = sys.executable
    logs = out / "logs"
    logs.mkdir(exist_ok=True)
    tag = out.name
    # held-out suite in its own process
    suite_code = ("import sys; sys.path.insert(0,'.'); import torch\n"
                  "from verqen.config import VerqenConfig\nfrom verqen.evaluation.suite import run_suite\n"
                  f"cfg=VerqenConfig.from_yaml({a.config!r})\n"
                  f"run_suite({str(out)!r}, cfg, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), {str(out / 'eval')!r})\n")
    rc = {"suite": sh([py, "-c", suite_code], str(logs / "suite.log"))}
    rc["probes"] = sh([py, "scripts/audit/run_probes.py", "--checkpoint", str(out), "--tag", f"probes_{tag}"], str(logs / "probes.log"))
    rc["cf_missing"] = sh([py, "scripts/audit/rerun_cf_missing.py", str(out), f"probes_{tag}"], str(logs / "cf_missing.log"))
    if Path("verqen_benchmark/dataset/benchmark_cases.jsonl").exists():
        rc["artifact_g"] = sh([py, "scripts/audit/repro_benchmark.py", "--checkpoint", str(out), "--tag", f"bench_{tag}"],
                              str(logs / "artifact_g.log"))
        rc["artifact_g_corrected"] = sh([py, "scripts/audit/corrected_eval.py", "--tag", f"bench_{tag}"], str(logs / "corrected.log"))
    if Path("evals/jev_parity/run.py").exists():
        rc["jev_parity"] = sh([py, "evals/jev_parity/run.py", "--checkpoint", str(out), "--out", str(out / "jev_parity.json")],
                              str(logs / "jev_parity.log"))
    card["eval_return_codes"] = rc
    rep = out / "eval" / "report.json"
    if rep.exists():
        r = json.load(open(rep))
        card["suite"] = {"real_mean": r["summary"]["mean_accuracy_real_benchmarks"],
                         "splits": {k: v["accuracy"] for k, v in r["splits"].items()},
                         "invariance": r.get("invariance"), "environments": r.get("environments"), "speed": r.get("speed")}
    for name, p in (("probes", f"runs/audit/probes_{tag}/probes.json"), ("artifact_g", f"runs/audit/bench_{tag}/corrected.json")):
        if Path(p).exists():
            card[name + "_path"] = p
    card["seconds"] = round(time.time() - t0)
    (out / "model_card.json").write_text(json.dumps(card, indent=2, default=str))
    for d in ("probes", "bench"):
        src = Path(f"runs/audit/{d}_{tag}")
        if src.exists():
            shutil.copytree(src, out / "audit" / src.name, dirs_exist_ok=True)
    if all(v == 0 for v in rc.values()):
        (out / "FINAL_OK").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    else:
        print(f"[finalize] some evaluations failed: {rc}", flush=True)
        (out / "FINAL_OK").write_text(f"with failures: {rc}")
    if a.upload:
        upload(out, cfg)
    print(f"[finalize] done in {(time.time() - t0) / 60:.1f} min: {json.dumps(rc)}", flush=True)


if __name__ == "__main__":
    main()
