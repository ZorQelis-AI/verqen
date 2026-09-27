"""T2 stage entry point (one process per rank under torchrun; plain python for one GPU / CPU).

    torchrun --standalone --nproc_per_node 2 scripts/v3/run_stage.py --config configs/v3/t2.yaml --stage pretrain
    torchrun --standalone --nproc_per_node 2 scripts/v3/run_stage.py --config configs/v3/t2.yaml --stage pretrain_ext
    torchrun --standalone --nproc_per_node 2 scripts/v3/run_stage.py --config configs/v3/t2.yaml --stage decisions
    python scripts/v3/run_stage.py --config configs/v3/t2.yaml --stage rlcd

Every stage is idempotent: it skips if its `final` marker exists, and resumes from its own
`resume/` directory after a crash. Stage outputs live under <run>/<stage>/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import torch  # noqa: E402

from verqen.config import VerqenConfig  # noqa: E402
from verqen.training import distributed as D  # noqa: E402

MARKERS = {"pretrain": "pretrain/final", "pretrain_ext": "pretrain_ext/final", "decisions": "decisions/final",
           "rlcd": "rlcd/final", "agentic_sft": "agentic_sft/final", "agentic_rlcd": "agentic_rlcd/final",
           "format_sft": "format_sft/final"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=list(MARKERS))
    ap.add_argument("--run", default=None)
    ap.add_argument("--hours", type=float, default=None, help="override the stage's wall-clock budget")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--init", default=None, help="agentic_sft: checkpoint to start from (default: the released choice)")
    ap.add_argument("--decision-dir", default=None, help="agentic_*: trajectory+replay mix (scripts/v3/build_trajectories.py)")
    ap.add_argument("--lr", type=float, default=5e-5, help="agentic_sft learning rate (decisions used 1e-4)")
    a = ap.parse_args()
    cfg = VerqenConfig.from_yaml(a.config)
    run = a.run or cfg.logging.output_dir
    info = D.init(prefer_cuda=not a.cpu)
    torch.backends.cuda.matmul.allow_tf32 = True
    marker = os.path.join(run, MARKERS[a.stage])
    with D.guarded(info):
        if os.path.exists(marker):
            if info.is_main:
                print(f"[stage] {a.stage}: already done ({marker}), skipping", flush=True)
            return
        t0 = time.time()
        hours = a.hours if a.hours is not None else float(STAGE_HOURS.get(a.stage, 1.0))
        if a.stage == "pretrain":
            from verqen.training.pretrain_v3 import run_pretrain_v3
            run_pretrain_v3(cfg, os.path.join(run, "pretrain"), hours, info, max_steps=a.max_steps)
        elif a.stage == "pretrain_ext":
            from verqen.training.pretrain_v3 import run_pretrain_v3
            cfg.model.rope_base = cfg.pretrain.ext_rope_base
            run_pretrain_v3(cfg, os.path.join(run, "pretrain_ext"), hours, info, stage="pretrain_ext",
                            init_core=os.path.join(run, "pretrain", "resume"), seq_len=cfg.pretrain.ext_seq_len,
                            micro_rows=cfg.pretrain.ext_micro_rows, max_steps=a.max_steps)
        elif a.stage == "decisions":
            from verqen.training.decisions_v3 import run_decisions_v3
            init = os.path.join(run, "pretrain_ext", "final")
            if not os.path.exists(init):
                init = os.path.join(run, "pretrain", "final")
            run_decisions_v3(cfg, init, os.path.join(run, "decisions"), hours, info, max_steps=a.max_steps)
        elif a.stage == "rlcd":
            from verqen.rlcd.run_v3 import run_rlcd_v3
            run_rlcd_v3(cfg, os.path.join(run, "decisions", "final"), os.path.join(run, "rlcd"), hours, info)
        elif a.stage in ("agentic_sft", "agentic_rlcd", "format_sft"):
            # agentic top-up (docs/production/NEXT_RUN.md item 1): trajectories + replay, from the best checkpoint
            if a.decision_dir:
                cfg.data.decision_dir = a.decision_dir
            if a.stage in ("agentic_sft", "format_sft"):
                from verqen.training.decisions_v3 import run_decisions_v3
                init = a.init or json.load(open(os.path.join(run, "release_choice.json")))["chosen"]
                run_decisions_v3(cfg, init, os.path.join(run, a.stage), hours, info, lr=a.lr, max_steps=a.max_steps)
            else:
                from verqen.rlcd.run_v3 import run_rlcd_v3
                run_rlcd_v3(cfg, os.path.join(run, "agentic_sft", "final"), os.path.join(run, "agentic_rlcd"), hours, info)
        if info.is_main:
            el = time.time() - t0
            print(f"[stage] {a.stage} finished in {el / 3600:.2f} h", flush=True)
            if el < 5 and a.max_steps is None:
                raise SystemExit(f"[stage] {a.stage} finished in {el:.1f}s: it did not run (LESSONS #36)")
            with open(os.path.join(run, f"{a.stage}.done.json"), "w") as f:
                json.dump({"stage": a.stage, "seconds": round(el), "world": info.world}, f)


STAGE_HOURS: dict = {}

if __name__ == "__main__":
    # stage budgets come from the config's `stages:` block (hours per stage)
    import yaml
    _cfg_path = sys.argv[sys.argv.index("--config") + 1] if "--config" in sys.argv else None
    if _cfg_path:
        STAGE_HOURS.update((yaml.safe_load(open(_cfg_path)) or {}).get("stages", {}))
    main()
