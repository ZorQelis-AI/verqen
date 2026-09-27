"""VerQen-T1 command line.

    python -m verqen pipeline  --config configs/v2/t1.yaml [--stage all|pretrain|decisions|rlcd|calibrate|eval]
    python -m verqen build-data [--procedural 450000]
    python -m verqen infer     --checkpoint runs/verqen-t1-v2/release --input request.json
    python -m verqen inspect   --checkpoint runs/verqen-t1-v2/release
    python -m verqen audit     --train data/decision/v2/train.jsonl --test data/decision/v2/eval/proc_test.jsonl
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _script(name: str, argv: list[str]) -> None:
    sys.argv = [name] + argv
    runpy.run_path(str(ROOT / "scripts" / name), run_name="__main__")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return
    cmd, rest = argv[0], argv[1:]
    if cmd == "pipeline":
        _script("run_pipeline.py", rest)
    elif cmd == "build-data":
        _script("build_decision_data.py", rest)
    elif cmd == "infer":
        sys.argv = ["verqen.inference"] + rest
        runpy.run_module("verqen.inference", run_name="__main__")
    elif cmd == "inspect":
        import argparse
        from verqen.utils.checkpoint import load_calibration, load_model
        ap = argparse.ArgumentParser()
        ap.add_argument("--checkpoint", required=True)
        a = ap.parse_args(rest)
        m = load_model(a.checkpoint, "cpu")
        cal = load_calibration(a.checkpoint)
        print(json.dumps({"config": m.config.__dict__, "parameters": m.count_parameters(),
                          "calibration": (cal or {}).get("calibration_report")}, indent=2))
    elif cmd == "audit":
        import argparse
        from verqen.data import audit
        from verqen.data.store import read_jsonl
        ap = argparse.ArgumentParser()
        ap.add_argument("--train", required=True)
        ap.add_argument("--test", required=True, nargs="+")
        ap.add_argument("--limit", type=int, default=200000)
        a = ap.parse_args(rest)
        train = read_jsonl(a.train, a.limit)
        print(json.dumps(audit.audit(train, {Path(t).stem: read_jsonl(t) for t in a.test}), indent=2))
    else:
        print(f"unknown command {cmd!r}\n{__doc__}")
        sys.exit(2)


if __name__ == "__main__":
    main()
