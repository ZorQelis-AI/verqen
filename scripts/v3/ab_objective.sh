#!/usr/bin/env bash
# Local gate 2 (T2_PLAN.md section 7): RTD vs MLM at equal wall-clock, then the same decision fine-tune.
# Arms run sequentially with the GPU to themselves (equal wall-clock is only fair on an idle GPU).
cd "$(dirname "$0")/../.."
set -u
for obj in rtd mlm; do
  for st in pretrain decisions; do
    python scripts/v3/run_stage.py --config configs/v3/ab_$obj.yaml --stage $st > logs/ab_${obj}_$st.log 2>&1
    echo "$obj $st exit=$?"
  done
  python evals/jev_parity/run.py --checkpoint runs/ab_obj/$obj/decisions/final --out evals/jev_parity/results/ab_$obj.json > logs/ab_${obj}_jev.log 2>&1
  echo "$obj jev exit=$?"
done
python - <<'PY'
import json
for o in ("rtd", "mlm"):
    s = json.load(open(f"runs/ab_obj/{o}/decisions/selection.json"))
    j = json.load(open(f"evals/jev_parity/results/ab_{o}.json"))["overall"]
    print(o, {k: v for k, v in s.items() if k != "history"}, "jev_parity", j["accuracy"], j["ci95"])
PY
