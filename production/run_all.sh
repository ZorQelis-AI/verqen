#!/usr/bin/env bash
# VerQen-T2 full production run on 2x RTX 5090. Idempotent: re-run it after any crash and every
# finished stage is skipped, every unfinished one resumes from its last wall-clock checkpoint.
#
#   cd /workspace/verqen && setsid nohup bash production/run_all.sh > logs/run_all.log 2>&1 < /dev/null &
#
# Launch detached exactly like that ([LESSONS] `timeout ssh` leaves orphans; idle SSH sessions get
# dropped [#30]). Monitor with short connections: `cat runs/verqen-t2/progress.json`.
set -uo pipefail
cd "$(dirname "$0")/.."
WORK=$(pwd)
PY="$WORK/.venv/bin/python"
TORCHRUN="$WORK/.venv/bin/torchrun"
CFG=configs/v3/t2.yaml
RUN=runs/verqen-t2
mkdir -p "$RUN" logs
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
PROGRESS="$RUN/progress.json"
T0=$(date +%s)

progress() {  # the real monitoring channel [#21]: one JSON, rewritten atomically
  "$PY" - "$PROGRESS" "$1" "$2" "$T0" <<'PYEOF'
import json, os, sys, time
path, stage, status, t0 = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
d = json.load(open(path)) if os.path.exists(path) else {"stages": {}}
d["stages"][stage] = {"status": status, "at": time.strftime("%H:%M:%S"), "elapsed_h": round((time.time() - t0) / 3600, 2)}
d["current"], d["updated"] = stage, time.strftime("%Y-%m-%d %H:%M:%S")
tmp = path + ".tmp"; json.dump(d, open(tmp, "w"), indent=1); os.replace(tmp, path)
PYEOF
}

run_stage() {  # name, marker, command...: skip if marker exists; a stage < 5 s did not run [#36]
  local name=$1 marker=$2; shift 2
  if [ -e "$marker" ]; then echo "[run_all] $name: done, skipping"; progress "$name" skipped; return 0; fi
  progress "$name" running
  local s=$(date +%s)
  "$@" > "logs/$name.log" 2>&1
  local rc=$? el=$(( $(date +%s) - s ))
  if [ $rc -ne 0 ]; then progress "$name" "failed rc=$rc"; echo "[run_all] $name FAILED (rc=$rc), see logs/$name.log"; tail -30 "logs/$name.log"; exit $rc; fi
  if [ ! -e "$marker" ]; then progress "$name" "no-marker"; echo "[run_all] $name finished in ${el}s without its marker $marker"; exit 3; fi
  progress "$name" "done ${el}s"
}

# 0+1. pre-flight (GPU) and the pretraining corpus (CPU/network) IN PARALLEL: data prep no longer
#      idles the GPUs. prepare_data.sh is idempotent, so a corpus prepared before launch is skipped.
if [ ! -e data/pretrain/v3/manifest.json ]; then
  progress data_prep running
  ( bash production/prepare_data.sh > logs/data_prep.log 2>&1 && progress data_prep done || progress data_prep failed ) &
  PREP_PID=$!
fi
run_stage preflight configs/v3/t2.box.yaml "$PY" production/preflight.py --config "$CFG"
CFG=configs/v3/t2.box.yaml
if [ -n "${PREP_PID:-}" ]; then echo "[run_all] waiting for the corpus"; wait "$PREP_PID" || true; fi
[ -e data/pretrain/v3/manifest.json ] || { echo "[run_all] data prep failed: logs/data_prep.log"; tail -20 logs/data_prep.log; exit 5; }

# the decision data (CPU only) builds in the background while pretraining runs; needed at stage 3
if [ ! -e data/decision/v3/build_report.json ]; then
  progress decision_data running
  ( "$PY" scripts/v3/build_data.py --out data/decision/v3 --workers 12 > logs/decision_data.log 2>&1       && progress decision_data done || progress decision_data failed ) &
  DATA_PID=$!
fi

# 2. pretraining (DDP x2), then context extension
run_stage pretrain "$RUN/pretrain/final" "$TORCHRUN" --standalone --nproc_per_node 2 scripts/v3/run_stage.py --config "$CFG" --stage pretrain
run_stage pretrain_ext "$RUN/pretrain_ext/final" "$TORCHRUN" --standalone --nproc_per_node 2 scripts/v3/run_stage.py --config "$CFG" --stage pretrain_ext

# 3. decision training needs the v3 decision data
if [ -n "${DATA_PID:-}" ]; then echo "[run_all] waiting for decision data build"; wait "$DATA_PID"; fi
[ -e data/decision/v3/build_report.json ] || { echo "[run_all] decision data build failed: logs/decision_data.log"; exit 4; }
run_stage decisions "$RUN/decisions/final" "$TORCHRUN" --standalone --nproc_per_node 2 scripts/v3/run_stage.py --config "$CFG" --stage decisions

# 4. RLCD on GPU 0 while GPU 1 evaluates the pre-RLCD checkpoint (the comparison RLCD must beat)
( CUDA_VISIBLE_DEVICES=1 "$PY" production/finalize.py --config "$CFG" --checkpoint "$RUN/decisions/final" \
    --out "$RUN/eval_pre_rlcd" > logs/eval_pre_rlcd.log 2>&1 ) &
EVAL_PID=$!
# a failed RLCD must not cost the release: the subshell contains run_stage's exit, and the
# selection below then falls back to the decision checkpoint
( run_stage rlcd "$RUN/rlcd/final" env CUDA_VISIBLE_DEVICES=0 "$PY" scripts/v3/run_stage.py --config "$CFG" --stage rlcd ) \
  || echo "[run_all] WARNING: RLCD failed; releasing the decision checkpoint"
wait "$EVAL_PID" || echo "[run_all] WARNING: pre-RLCD eval failed (logs/eval_pre_rlcd.log)"

# 5. release selection: RLCD vs pre-RLCD on the VALIDATION split, rule fixed in advance
#    (production/select_release.py); test splits and external suites are never consulted
run_stage select "$RUN/release_choice.json" "$PY" production/select_release.py --config "$CFG" \
  --baseline "$RUN/decisions/final" --candidate "$RUN/rlcd/final" --out "$RUN/release_choice.json"
CHOSEN=$("$PY" -c "import json; print(json.load(open('$RUN/release_choice.json'))['chosen'])")
echo "[run_all] releasing $CHOSEN"

# 6. calibrate (marginal + per-domain worst-group), evaluate everything, export, upload
run_stage finalize "$RUN/release/FINAL_OK" "$PY" production/finalize.py --config "$CFG" --checkpoint "$CHOSEN" \
  --out "$RUN/release" --calibrate --upload
progress all done
echo "[run_all] complete in $(( ($(date +%s) - T0) / 3600 )) h"
