#!/usr/bin/env bash
# Agentic top-up: a short multi-step pass on top of the main release (docs/production/NEXT_RUN.md item 1).
# Run it only AFTER production/run_all.sh has finished (release/FINAL_OK exists):
#
#   cd /workspace/verqen && setsid nohup bash production/agentic_topup.sh > logs/agentic_topup.log 2>&1 < /dev/null &
#
# Stages (idempotent like run_all.sh: each is skipped once its marker exists):
#   0. agentic_para   data/decision/info_para: conflict/unknown rows reworded in new phrasings
#   1. agentic_data   data/decision/agentic_mix: oracle trajectories with detours + REPLAY_FRAC (0.5)
#                     replay of the main decision train split + the paraphrase rows; env seeds disjoint
#   2. agentic_sft    supervised pass on the mix from INIT (default the release; 25 Sep: rlcd/final), 2 GPUs
#   3. agentic_rlcd   RLCD from agentic_sft/final on the mix, GPU0, while the SFT candidate is judged on GPU1
#   4. selection      each candidate vs the release: the SAME val rule as the main release (main v3 val,
#                     not the mix) + mean environment gain >= ENV_GAIN + no environment down > 0.02;
#                     the passing candidate with the higher environment mean ships
#   5. agentic_finalize only if one won: calibrate, evaluate, export to release_agentic, upload
# A losing top-up changes nothing: the main release stays the release.
set -uo pipefail
cd "$(dirname "$0")/.."
WORK=$(pwd)
PY=${PY:-"$WORK/.venv/bin/python"}          # overridable for the local smoke chain only
TORCHRUN=${TORCHRUN:-"$WORK/.venv/bin/torchrun"}
NPROC=${NPROC:-2}
CFG=${CFG:-configs/v3/t2.box2048.yaml}  # 26 Sep: 2048 context (long_state); micro 12 x 2048 probed at 22.55 GB
RUN=${RUN:-runs/verqen-t2}
MIX=${MIX:-data/decision/agentic_mix}
REPLAY=${REPLAY:-data/decision/v3}
SFT_HOURS=${SFT_HOURS:-1.0}  # a cap: the SFT schedule is 3 epochs and normally ends well before it
RLCD_HOURS=${RLCD_HOURS:-0.75}
EPISODES=${EPISODES:-6000}
REPLAY_FRAC=${REPLAY_FRAC:-0.5}
ENV_GAIN=${ENV_GAIN:-0.02}
export OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false
mkdir -p logs

stage() {  # name, marker, command...
  local name=$1 marker=$2; shift 2
  if [ -e "$marker" ]; then echo "[topup] $name: done, skipping"; return 0; fi
  echo "[topup] $name: start $(date +%T)"
  "$@" > "logs/$name.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "[topup] $name FAILED (rc=$rc)"; tail -30 "logs/$name.log"; exit $rc; fi
  if [ ! -e "$marker" ]; then echo "[topup] $name finished without its marker $marker"; exit 3; fi
  echo "[topup] $name: done $(date +%T)"
}

[ -e "$RUN/release/FINAL_OK" ] || { echo "[topup] main release not finalized yet; run after run_all.sh"; exit 2; }
BASE=$("$PY" -c "import json; print(json.load(open('$RUN/release_choice.json'))['chosen'])")
# INIT: where the top-up STARTS (default the release). INIT=runs/verqen-t2/rlcd/final keeps RLCD's
# environment gains when the val rule rejected RLCD itself: the 40% replay can win back single-step
# accuracy, and the candidate is still judged against the release by the same rule
INIT=${INIT:-$BASE}
echo "[topup] baseline (current release): $BASE; top-up starts from: $INIT"

# conflicting/unknown facts in 32 new phrasings (NEXT_RUN.md item 6); pool B stays out as val_B
PARA=${PARA:-data/decision/info_para}
stage agentic_para "$PARA/report.json" "$PY" scripts/v3/augment_info.py --src "$REPLAY" --out "$PARA" \
  --conflict "${PARA_CONFLICT:-30000}" --unknown "${PARA_UNKNOWN:-20000}"
# long states: decisions buried in 15-50 sentences of TRAINING prose (shard 000), for the 2048 context
LONG=${LONG:-data/decision/long_noise}
stage agentic_long "$LONG/report.json" "$PY" scripts/v3/augment_long.py --src "$REPLAY" --out "$LONG" \
  --n "${LONG_N:-30000}"
# gap sources: extra TRAIN-split rehearsal where Laya leads (ag_news, XNLI~mnli/snli/wanli, boolq, sst5)
stage agentic_data "$MIX/build_report.json" "$PY" scripts/v3/build_trajectories.py --out "$MIX" \
  --replay "$REPLAY" --episodes "$EPISODES" --replay-frac "$REPLAY_FRAC" \
  --extra-train "$PARA/train.jsonl" "$LONG/train.jsonl" --extra-val "$PARA/val_A.jsonl" \
  --gap-sources "${GAP_SOURCES-ag_news,mnli,snli,wanli,boolq,sst5}" --gap-per-source "${GAP_PER_SOURCE:-7000}"
stage agentic_sft "$RUN/agentic_sft/final" "$TORCHRUN" --standalone --nproc_per_node "$NPROC" scripts/v3/run_stage.py \
  --config "$CFG" --stage agentic_sft --init "$INIT" --decision-dir "$MIX" --hours "$SFT_HOURS" ${SFT_EXTRA:-}
# Two candidates, one pre-registered rule (fixed 25 Sep before any top-up result):
#   agentic_sft/final  judged on GPU1 WHILE agentic_rlcd trains on GPU0 (no extra wall-clock)
#   agentic_rlcd/final judged after it finishes
# each must pass select_release.py against the release (val overall/domain/variant rule, mean
# environment gain >= ENV_GAIN, no environment down > 0.02). Of those that pass, the higher mean
# environment score ships; if neither passes, the release stays. RLCD is included because it is
# what bought the environment gains, the SFT because RLCD is what cost single-step accuracy.
SEL="$PY production/select_release.py --config $CFG --baseline $BASE --require-env-gain $ENV_GAIN"
( export CUDA_VISIBLE_DEVICES="${SEL_GPU-1}"
  stage agentic_select_sft "$RUN/agentic_choice_sft.json" \
    $SEL --candidate "$RUN/agentic_sft/final" --out "$RUN/agentic_choice_sft.json" ) &
SEL_PID=$!
( stage agentic_rlcd "$RUN/agentic_rlcd/final" env CUDA_VISIBLE_DEVICES="${RLCD_GPU-0}" "$PY" scripts/v3/run_stage.py \
    --config "$CFG" --stage agentic_rlcd --decision-dir "$MIX" --hours "$RLCD_HOURS" ) \
  || echo "[topup] WARNING: agentic_rlcd failed; only the SFT candidate competes"
wait "$SEL_PID" || echo "[topup] WARNING: SFT candidate selection failed"
if [ -e "$RUN/agentic_rlcd/final/model.safetensors" ]; then
  stage agentic_select_rlcd "$RUN/agentic_choice_rlcd.json" \
    $SEL --candidate "$RUN/agentic_rlcd/final" --out "$RUN/agentic_choice_rlcd.json"
fi
"$PY" - "$RUN" "$BASE" <<'PYEOF'
import json, os, sys
run, base = sys.argv[1], sys.argv[2]
recs = [json.load(open(p)) for p in (f"{run}/agentic_choice_sft.json", f"{run}/agentic_choice_rlcd.json") if os.path.exists(p)]
won = [r for r in recs if r["chosen"] != base]
best = max(won, key=lambda r: r["environments"]["candidate"]["mean"]) if won else None
out = {"rule": "each candidate vs the release by select_release.py; highest mean env score among passers",
       "candidates": {r["candidate"]["checkpoint"]: {"passed": r["chosen"] != base, "val": r["candidate"]["accuracy"],
                                                     "env": r["environments"], "reasons": r["reasons"]} for r in recs if r["candidate"]},
       "baseline": base, "chosen": best["chosen"] if best else base,
       "reasons": ["passed; highest environment mean among passing candidates"] if best else
                  [f"{r['candidate']['checkpoint'] if r['candidate'] else '?'}: {'; '.join(r['reasons'])}" for r in recs]}
tmp = f"{run}/agentic_choice.json.tmp"; json.dump(out, open(tmp, "w"), indent=1); os.replace(tmp, f"{run}/agentic_choice.json")
print(json.dumps({k: out[k] for k in ("candidates", "chosen")}, indent=1))
PYEOF
CHOSEN=$("$PY" -c "import json; print(json.load(open('$RUN/agentic_choice.json'))['chosen'])")
if [ "$CHOSEN" = "$BASE" ]; then
  echo "[topup] top-up did not clear the rule; the main release stays: $BASE"
  "$PY" -c "import json; print('\n'.join(json.load(open('$RUN/agentic_choice.json'))['reasons']))"
  "$PY" production/finalize.py --config "$CFG" --upload-stages || true  # still archive the attempt
  exit 0
fi
echo "[topup] releasing the top-up: $CHOSEN"
stage agentic_finalize "$RUN/release_agentic/FINAL_OK" "$PY" production/finalize.py --config "$CFG" \
  --checkpoint "$CHOSEN" --out "$RUN/release_agentic" --calibrate --upload
"$PY" production/finalize.py --config "$CFG" --upload-stages || true
echo "[topup] complete $(date +%T)"
