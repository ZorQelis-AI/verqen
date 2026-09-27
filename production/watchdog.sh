#!/usr/bin/env bash
# Monitor + uploader for the unattended T2 run (launch detached right after run_all.sh):
#   setsid nohup bash production/watchdog.sh 25 > logs/watchdog.log 2>&1 < /dev/null &
#
# It NEVER kills a stage or stops the instance on its own (user decision, 25 Sep 2026: a stop at
# the deadline with one hour of training left would waste the run; credits cover ~28 h). It:
# * counts INSTANCE RUNNING TIME against the budget (starts from container uptime, persisted in
#   runs/verqen-t2/budget_used_s so a stop/restart resumes the count; relaunch after a restart);
# * at the budget: writes runs/verqen-t2/BUDGET_REACHED and warns loudly, every tick, and keeps going;
# * every 5 min: records GPU state, flags a stale log (allocator thrash is silent), uploads each
#   finished stage checkpoint and the final release to the private HF repo once a token exists
#   (`hf auth login` on the box, any time).
# Opt-in only: AUTO_STOP_WHEN_DONE=1 stops the instance after the release AND every stage checkpoint
# are uploaded and run_all.sh has exited.
set -u
cd "$(dirname "$0")/.."
HOURS=${1:-25}
AUTO_STOP_WHEN_DONE=${AUTO_STOP_WHEN_DONE:-0}
RUN=runs/verqen-t2
PY=$(pwd)/.venv/bin/python
CFG=configs/v3/t2.box.yaml
[ -f "$CFG" ] || CFG=configs/v3/t2.yaml
BUDGET_FILE="$RUN/budget_used_s"
LIMIT=$(( HOURS * 3600 ))
mkdir -p "$RUN" logs
log() { echo "[watchdog $(date '+%F %T')] $*"; }

if [ -f "$BUDGET_FILE" ]; then
  used=$(cat "$BUDGET_FILE")
  log "resuming budget: $((used / 60)) min of $((LIMIT / 60)) used"
else
  used=$(ps -o etimes= -p 1 | tr -dc 0-9)   # time the instance ran before this first launch
  used=${used:-0}
  log "first launch: counting $((used / 60)) min of instance uptime already spent"
fi
last=$(date +%s)

stop_instance() {
  local id
  id=$(tr -dc 0-9 < /root/.vast_containerlabel 2>/dev/null || true)
  if [ -n "$id" ] && [ -x /opt/instance-tools/bin/vastai ]; then
    log "stopping instance $id"
    /opt/instance-tools/bin/vastai stop instance "$id" --api-key "$(cat /root/.vast_api_key)" || log "stop failed"
  fi
}

uploads() {  # cheap without a token: one whoami call each
  CFG_NOW=configs/v3/t2.box.yaml; [ -f "$CFG_NOW" ] || CFG_NOW=configs/v3/t2.yaml
  timeout 3600 "$PY" production/finalize.py --config "$CFG_NOW" --upload-stages >> logs/upload.log 2>&1
  if [ -f "$RUN/release/FINAL_OK" ] && [ ! -f "$RUN/release/UPLOADED" ]; then
    timeout 3600 "$PY" production/finalize.py --config "$CFG_NOW" --out "$RUN/release" --upload-only >> logs/upload.log 2>&1 \
      && log "release uploaded"
  fi
}

all_done() {  # release + every existing stage checkpoint uploaded, and the pipeline has exited
  [ -f "$RUN/release/UPLOADED" ] || return 1
  pgrep -f 'production/run_al[l].sh' > /dev/null && return 1
  for st in pretrain pretrain_ext decisions rlcd; do
    [ -e "$RUN/$st/final/model.safetensors" ] && [ ! -f "$RUN/uploads/$st.done" ] && return 1
  done
  return 0
}

while :; do
  now=$(date +%s)
  used=$(( used + now - last )); last=$now
  echo "$used" > "$BUDGET_FILE.tmp" && mv "$BUDGET_FILE.tmp" "$BUDGET_FILE"
  uploads
  latest=$(ls -t logs/*.log 2>/dev/null | grep -v -e watchdog -e upload | head -1)
  if [ -n "$latest" ]; then
    age=$(( now - $(stat -c %Y "$latest") ))
    [ $age -gt 1800 ] && log "WARNING: $latest silent for $((age / 60)) min (stall or allocator thrash?)"
  fi
  echo "[gpu $(date +%T)] budget $((used / 60))/$((LIMIT / 60)) min | $(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | tr '\n' ' ')"
  if [ $used -ge $LIMIT ]; then
    [ -f "$RUN/BUDGET_REACHED" ] || date '+%F %T' > "$RUN/BUDGET_REACHED"
    log "WARNING: planned budget of $HOURS h reached; NOT stopping anything (decide manually)"
  fi
  if all_done; then
    if [ "$AUTO_STOP_WHEN_DONE" = "1" ]; then log "everything uploaded; AUTO_STOP_WHEN_DONE=1 -> stopping"; stop_instance; exit 0; fi
    log "everything finished and uploaded; the instance can be stopped"
  fi
  sleep 300
done
