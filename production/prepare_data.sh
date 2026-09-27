#!/usr/bin/env bash
# Pretraining corpus on the box, CPU/network only (no GPU), idempotent: every step has a marker.
#   bash production/prepare_data.sh          (run_all.sh starts it in the background next to preflight)
#
# Curated sources only, base plan size (3.95B tokens: fineweb-edu, cosmopedia-v2, finemath-4+,
# wikipedia) plus 400M tokens of engine-generated decision/structured text. Pulled by the box
# (never pushed, LESSONS #34), decontaminated against every real eval split while tokenizing.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="$(pwd)/.venv/bin/python"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
mkdir -p logs data/pretrain/v3

# CPU allocation, not the host's core count: a vast container sees all 144 host cores but gets a
# CFS quota (36 here); 144 workers would oversubscribe it and copy the 13-gram blocklist 144 times.
CPUS=$(awk '{ if ($1 == "max") print 0; else print int($1 / $2) }' /sys/fs/cgroup/cpu.max 2>/dev/null || echo 0)
[ "${CPUS:-0}" -gt 0 ] || CPUS=$(nproc)
W=$(( CPUS > 6 ? CPUS - 2 : 4 ))
echo "[prepare_data] cpu quota $CPUS -> $W workers"
t0=$(date +%s)
step() { echo "[prepare_data $(date +%T)] $*"; }

# 1. download and the real eval splits in parallel (both network-bound)
if [ ! -e data/pretrain/v3/.download_done ]; then
  step "download (base plan)"
  ( "$PY" scripts/v3/prepare_pretrain.py download --out data/pretrain/v3 --workers 12 > logs/download.log 2>&1 \
      && touch data/pretrain/v3/.download_done ) &
  DL=$!
fi
if [ ! -e data/decision/v3_evals/EVALS_OK ]; then
  step "real eval splits (the decontamination blocklist)"
  "$PY" scripts/v3/build_data.py --evals-only --out data/decision/v3_evals --workers 12 > logs/real_evals.log 2>&1
fi
# 2. synthetic text while the download finishes (CPU-bound, sharded)
if [ ! -e data/pretrain/v3/synthetic_text.jsonl ]; then
  step "synthetic text"
  "$PY" scripts/v3/prepare_pretrain.py synthetic --out data/pretrain/v3 --tokens 400000000 --workers "$W" > logs/synthetic_text.log 2>&1
fi
if [ -n "${DL:-}" ]; then wait "$DL"; fi
[ -e data/pretrain/v3/.download_done ] || { step "download FAILED (logs/download.log)"; exit 2; }
[ -e data/pretrain/v3/tokenizer_v3_32k.json ] || { step "tokenizer missing: it ships in the code package"; exit 3; }
# 3. tokenize + 13-gram decontamination
if [ ! -e data/pretrain/v3/manifest.json ]; then
  step "tokenize"
  "$PY" scripts/v3/prepare_pretrain.py tokenize --out data/pretrain/v3 --workers "$W" \
    --decision-dir data/decision/v3_evals > logs/tokenize.log 2>&1
fi
grep -h "dropped" logs/tokenize.log 2>/dev/null || true
step "done in $(( ($(date +%s) - t0) / 60 )) min"
