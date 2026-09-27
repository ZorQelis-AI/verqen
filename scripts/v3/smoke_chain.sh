#!/usr/bin/env bash
# End-to-end smoke of every T2 stage on CPU with 2 DDP ranks (Linux / WSL). Usage: bash scripts/v3/smoke_chain.sh
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2
rm -rf runs/smoke_v3; sync; sleep 1
for st in pretrain pretrain_ext decisions; do
  torchrun --standalone --nproc_per_node 2 scripts/v3/run_stage.py --config configs/v3/smoke.yaml --stage "$st" --cpu --max-steps 8 > "runs/smoke_$st.log" 2>&1
  echo "$st exit=$?"; grep -v -i warn "runs/smoke_$st.log" | grep -E '\] done|FAILED|Error|initialized' | cut -c1-180
done
python3 scripts/v3/run_stage.py --config configs/v3/smoke.yaml --stage rlcd --cpu --hours 0.003 > runs/smoke_rlcd.log 2>&1
echo "rlcd exit=$?"; grep -v -i warn runs/smoke_rlcd.log | grep -E 'rlcd\] done|FAILED|Error|finished' | cut -c1-180
ls runs/smoke_v3
