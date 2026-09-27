#!/usr/bin/env bash
# VerQen-T2 box setup (run ON the rented 2x RTX 5090 box, as root, once).
#
#   export CODE_URL='<signed CDN url printed by production/package.py>'
#   export CODE_MD5='<md5 printed by package.py>'
#   export HF_TOKEN='<your token>'            # used for pulling data and uploading results
#   curl -fsSL "$SETUP_URL" -o setup.sh && bash setup.sh
#
# Every step below exists because it failed once on a rented box (SoreQen/LESSONS.md numbers in [#]).
set -euo pipefail
WORK=${WORK:-/workspace/verqen}
PY=${PY:-python3}
log() { echo "[setup $(date +%H:%M:%S)] $*"; }

# [#9 #28] the model hub can be DNS-poisoned or TCP-blocked: probe it before anything else
HUB=https://huggingface.co
if ! curl -s -o /dev/null -m 15 -w '%{http_code}' https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu | grep -q 200; then
  log "huggingface.co unreachable; trying hf-mirror.com"
  if curl -s -o /dev/null -m 15 -A 'Mozilla/5.0' -w '%{http_code}' https://hf-mirror.com/api/datasets/HuggingFaceFW/fineweb-edu | grep -q 200; then
    HUB=https://hf-mirror.com
    export HF_ENDPOINT=$HUB
  else
    log "FATAL: neither huggingface.co nor hf-mirror.com reachable; data cannot be pulled"; exit 1
  fi
fi
log "model hub: $HUB"
curl -s -o /dev/null -m 15 -w 'pypi %{http_code}\n' https://pypi.org/simple/torch/ || true

# [#27] a 5090 exposes 31.4 GB usable; DDP replicates, so per-card numbers are what matter
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
N_GPU=$(nvidia-smi -L | wc -l)
[ "$N_GPU" -ge 2 ] || { log "FATAL: expected 2 GPUs, found $N_GPU"; exit 1; }
df -h /workspace | tail -1
FREE_GB=$(df -BG --output=avail /workspace | tail -1 | tr -dc 0-9)
[ "$FREE_GB" -ge 150 ] || log "WARNING: only ${FREE_GB} GB free on /workspace (plan needs ~150 GB)"

mkdir -p "$WORK" && cd "$WORK"
# [#34] never push data over SSH: pull the code package from a signed CDN URL, verify md5 [#19]
if [ ! -f .code_ok ]; then
  curl -fL --retry 5 -o code.tar.gz "$CODE_URL"
  echo "$CODE_MD5  code.tar.gz" | md5sum -c -
  tar xzf code.tar.gz && touch .code_ok
fi

# [#10 #29] dead IPv6: force IPv4 for every Python client, process-wide
if [ ! -d .venv ]; then
  $PY -m venv .venv
fi
VPY="$WORK/.venv/bin/python"
SITE=$($VPY -c 'import site; print(site.getsitepackages()[0])')
cat > "$SITE/sitecustomize.py" <<'PYEOF'
import socket
_orig = socket.getaddrinfo
def _v4(host, port, family=0, *a, **k):
    return _orig(host, port, socket.AF_INET, *a, **k)
socket.getaddrinfo = _v4
PYEOF

# Blackwell (sm_120) needs CUDA 12.8+ wheels
$VPY -m pip install -q --upgrade pip
$VPY -m pip install -q torch --index-url https://download.pytorch.org/whl/cu128
$VPY -m pip install -q -r production/requirements.txt
$VPY -c "import torch; assert torch.cuda.is_available() and torch.cuda.device_count() >= 2; print('torch', torch.__version__, torch.cuda.get_device_name(0))"

# [#20 #36] call venv binaries by full path; torchrun is not on PATH inside the venv
TORCHRUN="$WORK/.venv/bin/torchrun"
[ -x "$TORCHRUN" ] || { log "FATAL: torchrun missing at $TORCHRUN"; exit 1; }

# CPU-side processes must not spawn one thread per core on 100+ core hosts
grep -q OMP_NUM_THREADS ~/.bashrc || echo 'export OMP_NUM_THREADS=8' >> ~/.bashrc
export OMP_NUM_THREADS=8

log "unit tests"
$VPY -m pytest -q tests/ -x 2>&1 | tail -3
log "setup done. next: bash production/run_all.sh (it runs preflight first)"
