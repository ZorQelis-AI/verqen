"""Weight-space interpolation (WiSE-FT) between two checkpoints of the same architecture.

    python scripts/v3/wise_merge.py --a runs/verqen-t2/decisions/final --b runs/verqen-t2/agentic_sft/final \
        --alpha 0.5 --out runs/verqen-t2/wise_sft_050

out = (1 - alpha) * a + alpha * b, tensor by tensor. Why (26 Sep 2026): the agentic top-up gained
+0.40 mean environment optimal-action rate and +34 pts on info_missing, but lost 22 pts on
v2:incident_triage (a multi-factor severity rubric) and a few points on other v2 counting families.
That is fine-tuning drift; interpolating toward the pre-top-up release is the standard remedy and
usually keeps most of the gain (Wortsman et al., 2022). Both checkpoints descend from the same
decisions checkpoint (decisions -> rlcd -> top-up), so they share a basin. Alphas are fixed before
evaluation; the release rule decides; the final numbers come from test splits the rule never sees.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="the release (alpha=0 end)")
    ap.add_argument("--b", required=True, help="the fine-tuned candidate (alpha=1 end)")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--out", required=True)
    x = ap.parse_args()
    A, B = load_file(str(Path(x.a, "model.safetensors"))), load_file(str(Path(x.b, "model.safetensors")))
    if A.keys() != B.keys():
        raise SystemExit(f"key mismatch: {sorted(A.keys() ^ B.keys())[:5]}")
    out = {}
    for k, ta in A.items():
        tb = B[k]
        if ta.shape != tb.shape:
            raise SystemExit(f"shape mismatch at {k}: {ta.shape} vs {tb.shape}")
        if ta.is_floating_point():
            out[k] = ((1 - x.alpha) * ta.float() + x.alpha * tb.float()).to(ta.dtype)
        else:
            out[k] = tb.clone()
    o = Path(x.out)
    o.mkdir(parents=True, exist_ok=True)
    save_file(out, str(o / "model.safetensors"))
    for f in ("model_config.json", "tokenizer.json"):
        shutil.copy(Path(x.b, f), o / f)
    (o / "merge.json").write_text(json.dumps({"a": x.a, "b": x.b, "alpha": x.alpha}, indent=1))
    print(f"[wise] {len(out)} tensors, alpha={x.alpha} -> {o}")


if __name__ == "__main__":
    main()
