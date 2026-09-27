"""Run the controlled diagnostic probes against a checkpoint and write a JSON report.

    python scripts/audit/run_probes.py --checkpoint runs/verqen-t1-v2/release --tag baseline
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "audit"))
os.chdir(ROOT)

import probes as P  # noqa: E402
from verqen.data.collator import collate_rows, to_device  # noqa: E402
from verqen.inference.engine import VerqenInferenceEngine  # noqa: E402
from verqen.model.attention import apply_rotary  # noqa: E402
from verqen.model.attention_mask import SEG_STATE, build_attention_mask  # noqa: E402


def decide(engine, state, question, options):
    r = engine.decide_multi(state, [{"type": "choice", "text": question, "options": options}])[0]
    return r


@torch.no_grad()
def decide_attention(engine, state, question, options, span_text):
    """Fraction of the DECIDE token's attention (averaged over heads) that lands on the tokens of
    ``span_text`` inside the state, per layer. Recomputes attention probabilities from q/k."""
    m = engine.model
    norm = engine._normalize_questions([{"type": "choice", "text": question, "options": options}])
    ex = engine.packer.pack({"state": state, "questions": norm})
    b = to_device(collate_rows([[ex]], len(ex), 0), engine.device)
    ids = ex.ids
    span_ids = engine.tokenizer.encode(" " + span_text)
    # locate the span inside the state tokens (first exact match of the middle 60% of the span)
    core = span_ids[len(span_ids) // 5: len(span_ids) - len(span_ids) // 5] or span_ids
    start = next((i for i in range(len(ids) - len(core)) if ids[i:i + len(core)] == core), None)
    state_idx = [i for i, s in enumerate(ex.seg) if s == SEG_STATE]
    dec = ex.questions[0].decide_idx
    mask = build_attention_mask(b["seg"], b["doc"], b["qid"], b["oid"])
    x = m.transformer.embeddings(b["input_ids"], b["seg"])
    rope = m.transformer.rope(b["pos"])
    fracs, state_share = [], []
    for layer in m.transformer.layers:
        att = layer.attention
        h = layer.attention_norm(x)
        B, L, _ = h.shape
        q, k, v = att.qkv(h).view(B, L, 3, att.num_heads, att.head_dim).permute(2, 0, 3, 1, 4)
        q, k = att.q_norm(q), att.k_norm(k)
        q, k = apply_rotary(q, *rope), apply_rotary(k, *rope)
        s = (q[0, :, dec, None] @ k[0].transpose(-1, -2)).squeeze(1) / math.sqrt(att.head_dim)  # [H, L]
        s = s.masked_fill(~mask[0, 0, dec][None], float("-inf"))
        p = torch.softmax(s.float(), -1).mean(0)
        if start is not None:
            span = list(range(start - len(span_ids) // 5, start - len(span_ids) // 5 + len(span_ids)))
            span = [i for i in span if 0 <= i < L and ex.seg[i] == SEG_STATE]
            ps = float(p[state_idx].sum())
            fracs.append(float(p[span].sum()) / max(ps, 1e-9))
            state_share.append(len(span) / max(1, len(state_idx)))
        x = layer(x, rope, mask)
    return fracs, (state_share[0] if state_share else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/verqen-t1-v2/release")
    ap.add_argument("--tag", default="baseline")
    a = ap.parse_args()
    eng = VerqenInferenceEngine.from_checkpoint(a.checkpoint, "auto")
    out_dir = Path("runs/audit") / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = {}

    # ---- distractor ------------------------------------------------------------------
    prose = P.fineweb_sentences(3000)
    dp = P.distractor_probes(prose)
    cells = collections.defaultdict(lambda: {"n": 0, "correct": 0, "p_correct": [], "escalate": 0})
    att_by_cell = collections.defaultdict(list)
    for i, pr in enumerate(dp):
        r = decide(eng, pr["state"], pr["question"], pr["options"])
        c = cells[pr["cell"]]
        c["n"] += 1
        c["correct"] += r["selected"] == pr["answer"]
        c["p_correct"].append(r["p_correct"])
        c["escalate"] += r["should_escalate"]
        if i % 3 == 0:
            d_text = [s for s in pr["state"].split(". ") if "runbook step 4" in s]
            if d_text:
                fr, share = decide_attention(eng, pr["state"], pr["question"], pr["options"], d_text[0])
                if fr:
                    att_by_cell[pr["cell"]].append((fr, share))
    rep["distractor"] = {}
    for cell, c in sorted(cells.items()):
        atts = att_by_cell.get(cell, [])
        rep["distractor"][cell] = {
            "n": c["n"], "accuracy": round(c["correct"] / c["n"], 3), "mean_p_correct": round(float(np.mean(c["p_correct"])), 3),
            "escalation_rate": round(c["escalate"] / c["n"], 3),
            "decide_attention_on_directive_last4": round(float(np.mean([np.mean(f[-4:]) for f, _ in atts])), 3) if atts else None,
            "directive_token_share": round(float(np.mean([s for _, s in atts])), 3) if atts else None,
        }

    # ---- boundary --------------------------------------------------------------------
    bp = P.boundary_probes()
    by_delta = collections.defaultdict(lambda: [0, 0])
    by_tmpl = collections.defaultdict(lambda: [0, 0])
    pos_rate = collections.defaultdict(lambda: [0, 0])
    for pr in bp:
        r = decide(eng, pr["state"], pr["question"], pr["options"])
        ok = r["selected"] == pr["answer"]
        by_delta[pr["cell"]][0] += ok; by_delta[pr["cell"]][1] += 1
        by_tmpl[pr["template"]][0] += ok; by_tmpl[pr["template"]][1] += 1
        pos_rate[pr["cell"]][0] += r["selected"] == pr["options"][0]; pos_rate[pr["cell"]][1] += 1
    rep["boundary"] = {
        "accuracy_by_delta": {k: round(v[0] / v[1], 3) for k, v in sorted(by_delta.items(), key=lambda kv: int(kv[0].split("=")[1]))},
        "rate_first_option_by_delta": {k: round(v[0] / v[1], 3) for k, v in sorted(pos_rate.items(), key=lambda kv: int(kv[0].split("=")[1]))},
        "accuracy_by_template": {k: round(v[0] / v[1], 3) for k, v in by_tmpl.items()},
        "number_tokenization": {str(n): [eng.tokenizer.decode([t]) for t in eng.tokenizer.encode(f" {n}")] for n in (29, 30, 31, 649, 650, 651, 89, 90, 91, 1000)},
    }

    # ---- missing information ---------------------------------------------------------
    mp = P.missing_probes()
    mc = collections.defaultdict(lambda: {"n": 0, "correct": 0, "chose_info": 0, "p_correct": [], "escalate": 0, "entropy": []})
    for pr in mp:
        r = decide(eng, pr["state"], pr["question"], pr["options"])
        c = mc[pr["cell"]]
        c["n"] += 1; c["correct"] += r["selected"] == pr["answer"]; c["chose_info"] += r["selected"] == pr["options"][2]
        c["p_correct"].append(r["p_correct"]); c["escalate"] += r["should_escalate"]
        pv = np.array(list(r["probabilities"].values())); c["entropy"].append(float(-(pv * np.log(pv + 1e-12)).sum() / np.log(len(pv))))
    rep["missing"] = {k: {"n": c["n"], "accuracy": round(c["correct"] / c["n"], 3), "chose_request_info": round(c["chose_info"] / c["n"], 3),
                          "mean_p_correct": round(float(np.mean(c["p_correct"])), 3), "escalation_rate": round(c["escalate"] / c["n"], 3),
                          "mean_norm_entropy": round(float(np.mean(c["entropy"])), 3)} for k, c in mc.items()}

    # ---- counterfactual --------------------------------------------------------------
    cp = P.counterfactual_probes()
    cf = collections.defaultdict(lambda: {"pairs": 0, "flip": 0, "both_correct": 0, "logit_delta": []})
    for pr in cp:
        ra = decide(eng, pr["a"][0], pr["question"], pr["options"])
        rb = decide(eng, pr["b"][0], pr["question"], pr["options"])
        key = pr["pair"].rstrip("0123456789") + "|" + pr["kind"]
        c = cf[key]
        c["pairs"] += 1
        c["flip"] += ra["selected"] != rb["selected"]
        c["both_correct"] += (ra["selected"] == pr["a"][1]) and (rb["selected"] == pr["b"][1])
        o0 = pr["options"][0]
        la = math.log(ra["probabilities"][o0] + 1e-9) - math.log(1 - ra["probabilities"][o0] + 1e-9)
        lb = math.log(rb["probabilities"][o0] + 1e-9) - math.log(1 - rb["probabilities"][o0] + 1e-9)
        c["logit_delta"].append(abs(la - lb))
    rep["counterfactual"] = {k: {"pairs": c["pairs"], "decision_flip_rate": round(c["flip"] / c["pairs"], 3),
                                 "both_correct_rate": round(c["both_correct"] / c["pairs"], 3),
                                 "mean_abs_logit_delta": round(float(np.mean(c["logit_delta"])), 3)} for k, c in cf.items()}
    json.dump(rep, open(out_dir / "probes.json", "w"), indent=2)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
