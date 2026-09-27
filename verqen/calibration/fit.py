"""Stage 4: post-hoc calibration on a held-out calibration split (never trained on).

Produces ``calibration.json`` in the checkpoint:

* ``temperature`` per question type, fitted by NLL.
* ``selective_platt``: (a, b) mapping the selective-head logit to P(top-1 correct).
* ``conformal``: split-conformal threshold for prediction sets with >= 1 - alpha coverage.
* ``risk_control``: for each target error rate r, the confidence threshold whose
  accepted decisions have error <= r with probability >= 1 - delta (Clopper-Pearson
  upper bound, fixed-sequence testing). Decisions below the threshold are escalated.
  This is what turns a small model plus a frontier fallback into a system with a
  *guaranteed* error rate on the decisions it keeps.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from scipy.stats import beta as beta_dist

from verqen.calibration.metrics import expected_calibration_error
from verqen.data.collator import collate_rows, pack_rows, to_device
from verqen.data.store import PackedStore
from verqen.losses.decision import correctness
from verqen.model.attention_mask import build_attention_mask
from verqen.training.common import autocast
from verqen.training.decisions import token_budget_batches

TYPES = ["choice", "score", "noul"]


@torch.no_grad()
def collect_outputs(model, store: PackedStore, device, max_len: int = 512, rows: int = 32,
                    limit: int | None = None) -> dict[str, np.ndarray]:
    """Run the model over a store; returns padded logp/target/mask arrays in store order."""
    model.eval()
    n = len(store) if limit is None else min(limit, len(store))
    logps, targets, masks, qtypes, sels, ex_ids = [], [], [], [], [], []
    for idx in token_budget_batches(store, rows, max_len, 0, shuffle=False, epochs=1):
        idx = [i for i in idx if i < n]
        if not idx:
            break
        exs = [store[i] for i in idx]
        packed, _ = pack_rows(exs, max_len, None)
        order = [e for row in packed for e in row]
        ex_index = {id(e): i for i, e in zip(idx, exs)}
        b = to_device(collate_rows(packed, max_len, 0), device)
        with autocast(device):
            mask = build_attention_mask(b["seg"], b["doc"], b["qid"], b["oid"])
            h = model.transformer(b["input_ids"], b["seg"], b["pos"], mask)
            out = model.decision_readout(h, b)
        O = 256  # pad width across batches; must cover the widest option set (Banking77 has 77)
        lp = torch.full((len(b["qtype"]), O), float("-inf"))
        lp[:, : out["logp"].size(1)] = out["logp"].float().cpu()
        tg = torch.zeros(len(b["qtype"]), O)
        tg[:, : b["target"].size(1)] = b["target"].cpu()
        mk = torch.zeros(len(b["qtype"]), O, dtype=torch.bool)
        mk[:, : b["opt_mask"].size(1)] = b["opt_mask"].cpu()
        logps.append(lp); targets.append(tg); masks.append(mk)
        qtypes.append(b["qtype"].cpu()); sels.append(out["selective_logit"].float().cpu())
        for e in order:
            ex_ids += [ex_index[id(e)]] * len(e.questions)
        if max(idx) >= n - 1:
            break
    model.train()
    return {"logp": torch.cat(logps).numpy(), "target": torch.cat(targets).numpy(), "mask": torch.cat(masks).numpy(),
            "qtype": torch.cat(qtypes).numpy(), "sel": torch.cat(sels).numpy(), "example": np.asarray(ex_ids)}


def _softmax_T(logp: np.ndarray, mask: np.ndarray, T: float) -> np.ndarray:
    z = np.where(mask, logp / T, -np.inf)
    z = z - z.max(-1, keepdims=True)
    e = np.where(mask, np.exp(z), 0.0)
    return e / e.sum(-1, keepdims=True)


def fit_temperature(logp, target, mask) -> float:
    best_T, best = 1.0, float("inf")
    for T in np.exp(np.linspace(math.log(0.3), math.log(5.0), 60)):
        p = _softmax_T(logp, mask, T)
        nll = -(target * np.log(np.clip(p, 1e-12, 1))).sum(-1).mean()
        if nll < best:
            best, best_T = nll, float(T)
    return best_T


def fit_platt(logit: np.ndarray, y: np.ndarray, iters: int = 500) -> tuple[float, float]:
    x = torch.tensor(logit, dtype=torch.float64)
    t = torch.tensor(y, dtype=torch.float64)
    a = torch.ones(1, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([a, b], max_iter=iters)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(a * x + b, t)
        loss.backward()
        return loss
    opt.step(closure)
    return float(a), float(b)


def cp_upper(k: int, n: int, delta: float) -> float:
    """Clopper-Pearson one-sided upper bound on an error rate with k errors in n trials."""
    if n == 0:
        return 1.0
    return 1.0 if k >= n else float(beta_dist.ppf(1 - delta, k + 1, n - k))


def risk_thresholds(conf: np.ndarray, correct: np.ndarray, risks=(0.02, 0.05, 0.1, 0.2), delta: float = 0.05):
    """Fixed-sequence (Learn-then-Test style) scan from the most to least confident threshold."""
    order = np.argsort(-conf)
    c_sorted, err_sorted = conf[order], 1 - correct[order]
    cum_err = np.cumsum(err_sorted)
    out = {}
    for r in risks:
        best = None
        for k in range(len(order)):
            if k + 1 < 20:
                continue
            if cp_upper(int(cum_err[k]), k + 1, delta) <= r:
                best = k
            else:
                if best is not None:
                    break  # fixed sequence: stop at the first failure after successes
        if best is None:
            out[str(r)] = {"threshold": 1.01, "coverage": 0.0, "empirical_error": None}
        else:
            out[str(r)] = {"threshold": float(c_sorted[best]), "coverage": round((best + 1) / len(order), 4),
                           "empirical_error": round(float(cum_err[best] / (best + 1)), 4)}
    return out


def calibrate(outputs: dict[str, np.ndarray], alpha: float = 0.1, delta: float = 0.05) -> dict[str, Any]:
    logp, target, mask, qtype = outputs["logp"], outputs["target"], outputs["mask"], outputs["qtype"]
    temps, probs = {}, np.zeros_like(target)
    for t_id, name in enumerate(TYPES):
        sel = qtype == t_id
        if sel.sum() < 30:
            temps[name] = 1.0
        else:
            temps[name] = fit_temperature(logp[sel], target[sel], mask[sel])
        if sel.any():
            probs[sel] = _softmax_T(logp[sel], mask[sel], temps[name])
    correct = correctness(torch.tensor(probs), torch.tensor(target), torch.tensor(mask)).numpy()
    a, b = fit_platt(outputs["sel"], correct)
    p_correct = 1 / (1 + np.exp(-(a * outputs["sel"] + b)))

    # Split conformal (LAC score = 1 - p(true label)), hard label = argmax target
    y = target.argmax(-1)
    scores = 1 - probs[np.arange(len(y)), y]
    n = len(scores)
    q_level = min(1.0, math.ceil((n + 1) * (1 - alpha)) / n)
    qhat = float(np.quantile(scores, q_level, method="higher"))
    set_sizes = ((probs >= 1 - qhat) & mask).sum(-1)

    raw = np.where(mask, np.exp(logp), 0.0)
    report = {
        "n": int(n),
        "ece_raw": expected_calibration_error(raw, target, mask),
        "ece_calibrated": expected_calibration_error(probs, target, mask),
        "selective_auroc": _auroc(p_correct, correct),
        "conformal_avg_set_size": float(set_sizes.mean()),
    }
    return {
        "temperature": temps,
        "selective_platt": [a, b],
        "conformal": {"alpha": alpha, "qhat": qhat},
        "risk_control": {"delta": delta, "by_risk": risk_thresholds(p_correct, correct, delta=delta)},
        "default_risk": "0.1",
        "calibration_report": report,
    }


def worst_group_risk_control(outputs: dict[str, np.ndarray], groups: np.ndarray, calib: dict[str, Any],
                             risks=(0.02, 0.05, 0.1, 0.2), delta: float = 0.05, min_group: int = 150) -> dict[str, Any]:
    """Fix C (ROOT_CAUSES RC-4): escalation thresholds that bound risk on *every* calibration group.

    The default ``risk_control`` bounds the risk of the pooled calibration mixture, so easy groups
    (procedural, MNLI) subsidise hard ones: measured accuracy-on-kept at a target risk of 0.1 was
    0.34 on MMLU. Here each group g with n_g >= ``min_group`` gets its own fixed-sequence
    Clopper-Pearson threshold at level delta / G (Bonferroni over groups), and the deployed
    threshold is the maximum over groups. That is exact for each group's own threshold and holds
    at the max under the usual assumption that risk does not increase as confidence rises; the
    per-group empirical risk at the chosen threshold is recorded so the assumption is checkable.
    Groups for which no threshold reaches the target force ``threshold > 1`` (always escalate):
    at that risk level the model cannot certify inputs like them.

    The escalation score is chosen, on this calibration split only, as whichever of the selective
    head (Platt p_correct) or the calibrated max-probability has the higher *worst-group* AUROC.
    """
    probs, p_correct = apply_calibration(torch.tensor(outputs["logp"]), torch.tensor(outputs["mask"]),
                                         torch.tensor(outputs["qtype"]), torch.tensor(outputs["sel"]), calib)
    probs, p_correct = probs.numpy(), p_correct.numpy()
    correct = correctness(torch.tensor(probs), torch.tensor(outputs["target"]), torch.tensor(outputs["mask"])).numpy()
    scores = {"p_correct": p_correct, "max_prob": np.where(outputs["mask"], probs, 0).max(-1)}
    names, counts = np.unique(groups, return_counts=True)
    big = [g for g, n in zip(names, counts) if n >= min_group]
    auroc = {k: {str(g): _auroc(v[groups == g], correct[groups == g]) for g in big} for k, v in scores.items()}
    worst = {k: float(np.nanmin(list(a.values()))) for k, a in auroc.items()}
    score_name = max(worst, key=worst.get)
    conf = scores[score_name]
    d_g = delta / max(1, len(big))
    by_risk = {}
    for r in risks:
        per = {str(g): risk_thresholds(conf[groups == g], correct[groups == g], risks=(r,), delta=d_g)[str(r)]["threshold"] for g in big}
        thr = max(per.values())
        kept = conf >= thr
        by_risk[str(r)] = {
            "threshold": float(thr), "per_group_threshold": per, "coverage": round(float(kept.mean()), 4),
            "per_group_coverage": {str(g): round(float(kept[groups == g].mean()), 4) for g in big},
            "per_group_empirical_error": {str(g): (round(float(1 - correct[(groups == g) & kept].mean()), 4)
                                                   if kept[groups == g].any() else None) for g in big},
        }
    return {"score": score_name, "delta": delta, "min_group": min_group, "groups": {str(g): int(n) for g, n in zip(names, counts)},
            "worst_group_auroc": worst, "auroc_by_group": auroc, "by_risk": by_risk}


def _auroc(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg])))
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg)))


def apply_calibration(logp: torch.Tensor, mask: torch.Tensor, qtype: torch.Tensor, sel_logit: torch.Tensor,
                      calib: dict[str, Any] | None):
    """Return calibrated probs and P(correct). Works on torch tensors (inference path)."""
    if not calib:
        return logp.exp(), torch.sigmoid(sel_logit)
    temp_dict = calib.get("temperature", {})
    T = torch.tensor([temp_dict.get(t, 1.0) for t in TYPES], device=logp.device)[qtype]
    z = (logp / T[:, None]).masked_fill(~mask, float("-inf"))
    probs = torch.softmax(z, -1)
    platt = calib.get("selective_platt", (1.0, 0.0))
    a, b = platt[0], platt[1]
    return probs, torch.sigmoid(a * sel_logit + b)
