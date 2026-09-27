"""Tests for the root-cause fixes (docs/audit/ROOT_CAUSES.md): data generators, decontamination,
worst-group risk control, cost-aware escalation, and the guarantees they must not break."""

import json
import os
import random
import re

import numpy as np
import pytest
import torch

from verqen.calibration.fit import TYPES, risk_thresholds, worst_group_risk_control
from verqen.config import ModelConfig
from verqen.data.procedural.discourse import ACTIONS as DISC_ACTIONS
from verqen.data.procedural.discourse import discourse_families
from verqen.data.procedural.rules import HOLDS, NUMERIC, RELATIONS, ROLE_SETS, rule_families
from verqen.inference.engine import VerqenInferenceEngine
from verqen.model.verqen import VerqenModel
from verqen.tokenizer import VerqenTokenizer

TOK = "data/tokenizer/verqen_32k.json"
BENCH = "verqen_benchmark/dataset/benchmark_cases.jsonl"


def _label(ex):
    q = ex["questions"][0]
    return q["options"][q["target"].index(max(q["target"]))]


# ── Fix A: rule curriculum ──────────────────────────────────────────────────────────
def test_rule_families_are_counterfactual_and_share_options():
    rng = random.Random(0)
    for _ in range(300):
        fam = rule_families(rng, 1)
        assert len(fam) == 4
        base, flip, irr, info = fam
        opts = {frozenset(e["questions"][0]["options"]) for e in fam}
        assert len(opts) == 1, "every variant must offer the same option set"
        assert _label(base) != _label(flip), "moving the decisive variable across the rule must flip the label"
        assert _label(base) == _label(irr), "an irrelevant change must not change the label"
        assert _label(info) not in (_label(base), _label(flip)), "missing/unknown/conflicting info -> gather option"
        assert base["state"] != irr["state"]


def test_numeric_rule_labels_are_computed_from_the_stated_rule():
    """Metamorphic check: parse the threshold, relation and value back out of the text and recompute."""
    rel_re = {k: [re.escape(p).replace(r"\{T\}", r"(\d+)").replace(r"\{u\}", r"[\w/ ]+?") for p in v] for k, v in RELATIONS.items()}
    rng, checked = random.Random(1), 0
    while checked < 200:
        fam = rule_families(rng, 1)
        if fam[0]["family"] != "rule_numeric":
            continue
        for ex in fam[:2]:  # base and flip have a single clean value
            st = ex["state"]
            scen = next(s for s in NUMERIC if s[0] in st and s[5][0] in ex["questions"][0]["options"])
            # earliest match wins, so "no more than" is not read as "more than"
            hits = [(m.start(), -len(m.group(0)), k, m) for k, pats in rel_re.items() for p in pats for m in [re.search(p, st)] if m]
            assert hits, st
            _, _, relk, m = min(hits, key=lambda h: h[:2])
            T = int(m.group(1))
            rest = st.replace(m.group(0), "")
            v = int(re.search(rf"(\d+) {re.escape(scen[1])}", rest).group(1))
            want = scen[5][0] if HOLDS[relk](v, T) else scen[5][1]
            assert _label(ex) == want, (st, _label(ex))
            checked += 1


def test_role_rule_labels():
    rng = random.Random(2)
    n = 0
    while n < 100:
        fam = rule_families(rng, 1)
        if fam[0]["family"] != "rule_role":
            continue
        allowed, others, act, (ok, bad, info) = next(r for r in ROLE_SETS if r[3][0] in fam[0]["questions"][0]["options"])
        for ex in fam[:3]:
            # the case sentence names exactly one role; the rule sentence names the allowed roles
            case = [s for s in re.split(r"(?<=[.])\s+", ex["state"]) if "asks" in s or "wants" in s or "request" in s][0]
            is_allowed = any(re.search(rf"\b{re.escape(r)}\b", case) for r in allowed)
            assert _label(ex) == (ok if is_allowed else bad), ex["state"]
        n += 1


# ── Fix B: role-aware distractors ─────────────────────────────────────────────────
def test_discourse_twins_move_the_label_with_the_directive():
    rng = random.Random(3)
    for _ in range(300):
        a, b = discourse_families(rng, 1, prose=["Filler sentence one about nothing in particular here.",
                                                 "Another unrelated sentence for padding the state out."])
        assert set(a["questions"][0]["options"]) == set(b["questions"][0]["options"])
        assert _label(a) != _label(b)
        for ex in (a, b):
            phrase = next(p for p, oid in DISC_ACTIONS if oid == _label(ex))
            assert phrase in ex["state"], "the correct action is always named by the directive"
            # at least one wrong option is also mentioned (bait is always present)
            wrong = [p for p, oid in DISC_ACTIONS if oid in ex["questions"][0]["options"] and oid != _label(ex)]
            assert any(p in ex["state"] for p in wrong)


# ── decontamination ──────────────────────────────────────────────────────────────
@pytest.mark.skipif(not os.path.exists(BENCH), reason="benchmark not present")
def test_fix_vocabulary_is_disjoint_from_benchmark_and_probes():
    import sys
    sys.path.insert(0, "scripts/audit")
    import probes as P
    bench = set()
    for line in open(BENCH, encoding="utf-8"):
        c = json.loads(line)
        bench |= set(c.get("allowed_decisions") or [])
        for q in c["input"]["questions"]:
            bench |= set(q.get("options") or [])
    probe = {a[1] for a in P.ACTIONS}
    for pr in P.boundary_probes() + P.missing_probes() + P.missing_probes_v2():
        probe |= set(pr["options"])
    ours = {oid for _, oid in DISC_ACTIONS} | {o for s in NUMERIC for o in s[5]} | {o for r in ROLE_SETS for o in r[3]}
    assert not (ours & bench), sorted(ours & bench)
    assert not (ours & probe), sorted(ours & probe)


# ── Fix C: worst-group risk control ──────────────────────────────────────────────────
def _synthetic(seed=0):
    """Group 'easy' (95% right) and group 'hard' (60% right); the score is informative in both."""
    rng = np.random.default_rng(seed)
    n_e, n_h = 3000, 1500
    y = np.concatenate([rng.random(n_e) < 0.95, rng.random(n_h) < 0.6]).astype(float)
    groups = np.array(["easy"] * n_e + ["hard"] * n_h)
    score = np.clip(0.5 + 0.35 * (y - 0.5) + rng.normal(0, 0.12, len(y)) + 0.15 * (groups == "easy"), 0.01, 0.99)
    # express as two-option outputs whose calibrated max-prob equals the score and whose target is correct iff y
    logp = np.log(np.stack([score, 1 - score], -1))
    target = np.stack([y, 1 - y], -1)
    mask = np.ones_like(target, dtype=bool)
    qtype = np.full(len(y), TYPES.index("choice"))
    sel = np.log(score / (1 - score))
    calib = {"temperature": {t: 1.0 for t in TYPES}, "selective_platt": [1.0, 0.0]}
    return {"logp": logp, "target": target, "mask": mask, "qtype": qtype, "sel": sel}, groups, calib, y, score


def test_worst_group_threshold_controls_the_hard_group():
    outs, groups, calib, y, score = _synthetic()
    wg = worst_group_risk_control(outs, groups, calib, risks=(0.1,), delta=0.05, min_group=150)
    thr = wg["by_risk"]["0.1"]["threshold"]
    pooled = risk_thresholds(score, y, risks=(0.1,), delta=0.05)["0.1"]["threshold"]
    assert thr >= pooled, "worst-group control can only be more conservative than the pooled one"
    hard_kept = (groups == "hard") & (score >= thr)
    if hard_kept.any():
        assert 1 - y[hard_kept].mean() <= 0.1 + 0.03
    pooled_hard = (groups == "hard") & (score >= pooled)
    assert 1 - y[pooled_hard].mean() > 1 - y[hard_kept].mean() if hard_kept.any() else True


def test_worst_group_on_unachievable_risk_escalates_everything():
    outs, groups, calib, y, score = _synthetic(1)
    wg = worst_group_risk_control(outs, groups, calib, risks=(0.001,), delta=0.05)
    assert wg["by_risk"]["0.001"]["threshold"] > 1.0


# ── Fix D and API defaults ─────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def engine():
    tok = VerqenTokenizer.from_pretrained(TOK)
    torch.manual_seed(0)
    cfg = ModelConfig(hidden_size=64, num_layers=2, num_heads=2, head_dim=32, ffn_size=128, vocab_size=tok.vocab_size, set_layers=1)
    calib = {"temperature": {t: 1.0 for t in TYPES}, "selective_platt": [1.0, 0.0], "default_risk": "0.1",
             "risk_control": {"delta": 0.05, "by_risk": {"0.1": {"threshold": 0.0}}}}
    return VerqenInferenceEngine(VerqenModel(cfg).eval(), tok, "cpu", 512, calib)


STATE = "The courier left the parcel at the wrong address and the customer wants it redelivered."
OPTS = ["redeliver", "refund", "ignore"]


def test_default_output_is_unchanged_without_cost(engine):
    r = engine.decide(STATE, "What should support do?", OPTS)
    assert set(r) == {"type", "selected", "probabilities", "p_correct", "margin", "should_escalate", "confidence"}
    assert r["should_escalate"] is False  # threshold 0.0


def test_cost_aware_escalation_uses_expected_loss(engine):
    base = engine.decide(STATE, "What should support do?", OPTS)
    p = base["p_correct"]
    hi = engine.decide(STATE, "What should support do?", OPTS, cost={"error_cost": 100.0, "escalation_cost": 0.5})
    lo = engine.decide(STATE, "What should support do?", OPTS, cost={"error_cost": 1.0, "escalation_cost": 5.0})
    assert hi["selected"] == base["selected"] and hi["probabilities"] == base["probabilities"], "cost never changes the choice"
    assert hi["should_escalate"] == ((1 - p) * 100.0 > 0.5)
    assert lo["should_escalate"] is False
    assert abs(hi["expected_loss_if_acting"] - (1 - p) * 100.0) < 100 * 1e-4  # p_correct is rounded to 4 dp


def test_worst_group_mode_requires_its_calibration(engine):
    with pytest.raises(ValueError):
        VerqenInferenceEngine(engine.model, engine.tokenizer, "cpu", 512, engine.calibration, risk_mode="worst_group")
    with pytest.raises(ValueError):
        VerqenInferenceEngine(engine.model, engine.tokenizer, "cpu", 512, engine.calibration, risk_mode="bogus")


def test_option_order_invariance_holds_with_cost(engine):
    a = engine.decide(STATE, "What should support do?", OPTS, cost={"error_cost": 10, "escalation_cost": 1})
    b = engine.decide(STATE, "What should support do?", OPTS[::-1], cost={"error_cost": 10, "escalation_cost": 1})
    for o in OPTS:
        assert abs(a["probabilities"][o] - b["probabilities"][o]) < 1e-5
    assert a["should_escalate"] == b["should_escalate"]


def test_determinism_repeated_calls(engine):
    outs = [engine.decide(STATE, "What should support do?", OPTS) for _ in range(5)]
    assert all(o == outs[0] for o in outs)


def test_group_conditional_threshold(engine):
    cal = dict(engine.calibration)
    cal["risk_control_worst_group"] = {"score": "p_correct", "groups": {"easy": 500, "hard": 500},
                                       "by_risk": {"0.1": {"threshold": 1.01, "per_group_threshold": {"easy": 0.3, "hard": 1.01}}}}
    worst = VerqenInferenceEngine(engine.model, engine.tokenizer, "cpu", 512, cal, risk_mode="worst_group")
    easy = VerqenInferenceEngine(engine.model, engine.tokenizer, "cpu", 512, cal, risk_mode="worst_group", risk_domain="easy")
    hard = VerqenInferenceEngine(engine.model, engine.tokenizer, "cpu", 512, cal, risk_mode="worst_group", risk_domain="hard")
    assert worst.threshold == 1.01 and easy.threshold == 0.3 and hard.threshold == 1.01
    assert hard.decide(STATE, "What should support do?", OPTS)["should_escalate"] is True
    with pytest.raises(ValueError):
        VerqenInferenceEngine(engine.model, engine.tokenizer, "cpu", 512, cal, risk_mode="worst_group", risk_domain="unseen")
    with pytest.raises(ValueError):
        VerqenInferenceEngine(engine.model, engine.tokenizer, "cpu", 512, cal, risk_domain="easy")
