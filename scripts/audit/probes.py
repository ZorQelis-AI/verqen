"""Controlled diagnostic probes for VerQen-T1 (independent of the Artifact-G benchmark).

Every probe is generated here from generic templates plus random FineWeb-Edu prose. No
benchmark text, template or label is used, so these probes can guide engineering without
leaking the locked test set.

Families (each yields paired variants so effects can be isolated):
  distractor   directive clause + k distractor sentences; varies position of the directive,
               k, whether distractors *name* the wrong options (lexical bait), and duplication
  boundary     "requires value OP T" with value swept over T-5..T+5, several phrasings/units
  missing      complete case vs decisive variable removed / replaced by "unknown" / contradicted
  counterfact  minimal pairs: change the decisive variable vs change an irrelevant variable
"""

from __future__ import annotations

import glob
import random
import re
from typing import Any

import pyarrow.parquet as pq

ACTIONS = [
    ("restart the payment service", "restart_payment_service"),
    ("rotate the leaked API keys", "rotate_api_keys"),
    ("roll back the latest deployment", "rollback_deployment"),
    ("freeze the customer account", "freeze_account"),
    ("page the database on-call engineer", "page_db_oncall"),
    ("quarantine the infected laptop", "quarantine_laptop"),
    ("issue a full refund", "issue_full_refund"),
    ("reorder stock from the backup supplier", "reorder_from_backup_supplier"),
    ("schedule a structural inspection", "schedule_inspection"),
    ("disable the compromised user", "disable_user"),
    ("renew the TLS certificate", "renew_tls_certificate"),
    ("evacuate the east wing", "evacuate_east_wing"),
]


PROBE_SHARD = "sample/10BT/002_00000.parquet"  # pinned: every audit number so far used this shard's prose


def probe_prose_path() -> str:
    """The pinned FineWeb-Edu shard, fetched by name if this machine lacks it (e.g. the rented box)."""
    import os
    for cand in ("data/pretrain/raw/" + PROBE_SHARD, "data/pretrain/probe_shard/" + PROBE_SHARD):
        if os.path.exists(cand):
            return cand
    from huggingface_hub import hf_hub_download
    return hf_hub_download("HuggingFaceFW/fineweb-edu", PROBE_SHARD, repo_type="dataset", local_dir="data/pretrain/probe_shard")


def fineweb_sentences(n: int = 4000, seed: int = 0) -> list[str]:
    path = probe_prose_path()
    pf = pq.ParquetFile(path)
    out = []
    rng = random.Random(seed)
    for batch in pf.iter_batches(batch_size=512, columns=["text"]):
        for t in batch.column(0).to_pylist():
            for s in re.split(r"(?<=[.!?])\s+", t):
                s = s.strip()
                if 60 <= len(s) <= 200 and not re.search(r"\d{3,}", s):
                    out.append(s)
        if len(out) >= n * 3:
            break
    rng.shuffle(out)
    return out[:n]


def distractor_probes(prose: list[str], n_per_cell: int = 30, seed: int = 1) -> list[dict[str, Any]]:
    """Cells: k in {0,1,3,6} x position in {first, middle, last} x bait in {none, named}."""
    rng = random.Random(seed)
    probes = []
    for k in (0, 1, 3, 6):
        for position in ("first", "middle", "last"):
            for bait in ("none", "named"):
                if k == 0 and (position != "first" or bait != "none"):
                    continue
                for i in range(n_per_cell):
                    right, *wrong = rng.sample(ACTIONS, 4)
                    directive = f"Incident runbook step 4 requires the operator to {right[0]} now."
                    fillers = rng.sample(prose, k)
                    if bait == "named":  # distractor sentences mention the wrong actions, as history/noise
                        fillers = [f"{s} Last quarter someone suggested we {w[0]} at some point." if j < len(wrong) else s
                                   for j, (s, w) in enumerate(zip(fillers, wrong + wrong))]
                    if position == "first":
                        sents = [directive] + fillers
                    elif position == "last":
                        sents = fillers + [directive]
                    else:
                        sents = fillers[: k // 2] + [directive] + fillers[k // 2:]
                    opts = [right[1]] + [w[1] for w in wrong]
                    rng.shuffle(opts)
                    probes.append({"probe": "distractor", "cell": f"k={k}|pos={position}|bait={bait}",
                                   "state": " ".join(sents), "question": "Which action does the runbook require now?",
                                   "options": opts, "answer": right[1]})
    return probes


BOUNDARY_TEMPLATES = [
    ("Returns are accepted within {T} days of delivery. This request arrives {v} days after delivery.",
     "Is the return eligible?", ["eligible", "not_eligible"], lambda v, T: v <= T),
    ("The loan requires a credit score of at least {T}. The applicant's score is {v}.",
     "Does the applicant meet the requirement?", ["meets_requirement", "below_requirement"], lambda v, T: v >= T),
    ("The alarm must fire when the reactor temperature exceeds {T} degrees. The sensor reads {v} degrees.",
     "Should the alarm fire?", ["fire_alarm", "stay_silent"], lambda v, T: v > T),
    ("A vessel needs at least {T} lifeboat seats. This vessel carries {v} seats.",
     "Is the vessel compliant?", ["compliant", "non_compliant"], lambda v, T: v >= T),
]


def boundary_probes(seed: int = 2) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    probes = []
    for ti, (tmpl, q, opts, rule) in enumerate(BOUNDARY_TEMPLATES):
        for T in rng.sample([12, 30, 45, 90, 120, 250, 600, 650, 720, 1000], 6):
            for d in range(-5, 6):
                v = T + d
                ok = rule(v, T)
                probes.append({"probe": "boundary", "cell": f"delta={d:+d}", "template": ti, "T": T, "v": v,
                               "state": tmpl.format(T=T, v=v), "question": q, "options": list(opts),
                               "answer": opts[0] if ok else opts[1]})
    return probes


MISSING_CASES = [
    # (complete state, missing, unknown, contradictory, question, options, complete_answer, info_option)
    ("Warranty covers defects within 12 months of purchase. The laptop was bought {m} months ago and the screen is defective.",
     "Warranty covers defects within 12 months of purchase. The screen of the laptop is defective.",
     "Warranty covers defects within 12 months of purchase. The purchase date of the laptop is unknown and the screen is defective.",
     "Warranty covers defects within 12 months of purchase. The receipt says the laptop was bought 3 months ago but the store record says 20 months ago. The screen is defective.",
     "What should the support agent do?", ["approve_warranty_repair", "deny_warranty", "request_proof_of_purchase_date"],
     lambda m: "approve_warranty_repair" if m <= 12 else "deny_warranty", "request_proof_of_purchase_date"),
    ("Deliveries over 20 kg need a freight truck. The parcel weighs {m} kg.",
     "Deliveries over 20 kg need a freight truck. The parcel is ready at the dock.",
     "Deliveries over 20 kg need a freight truck. The parcel weight was not recorded.",
     "Deliveries over 20 kg need a freight truck. The label says 5 kg but the dock scale shows 40 kg.",
     "Which vehicle should be booked?", ["book_van", "book_freight_truck", "weigh_parcel_first"],
     lambda m: "book_freight_truck" if m > 20 else "book_van", "weigh_parcel_first"),
    ("Visitors under {x} years need a guardian; the rule is 16 years. The visitor is {m} years old.",
     "The rule says visitors under 16 years need a guardian. A visitor arrives alone.",
     "The rule says visitors under 16 years need a guardian. The visitor's age is not known.",
     "The rule says visitors under 16 years need a guardian. The visitor says they are 19 but their ID shows 12.",
     "What should the front desk do?", ["admit_visitor", "require_guardian", "verify_age_first"],
     lambda m: "require_guardian" if m < 16 else "admit_visitor", "verify_age_first"),
]


MISSING_RULES = [
    # (rule, value sentence, variable phrase, question, [ok, not_ok, info], ok(v), values)
    ("Orders above $500 require a second approver.", "The order total is ${v}.", "the order total",
     "How should the order be processed?", ["process_with_one_approver", "get_second_approver", "confirm_order_total"], lambda v: v <= 500, [120, 480, 520, 900]),
    ("Drivers must have at least 2 years of experience to drive the bus route.", "The applicant has {v} years of driving experience.",
     "the applicant's driving experience", "Can the applicant be assigned to the bus route?", ["assign_route", "reject_assignment", "check_driving_record"],
     lambda v: v >= 2, [1, 3, 6, 0]),
    ("Medication doses above 40 mg need a pharmacist review.", "The prescribed dose is {v} mg.", "the prescribed dose",
     "What should the nurse do?", ["administer_dose", "request_pharmacist_review", "confirm_prescribed_dose"], lambda v: v <= 40, [10, 35, 45, 80]),
    ("Servers running hotter than 85 degrees must be shut down.", "Server 7 reports {v} degrees.", "the server temperature",
     "What should the operator do with server 7?", ["keep_running", "shut_down_server", "read_temperature_again"], lambda v: v <= 85, [60, 80, 90, 99]),
    ("Loans need a credit score of at least 650.", "The applicant's credit score is {v}.", "the applicant's credit score",
     "What should the loan officer do?", ["approve_loan", "decline_loan", "pull_credit_report"], lambda v: v >= 650, [560, 640, 700, 780]),
]


def missing_probes_v2(seed: int = 5) -> list[dict[str, Any]]:
    """Generic rule cases: complete / value missing / value unknown / two conflicting values."""
    rng = random.Random(seed)
    probes = []
    for ci, (rule, val, var, q, opts, ok, values) in enumerate(MISSING_RULES):
        filler = "The request was filed this morning by the operations team."
        for v in values:
            probes.append({"probe": "missing", "cell": "complete", "case": ci, "state": f"{rule} {val.format(v=v)} {filler}",
                           "question": q, "options": list(opts), "answer": opts[0] if ok(v) else opts[1]})
        a, b = rng.sample(values, 2)
        while ok(a) == ok(b):
            a, b = rng.sample(values, 2)
        for cell, st in (("missing", f"{rule} {filler}"),
                         ("unknown", f"{rule} {var[0].upper() + var[1:]} has not been recorded. {filler}"),
                         ("contradictory", f"{rule} One system shows {var} as {a}, another shows {b}. {filler}")):
            probes.append({"probe": "missing", "cell": cell, "case": ci, "state": st, "question": q,
                           "options": list(opts), "answer": opts[2]})
    return probes


def missing_probes(seed: int = 3) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    probes = []
    for ci, (full, miss, unk, contra, q, opts, ans, info) in enumerate(MISSING_CASES):
        for _ in range(8):
            m = rng.choice([2, 5, 8, 14, 19, 25, 30, 45])
            probes.append({"probe": "missing", "cell": "complete", "case": ci, "state": full.format(m=m, x=16), "question": q,
                           "options": list(opts), "answer": ans(m)})
        for cell, st in (("missing", miss), ("unknown", unk), ("contradictory", contra)):
            for _ in range(8):
                probes.append({"probe": "missing", "cell": cell, "case": ci, "state": st, "question": q,
                               "options": list(opts), "answer": info})
    return probes


def counterfactual_probes(seed: int = 4) -> list[dict[str, Any]]:
    """Pairs (a, b): b changes only the decisive variable (answer flips) or only an irrelevant one."""
    rng = random.Random(seed)
    out = []
    names = ["Priya", "Lucas", "Amara", "Kenji", "Sofia", "Omar"]
    for i in range(40):
        role_ok, role_bad = rng.sample(["auditor", "admin", "analyst", "intern", "contractor", "guest"], 2)
        name, name2 = rng.sample(names, 2)
        base = "Only the {role_ok} role may read the audit log. {name} has the {role} role and asks to read the audit log."
        q, opts = "Should access be granted?", ["grant_access", "deny_access"]
        a = base.format(role_ok=role_ok, name=name, role=role_ok)
        b_rel = base.format(role_ok=role_ok, name=name, role=role_bad)
        b_irr = base.format(role_ok=role_ok, name=name2, role=role_ok)
        out.append({"pair": f"role{i}", "kind": "relevant", "a": (a, "grant_access"), "b": (b_rel, "deny_access"), "question": q, "options": opts})
        out.append({"pair": f"role{i}", "kind": "irrelevant", "a": (a, "grant_access"), "b": (b_irr, "grant_access"), "question": q, "options": opts})
        T = rng.choice([30, 60, 90])
        v_in, v_out = rng.randint(5, T - 10), rng.randint(T + 10, T + 60)
        base2 = "The response target for urgent tickets is {T} minutes. Ticket {tid} has been open for {v} minutes."
        q2, opts2 = "Is the response target breached?", ["target_breached", "within_target"]
        tid = rng.randint(100, 999)  # same ticket id in both halves: only the decisive variable may change
        a2 = base2.format(T=T, tid=tid, v=v_in)
        out.append({"pair": f"sla{i}", "kind": "relevant", "a": (a2, "within_target"),
                    "b": (base2.format(T=T, tid=tid, v=v_out), "target_breached"), "question": q2, "options": opts2})
        out.append({"pair": f"sla{i}", "kind": "irrelevant", "a": (a2, "within_target"),
                    "b": (a2.replace("urgent tickets", "urgent support tickets"), "within_target"), "question": q2, "options": opts2})
    return out
