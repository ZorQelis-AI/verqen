"""Fix A: compositional rule curriculum emitted as counterfactual families (ROOT_CAUSES RC-2, RC-3).

The v2 data taught each rule type in roughly one fixed template (refund thresholds only in
``refund_policy``; no role/permission rules; no operator paraphrases), so the model learned
template-specific features. Here the rule is *composed*:

    predicate  = numeric threshold (4 relations x several phrasings each) | role membership
    rule       = "<subject> <requirement>" or "if <condition>, <consequence>"
    case facts = one fact per variable, in random order, plus optional unrelated filler

Every sampled rule is emitted as a family of minimal variants, the counterfactual structure
the model otherwise never sees:

    base         decisive variable on a random side of the rule
    flip         decisive variable moved across the rule; the label flips (often by 1-3 units)
    irrelevant   a non-decisive detail changed (name / id / filler); the label is unchanged
    info         decisive variable missing / "not recorded" / two conflicting values;
                 the label is the information-gathering option

All variants offer the same three options (act, don't act, gather information), so neither
the option set nor the mere presence of the information option predicts the label.
"""

from __future__ import annotations

import random
from typing import Any

from .common import FIRST_NAMES, choice_q, example, noul_q

# Relation phrasings: (text template, relation). {T} threshold, {u} unit.
RELATIONS = {
    ">=": ["at least {T} {u}", "no less than {T} {u}", "a minimum of {T} {u}", "{T} {u} or more"],
    ">": ["more than {T} {u}", "over {T} {u}", "above {T} {u}", "in excess of {T} {u}"],
    "<=": ["at most {T} {u}", "no more than {T} {u}", "a maximum of {T} {u}", "{T} {u} or less"],
    "<": ["less than {T} {u}", "under {T} {u}", "below {T} {u}", "fewer than {T} {u}"],
}
HOLDS = {">=": lambda v, T: v >= T, ">": lambda v, T: v > T, "<=": lambda v, T: v <= T, "<": lambda v, T: v < T}

# Numeric scenarios: variable phrase, unit, threshold pool, subject, gated activity, option ids
# (act, refuse, gather). Vocabulary is generic and disjoint from the audit probes and the
# Artifact-G benchmark options (enforced in build_rule_data via the decontamination list).
NUMERIC = [
    ("tenure", "months", [3, 6, 12, 18, 24], "a staff member", "join the mentoring pool", ("enroll_mentor", "decline_mentor", "check_tenure_record")),
    ("attendance", "sessions", [4, 8, 10, 15], "a trainee", "sit the final exam", ("allow_exam", "bar_from_exam", "confirm_attendance_count")),
    ("battery charge", "percent", [20, 30, 50, 80], "a delivery drone", "start a flight", ("launch_drone", "keep_grounded", "read_battery_level")),
    ("wind speed", "km/h", [25, 40, 55, 70], "a crane crew", "lift loads", ("begin_lift", "suspend_lift", "measure_wind_again")),
    ("water depth", "cm", [15, 30, 45, 60], "a road crossing", "stay open to cars", ("keep_road_open", "close_road", "send_depth_survey")),
    ("load", "kg", [50, 120, 300, 750], "a shelf unit", "be used for storage", ("store_items", "move_items_elsewhere", "weigh_the_load")),
    ("queue length", "people", [10, 25, 40, 60], "a ticket counter", "open a second window", ("open_second_window", "keep_one_window", "count_the_queue")),
    ("soil moisture", "percent", [15, 25, 35, 45], "a greenhouse bed", "be watered today", ("water_bed", "skip_watering", "probe_soil_moisture")),
    ("invoice amount", "dollars", [250, 800, 1500, 5000], "a purchase", "be paid without review", ("pay_invoice", "hold_for_review", "confirm_invoice_amount")),
    ("sample size", "responses", [30, 100, 250, 400], "a survey result", "be published", ("publish_result", "withhold_result", "recount_responses")),
    ("blood oxygen", "percent", [88, 90, 92, 94], "a clinic patient", "be sent home", ("discharge_patient", "keep_under_observation", "repeat_oxygen_reading")),
    ("noise level", "decibels", [55, 65, 75, 85], "a work site", "operate after 8 pm", ("allow_night_work", "stop_night_work", "take_noise_reading")),
    ("file size", "megabytes", [10, 25, 100, 500], "an upload", "go through the fast lane", ("use_fast_lane", "use_bulk_queue", "check_file_size")),
    ("response time", "milliseconds", [100, 250, 500, 900], "a web service", "stay in the pool", ("keep_in_pool", "drain_from_pool", "rerun_latency_probe")),
    ("credit balance", "points", [100, 500, 1000, 2500], "a member", "redeem the premium reward", ("redeem_reward", "refuse_redemption", "look_up_balance")),
    ("mileage", "km", [5000, 10000, 15000, 30000], "a rental car", "go out without a service", ("rent_out_car", "send_for_service", "read_odometer")),
    ("fever", "degrees Celsius", [37, 38, 39], "a camp child", "join the hike", ("join_hike", "stay_at_camp", "retake_temperature")),
    ("stock level", "units", [20, 50, 100, 200], "a warehouse item", "skip reordering this week", ("skip_reorder", "place_reorder", "run_stock_count")),
    ("delay", "minutes", [15, 30, 45, 90, 120], "a train service", "keep its connection guarantee", ("honor_connection", "rebook_passengers", "confirm_delay_minutes")),
    ("pressure", "bar", [2, 4, 6, 10], "a gas line", "stay in service", ("keep_line_live", "isolate_line", "read_pressure_gauge")),
    ("rainfall", "mm", [10, 20, 40, 80], "a stadium match", "go ahead", ("play_match", "postpone_match", "check_rain_gauge")),
    ("error rate", "percent", [1, 2, 5, 10], "a model release", "ship to users", ("ship_release", "block_release", "rerun_error_eval")),
    ("vote count", "votes", [3, 5, 7, 12], "a board motion", "pass", ("record_passed", "record_failed", "recount_votes")),
]

ROLE_SETS = [
    (["pharmacist", "senior nurse"], ["porter", "receptionist", "student nurse", "volunteer"], "dispense controlled drugs",
     ("allow_dispensing", "refuse_dispensing", "verify_staff_role")),
    (["treasurer", "finance director"], ["event planner", "facilities lead", "designer", "receptionist"], "sign cheques",
     ("accept_signature", "reject_signature", "confirm_signatory_role")),
    (["licensed electrician"], ["painter", "apprentice", "site visitor", "plumber"], "open the switchboard",
     ("grant_switchboard_access", "deny_switchboard_access", "check_trade_license")),
    (["referee", "match commissioner"], ["coach", "spectator", "ball kid", "photographer"], "stop the match",
     ("accept_stoppage", "ignore_stoppage", "confirm_official_status")),
    (["data steward", "privacy officer"], ["sales rep", "marketing analyst", "trainee", "copywriter"], "export customer records",
     ("approve_export", "block_export", "verify_requester_role")),
    (["pilot in command"], ["cabin crew", "passenger", "ground handler", "caterer"], "delay the departure",
     ("hold_departure", "keep_schedule", "confirm_crew_role")),
    (["registrar", "dean"], ["tutor", "librarian", "student", "alumni officer"], "change a final grade",
     ("apply_grade_change", "reject_grade_change", "verify_staff_title")),
]

RULE_FRAMES = [
    "Policy: {subj} may {act} only with {rel} of {var}.",
    "Rule {rid}: to {act}, {subj} needs {rel} of {var}.",
    "{Subj} is permitted to {act} when its {var} is {rel}.",
    "Guideline: {subj} can {act} only if the {var} is {rel}.",
    "Under the site standard, {subj} must have {rel} of {var} to {act}.",
]
FACT_FRAMES = [
    "The recorded {var} is {v} {u}.",
    "Current reading: {var} = {v} {u}.",
    "{Name} reports a {var} of {v} {u}.",
    "According to the log, the {var} stands at {v} {u}.",
    "The {var} was measured at {v} {u} this morning.",
]
MISSING_FACT = [
    "The {var} has not been recorded yet.",
    "Nobody has checked the {var}.",
    "The {var} field in the form is blank.",
]
CONFLICT_FACT = [
    "One log shows a {var} of {a} {u}, but the second log shows {b} {u}.",
    "{Name} reports a {var} of {a} {u}; the sensor says {b} {u}.",
]
FILLERS = [
    "The request was filed by {Name} earlier today.",
    "Reference number {rid} is attached to this case.",
    "{Name} has handled similar cases before.",
    "The weather at the site is mild.",
    "A follow-up meeting is planned for next week.",
    "The paperwork was scanned and uploaded.",
]
QUESTIONS = ["What is the correct action?", "Which action should be taken?", "What should happen next?",
             "Which decision follows from the rule?"]


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _numeric_family(rng: random.Random, variant: int) -> list[dict[str, Any]]:
    var, u, pool, subj, act, (ok, bad, info) = rng.choice(NUMERIC)
    T = rng.choice(pool)
    relk = rng.choice(list(RELATIONS))
    rel = rng.choice(RELATIONS[relk]).format(T=T, u=u)
    rid = rng.randint(10, 99)
    rule = rng.choice(RULE_FRAMES).format(subj=subj, Subj=_cap(subj), act=act, rel=rel, var=var, rid=rid)
    # the rule text may read "the var is at least T units" or "at least T units of var"; both are fine English
    name, name2 = rng.sample(FIRST_NAMES, 2)
    fact_f = rng.choice(FACT_FRAMES)
    filler_t = rng.choice(FILLERS)
    filler = filler_t.format(Name=name, rid=rng.randint(100, 999))
    holds = HOLDS[relk]

    def value(side_ok: bool, near: bool) -> int:
        for _ in range(100):
            d = rng.randint(0, 3) if near else rng.randint(4, max(6, T // 2))
            v = T + d if rng.random() < 0.5 else T - d
            if v >= 0 and holds(v, T) == side_ok:
                return v
        return T

    # One layout per family, so the variants are minimal pairs: only the named variable changes.
    rule_first = rng.random() < 0.7
    fill_slot = rng.randint(0, 2) if rng.random() < 0.6 else None

    def state(fact: str, fill: str | None = filler, force_fill: bool = False) -> str:
        parts = [rule, fact] if rule_first else [fact, rule]
        slot = fill_slot if fill_slot is not None else (2 if force_fill else None)
        if slot is not None and fill:
            parts.insert(min(slot, len(parts)), fill)
        return " ".join(p for p in parts if p)

    q = rng.choice(QUESTIONS)
    opts = [ok, bad, info]
    base_ok = rng.random() < 0.5
    v0 = value(base_ok, rng.random() < 0.5)
    v1 = value(not base_ok, rng.random() < 0.6)
    fact0 = fact_f.format(var=var, v=v0, u=u, Name=name)
    out = []

    def add(st, target_idx, sat=None):
        qs = [choice_q(q, opts, target_idx, rng)]
        if sat is not None and rng.random() < 0.3:
            qs.append(noul_q(f"Does the {var} satisfy the rule?", sat))
        out.append(example("rule_numeric", st, qs, variant))

    add(state(fact0), 0 if base_ok else 1, base_ok)
    add(state(fact_f.format(var=var, v=v1, u=u, Name=name)), 1 if base_ok else 0, not base_ok)
    # irrelevant change: a different filler (always present) and, half the time, a paraphrased fact
    other_fill = rng.choice([f for f in FILLERS if f != filler_t]).format(Name=name2, rid=rng.randint(100, 999))
    fact_irr = (rng.choice([f for f in FACT_FRAMES if f != fact_f]) if rng.random() < 0.5 else fact_f).format(var=var, v=v0, u=u, Name=name2)
    add(state(fact_irr, other_fill, force_fill=True), 0 if base_ok else 1)
    kind = rng.choice(["missing", "unknown", "conflict"])
    if kind == "missing":
        st = state("")
    elif kind == "unknown":
        st = state(rng.choice(MISSING_FACT).format(var=var))
    else:
        a, b = (v0, v1) if rng.random() < 0.5 else (v1, v0)
        st = state(rng.choice(CONFLICT_FACT).format(var=var, a=a, b=b, u=u, Name=name))
    add(st, 2)
    return out


ROLE_RULES = [
    "Only a {roles} may {act}.",
    "The authority to {act} is limited to the {roles}.",
    "Staff other than the {roles} must not {act}.",
]
ROLE_FACTS = [
    "{Name}, who works as a {role}, asks to {act}.",
    "A request to {act} comes from {Name}; {Name}'s role is {role}.",
    "{Name} ({role}) wants to {act}.",
]
ROLE_UNKNOWN = [
    "{Name} asks to {act}; {Name}'s role is not listed in the directory.",
    "Someone called {Name} asks to {act} but gives no job title.",
]
ROLE_CONFLICT = [
    "{Name} says they are a {r1}, but the staff roster lists {Name} as a {r2}; {Name} asks to {act}.",
]


def _role_family(rng: random.Random, variant: int) -> list[dict[str, Any]]:
    allowed, others, act, (ok, bad, info) = rng.choice(ROLE_SETS)
    roles = " or ".join(allowed) if len(allowed) > 1 else allowed[0]
    rule = rng.choice(ROLE_RULES).format(roles=roles, act=act)
    name, name2 = rng.sample(FIRST_NAMES, 2)
    ff = rng.choice(ROLE_FACTS)
    good_r, bad_r = rng.choice(allowed), rng.choice(others)
    base_ok = rng.random() < 0.5
    r0, r1 = (good_r, bad_r) if base_ok else (bad_r, good_r)
    q = rng.choice(QUESTIONS)
    opts = [ok, bad, info]
    order = rng.random() < 0.7
    st = lambda fact: f"{rule} {fact}" if order else f"{fact} {rule}"
    out = [example("rule_role", st(ff.format(Name=name, role=r0, act=act)), [choice_q(q, opts, 0 if base_ok else 1, rng)], variant),
           example("rule_role", st(ff.format(Name=name, role=r1, act=act)), [choice_q(q, opts, 1 if base_ok else 0, rng)], variant),
           example("rule_role", st(ff.format(Name=name2, role=r0, act=act)), [choice_q(q, opts, 0 if base_ok else 1, rng)], variant)]
    if rng.random() < 0.5:
        fact = rng.choice(ROLE_UNKNOWN).format(Name=name, act=act)
    else:
        fact = rng.choice(ROLE_CONFLICT).format(Name=name, r1=good_r, r2=bad_r, act=act)
    out.append(example("rule_role", st(fact), [choice_q(q, opts, 2, rng)], variant))
    return out


def rule_families(rng: random.Random, variant: int) -> list[dict[str, Any]]:
    return _numeric_family(rng, variant) if rng.random() < 0.7 else _role_family(rng, variant)
