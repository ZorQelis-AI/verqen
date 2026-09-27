"""VerQen data v3: the scenario engine behind the 24-domain synthetic curriculum.

A *spec* describes one subdomain as data: typed variables, an ordered policy (clauses with
explicit precedence and a default), the action vocabulary, and phrasing. The engine turns a
spec into labelled decision examples. Every quality property below is enforced here once,
instead of being hand-written (or forgotten) per family as in v2:

* Computed labels. The policy is evaluated on the sampled values; text is rendered from
  the same values, and ``verify`` re-checks both with an independent evaluator.
* Capabilities Jev says it lacks. ``DateVar`` needs date arithmetic, ``CountVar`` needs
  counting items in a list, ``ProductVar`` needs multiplication. Multi-clause precedence
  is multi-step reasoning.
* Counterfactual families. base, decisive flip (label changes), irrelevant change (label
  fixed), information-sufficiency variant (the decisive variable missing / unknown /
  conflicting, so the answer is the gather option), and a bait twin (wrong actions
  mentioned in non-authoritative roles).
* Near-boundary values. Numeric values are proposed close to active thresholds half the time.
* Frame hold-out. Every phrasing slot has several frames; frame 0 of each slot is reserved
  for held-out evaluation ("unseen phrasing"), and training never renders it.
* Number formats. Integers with or without separators, money in 3 styles, percentages in
  2 styles, dates in 3 styles.
"""

from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from verqen.data.procedural.common import FIRST_NAMES, choice_q, noul_q

# ── phrasing banks (index 0 of each list is held out for evaluation) ───────────────────────
RELATIONS = {
    ">=": ["{T} {u} or more", "at least {T} {u}", "no less than {T} {u}", "a minimum of {T} {u}"],
    ">": ["above {T} {u}", "more than {T} {u}", "over {T} {u}", "in excess of {T} {u}"],
    "<=": ["{T} {u} or less", "at most {T} {u}", "no more than {T} {u}", "a maximum of {T} {u}"],
    "<": ["below {T} {u}", "less than {T} {u}", "under {T} {u}", "fewer than {T} {u}"],
}
CLAUSE_FRAMES = [
    "Whenever {cond}, the required action is: {act}.",
    "If {cond}, {act}.",
    "When {cond}, the correct action is to {act}.",
    "In cases where {cond}, staff must {act}.",
    "{Act} if {cond}.",
]
PRECEDENCE = [
    "Rules are ranked: a lower-numbered rule overrides any later one. With no applicable rule: {default}.",
    "Check the rules in order and apply the first one that matches. If none matches, {default}.",
    "The first rule that applies decides the case; otherwise {default}.",
    "Earlier rules take priority over later rules. If no rule applies, {default}.",
]
RULE_LABELS = [lambda i: f"({chr(96 + i)})", lambda i: f"Rule {i}:", lambda i: f"{i}.", lambda i: f"R{i} -"]
MISSING = ["The {phrase} is not stated anywhere in the file.", "The {phrase} has not been recorded.",
           "Nobody has confirmed the {phrase} yet.", "The {phrase} field was left blank."]
CONFLICT = ["Two sources disagree on the {phrase}: one says {a}, the other says {b}.",
            "The {phrase} is {a} according to the form, but {b} according to the system.",
            "One record gives the {phrase} as {a}; a second record gives {b}.",
            "The {phrase} was reported as {a} and later as {b}, and nobody has resolved which is right."]
BAIT_ROLES = [
    "Someone suggested that we {a}; that suggestion was rejected.",
    "Last year a similar case ended with the decision to {a}.",
    "A colleague wants to {a}, but that colleague is not responsible for this case.",
    "Do not {a} just because it was done in an earlier, unrelated case.",
    "If the policy were different, the answer would be to {a}.",
    "An outdated note says to {a}; it was superseded by the current policy.",
    "The customer insists that we {a}.",
]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


# ── number and date rendering ─────────────────────────────────────────────────────
def fmt_num(v: float, kind: str, rng: random.Random, unit: str = "") -> str:
    if kind == "money":
        s = rng.randrange(3)
        body = f"{v:,.2f}" if (v != int(v) or rng.random() < 0.4) else f"{int(v):,}"
        return [f"${body}", f"{body} dollars", f"USD {body}"][s]
    if kind == "pct":
        return f"{v:g}%" if rng.random() < 0.6 else f"{v:g} percent"
    if kind == "dec":
        return f"{v:g} {unit}".strip()
    iv = int(v)
    body = f"{iv:,}" if (iv >= 1000 and rng.random() < 0.6) else str(iv)
    return f"{body} {unit}".strip()


def fmt_date(d: dt.date, style: int) -> str:
    return [d.isoformat(), f"{MONTHS[d.month - 1]} {d.day}, {d.year}", f"{d.day} {MONTHS[d.month - 1][:3]} {d.year}"][style % 3]


# ── variables ─────────────────────────────────────────────────────────────────────
@dataclass
class Var:
    name: str
    phrase: str  # "the order total"

    def sample(self, rng: random.Random) -> Any: ...
    def near(self, threshold: Any, rng: random.Random) -> Any:
        return self.sample(rng)
    def render(self, value: Any, rng: random.Random, frame: int, ctx: dict) -> str: ...
    def show(self, value: Any, rng: random.Random) -> str:
        """Short value text used in conflict sentences."""
        return str(value)


@dataclass
class NumVar(Var):
    lo: float = 0
    hi: float = 100
    unit: str = ""
    kind: str = "int"  # int | money | pct | dec
    step: float = 1
    facts: Sequence[str] = ("The {phrase} is {v}.", "Recorded {phrase}: {v}.", "{Name} reports that the {phrase} is {v}.",
                            "According to the log, the {phrase} stands at {v}.")

    def _q(self, x: float) -> float:
        x = round(round(x / self.step) * self.step, 2)
        return int(x) if self.kind in ("int",) or (float(x).is_integer() and self.kind != "money") else x

    def sample(self, rng):
        return self._q(rng.uniform(self.lo, self.hi))

    def near(self, threshold, rng):
        d = self.step * rng.choice([0, 1, 1, 2, 3])
        return self._q(min(self.hi, max(self.lo, threshold + rng.choice([-1, 1]) * d)))

    def show(self, value, rng):
        return fmt_num(value, self.kind, rng, self.unit)

    def render(self, value, rng, frame, ctx):
        return self.facts[frame % len(self.facts)].format(phrase=self.phrase, Phrase=_cap(self.phrase),
                                                          v=self.show(value, rng), Name=ctx["name"])


@dataclass
class DateVar(Var):
    """Value = days between an event date and 'today'; both dates are shown, never the difference."""
    lo: int = 0
    hi: int = 120
    event: str = "the purchase"
    facts: Sequence[str] = ("{Event} took place on {d1}. Today's date is {d2}.",
                            "Date of {event_bare}: {d1}. Current date: {d2}.",
                            "{Event} happened on {d1}, and it is now {d2}.",
                            "Today is {d2}; {event} was on {d1}.")
    unit: str = "days"
    kind: str = "int"
    step: int = 1

    def sample(self, rng):
        return rng.randint(self.lo, self.hi)

    def near(self, threshold, rng):
        return int(min(self.hi, max(self.lo, threshold + rng.choice([-3, -2, -1, 0, 1, 2, 3]))))

    def show(self, value, rng):
        return f"{value} days"

    def render(self, value, rng, frame, ctx):
        today = ctx["today"]
        d1 = today - dt.timedelta(days=int(value))
        style = rng.randrange(3)
        bare = self.event.removeprefix("the ")
        return self.facts[frame % len(self.facts)].format(Event=_cap(self.event), event=self.event, event_bare=bare,
                                                          d1=fmt_date(d1, style), d2=fmt_date(today, style))


@dataclass
class CountVar(Var):
    """Value = number of matching items in a rendered list mixed with non-matching items."""
    lo: int = 0
    hi: int = 8
    match_items: Sequence[str] = ()
    other_items: Sequence[str] = ()
    unit: str = ""
    kind: str = "int"
    step: int = 1
    facts: Sequence[str] = ("Event log:\n{items}", "Recent activity:\n{items}", "Entries on file:\n{items}",
                            "The record lists:\n{items}")

    def sample(self, rng):
        return rng.randint(self.lo, self.hi)

    def near(self, threshold, rng):
        return int(min(self.hi, max(self.lo, threshold + rng.choice([-2, -1, 0, 1, 2]))))

    def show(self, value, rng):
        return f"{value} such entries"

    def render(self, value, rng, frame, ctx):
        items = [rng.choice(self.match_items).format(n=rng.randint(2, 250)) for _ in range(int(value))]
        items += [rng.choice(self.other_items).format(n=rng.randint(2, 250)) for _ in range(rng.randint(1, 5))]
        rng.shuffle(items)
        bullet = rng.choice(["- ", "* ", ""])
        return self.facts[frame % len(self.facts)].format(items="\n".join(bullet + i for i in items))


@dataclass
class ProductVar(Var):
    """Value = quantity x unit amount; only the two factors are shown."""
    qty: tuple = (1, 20)
    each: tuple = (5, 200)
    item: str = "units"
    kind: str = "money"
    unit: str = ""
    step: float = 1
    facts: Sequence[str] = ("The order contains {q} {item} at {e} each.", "{q} {item} were ordered, priced at {e} per unit.",
                            "Quantity: {q} {item}. Unit price: {e}.", "{Name} is buying {q} {item} for {e} apiece.")

    def sample(self, rng):
        q, e = rng.randint(*self.qty), rng.randint(*self.each)
        return (q, e)

    def near(self, threshold, rng):
        for _ in range(50):
            q = rng.randint(*self.qty)
            e = max(self.each[0], min(self.each[1], round(threshold / q) + rng.choice([-1, 0, 1])))
            if e > 0:
                return (q, e)
        return self.sample(rng)

    @staticmethod
    def value(v):
        return v[0] * v[1]

    def show(self, value, rng):
        return fmt_num(self.value(value), self.kind, rng)

    def render(self, value, rng, frame, ctx):
        q, e = value
        return self.facts[frame % len(self.facts)].format(q=q, item=self.item, e=fmt_num(e, self.kind, rng),
                                                          Name=ctx["name"])


@dataclass
class CatVar(Var):
    values: Sequence[str] = ()
    facts: Sequence[str] = ("The {phrase} is {v}.", "{Phrase}: {v}.", "It is recorded that the {phrase} is {v}.",
                            "{Name} confirms the {phrase} is {v}.")

    def sample(self, rng):
        return rng.choice(list(self.values))

    def render(self, value, rng, frame, ctx):
        return self.facts[frame % len(self.facts)].format(phrase=self.phrase, Phrase=_cap(self.phrase), v=value,
                                                          Name=ctx["name"])


@dataclass
class FlagVar(Var):
    true_text: Sequence[str] = ("yes",)
    false_text: Sequence[str] = ("no",)
    cond_true: str = ""   # phrase used inside rules for the True case
    cond_false: str = ""

    def sample(self, rng):
        return rng.random() < 0.5

    def show(self, value, rng):
        return "yes" if value else "no"

    def render(self, value, rng, frame, ctx):
        bank = self.true_text if value else self.false_text
        return bank[frame % len(bank)].format(Name=ctx["name"])


def numeric_value(var: Var, v: Any) -> float:
    return ProductVar.value(v) if isinstance(var, ProductVar) else v


# ── policy ────────────────────────────────────────────────────────────────────────
@dataclass
class Cond:
    var: str
    op: str                 # >=, >, <=, <, in, is
    thresholds: Sequence = ()   # candidate thresholds / value sets; one is sampled per instance

    def holds(self, value: Any, param: Any, var: Var) -> bool:
        if self.op == "is":
            return bool(value) == bool(param)
        if self.op == "in":
            return value in param
        x = numeric_value(var, value)
        return {">=": x >= param, ">": x > param, "<=": x <= param, "<": x < param}[self.op]


@dataclass
class Clause:
    conds: Sequence[Cond]
    action: str


@dataclass
class Spec:
    domain: str
    subdomain: str
    variables: Sequence[Var]
    clauses: Sequence[Clause]
    default: str
    actions: dict            # key -> (option_id, verb phrase)
    gather: tuple            # (option_id, verb phrase): the information-gathering action
    questions: Sequence[str] = ("What is the correct action?", "Which action does the policy require?",
                                "What should be done with this case?", "Which decision follows from the rules?")
    context: Sequence[str] = ()   # optional scene-setting sentences (irrelevant to the label)
    weight: float = 1.0

    def var(self, name: str) -> Var:
        return next(v for v in self.variables if v.name == name)


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def keyed(seed: float, key: str) -> random.Random:
    """Deterministic RNG per (layout, key): variants of one family render unchanged values identically."""
    return random.Random(f"{seed!r}:{key}")


def _render_cond(c: Cond, param: Any, var: Var, rng: random.Random, frames: Callable[[int], int]) -> str:
    if c.op == "is":
        return var.cond_true if param else var.cond_false
    if c.op == "in":
        vals = list(param)
        joined = vals[0] if len(vals) == 1 else ", ".join(vals[:-1]) + " or " + vals[-1]
        return [f"the {var.phrase} is {joined}", f"the {var.phrase} is one of: {', '.join(vals)}",
                f"{var.phrase} equals {joined}", f"the {var.phrase} is listed as {joined}"][frames(4)]
    unit = getattr(var, "unit", "")
    kind = getattr(var, "kind", "int")
    T = fmt_num(param, kind, rng) if kind in ("money", "pct") else (f"{param:,}" if isinstance(param, int) and param >= 1000 and rng.random() < 0.5 else str(param))
    rel = RELATIONS[c.op][frames(len(RELATIONS[c.op]))].format(T=T, u=unit if kind not in ("money", "pct") else "").replace("  ", " ").strip()
    if isinstance(var, DateVar):
        return [f"{var.event} was {rel} ago", f"{rel} have passed since {var.event}", f"the time since {var.event} is {rel}",
                f"{var.event} is {rel} old"][frames(4)]
    if isinstance(var, CountVar):
        bare = rel.replace(f" {unit}", "").strip() if unit else rel.strip()
        return [f"the number of {var.phrase} is {rel}", f"{bare} {var.phrase} are listed",
                f"the count of {var.phrase} is {rel}", f"{var.phrase} number {rel}"][frames(4)]
    return [f"the {var.phrase} is {rel}", f"the {var.phrase} comes to {rel}", f"{var.phrase} is {rel}",
            f"the recorded {var.phrase} is {rel}"][frames(4)]


# ── the engine ────────────────────────────────────────────────────────────────────
@dataclass
class Instance:
    spec: Spec
    params: list            # per clause: list of sampled params (one per condition)
    values: dict
    label: str
    name: str
    frames: dict = field(default_factory=dict)


class Engine:
    def __init__(self, spec: Spec, split: str, prose: Sequence[str] = (), seed: int = 0):
        self.spec, self.split, self.prose = spec, split, list(prose)
        self.rng = random.Random(seed)
        self.heldout = split in ("val", "test", "test_unseen")

    def _frame(self, n: int) -> int:
        """Frame 0 is held out: unseen-phrasing splits always use it; training never does."""
        if n <= 1:
            return 0
        return 0 if self.split == "test_unseen" else self.rng.randrange(1, n)

    def evaluate(self, params, values) -> str:
        for clause, ps in zip(self.spec.clauses, params):
            if all(c.holds(values[c.var], p, self.spec.var(c.var)) for c, p in zip(clause.conds, ps)):
                return clause.action
        return self.spec.default

    def _sample_params(self):
        out = []
        for clause in self.spec.clauses:
            ps = []
            for c in clause.conds:
                if c.op == "is":
                    ps.append(True if not c.thresholds else self.rng.choice(list(c.thresholds)))
                elif c.op == "in":
                    ps.append(tuple(self.rng.choice(list(c.thresholds))))
                else:
                    ps.append(self.rng.choice(list(c.thresholds)))
            out.append(ps)
        return out

    def _propose(self, params) -> dict:
        vals = {}
        active = [(c, p) for clause, ps in zip(self.spec.clauses, params) for c, p in zip(clause.conds, ps)]
        for v in self.spec.variables:
            near = [p for c, p in active if c.var == v.name and c.op not in ("in", "is")]
            vals[v.name] = v.near(self.rng.choice(near), self.rng) if near and self.rng.random() < 0.55 else v.sample(self.rng)
        return vals

    def sample(self, want: str | None = None) -> Instance | None:
        reachable = sorted({c.action for c in self.spec.clauses} | {self.spec.default})
        want = want or self.rng.choice(reachable)
        for _ in range(400):
            params = self._sample_params()
            vals = self._propose(params)
            if self.evaluate(params, vals) == want:
                return Instance(self.spec, params, vals, want, self.rng.choice(FIRST_NAMES))
        return None

    def decisive(self, inst: Instance) -> list[str]:
        """Variables whose value alone can change the label (given the others)."""
        out = []
        for v in self.spec.variables:
            for _ in range(30):
                alt = dict(inst.values)
                alt[v.name] = v.sample(self.rng) if self.rng.random() < 0.5 else self._propose(inst.params)[v.name]
                if self.evaluate(inst.params, alt) != inst.label:
                    out.append(v.name)
                    break
        return out

    # rendering ---------------------------------------------------------------
    def render(self, inst: Instance, override: dict | None = None, bait: bool = False, filler: bool = True,
               layout: dict | None = None) -> str:
        """Policy + case facts (+ optional filler and bait). ``override`` maps var -> ("missing"|"unknown"|("conflict", a, b)).

        Every rendering choice is stored in ``layout`` or drawn from an RNG keyed by (layout seed, slot), so
        two variants rendered with the same layout differ only where their values differ (minimal pairs).
        """
        rng, spec = self.rng, self.spec
        L = layout if layout is not None else {}
        seed = L.setdefault("seed", self.rng.random())
        lab = L.setdefault("lab", self._frame(len(RULE_LABELS)))
        rules = []
        for i, (clause, ps) in enumerate(zip(spec.clauses, inst.params), 1):
            cf = [L.setdefault(f"c{i}_{j}", self._frame(4)) for j in range(len(clause.conds))]
            cond = " and ".join(_render_cond(c, p, spec.var(c.var), keyed(seed, f"c{i}_{j}"), (lambda n, f=f: f % n))
                                for j, (c, p, f) in enumerate(zip(clause.conds, ps, cf)))
            fr = CLAUSE_FRAMES[L.setdefault(f"cf{i}", self._frame(len(CLAUSE_FRAMES)))]
            act = spec.actions[clause.action][1]
            rules.append(f"{RULE_LABELS[lab](i)} " + fr.format(cond=cond, act=act, Act=_cap(act)))
        prec = PRECEDENCE[L.setdefault("prec", self._frame(len(PRECEDENCE)))].format(default=spec.actions[spec.default][1])
        policy = "Policy:\n" + "\n".join(rules) + "\n" + prec
        ctx = {"name": inst.name, "today": dt.date(2026, 1, 1) + dt.timedelta(days=keyed(seed, "today").randrange(365))}
        facts = []  # (slot, text)
        for v in spec.variables:
            o = (override or {}).get(v.name)
            fr = L.setdefault(f"ff_{v.name}", self._frame(4))
            r = keyed(seed, f"v_{v.name}")
            if o == "missing":
                continue
            if o == "unknown":
                facts.append((v.name, MISSING[fr % len(MISSING)].format(phrase=v.phrase)))
            elif isinstance(o, tuple) and o[0] == "conflict":
                facts.append((v.name, CONFLICT[fr % len(CONFLICT)].format(phrase=v.phrase, a=v.show(o[1], r), b=v.show(o[2], r))))
            else:
                facts.append((v.name, v.render(inst.values[v.name], r, fr, ctx)))
        if spec.context:
            facts.insert(0, ("ctx", keyed(seed, "ctx").choice(list(spec.context)).format(Name=inst.name)))
        order_rng = keyed(seed, "order")
        slots = [k for k, _ in facts]
        if L.setdefault("shuffle_facts", order_rng.random() < 0.5):
            slots = sorted(slots, key=lambda k: keyed(seed, "pos_" + k).random())
        texts = dict(facts)
        lines = [texts[k] for k in slots]
        if filler and self.prose and L.setdefault("filler", order_rng.random() < 0.35):
            fr_ = keyed(seed, "filler")
            lines.insert(fr_.randint(0, len(lines)), fr_.choice(self.prose))
        if bait:
            wrong = [k for k in spec.actions if k != inst.label]
            for k in rng.sample(wrong, min(len(wrong), rng.randint(1, 2))):
                lines.insert(rng.randint(0, len(lines)), rng.choice(BAIT_ROLES).format(a=spec.actions[k][1]))
        case = "Case:\n" + "\n".join(lines)
        return f"{policy}\n\n{case}" if L.setdefault("policy_first", order_rng.random() < 0.75) else f"{case}\n\n{policy}"

    def _options(self):
        spec = self.spec
        return [spec.actions[k][0] for k in spec.actions] + [spec.gather[0]]

    def _example(self, state: str, label_opt: str, kind: str, inst: Instance, extra_q: list | None = None,
                 decisive: Var | None = None) -> dict:
        opts = self._options()
        q = self.spec.questions[self._frame(len(self.spec.questions))]
        qs = [choice_q(q, opts, opts.index(label_opt), self.rng)] + (extra_q or [])
        return {"source": "procedural_v3", "family": f"{self.spec.domain}/{self.spec.subdomain}", "variant": kind,
                "state": state, "questions": qs,
                "meta": {"domain": self.spec.domain, "subdomain": self.spec.subdomain, "kind": kind, "split": self.split,
                         "decisive": type(decisive).__name__ if decisive is not None else None,
                         "var_types": sorted({type(v).__name__ for v in self.spec.variables})}}

    def family(self) -> list[dict]:
        """base + flip + irrelevant + info-sufficiency + bait twin, sharing one layout (minimal pairs)."""
        spec, rng = self.spec, self.rng
        inst = self.sample()
        if inst is None:
            return []
        layout: dict = {}
        opt = lambda key: spec.actions[key][0]
        out = []
        extra = []
        if rng.random() < 0.3 and spec.clauses:  # a noul question about one rule, computed exactly
            k = rng.randrange(len(spec.clauses))
            c = spec.clauses[k]
            fired = all(cc.holds(inst.values[cc.var], p, spec.var(cc.var)) for cc, p in zip(c.conds, inst.params[k]))
            extra.append(noul_q(f"Do the conditions of rule {k + 1} hold for this case?", fired))
        base_state = self.render(inst, layout=layout)
        out.append(self._example(base_state, opt(inst.label), "base", inst, extra))
        dec = self.decisive(inst)
        if dec:
            v = spec.var(rng.choice(dec))
            for _ in range(60):
                alt = dict(inst.values)
                alt[v.name] = self._propose(inst.params)[v.name] if rng.random() < 0.6 else v.sample(rng)
                lab = self.evaluate(inst.params, alt)
                if lab != inst.label:
                    flip = Instance(spec, inst.params, alt, lab, inst.name)
                    out.append(self._example(self.render(flip, layout=layout), opt(lab), "flip", flip, decisive=v))
                    break
            # information sufficiency: the decisive variable is missing / unknown / conflicting
            kind = rng.choice(["missing", "unknown", "conflict"])
            if kind == "conflict":
                for _ in range(60):
                    a, b = inst.values[v.name], v.sample(rng)
                    alt = dict(inst.values)
                    alt[v.name] = b
                    if self.evaluate(inst.params, alt) != inst.label and a != b:
                        st = self.render(inst, override={v.name: ("conflict", a, b) if rng.random() < 0.5 else ("conflict", b, a)},
                                         layout=layout)
                        out.append(self._example(st, spec.gather[0], "info_conflict", inst, decisive=v))
                        break
            else:
                st = self.render(inst, override={v.name: kind}, layout=layout)
                out.append(self._example(st, spec.gather[0], f"info_{kind}", inst, decisive=v))
        nondec = [x for x in spec.variables if x.name not in dec]
        irr = Instance(spec, inst.params, dict(inst.values), inst.label, rng.choice([n for n in FIRST_NAMES if n != inst.name]))
        if nondec and rng.random() < 0.7:
            v = rng.choice(nondec)
            irr.values[v.name] = v.sample(rng)
        if self.evaluate(irr.params, irr.values) == inst.label:
            st = self.render(irr, layout=layout)
            if st != base_state:
                out.append(self._example(st, opt(inst.label), "irrelevant", irr))
        out.append(self._example(self.render(inst, bait=True, layout=layout), opt(inst.label), "bait", inst))
        return out


import operator as _op

_OPS = {">=": _op.ge, ">": _op.gt, "<=": _op.le, "<": _op.lt}


# words run together, doubled spaces or unfilled placeholders (e.g. the "atleast2" bug)
_TEXT_BUGS = __import__("re").compile(r"\b(atleast|atmost|lessthan|morethan|nomore|noless|fewerthan|orless|ormore)\w*|"
                                      r"[a-z]\d{1,3}\s(missed|failed|approvals|entries)|  +|\{[a-zA-Z_]+\}")


def _visible(var: Var, value: Any, text: str) -> bool:
    """The value the label depends on must be readable from the text."""
    flat = text.replace(",", "")
    if isinstance(var, NumVar):
        return str(int(value)) in flat
    if isinstance(var, ProductVar):
        return str(value[0]) in flat and str(value[1]) in flat
    if isinstance(var, CatVar):
        return str(value) in text
    return True  # dates, counts and flags are rendered indirectly by construction


def verify(engine: Engine, n: int = 200) -> dict:
    """Independent checks on a spec: labels recomputed with a second evaluator, every sampled
    value visible in the text, and each family kind has the label relation it claims."""
    errors, fams = [], 0
    for _ in range(n):
        inst = engine.sample()
        if inst is None:
            errors.append("unsatisfiable target")
            continue
        # second evaluator: explicit loop, no shared helper with Engine.evaluate
        label = engine.spec.default
        for clause, ps in zip(engine.spec.clauses, inst.params):
            ok = True
            for c, p in zip(clause.conds, ps):
                var = engine.spec.var(c.var)
                x = inst.values[c.var]
                if c.op == "is":
                    ok &= bool(x) == bool(p)
                elif c.op == "in":
                    ok &= x in p
                else:
                    xv = x[0] * x[1] if isinstance(var, ProductVar) else x
                    ok &= _OPS[c.op](xv, p)
            if ok:
                label = clause.action
                break
        if label != inst.label:
            errors.append(f"label mismatch {label} vs {inst.label}")
        layout: dict = {}
        txt = engine.render(inst, layout=layout, filler=False)
        bad = _TEXT_BUGS.search(txt)
        if bad:
            errors.append(f"malformed text near {bad.group(0)!r}")
        for v in engine.spec.variables:
            if not _visible(v, inst.values[v.name], txt):
                errors.append(f"value of {v.name} not visible")
        if engine.render(inst, layout=layout, filler=False) != txt:
            errors.append("render is not deterministic for a fixed layout")
        fam = engine.family()
        fams += 1
        kinds = {e["variant"]: e for e in fam}
        lab = lambda e: e["questions"][0]["options"][e["questions"][0]["target"].index(1.0)]
        if "flip" in kinds and lab(kinds["flip"]) == lab(kinds["base"]):
            errors.append("flip did not change the label")
        if "irrelevant" in kinds and lab(kinds["irrelevant"]) != lab(kinds["base"]):
            errors.append("irrelevant change changed the label")
        for k in ("info_missing", "info_unknown", "info_conflict"):
            if k in kinds and lab(kinds[k]) != engine.spec.gather[0]:
                errors.append(f"{k} not labelled gather")
        if "bait" in kinds and lab(kinds["bait"]) != lab(kinds["base"]):
            errors.append("bait changed the label")
    return {"n": n, "families": fams, "errors": errors[:10], "n_errors": len(errors)}
