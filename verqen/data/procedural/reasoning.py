"""Reasoning families: arithmetic, word problems, time, logic, space, tracking, counting, code."""

from __future__ import annotations

import contextlib
import io
import random

from .common import COLORS, FIRST_NAMES, ITEMS, choice_q, example, noul_q, numeric_options, pick


def arithmetic(rng: random.Random, variant: int):
    a, b, c = rng.randint(2, 60), rng.randint(2, 40), rng.randint(2, 12)
    form = rng.choice(["{a} + {b} * {c}", "({a} + {b}) * {c}", "{a} * {c} - {b}", "{a} - {b} + {c}", "({a} - {b}) * {c}"])
    expr = form.format(a=a, b=b, c=c)
    val = eval(expr)  # expression is built from integers only
    opts, idx = numeric_options(val, rng, spread=max(4, abs(val) // 5 + 2))
    qs = [choice_q(f"What is the value of {expr}?", opts, idx, rng)]
    k = rng.choice([2, 3, 4, 5, 7])
    qs.append(noul_q(f"Is {val} divisible by {k}?", val % k == 0))
    exprs = [form2.format(a=rng.randint(2, 30), b=rng.randint(2, 30)) for form2 in
             rng.sample(["{a} * {b}", "{a} + {b} + {a}", "{a} ** 2 - {b}", "{b} * 3 + {a}"], 3)]
    vals = [eval(e) for e in exprs]
    if len(set(vals)) == 3:
        qs.append(choice_q("Which expression has the largest value?", exprs, vals.index(max(vals)), rng))
    state = "Evaluate using standard operator precedence (multiplication before addition, ** is a power)."
    return example("arithmetic", state, qs, variant)


def shopping(rng: random.Random, variant: int):
    item = pick(rng, ITEMS)
    price = rng.randint(5, 400)
    qty = rng.randint(1, 6)
    disc = rng.choice([0, 5, 10, 15, 20, 25, 50])
    ship = rng.choice([0, 4, 7, 12])
    free_over = rng.choice([None, 50, 100, 200])
    sub = price * qty * (100 - disc) / 100
    ship_cost = 0 if (free_over is not None and sub >= free_over) else ship
    total = round(sub + ship_cost, 2)
    state = (f"{pick(rng, FIRST_NAMES)} buys {qty} x {item} at ${price} each. A {disc}% discount applies to the items. "
             f"Shipping costs ${ship}" + (f", free for discounted item totals of ${free_over} or more." if free_over else "."))
    errors = [price * qty + ship_cost, sub + ship, sub, price * (100 - disc) / 100 + ship_cost, (price * qty + ship) * (100 - disc) / 100]
    opts, idx = numeric_options(float(total), rng, spread=max(3, int(total * 0.15) + 2), fmt=lambda v: f"${v:.2f}", extra=errors)
    qs = [choice_q("What is the total cost?", opts, idx, rng)]
    if free_over:
        qs.append(noul_q("Does the order qualify for free shipping?", sub >= free_over))
    return example("shopping", state, qs, variant)


def comparison_shopping(rng: random.Random, variant: int):
    n = rng.randint(3, 5)
    stores = pick(rng, ["ShopA", "MegaMart", "QuickBuy", "ValueHub", "Crate&Co", "Buyly", "NetStore"], n)
    rows, totals = [], {}
    for s in stores:
        p = rng.randint(40, 200)
        d = rng.choice([0, 10, 20, 30])
        sh = rng.choice([0, 5, 9, 15])
        totals[s] = round(p * (100 - d) / 100 + sh, 2)
        rows.append(f"{s}: ${p}, {d}% off, shipping ${sh}")
    best = min(totals.values())
    state = f"Same {pick(rng, ITEMS)} at several stores. " + "; ".join(rows) + "."
    qs = [choice_q("Which store has the lowest total price?", stores, [1.0 if totals[s] == best else 0.0 for s in stores], rng)]
    return example("comparison_shopping", state, qs, variant)


def rates(rng: random.Random, variant: int):
    speed = rng.randint(20, 120)
    hours = rng.choice([0.5, 1, 1.5, 2, 2.5, 3, 4])
    dist = speed * hours
    dist = int(dist) if dist == int(dist) else dist
    who = pick(rng, FIRST_NAMES)
    if rng.random() < 0.5:
        state = f"{who} drives at a constant {speed} km/h for {hours} hours."
        opts, idx = numeric_options(dist, rng, fmt=lambda v: f"{v} km")
        q = choice_q("How far does " + who + " travel?", opts, idx, rng)
    else:
        state = f"{who} covers {dist} km at a constant {speed} km/h."
        opts, idx = numeric_options(float(hours), rng, k=4, spread=2, fmt=lambda v: f"{v:g} hours")
        q = choice_q("How long does the trip take?", opts, idx, rng)
    return example("rates", state, [q], variant)


def temporal(rng: random.Random, variant: int):
    names = pick(rng, ["standup", "design review", "lunch", "client call", "code freeze", "gym", "demo", "1:1"], rng.randint(3, 5))
    events = {}
    for n in names:
        start = rng.randint(8 * 60, 17 * 60) // 15 * 15
        dur = rng.choice([15, 30, 45, 60, 90])
        events[n] = (start, start + dur)
    fmt = lambda m: f"{m // 60:02d}:{m % 60:02d}"
    state = "Today's schedule: " + "; ".join(f"{n} starts at {fmt(s)} and lasts {e - s} minutes" for n, (s, e) in events.items()) + "."
    first = min(events.values(), key=lambda x: x[0])[0]
    qs = [choice_q("Which event starts first?", names, [1.0 if events[n][0] == first else 0.0 for n in names], rng)]
    last_end = max(e for _, e in events.values())
    qs.append(choice_q("Which event ends last?", names, [1.0 if events[n][1] == last_end else 0.0 for n in names], rng))
    a, b = pick(rng, names, 2)
    (s1, e1), (s2, e2) = events[a], events[b]
    qs.append(noul_q(f"Do the {a} and the {b} overlap in time?", s1 < e2 and s2 < e1))
    return example("temporal", state, qs, variant)


NONCE = ["blicket", "wug", "dax", "fep", "toma", "zorb", "glim", "prax", "vell", "quon", "snib", "krell"]


def logic(rng: random.Random, variant: int):
    a, b, c = pick(rng, NONCE, 3)
    x = pick(rng, FIRST_NAMES)
    ans = rng.choice(["true", "false", "cannot be determined"])  # label-first: balanced answers
    kind = rng.choice({"true": ["chain", "modus_ponens"], "false": ["negation", "modus_tollens"],
                       "cannot be determined": ["some", "affirm_consequent"]}[ans])
    if kind == "chain":
        prem, q = f"All {a}s are {b}s. All {b}s are {c}s. {x} is a {a}.", f"Is {x} a {c}?"
    elif kind == "modus_ponens":
        prem, q = f"If something is a {a}, then it is a {b}. {x} is a {a}.", f"Is {x} a {b}?"
    elif kind == "negation":
        prem, q = f"All {a}s are {b}s. No {b}s are {c}s. {x} is a {a}.", f"Is {x} a {c}?"
    elif kind == "modus_tollens":
        prem, q = f"If something is a {a}, then it is a {b}. {x} is not a {b}.", f"Is {x} a {a}?"
    elif kind == "some":
        prem, q = f"Some {a}s are {b}s. All {b}s are {c}s. {x} is a {a}.", f"Is {x} a {c}?"
    else:  # affirming the consequent: a classic fallacy
        prem, q = f"If something is a {a}, then it is a {b}. {x} is a {b}.", f"Is {x} a {a}?"
    opts = ["true", "false", "cannot be determined"]
    state = f"Premises (assume they are all true; the words are made up): {prem}"
    return example("logic", state, [choice_q(q, opts, opts.index(ans), rng)], variant)


def spatial(rng: random.Random, variant: int):
    objs = pick(rng, ITEMS, rng.randint(3, 5))
    order = objs[:]
    rng.shuffle(order)
    facts = [f"the {order[i]} is left of the {order[i + 1]}" for i in range(len(order) - 1)]
    rng.shuffle(facts)
    state = "On a shelf, " + "; ".join(facts) + "."
    qs = [choice_q("Which object is furthest to the right?", objs, objs.index(order[-1]), rng),
          choice_q("Which object is furthest to the left?", objs, objs.index(order[0]), rng)]
    i, j = rng.sample(range(len(order)), 2)
    qs.append(noul_q(f"Is the {order[i]} left of the {order[j]}?", i < j))
    return example("spatial", state, qs, variant)


def entity_tracking(rng: random.Random, variant: int):
    boxes = [f"{c} box" for c in pick(rng, COLORS, rng.randint(3, 4))]
    objs = pick(rng, ITEMS, rng.randint(2, 4))
    loc = {o: pick(rng, boxes) for o in objs}
    lines = [f"The {o} is in the {b}." for o, b in loc.items()]
    for _ in range(rng.randint(1, 5)):
        o = pick(rng, objs)
        b = pick(rng, [x for x in boxes if x != loc[o]])
        lines.append(f"{pick(rng, FIRST_NAMES)} moves the {o} to the {b}.")
        loc[o] = b
    state = " ".join(lines)
    o = pick(rng, objs)
    qs = [choice_q(f"Where is the {o} now?", boxes, boxes.index(loc[o]), rng)]
    b = pick(rng, boxes)
    qs.append(noul_q(f"Is the {b} empty?", all(v != b for v in loc.values())))
    return example("entity_tracking", state, qs, variant)


def counting(rng: random.Random, variant: int):
    pool = pick(rng, ITEMS, rng.randint(3, 5))
    seq = [pick(rng, pool) for _ in range(rng.randint(6, 16))]
    counts = {p: seq.count(p) for p in pool}
    top = max(counts.values())
    state = "Inventory scan log: " + ", ".join(seq) + "."
    qs = [choice_q("Which item was scanned most often?", pool, [1.0 if counts[p] == top else 0.0 for p in pool], rng)]
    p = pick(rng, pool)
    opts, idx = numeric_options(counts[p], rng, spread=3)
    if all(int(o) >= 0 for o in opts):
        qs.append(choice_q(f"How many times was the {p} scanned?", opts, idx, rng))
    return example("counting", state, qs, variant)


def _program(rng: random.Random) -> str:
    v = pick(rng, ["x", "n", "total", "acc", "s", "k"])
    kind = rng.randint(0, 3)
    a, b = rng.randint(1, 9), rng.randint(2, 6)
    if kind == 0:
        return f"{v} = {a}\nfor i in range({b}):\n    {v} += i\nprint({v})"
    if kind == 1:
        return f"items = {[rng.randint(0, 20) for _ in range(rng.randint(3, 6))]}\nprint(sum(t for t in items if t % 2 == {rng.randint(0, 1)}))"
    if kind == 2:
        return f"{v} = {a}\nwhile {v} < {a * b + rng.randint(1, 20)}:\n    {v} *= 2\nprint({v})"
    return f"word = '{pick(rng, ['banana', 'kernel', 'decision', 'matrix', 'pipeline', 'router'])}'\nprint(len(word) * {b} - word.count('{pick(rng, 'aeinrt')}'))"


def code_exec(rng: random.Random, variant: int):
    src = _program(rng)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(src, {"__builtins__": {"range": range, "print": print, "sum": sum, "len": len}})  # generated code only
    val = int(buf.getvalue().strip())
    opts, idx = numeric_options(val, rng, spread=max(3, abs(val) // 4 + 1))
    state = f"Python program:\n{src}"
    return example("code_exec", state, [choice_q("What does the program print?", opts, idx, rng)], variant)


FAMILIES = {
    "arithmetic": arithmetic,
    "shopping": shopping,
    "comparison_shopping": comparison_shopping,
    "rates": rates,
    "temporal": temporal,
    "logic": logic,
    "spatial": spatial,
    "entity_tracking": entity_tracking,
    "counting": counting,
    "code_exec": code_exec,
}
