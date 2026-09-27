"""Fix B: role-aware distractors (ROOT_CAUSES RC-1).

v2 never saw an option that is *mentioned* in the state but wrong because of the role the
mention plays, so "option words appear in the text" stayed a rewarded feature: in real MC
data P(mention | correct) = 0.14 vs P(mention | wrong) = 0.07. Here every state has exactly
one authoritative directive naming the correct action. Wrong actions are mentioned, often
more often and more recently than the right one, in non-authoritative roles:

    rejected suggestion, past event, prohibition, hypothetical, superseded instruction,
    third-party demand, generic advice

Each example is emitted together with a *role-swap twin*: the same actions, where the
directive now names a previously-distractor action and the old answer moves into a
non-authoritative sentence, so the label moves while the set of mentioned words stays the
same. Only the discourse role separates the answers.
"""

from __future__ import annotations

import random
from typing import Any

from .common import FIRST_NAMES, choice_q, example

# (verb phrase, option id). Disjoint from the audit-probe ACTIONS and Artifact-G options
# (enforced by decontamination in build_fix_data).
ACTIONS = [
    ("repaint the loading bay", "repaint_loading_bay"), ("reseed the north lawn", "reseed_north_lawn"),
    ("recalibrate the scales", "recalibrate_scales"), ("reroute the night bus", "reroute_night_bus"),
    ("archive the old tickets", "archive_old_tickets"), ("replace the water filter", "replace_water_filter"),
    ("reprint the shipping labels", "reprint_shipping_labels"), ("reset the door codes", "reset_door_codes"),
    ("defrost the walk-in freezer", "defrost_walkin_freezer"), ("update the price tags", "update_price_tags"),
    ("tune the stage lights", "tune_stage_lights"), ("swap the forklift battery", "swap_forklift_battery"),
    ("clean the air vents", "clean_air_vents"), ("restock the first aid kits", "restock_first_aid_kits"),
    ("lock the east gate", "lock_east_gate"), ("flush the cooling pipes", "flush_cooling_pipes"),
    ("resend the welcome email", "resend_welcome_email"), ("merge the duplicate records", "merge_duplicate_records"),
    ("trim the hedges", "trim_hedges"), ("rebalance the delivery routes", "rebalance_delivery_routes"),
    ("reindex the search catalog", "reindex_search_catalog"), ("move the pallets to bay 3", "move_pallets_bay3"),
    ("sharpen the kitchen knives", "sharpen_kitchen_knives"), ("renew the parking permits", "renew_parking_permits"),
    ("test the fire alarm bells", "test_alarm_bells"), ("clear the gutter drains", "clear_gutter_drains"),
    ("print the seating chart", "print_seating_chart"), ("cancel the catering order", "cancel_catering_order"),
    ("call the elevator technician", "call_elevator_technician"), ("switch to the spare generator", "switch_spare_generator"),
    ("pause the newsletter", "pause_newsletter"), ("book the large meeting room", "book_large_meeting_room"),
    ("rotate the stock shelves", "rotate_stock_shelves"), ("sweep the parking deck", "sweep_parking_deck"),
    ("upgrade the kiosk software", "upgrade_kiosk_software"), ("order more cardboard boxes", "order_cardboard_boxes"),
    ("disable the guest wifi", "disable_guest_wifi"), ("inspect the roof tiles", "inspect_roof_tiles"),
    ("reorder the chess clocks", "reorder_chess_clocks"), ("relabel the spice jars", "relabel_spice_jars"),
]

ROLES = {  # non-authoritative mentions of a wrong action {a}
    "rejected": ["{Name} proposed that we {a}, but the proposal was turned down.",
                 "Someone asked whether to {a}; the answer was no.",
                 "The idea to {a} came up in the meeting and was rejected."],
    "past": ["Last spring the team had to {a}.", "Two years ago the crew decided to {a}.",
             "In an earlier incident, staff chose to {a}."],
    "prohibition": ["Do not {a} today.", "Nobody should {a} this week.", "Under no circumstances {a} before the audit."],
    "hypothetical": ["If the delivery had been late, we would have had to {a}.",
                     "Had the inspector found a fault, the plan would be to {a}."],
    "superseded": ["The earlier instruction to {a} has been cancelled.",
                   "This morning's plan to {a} was withdrawn an hour ago."],
    "demand": ["A customer keeps insisting that we {a}.", "{Name} from another department wants us to {a}."],
    "advice": ["Online forums often say you should {a}.", "A magazine article recommends that every site {a}."],
}
DIRECTIVES = [
    "The shift manager's final decision: {a} now.",
    "Approved instruction for today: {a}.",
    "{Name}, who is in charge today, has ordered the team to {a}.",
    "The current work order says to {a}.",
    "Decision confirmed by the supervisor: {a}.",
]
QUESTIONS = ["Which action should the team take now?", "What does the current instruction require?",
             "Which task must be done now?", "What is the approved action?"]


def _render(rng, right, mentioned, filler, name_pool):
    """One directive for ``right`` plus role sentences for each wrong action (1-3 mentions)."""
    sents = []
    for w in mentioned:
        for _ in range(rng.choice([1, 1, 2, 3])):
            role = rng.choice(list(ROLES))
            sents.append(rng.choice(ROLES[role]).format(a=w[0], Name=rng.choice(name_pool)))
    sents += filler
    rng.shuffle(sents)
    directive = rng.choice(DIRECTIVES).format(a=right[0], Name=rng.choice(name_pool))
    sents.insert(rng.randint(0, len(sents)), directive)
    return " ".join(sents)


def discourse_families(rng: random.Random, variant: int, prose: list[str] | None = None,
                       mention_all: bool = False) -> list[dict[str, Any]]:
    """``mention_all``: every wrong option is named somewhere (v3). Otherwise an option never named
    in the text is never correct, which is itself a learnable elimination shortcut."""
    k = rng.randint(3, 5)
    acts = rng.sample(ACTIONS, k)
    right, *wrong = acts
    n_mention = len(wrong) if mention_all else rng.randint(1, len(wrong))
    mentioned = wrong[:n_mention]
    filler = rng.sample(prose, min(len(prose), rng.randint(0, 4))) if prose else []
    names = rng.sample(FIRST_NAMES, 3)
    q = rng.choice(QUESTIONS)
    opts = [a[1] for a in acts]
    st1 = _render(rng, right, mentioned, filler, names)
    ex1 = example("discourse_role", st1, [choice_q(q, opts, 0, rng)], variant)
    # twin: a previously-mentioned wrong action becomes the directive, the old answer is demoted
    new_right = mentioned[0]
    demoted = [right] + mentioned[1:]
    st2 = _render(rng, new_right, demoted, filler, names)
    ex2 = example("discourse_role", st2, [choice_q(q, opts, opts.index(new_right[1]), rng)], variant)
    return [ex1, ex2]
