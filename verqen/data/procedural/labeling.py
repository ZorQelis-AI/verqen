"""Data-labelling families. The label is sampled first and the text is composed to imply it,
including negation, contrast ("..., but ...") and intensity, so keyword matching fails."""

from __future__ import annotations

import random

from .common import FIRST_NAMES, ITEMS, choice_q, example, pick

POS = ["excellent", "great", "reliable", "delightful", "fast", "sturdy", "comfortable", "intuitive"]
NEG = ["terrible", "flimsy", "slow", "confusing", "noisy", "broken", "overpriced", "unreliable"]
ASPECTS = ["battery", "build quality", "delivery", "price", "screen", "customer service", "setup", "sound"]


def _phrase(rng, polarity: str) -> str:
    aspect = pick(rng, ASPECTS)
    if polarity == "pos":
        return rng.choice([f"the {aspect} is {pick(rng, POS)}", f"the {aspect} is not {pick(rng, NEG)} at all",
                           f"I was impressed by the {aspect}"])
    return rng.choice([f"the {aspect} is {pick(rng, NEG)}", f"the {aspect} is not {pick(rng, POS)}",
                       f"I regret the {aspect}"])


def sentiment(rng: random.Random, variant: int):
    label = rng.choice(["positive", "negative", "mixed", "neutral"])
    item = pick(rng, ITEMS)
    if label == "positive":
        parts = [_phrase(rng, "pos") for _ in range(rng.randint(1, 3))]
        if rng.random() < 0.4:  # contrast where the later, stronger clause wins
            parts = [f"at first I thought {_phrase(rng, 'neg')}", f"but after a week {_phrase(rng, 'pos')} and {_phrase(rng, 'pos')}"]
    elif label == "negative":
        parts = [_phrase(rng, "neg") for _ in range(rng.randint(1, 3))]
        if rng.random() < 0.4:
            parts = [f"it looked promising and {_phrase(rng, 'pos')}", f"but {_phrase(rng, 'neg')} and {_phrase(rng, 'neg')}"]
    elif label == "mixed":
        parts = [_phrase(rng, "pos"), _phrase(rng, "neg")]
        rng.shuffle(parts)
    else:
        parts = [f"the {item} arrived on {pick(rng, ['Monday', 'Tuesday', 'Friday'])}",
                 f"it comes in a {pick(rng, ['box', 'bag', 'case'])}", f"it weighs about {rng.randint(1, 9)} kg"][: rng.randint(2, 3)]
    text = "; ".join(parts)
    state = f"Review of a {item} by {pick(rng, FIRST_NAMES)}: \"{text[0].upper() + text[1:]}.\""
    opts = ["positive", "negative", "mixed", "neutral"]
    guide = ("Label guide: mixed = clear praise and clear criticism with neither dominating; when the review changes its "
             "mind, the final verdict decides; neutral = no opinion expressed.")
    state = f"{guide}\n{state}" if variant % 2 else f"{state}\n{guide}"
    return example("sentiment", state, [choice_q("What sentiment label fits this review?", opts, opts.index(label), rng)], variant)


INTENTS = {
    "cancel_subscription": ["I want to stop my plan", "please end my membership", "how do I cancel before the next charge"],
    "update_payment": ["my card expired, I need to add a new one", "change the credit card on file", "switch billing to PayPal"],
    "track_order": ["where is my package", "my order still hasn't arrived", "can you check the delivery status"],
    "report_bug": ["the app crashes when I upload a photo", "the save button does nothing", "I get an error 500 at checkout"],
    "request_refund": ["I was charged twice and want my money back", "refund the item I returned", "this was not as described, refund please"],
    "reset_password": ["I can't log in, forgot my password", "the reset link never arrives", "locked out of my account"],
    "upgrade_plan": ["I need more storage", "what does the pro tier cost, I want it", "move me to the business plan"],
}


def intent(rng: random.Random, variant: int):
    labels = pick(rng, list(INTENTS), rng.randint(3, 6))
    label = pick(rng, labels)
    msg = pick(rng, INTENTS[label])
    filler = pick(rng, ["", "Hi there. ", "Hello, quick question. ", "Ugh. ", "Good morning! "])
    distract = rng.random() < 0.3
    if distract:  # mention of another intent's topic that is explicitly not the request
        other = pick(rng, [l for l in INTENTS if l != label])
        msg = f"{msg}. (My {other.split('_')[1]} is fine, that's not the issue.)"
    state = f"Customer message: \"{filler}{msg}\""
    return example("intent", state, [choice_q("Which intent label applies?", labels, labels.index(label), rng)], variant)


def moderation(rng: random.Random, variant: int):
    label = rng.choice(["allow", "remove", "age-restrict"])
    name = pick(rng, FIRST_NAMES)
    if label == "allow":
        text = rng.choice([f"{name}, your argument is wrong and here is the data that shows why.",
                           "This movie was so bad I wanted to scream, one star.",
                           "Historical photo of a battlefield memorial, for a history class."])
    elif label == "remove":
        text = rng.choice([f"{name} lives at 42 Elm Street, go show up at their door.",
                           f"Everyone should harass {name} until they quit.",
                           "Buy followers here, send your password to verify."])
    else:
        text = rng.choice(["Graphic description of a surgical procedure, shared by a nursing student.",
                           "Review of a horror film with detailed gore effects.",
                           "Guide to mixing strong cocktails at home."])
    policy = ("Policy: remove doxxing, targeted harassment and credential scams; age-restrict graphic or adult-oriented "
              "but legitimate content; allow criticism, opinions and educational material.")
    state = f"{policy}\nPost: \"{text}\""
    opts = ["allow", "remove", "age-restrict"]
    return example("moderation", state, [choice_q("What moderation action applies?", opts, opts.index(label), rng)], variant)


FAMILIES = {"sentiment": sentiment, "intent": intent, "moderation": moderation}
