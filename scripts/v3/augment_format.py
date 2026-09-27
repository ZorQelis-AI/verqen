"""Format diversity for the real-data tasks: the same example, many layouts.

    python scripts/v3/augment_format.py --src data/decision/v3 --out data/decision/format_div

Why (26 Sep 2026, T2 vs Laya on Laya's own benchmark): T2 learned each real task in ONE layout.
NLI was always `Text: <premise>\\nHow does the hypothesis below relate to the text?` + `Hypothesis:
<h>`; given `Premise: ... Hypothesis: ...` T2 fell to 40.7% on XNLI (76% on its own MNLI layout),
answering "neutral" 64% of the time. banking77 fell 92.6% -> 65.9%, MASSIVE 88.2% -> 69.5%.

What it does: parses each real train row back into its fields (premise/hypothesis, text, passage/
question) and re-renders it with a random layout from pool A: field names, where each field sits,
question wording, and (NLI / yes-no) label words. The gold answer is carried through every remap.

Leakage discipline: Laya's exact layouts are excluded from pool A (the `Premise:/Hypothesis:` NLI
layout and its six question strings; see LAYA_EXCLUDED). Pool B is held out of training entirely;
val_B.jsonl measures transfer to unseen layouts and is the pre-registered selection metric for this
round. Laya's benchmark is used only for the final report.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
from pathlib import Path

LAYA_EXCLUDED = {"What is the topic of this news article?", "Which emotion does the author express?",
                 "What does the customer want to do?", "What is the sentiment of this review?",
                 "How does the hypothesis relate to the premise?", "What is the user's intent?"}

NLI_SRC = {"mnli", "snli", "wanli"}
TASK = {"ag_news": "topic", "dbpedia": "topic", "emotion": "emotion", "go_emotions": "emotion", "sst5": "sentiment",
        "tweet_sentiment": "sentiment", "banking77": "intent", "massive_en": "intent", "clinc": "intent"}
YESNO = {"boolq"}
CAPS = {"mnli": 8000, "snli": 6000, "wanli": 6000, "ag_news": 5000, "dbpedia": 2000, "emotion": 4000, "go_emotions": 2000,
        "sst5": 4000, "tweet_sentiment": 2000, "banking77": 5000, "massive_en": 5000, "clinc": 3000, "boolq": 5000}

# NLI label words, index order (entailment, neutral, contradiction)
NLI_WORDS = [("entailment", "neutral", "contradiction"), ("yes", "maybe", "no"), ("true", "unknown", "false"),
             ("entailed", "undetermined", "contradicted"), ("follows", "cannot tell", "contradicts"),
             ("supported", "not enough information", "refuted")]
NLI_A = [  # (state template, question template); {p} premise, {h} hypothesis
    ("Sentence 1: {p}\nSentence 2: {h}", "What is the relationship between sentence 1 and sentence 2?"),
    ("Sentence 1: {p}\nSentence 2: {h}", "Does sentence 1 imply sentence 2?"),
    ("Context: {p}", "Statement: {h}\nGiven the context, is the statement true?"),
    ("{p}\n\n{h}", "Is the second text entailed by, contradicted by, or neutral to the first?"),
    ("Passage: {p}\nClaim: {h}", "What does the passage imply about the claim?"),
    ("Source: {p}\nInference: {h}", "Is the inference valid given the source?"),
    ("Text A: {p}\nText B: {h}", "Classify the relation from text A to text B."),
    ("Background: {p}", "Conclusion: {h}\nDoes the background support the conclusion?"),
    ("Statement: {h}\nEvidence: {p}", "Is the statement supported by the evidence?"),
    ("Text: {p}\nHow does the hypothesis below relate to the text?", "Hypothesis: {h}"),
    ("{p}", "Based on the text above, what about this: {h}"),
]
NLI_B = [("First sentence: {p}\nSecond sentence: {h}", "Does the first sentence entail the second?"),
         ("Given: {p}\nQuery: {h}", "Which relation holds?")]

FIELD_A = {"topic": ["", "Article", "News", "Story", "Text", "Document", "Headline and summary"],
           "emotion": ["", "Text", "Message", "Post", "Statement", "Sentence", "Tweet"],
           "sentiment": ["", "Review", "Text", "Comment", "Opinion", "Feedback", "Sentence"],
           "intent": ["", "Message", "Query", "User", "Utterance", "Request", "Customer", "User says"]}
QUESTION_A = {"topic": ["Which category does this article belong to?", "What is this news story about?", "Pick the topic.",
                        "Classify the article by subject.", "Which section should this article be filed under?",
                        "What subject area fits this text best?", "Label the topic of the text.", "Which category fits?"],
              "emotion": ["What emotion is expressed?", "How does the writer feel?", "Which feeling best describes the text?",
                          "Classify the emotion.", "What is the emotional tone?", "Which emotion does the comment express?",
                          "Which emotion fits this text?"],
              "sentiment": ["How positive or negative is this?", "Rate the sentiment.", "What is the overall sentiment?",
                            "Classify the opinion expressed.", "How does the reviewer feel about it?",
                            "What is the sentiment of the post?", "Which sentiment label fits?"],
              "intent": ["What is the user trying to do?", "Which intent matches this request?", "Route this message to the right intent.",
                         "What does the person want?", "Classify the intent of the message.", "Which action is being requested?",
                         "Route this customer message to the right intent.", "Which request type is this?"]}
FIELD_B = {k: ["Input text"] for k in FIELD_A}
QUESTION_B = {"topic": ["Assign a news category."], "emotion": ["Name the emotion."], "sentiment": ["Give the sentiment rating."],
              "intent": ["Identify the request type."]}

YESNO_A = [("{p}", "{q}", ("yes", "no")), ("Passage: {p}", "Question: {q}\nAnswer yes or no.", ("yes", "no")),
           ("{p}\n\nQuestion: {q}", "Is the answer yes or no?", ("yes", "no")), ("Context: {p}", "{q}", ("true", "false")),
           ("Text: {p}", "True or false: {q}", ("true", "false")), ("Article: {p}", "{q}", ("Yes", "No"))]
YESNO_B = [("Reference: {p}", "{q} (yes/no)", ("yes", "no"))]

for _t in [q for v in QUESTION_A.values() for q in v] + [q for _, q in NLI_A]:
    assert _t not in LAYA_EXCLUDED, _t
assert not any(s.startswith("Premise:") for s, _ in NLI_A)

LABEL_PREFIX = re.compile(r"^[A-Z][A-Za-z ]{0,24}: ")


def gold_index(q):
    t = q["target"]
    return max(range(len(t)), key=lambda i: t[i])


def parse(r):
    """-> (kind, fields, options, gold) or None"""
    src, q = r["source"], r["questions"][0]
    # boolq is stored as `noul` (fixed no/yes head); its restyled copies become ordinary `choice`
    # questions, the form any caller (and Laya's benchmark) uses for a yes/no question
    if q.get("type") not in ("choice", "noul") or not q["options"] or (q["type"] == "noul" and src not in YESNO):
        return None
    g = gold_index(q)
    if src in NLI_SRC:
        m1 = re.match(r"^Text: (.*)\nHow does the hypothesis below relate to the text\?$", r["state"], re.S)
        m2 = re.match(r"^Hypothesis: (.*)$", q["text"], re.S)
        if not (m1 and m2) or q["options"] != ["entailment", "neutral", "contradiction"]:
            return None
        return "nli", {"p": m1.group(1).strip(), "h": m2.group(1).strip()}, q["options"], g
    if src in TASK:
        text = LABEL_PREFIX.sub("", r["state"], count=1).strip()
        return TASK[src], {"t": text}, q["options"], g
    if src in YESNO:
        if [o.lower() for o in q["options"]] not in (["yes", "no"], ["no", "yes"]):
            return None
        return "yesno", {"p": r["state"].strip(), "q": q["text"].strip().rstrip("?") + "?"}, q["options"], g
    return None


def render(r, parsed, pool, rng):
    kind, f, opts, g = parsed
    q0 = r["questions"][0]
    if kind == "nli":
        st, qt = rng.choice(NLI_A if pool == "A" else NLI_B)
        words = rng.choice(NLI_WORDS) if pool == "A" else NLI_WORDS[0]
        order = list(range(3))
        if rng.random() < 0.5:
            rng.shuffle(order)
        new_opts = [words[i] for i in order]
        target = [1.0 if i == g else 0.0 for i in order]
        state, question = st.format(**f), qt.format(**f)
    elif kind == "yesno":
        st, qt, words = rng.choice(YESNO_A if pool == "A" else YESNO_B)
        yes_is_gold = opts[g].lower() == "yes"
        order = [0, 1] if rng.random() < 0.5 else [1, 0]  # 0 = yes-word, 1 = no-word
        new_opts = [words[i] for i in order]
        target = [1.0 if (i == 0) == yes_is_gold else 0.0 for i in order]
        state, question = st.format(**f), qt.format(**f)
    else:
        field = rng.choice(FIELD_A[kind] if pool == "A" else FIELD_B[kind])
        state = f"{field}: {f['t']}" if field else f["t"]
        question = rng.choice(QUESTION_A[kind] if pool == "A" else QUESTION_B[kind])
        case = rng.random()
        new_opts = [o.lower() for o in opts] if case < 0.2 else ([o.title() for o in opts] if case < 0.35 else list(opts))
        if len(set(new_opts)) != len(new_opts):
            new_opts = list(opts)
        target = list(q0["target"])
    out = dict(r, state=state, questions=[dict(q0, type="choice", text=question, options=new_opts, target=target)],
               source=r["source"] + "+fmt", meta=dict(r.get("meta") or {}, format_pool=pool))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/decision/v3")
    ap.add_argument("--out", default="data/decision/format_div")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--val-per-source", type=int, default=150)
    a = ap.parse_args()
    rng = random.Random(9090)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {}
    for split, src_file, pool in (("train", "train.jsonl", "A"), ("val_A", "val.jsonl", "A"), ("val_B", "val.jsonl", "B")):
        by = collections.defaultdict(list)
        with open(os.path.join(a.src, src_file), encoding="utf-8") as f:
            for line in f:
                i = line.find('"source": "')
                s = line[i + 11:line.find('"', i + 11)] if i >= 0 else ""
                if s in CAPS:
                    by[s].append(line)
        rows, counts = [], {}
        for s, lines in sorted(by.items()):
            rng.shuffle(lines)
            cap = int(CAPS[s] * a.scale) if split == "train" else a.val_per_source
            made = 0
            for line in lines:
                if made >= cap:
                    break
                r = json.loads(line)
                p = parse(r)
                if p is None:
                    continue
                rows.append(json.dumps(render(r, p, pool, rng), ensure_ascii=False) + "\n")
                made += 1
            counts[s] = {"available": len(lines), "made": made}
        rng.shuffle(rows)
        tmp = out / f"{split}.jsonl.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(rows)
        os.replace(tmp, out / f"{split}.jsonl")
        rep[split] = counts
    json.dump(rep, open(out / "report.json", "w"), indent=1)
    print(json.dumps({k: sum(v["made"] for v in c.values()) for k, c in rep.items()}), json.dumps(rep["train"]))


if __name__ == "__main__":
    main()
