"""Curated real decision data for v3: domain-mapped, license-tagged, capped per source.

Rules (docs/production/T2_PLAN.md section 5):
* Train only from official train splits; eval only from official validation/test splits.
* Every source carries its license; `--commercial-only` builds drop non-commercial ones.
* Per-source caps keep any one dataset from dominating (T1: MNLI + RACE were ~40%).
* Classification with many labels (CLINC 151 intents, LEDGAR 100 clause types) becomes a
  routing decision over a sampled candidate set (gold + distractors, <= 8 options), which
  matches how decisions arrive in practice and stays under the head's option limit.
* Only datasets that load as plain parquet are included (script-based ones fail on the
  box and are not worth the risk).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable, Iterator

from datasets import load_dataset

from verqen.data.procedural.common import numeric_options
from verqen.data.real import sources as v2


@dataclass
class Source:
    name: str
    domain: str
    license: str
    commercial: bool
    train: Callable[[], Iterator[dict]] | None
    eval: Callable[[], Iterator[dict]] | None
    train_cap: int | None
    eval_cap: int | None = 2000


def _ex(*a, **kw):
    return v2._ex(*a, **kw)


def _route(source, idx, text, names, gold, rng, k=8, question="Which category does this belong to?", family=None):
    others = [n for i, n in enumerate(names) if i != gold]
    opts = rng.sample(others, min(k - 1, len(others))) + [names[gold]]
    rng.shuffle(opts)
    return _ex(source, idx, text, question, [o.replace("_", " ") for o in opts], opts.index(names[gold]), family=family)


def _features(ds, col):
    return ds.features[col].names


# ── loaders ─────────────────────────────────────────────────────────────────────────
def clinc(split):
    ds = load_dataset("clinc/clinc_oos", "plus", split=split)
    names, rng = _features(ds, "intent"), random.Random(11)
    for i, r in enumerate(ds):
        yield _route("clinc", i, f"User message: {r['text']}", names, r["intent"], rng,
                     question="Which intent should this message be routed to?", family="intent_routing")


def go_emotions(split):
    ds = load_dataset("google-research-datasets/go_emotions", "simplified", split=split)
    names, rng = ds.features["labels"].feature.names, random.Random(12)
    for i, r in enumerate(ds):
        if len(r["labels"]) != 1:
            continue  # single-label items only: the answer must be unambiguous
        yield _route("go_emotions", i, f"Comment: {r['text']}", names, r["labels"][0], rng, k=6,
                     question="Which emotion does the comment express?", family="emotion")


def ag_news(split):
    ds = load_dataset("fancyzhx/ag_news", split=split)
    names = _features(ds, "label")
    for i, r in enumerate(ds):
        yield _ex("ag_news", i, r["text"], "Which section should this article be filed under?", names, r["label"], family="topic")


def dbpedia(split):
    ds = load_dataset("fancyzhx/dbpedia_14", split=split)
    names, rng = _features(ds, "label"), random.Random(13)
    for i, r in enumerate(ds):
        yield _route("dbpedia", i, f"{r['title']}: {r['content']}", names, r["label"], rng, k=6,
                     question="What kind of entity is described?", family="topic")


# Laya's benchmark families (added 25 Sep 2026, disclosed in evals/HEAD_TO_HEAD.md): TRAIN splits
# only; the test splits are the head-to-head eval and are decontaminated against by build_data.py.
# Question wording deliberately differs from evals/laya_bench/build.py, so the benchmark measures
# transfer across phrasing, not a memorized prompt.
def _wide(source, i, text, names, gold, rng, questions, family, full_frac=0.5, k=(8, 20)):
    """Half the rows carry the FULL label set (Banking77: 77, MASSIVE: 59 options), half a sampled
    8-20 subset: wide choice sets are the skill Laya is documented to lack."""
    q = rng.choice(questions)
    if rng.random() < full_frac:
        return _ex(source, i, text, q, [n.replace("_", " ") for n in names], gold, family=family)
    return _route(source, i, text, names, gold, rng, k=rng.randint(*k), question=q, family=family)


def emotion(split):
    ds = load_dataset("dair-ai/emotion", "split", split=split)
    names, rng = _features(ds, "label"), random.Random(21)
    qs = ["What feeling is this text conveying?", "Which emotion best describes the writer?",
          "Label the emotion in this message."]
    for i, r in enumerate(ds):
        yield _ex("emotion", i, f"Text: {r['text']}", rng.choice(qs), names, r["label"], family="emotion")


def banking77(split):
    ds = load_dataset("mteb/banking77", split=split)
    m = {int(l): t for l, t in zip(ds["label"], ds["label_text"])}
    names, rng = [m[k] for k in sorted(m)], random.Random(22)
    qs = ["Which banking request is this?", "Route this customer message to the right intent.",
          "What is the customer asking the bank about?"]
    for i, r in enumerate(ds):
        yield _wide("banking77", i, f"Customer message: {r['text']}", names, int(r["label"]), rng, qs, "intent_routing")


def sst5(split):
    ds = load_dataset("SetFit/sst5", split=split)
    names, rng = ["very negative", "negative", "neutral", "positive", "very positive"], random.Random(23)
    qs = ["How positive or negative is this review?", "Rate the review's overall sentiment.",
          "Which sentiment level fits this review?"]
    for i, r in enumerate(ds):
        yield _ex("sst5", i, f"Review: {r['text']}", rng.choice(qs), names, int(r["label"]), family="sentiment")


def massive_en(split):
    ds = load_dataset("mteb/amazon_massive_intent", "en", split=split)
    names = sorted({v for v in ds["label"]})
    rng = random.Random(24)
    qs = ["What does the user want the assistant to do?", "Which command is this?",
          "Classify this request to a voice assistant."]
    for i, r in enumerate(ds):
        yield _wide("massive_en", i, f"User request: {r['text']}", names, names.index(r["label"]), rng, qs, "intent_routing")


def nli(name, rid, split, pcol="premise", hcol="hypothesis", lcol="label", names=None):
    labels = ["entailment", "neutral", "contradiction"]
    for i, r in enumerate(load_dataset(rid, split=split)):
        lab = r[lcol]
        if isinstance(lab, str):
            lab = labels.index(lab) if lab in labels else -1
        if lab is None or lab < 0:
            continue
        yield _ex(name, i, f"Text: {r[pcol]}\nHow does the hypothesis below relate to the text?", f"Hypothesis: {r[hcol]}",
                  labels, lab, family="inference")


def vitaminc(split):
    labels = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
    opts = ["supported", "refuted", "not enough information"]
    for i, r in enumerate(load_dataset("tals/vitaminc", split=split)):
        if r["label"] not in labels:
            continue
        yield _ex("vitaminc", i, f"Evidence: {r['evidence']}", f"Claim: {r['claim']}\nWhat does the evidence say about the claim?",
                  opts, labels[r["label"]], family="fact_verification")


def gsm8k(split):
    rng = random.Random(14)
    for i, r in enumerate(load_dataset("openai/gsm8k", "main", split=split)):
        m = re.search(r"####\s*([-\d,\.]+)", r["answer"])
        if not m:
            continue
        try:
            ans = float(m.group(1).replace(",", ""))
        except ValueError:
            continue
        ans = int(ans) if ans == int(ans) else ans
        opts, idx = numeric_options(ans, rng, k=4)
        yield _ex("gsm8k", i, r["question"], "What is the answer?", opts, idx, family="math_word")


def pubmedqa(split):
    opts = ["yes", "no", "maybe"]
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    for i, r in enumerate(ds):
        if (split == "train") != (i % 5 != 0):  # the only split is 'train' (1,000 rows): hold out every 5th
            continue
        ctx = " ".join(r["context"]["contexts"])
        yield _ex("pubmedqa", i, ctx, r["question"], opts, opts.index(r["final_decision"]), family="biomedical")


def ledgar(split):
    ds = load_dataset("coastalcph/lex_glue", "ledgar", split=split)
    names, rng = _features(ds, "label"), random.Random(15)
    for i, r in enumerate(ds):
        yield _route("ledgar", i, f"Contract provision: {r['text']}", names, r["label"], rng,
                     question="What type of clause is this provision?", family="contract_clause")


def case_hold(split):
    for i, r in enumerate(load_dataset("coastalcph/lex_glue", "case_hold", split=split)):
        yield _ex("case_hold", i, r["context"], "Which holding completes the citation?", r["endings"], int(r["label"]),
                  family="legal_holding")


def tweet_sentiment(split):
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment", split=split)
    names = _features(ds, "label")
    for i, r in enumerate(ds):
        yield _ex("tweet_sentiment", i, f"Post: {r['text']}", "What is the sentiment of the post?", names, r["label"], family="sentiment")


def civil_comments(split):
    rng = random.Random(16)
    for i, r in enumerate(load_dataset("google/civil_comments", split=split)):
        tox = r["toxicity"]
        if 0.2 < tox < 0.6:
            continue  # drop the ambiguous middle: labels there are annotator coin flips
        if tox <= 0.2 and rng.random() > 0.03:
            continue  # ~2.5% of comments score >= 0.6; keeping 3% of clearly clean ones gives ~50/50
        yield _ex("civil_comments", i, f"Comment: {r['text']}", "Should this comment be removed as toxic?", [], int(tox >= 0.6),
                  qtype="noul", family="moderation")


def _v2(name):
    tr, cap, ev, ecap = v2.SOURCES[name]
    return tr, ev, cap, ecap


# ── the catalog ─────────────────────────────────────────────────────────────────────
def catalog() -> list[Source]:
    S = []
    # v2 sources, now domain-mapped and license-tagged
    for name, dom, lic, com in [("race", "reading", "other (research)", False), ("mnli", "reading", "cc-by-3.0/cc-by-sa-3.0", True),
                                ("boolq", "reading", "cc-by-sa-3.0", True), ("hellaswag", "commonsense", "mit", True),
                                ("winogrande", "commonsense", "cc-by", True), ("csqa", "commonsense", "mit", True),
                                ("piqa", "commonsense", "afl-3.0", True), ("siqa", "commonsense", "cc-by-4.0", True),
                                ("sciq", "science", "cc-by-nc-3.0", False), ("arc_easy", "science", "cc-by-sa-4.0", True),
                                ("arc_challenge", "science", "cc-by-sa-4.0", True), ("obqa", "science", "apache-2.0", True),
                                ("qasc", "science", "cc-by-4.0", True), ("quartz", "science", "cc-by-4.0", True),
                                ("medmcqa", "healthcare_ops", "apache-2.0", True), ("aqua", "quantitative", "apache-2.0", True)]:
        tr, ev, cap, ecap = _v2(name)
        S.append(Source(name, dom, lic, com, tr, ev, {"mnli": 60_000, "race": 40_000}.get(name, cap), ecap))
    S.append(Source("mmlu", "knowledge", "mit", True, None, v2.SOURCES["mmlu"][2], None, 3000))
    # v3 additions
    S += [
        Source("clinc", "customer_support", "cc-by-3.0", True, lambda: clinc("train"), lambda: clinc("test"), 15_000),
        Source("go_emotions", "labeling", "apache-2.0", True, lambda: go_emotions("train"), lambda: go_emotions("test"), 20_000),
        Source("ag_news", "labeling", "unknown", False, lambda: ag_news("train"), lambda: ag_news("test"), 15_000),
        Source("dbpedia", "labeling", "cc-by-sa-3.0", True, lambda: dbpedia("train"), lambda: dbpedia("test"), 10_000),
        Source("snli", "reading", "cc-by-sa-4.0", True, lambda: nli("snli", "stanfordnlp/snli", "train"),
               lambda: nli("snli", "stanfordnlp/snli", "test"), 30_000),
        Source("wanli", "reading", "cc-by-4.0", True, lambda: nli("wanli", "alisawuffles/WANLI", "train", lcol="gold"),
               lambda: nli("wanli", "alisawuffles/WANLI", "test", lcol="gold"), 40_000),
        Source("vitaminc", "reading", "cc-by-sa-3.0", True, lambda: vitaminc("train"), lambda: vitaminc("test"), 40_000),
        Source("gsm8k", "quantitative", "mit", True, lambda: gsm8k("train"), lambda: gsm8k("test"), None),
        Source("pubmedqa", "healthcare_ops", "mit", True, lambda: pubmedqa("train"), lambda: pubmedqa("eval"), None),
        Source("ledgar", "legal", "cc-by-4.0", True, lambda: ledgar("train"), lambda: ledgar("test"), 20_000),
        Source("case_hold", "legal", "cc-by-4.0", True, lambda: case_hold("train"), lambda: case_hold("test"), 20_000),
        Source("tweet_sentiment", "labeling", "unknown", False, lambda: tweet_sentiment("train"),
               lambda: tweet_sentiment("test"), 15_000),
        Source("civil_comments", "safety", "cc0-1.0", True, lambda: civil_comments("train"), lambda: civil_comments("test"), 30_000),
        # Laya's benchmark families (TRAIN splits; test splits = the head-to-head eval, decontaminated)
        Source("emotion", "labeling", "unknown", False, lambda: emotion("train"), lambda: emotion("test"), 16_000),
        Source("banking77", "customer_support", "cc-by-4.0", True, lambda: banking77("train"), lambda: banking77("test"), 10_003),
        Source("sst5", "labeling", "unknown", False, lambda: sst5("train"), lambda: sst5("test"), 8_544),
        Source("massive_en", "tools", "cc-by-4.0", True, lambda: massive_en("train"), lambda: massive_en("test"), 11_514),
    ]
    return S
