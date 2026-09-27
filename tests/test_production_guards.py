"""Regression tests for defects found by running the T2 pipeline locally (T2_PLAN.md section 4.1)."""

import importlib.util
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verqen.data.audit import ngrams  # noqa: E402

TEXT = ("the committee reviewed the annual report and noted that the budget for the next year "
        "must be approved before the end of the quarter by the finance director")


def _hits(grams):
    return len(ngrams(TEXT, 13) & grams)


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_ngram_hash_is_stable_across_processes():
    """P15: built-in hash() is salted per process, so a spawn worker never matched the blocklist."""
    grams = ngrams(TEXT, 13)
    assert grams
    with mp.get_context("spawn").Pool(1) as pool:
        assert pool.apply(_hits, (grams,)) == len(grams)


def _square(x):
    return x * x


def test_bounded_map_keeps_order_and_window():
    """P20: Executor.map submitted the whole corpus at once; the bounded map must keep order."""
    pp = _load("scripts/v3/prepare_pretrain.py", "prepare_pretrain")
    with ProcessPoolExecutor(2) as ex:
        assert list(pp._bounded_map(ex, _square, iter(range(50)), window=4)) == [i * i for i in range(50)]


def test_release_rule():
    """P22/P24: RLCD ships only without an overall, per-domain or per-variant val regression."""
    sr = _load("production/select_release.py", "select_release")
    base = {"accuracy": 0.80, "by_domain": {"banking": {"n": 300, "accuracy": 0.80}},
            "by_variant": {"bait": {"n": 400, "accuracy": 0.78}}}
    good = {"accuracy": 0.81, "by_domain": {"banking": {"n": 300, "accuracy": 0.79}},
            "by_variant": {"bait": {"n": 400, "accuracy": 0.77}}}
    assert sr.decide(base, good)[0] == "candidate"
    assert sr.decide(base, dict(good, accuracy=0.79))[0] == "baseline"
    assert sr.decide(base, dict(good, by_domain={"banking": {"n": 300, "accuracy": 0.76}}))[0] == "baseline"
    assert sr.decide(base, dict(good, by_variant={"bait": {"n": 400, "accuracy": 0.74}}))[0] == "baseline"
    small = dict(base, by_variant={"bait": {"n": 50, "accuracy": 0.78}})
    assert sr.decide(small, dict(good, by_variant={"bait": {"n": 50, "accuracy": 0.60}}))[0] == "candidate"
    assert sr.decide(base, None)[0] == "baseline"
