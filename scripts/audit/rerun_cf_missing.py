"""Re-run counterfactual (confound fixed) and the extended missing-information probes."""
import collections, json, math, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path[:0] = [str(ROOT), str(ROOT / "scripts" / "audit")]; os.chdir(ROOT)
import probes as P
from verqen.inference.engine import VerqenInferenceEngine

ckpt = sys.argv[1] if len(sys.argv) > 1 else "runs/verqen-t1-v2/release"
tag = sys.argv[2] if len(sys.argv) > 2 else "probes_baseline"
eng = VerqenInferenceEngine.from_checkpoint(ckpt, "auto")
d = lambda st, q, o: eng.decide_multi(st, [{"type": "choice", "text": q, "options": o}])[0]
rep = {"counterfactual": {}, "missing_v2": {}}
cf = collections.defaultdict(lambda: collections.Counter())
for pr in P.counterfactual_probes():
    ra, rb = d(pr["a"][0], pr["question"], pr["options"]), d(pr["b"][0], pr["question"], pr["options"])
    c = cf[pr["pair"].rstrip("0123456789") + "|" + pr["kind"]]
    c["pairs"] += 1; c["flip"] += ra["selected"] != rb["selected"]
    c["a_correct"] += ra["selected"] == pr["a"][1]; c["b_correct"] += rb["selected"] == pr["b"][1]
    c["both_correct"] += ra["selected"] == pr["a"][1] and rb["selected"] == pr["b"][1]
    c[f"{ra['selected']}->{rb['selected']}"] += 1
rep["counterfactual"] = {k: dict(v) for k, v in cf.items()}
mc = collections.defaultdict(lambda: collections.Counter())
for pr in P.missing_probes_v2():
    r = d(pr["state"], pr["question"], pr["options"])
    c = mc[pr["cell"]]; c["n"] += 1; c["correct"] += r["selected"] == pr["answer"]
    c["chose_info"] += r["selected"] == pr["options"][2]; c["escalate"] += r["should_escalate"]
rep["missing_v2"] = {k: dict(v) for k, v in mc.items()}
Path(f"runs/audit/{tag}").mkdir(parents=True, exist_ok=True)
json.dump(rep, open(f"runs/audit/{tag}/cf_missing.json", "w"), indent=2)
print(json.dumps(rep, indent=1))
