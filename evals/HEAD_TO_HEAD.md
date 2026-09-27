# VerQen vs Laya (and Jev): head-to-head harness

Built 25 Sep 2026. Everything runs locally, on the same GPU, with identical inputs.

## Pieces

| Piece | What it does |
|---|---|
| `jev_parity/cases.jsonl` | Our suite: 7,840 cases on held-out phrasing, 10 capabilities × 22 domains × 105 subdomains |
| `laya_bench/cases.jsonl` | Laya's own public benchmarks, built by `laya_bench/build.py` (2,000 cases each): ag_news, emotion, banking77 (77 options), boolq, sst5, xnli_en, massive_en (59 options) |
| `jev_parity/run.py` | Runs one system on one case set. Reports accuracy with a Wilson CI, ECE, AURC, p50/p95 latency, cost and invalid count |
| `jev_parity/adapters.py` | `verqen` (local checkpoint), `laya` (via its `laya` package), `jev` (API, needs keys), `anthropic` |
| `compare.py` | Paired A-vs-B on the same cases: difference, bootstrap 95% CI, exact McNemar. WIN/LOSS only when the CI excludes 0 |

Laya runs in the isolated venv `Desktop/Language_Models/laya_env` (its package is not installed in the VerQen environment).

## Commands

```bash
# Laya (in its own venv)
../laya_env/Scripts/python.exe evals/jev_parity/run.py --adapter laya --cases evals/laya_bench/cases.jsonl --out evals/results/laya_laya_bench.json
../laya_env/Scripts/python.exe evals/jev_parity/run.py --adapter laya --out evals/results/laya_jev_parity.json
# VerQen-T2 (after the run: the release copied to runs/verqen-t2-remote/release)
python evals/jev_parity/run.py --checkpoint runs/verqen-t2-remote/release --cases evals/laya_bench/cases.jsonl --out evals/results/verqen_t2_laya_bench.json
python evals/jev_parity/run.py --checkpoint runs/verqen-t2-remote/release --out evals/results/verqen_t2_jev_parity.json
# Paired comparisons
python evals/compare.py evals/results/verqen_t2_laya_bench.json evals/results/laya_laya_bench.json
python evals/compare.py evals/results/verqen_t2_jev_parity.json evals/results/laya_jev_parity.json
```

## Fairness notes (read before quoting a number)

- **The prompts are ours.** Laya's exact benchmark prompts are not public, so its published numbers will not reproduce exactly. Both models get the same wording.
- **Training exposure.**
  - VerQen-T2 trained on the **train** splits of ag_news, boolq, emotion, banking77, sst5 and massive_en. The last four were added on 25 Sep 2026 (Option A), to match how Laya's headline numbers were produced.
  - T2 also trained on MNLI, SNLI and WANLI. XNLI's English training set *is* MNLI, so there is no separate XNLI training data.
  - Test splits were never trained on. Training rows whose text exactly matches a head-to-head case were removed (41 rows; `exact_overlap_removed_vs_laya_bench` in the build report).
  - The four added test splits are not in the *pretraining* 13-gram blocklist, because the corpus was tokenized before they were added. The build report lists them.
  - Laya's training data is unpublished.
  - Our 105-subdomain suite shares domains, but never phrasing frames, with VerQen's training data. That is a home advantage and should be named as such when quoting it.
- **Context.** Laya's English checkpoint reads 512 tokens and truncates the rest (the `long_state` cases are about 1,500 tokens). VerQen-T2 reads 2,048. This is a real capability difference, not a harness artifact.
- **Wide option sets.** Laya's default budget for question plus options is 192 tokens. The adapter retries with 256, 384 and max_len/2 before scoring a case as invalid. Invalid counts are reported.
- **Cost.** Both local models are priced on one basis: $0.80/h for one GPU, one decision at a time, at the measured mean latency. Batched serving would lower both.
