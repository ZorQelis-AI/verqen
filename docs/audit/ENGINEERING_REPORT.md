# VerQen-T1: Artifact-G audit, repair and re-benchmark

24 Sep 2026. Workflow: audit → reproduce → localize → root cause → design → implement → test → ablate → benchmark → document.

Every claim below is labelled **OBSERVED** (measured), **ROOT CAUSE** (isolated by a controlled test), **HYPOTHESIS** (not isolated) or **VERIFIED EFFECT** (a fix's effect against the matched control).

Root-cause evidence is in `docs/audit/ROOT_CAUSES.md`. Raw artifacts are in `runs/audit/*` and `runs/fix/*`.

## Summary

1. **The evaluation report overstates what it measures.**
   - Two "baselines" are oracles: they read `ground_truth`.
   - ECE defaults to 0 for non-probabilistic systems.
   - The counterfactual metric uses 2 pairs.
   - 363 cases collapse to **81 templates**, so the honest 95% CI for the headline 41.6% is **[0.28, 0.56]**, not [0.37, 0.47].
   - Expected chance accuracy on this benchmark is **0.37**.
   - 71% of locked-test cases share a template with dev/val. That is why a 35-case TF-IDF matcher "beats" VerQen.
2. **The model had four real defects:**
   - lexical bait (it picks options *mentioned* in the state);
   - rule skills bound to training templates;
   - no information-sufficiency signal;
   - a risk-control guarantee that holds only on average over an easy calibration mixture.

   The recency-bias and number-tokenization hypotheses were tested and **refuted** as primary causes.
3. **Data-level fixes** were trained in a fixed-compute 2×2 ablation against a matched control. The pre-registered winner, **AB** (compositional rules + role-aware distractors), gives:
   - **replicated across 2 seeds:** held-out probe aggregate **+7.5 pt** vs control; distractors **+21 pt** (bait +27 pt); missing information +12 pt;
   - no regression on 16 real benchmarks or procedural tests, and invariance intact;
   - one replicated cost: stability under *irrelevant* edits −10 pt;
   - the counterfactual both-correct gain seen on seed 0 (+19 pt) did **not** replicate.

   **Artifact-G shows no effect:** AB − BASE is −0.4 pt over two seeds, while two *control* seeds alone differ by 2.5 pt. The single-seed "+2.2 pt" was noise.
4. **Escalation fixes:**
   - **Cost-aware escalation:** silent high-stakes failures 17 → 0 and economic loss 1142 → 48.7. This is an opt-in configuration that uses the declared decision costs.
   - **Group-conditional risk control:** makes the guarantee honest. It shows the model *cannot certify* 7 of 13 real domains at ≤ 20% risk.
5. **Not fixed.** Threshold and boundary comparison on unseen phrasings stays near chance (0.48 → 0.54). The fix taught the generator's templates, not the abstraction.

---

## A. Root-cause matrix

| Observed failure (Artifact-G) | Root cause | Evidence | Status |
|---|---|---|---|
| Oracle baselines "beat" VerQen; baseline ECE = 0 | E1, E2: harness constructs baselines from `ground_truth`; ECE defaults to 0 | `harness/baselines.py`, `metrics.py` | **ROOT CAUSE**, reported next to the original metrics in `corrected_eval.py`; the harness is unchanged |
| CF sensitivity 0% | E3: only 2 valid `pair_cf` pairs; 23 one-sided pair ids | `corrected.json` pairs | **ROOT CAUSE** (metric). The model's CF defect is measured by a separate 80-pair probe (RC-2) |
| "15 catastrophic errors on FAM-N" | E4: counted over all families (M 8, B 1, E 1, G 2, N 2, A 1) | `corrected.json` high_stakes | **ROOT CAUSE** (attribution) |
| Narrow CIs, TF-IDF 57% | E5: 81 templates; 71% dev/test template overlap | `corrected.json` | **ROOT CAUSE** |
| FAM-F 0%, D/M/N pick one named option | RC-1: lexical bait; training data rewards mention (P(mention\|correct) 0.14 vs 0.07) and never shows mentioned-but-wrong | Bait isolation: 64% of errors on the mentioned option vs 33% chance | **ROOT CAUSE**; Fix B **VERIFIED** on held-out probes |
| Recency ("last sentence wins") | H1 | Directive-last is the *worst* position, not the best | **Refuted** |
| FAM-B/K boundary errors, parity always "yes" | RC-2: comparison skill bound to training template | Refund template 90–100% at ±1; new phrasings at chance at every Δ | **ROOT CAUSE**; Fix A **not verified** (see E) |
| Number tokenization | H3 | 29/30/31 are single tokens; in-template ±1 is perfect | **Refuted** as primary cause |
| Role/permission flips wrong, SLA moves the wrong way | RC-2 (binding) | Role mismatch granted 34/40, right 0/40; SLA wrong sign 19/40 | **ROOT CAUSE**; **not fixed** (AB both-correct 0.23 on seed 0 but 0.05 on seed 1 vs control 0.03) |
| FAM-D missing information picks an action | RC-3: no info-sufficiency family | Info option chosen as often when info is present | **ROOT CAUSE** (n = 8 base cases); AB improves the probe 0.41 → 0.53 (2-seed mean, replicated) |
| Accuracy-on-kept ≈ 0.54 at every risk level | RC-4: risk threshold is marginal over the calibration mixture; selective head AUROC below max-prob | Accuracy-on-kept at target 0.1 is < 0.9 on 15/16 real benchmarks | **ROOT CAUSE**; Fix C makes it honest; it does not make it good |
| High-stakes silent failures | RC-5: escalation is cost-blind | 15 silent failures, error_cost ≥ 20 | Fix D **VERIFIED**: 17 → 0 (opt-in config) |
| OOD (FAM-L) | Overlaps RC-2 | — | **HYPOTHESIS**: no separate mechanism isolated; FAM-L 0.47 → 0.53 (base) → 0.53 (AB), unchanged |

## B. Code changes

Nothing in `verqen_benchmark/` was modified: no labels, no cases, no harness, no scoring.

| File | Change | Default behaviour changed? |
|---|---|---|
| `verqen/data/procedural/rules.py` (new) | Fix A: compositional threshold and role rules emitted as counterfactual families (base / flip / irrelevant / missing-unknown-conflict), identical option sets | n/a (training data) |
| `verqen/data/procedural/discourse.py` (new) | Fix B: one authoritative directive plus wrong actions in 7 non-authoritative roles, with role-swap twins | n/a |
| `scripts/build_fix_data.py` (new) | Builds arms; **decontamination**: option ids must be disjoint from benchmark + probes (hard fail), 13-gram filter, FineWeb shard separation, held-out split de-duplication | n/a |
| `verqen/calibration/fit.py` | `worst_group_risk_control` (Fix C): per-group CP thresholds with Bonferroni; score chosen by worst-group AUROC **on the calibration split** | No (new key) |
| `verqen/inference/engine.py`, `api.py` | `risk_mode` ("marginal" default / "worst_group"), `risk_domain` (group-conditional), per-question `cost` (Fix D), `fast` flag exposed on `from_pretrained` | **No.** Default output is byte-identical (tested) |
| `verqen/training/decisions.py` | `max_steps` override for fixed-compute runs | No |
| `scripts/run_ablation.py`, `scripts/recalibrate_worst_group.py` (new) | Ablation driver (each GPU stage in its own process) and Fix C calibration | — |
| `scripts/audit/*` (new) | `repro_benchmark.py`, `probes.py`, `run_probes.py` (attention instrument validated against SDPA to 6e-7), `confirm_root_causes.py`, `rerun_cf_missing.py`, `corrected_eval.py`, `compare_arms.py` | — |
| `tests/test_audit_fixes.py` (new, 13 tests) | Generator label correctness (metamorphic re-parse), minimal-pair structure, decontamination, worst-group control, cost-aware escalation, default-unchanged, option-order invariance with cost, determinism, group-conditional thresholds | — |

**New hyperparameters**

| Name | Default | Purpose and rationale | Tuning protocol and validation |
|---|---|---|---|
| Fix-data size | 40K rules + 40K discourse | About 15% each of the fine-tune mix; enough to be seen ~1.5× in 1500 steps | Fixed a priori; not tuned |
| Fine-tune | 1500 steps, LR 5e-5, head ×2 | Short continued training from release, identical across arms | Fixed a priori; not tuned |
| `min_group` | 150 | Smallest calibration group with a usable CP bound | Fixed a priori |
| `delta` | 0.05 | Same as the existing risk control | Unchanged |
| Invariance tolerance for selection | TV ≤ 0.02 | Admits bf16 noise only (release max 0.0095) | Set before results |

## C. Before / after metrics

The **control** is BASE: the same init, steps, optimizer and replay data, with no new families. "Release" is the shipped v2 model. Real = mean over 16 held-out official benchmarks (≤ 3000 each).

| Metric | Release | BASE (control) | A | B | **AB (selected)** | Δ AB − BASE |
|---|---|---|---|---|---|---|
| Held-out probe aggregate | 0.470 | 0.454 | 0.463 | 0.488 | **0.563** | **+0.109** |
| · distractor (570 probes) | 0.660 | 0.588 | 0.535 | 0.839 | **0.835** | +0.247 |
| · distractor with bait | 0.504 | 0.437 | 0.396 | 0.733 | **0.752** | +0.315 |
| · boundary (264 probes) | 0.485 | 0.477 | 0.515 | 0.496 | 0.538 | +0.061 |
| · missing info (131 probes) | 0.345 | 0.405 | 0.502 | 0.382 | **0.566** | +0.160 |
| · CF relevant both-correct (80 pairs) | 0.013 | 0.038 | 0.100 | 0.025 | 0.225 | +0.188 (**did not replicate**: seed 1 0.05) |
| · CF irrelevant stable (80 pairs) | 0.850 | 0.763 | 0.663 | 0.700 | 0.650 | −0.113 |
| Real benchmarks mean | 0.5065 | 0.4981 | 0.4961 | 0.5002 | 0.5018 | +0.004 |
| Procedural test / OOD / adversarial | .959/.902/.961 | .964/.900/.963 | .961/.900/.964 | .967/.901/.963 | .965/.900/.964 | ±0.001 |
| Artifact-G accuracy (harness) | 0.416 | 0.4325 | 0.4545 | 0.4022 | 0.4545 | +0.022 (p = 0.33) |
| Artifact-G cluster CI95 | [.28, .56] | [.31, .56] | [.34, .58] | [.27, .53] | [.33, .57] | — |
| Artifact-G locked-test only | 0.434 | 0.458 | 0.438 | 0.422 | 0.474 | +0.016 |
| Artifact-G ECE | 0.215 | 0.192 | 0.150 | 0.248 | 0.171 | −0.021 |
| Artifact-G silent high-stakes | 15 | 20 | 10 | 27 | 17 | −3 |
| Artifact-G economic loss | 711 | 1228 | 585 | 1249 | 1142 | −86 |

**AB escalation configurations** (the same weights, reported separately):

| Configuration | Coverage | Accuracy on kept | Economic loss | Silent high-stakes |
|---|---|---|---|---|
| Default (marginal risk 0.1) | 0.51 | 0.543 | 1142 | 17 |
| + cost-aware (Fix D, uses declared costs) | 0.25 | 0.644 | **48.7** | **0** |
| Worst-group (Fix C), domain unknown | 0.00 | — | 69.7 | 0 |

**Latency.** The default eager path takes 39.5 ms. `fast=True` takes **4.6 ms** (8.5× faster) with 98.6% decision agreement with eager (bf16 weights; max |Δp| 0.048). It stays opt-in because it is not bit-identical.

**Training cost.** Each arm took about 20 min of GPU time on one RTX 5060 Ti. Parameters are unchanged (125.4M), and VRAM at inference is unchanged.

### BEFORE → AFTER → DELTA → CONFIDENCE → REGRESSION

| Change | Before (control) | After | Delta | Confidence | Regression |
|---|---|---|---|---|---|
| Fix B alone (1 seed) | Distractor probes 0.588 | 0.839 | +0.25 | n = 570, SE ≈ 0.02 | Artifact-G FAM-F 0.76 → 0.18 (one template near a tie, see E2) |
| Fix A + B (2 seeds) | Distractor probes 0.609 (mean) | 0.824 | **+0.215** | **Replicated**: both AB seeds (0.835 / 0.812) > both control seeds (0.588 / 0.630) | CF-irrelevant stability −0.106 (replicated) |
| Fix A + B (2 seeds) | Missing-info probes 0.405 | 0.528 | +0.123 | Replicated: AB 0.566 / 0.491 vs control 0.405 / 0.405 | — |
| Fix A + B (2 seeds) | Probe aggregate 0.466 | 0.541 | +0.075 | Replicated: AB 0.563 / 0.519 vs control 0.454 / 0.478 | — |
| Fix A + B (2 seeds) | Artifact-G 0.4449 | 0.4408 | −0.004 | **No effect**: the control seed spread alone is 0.025 | — |
| Fix C (worst-group) | Marginal: 15/16 real sources violate target 0.1 | Group-certified or escalate | — | Exact per group (CP, Bonferroni) | Coverage → 0 when the domain is unknown |
| Fix D (cost-aware) | 17 silent high-stakes, loss 1142 | 0, loss 48.7 | −17 | Deterministic rule | Coverage 0.51 → 0.25 |

## D. Regression report

- **Permutation invariance:** pytest (exact in fp32) passes. Suite option-reversal TV max: release 0.0079, AB 0.0066. The harness invariance pairs are 12/12 consistent for every arm.
- **Question isolation:** prepend-question TV max: release 0.0095, AB 0.0095.
- **Determinism:** the repeated-call test passes. Harness repeated stability is unchanged in code path.
- **Contradiction handling (FAM-E).** Case-level accuracy looks like a regression: release 0.83, BASE 0.83, A 0.39, B 0.17, AB 0.39. **Template-level, it is not.** 13 of the 18 FAM-E cases are one cargo-weight template sitting at a three-way near-tie (quarantine 0.33, discard 0.35, accept 0.30), which AB tips the other way. On the 6 distinct templates, template-macro accuracy is BASE 0.500 and AB 0.551. AB newly gets the identity-fraud conflict right; both miss the VIP-vs-final-sale and AV-vs-EDR templates. **Contradiction handling is preserved** (VERIFIED at template level), but this template is fragile in every arm.

Template-macro accuracy per family (each template weighted equally; `runs/fix/template_macro_by_family.json`):

| Family | Release | BASE | A | B | AB |
|---|---|---|---|---|---|
| A core | .382 | .407 | .423 | .431 | .365 |
| B boundary | .150 | .233 | .150 | .300 | .183 |
| C ambiguous | .286 | .286 | .286 | .000 | .333 |
| D missing info | .429 | .429 | .714 | .571 | .571 |
| E contradictory | .500 | .500 | .551 | .500 | .551 |
| F distractors | .000 | .289 | .333 | .067 | .311 |
| G adversarial | .500 | .500 | .611 | .389 | .611 |
| H counterfactual | .500 | .278 | .211 | .211 | .378 |
| I invariance | .562 | .625 | .662 | .719 | .656 |
| J | .750 | .750 | .750 | .750 | .750 |
| K | .194 | .375 | .236 | .417 | .292 |
| L OOD | .449 | .472 | .699 | .426 | .517 |
| M high-stakes | .333 | .333 | .356 | .333 | .422 |
| N high-stakes | .250 | .500 | .500 | .500 | .250 |
| **All templates** | **.372** | **.416** | **.456** | **.396** | **.429** |

- **Real benchmarks:** AB − BASE is within ±1.5 pt on each of the 16, with a mean of +0.4 pt.
- **Environments:** graph-route optimal-action rate fell 0.84 (release) → 0.68 (BASE) / 0.70 (AB). The drop appears in the control too, so it comes from fine-tuning without the RLCD stage, not from the fixes. **Open:** re-run environment RLCD on the selected model.
- **Tests:** 24/24 passing: 11 core plus 13 new.
- **Seed replication** (`runs/fix/seed_replication.json`). BASE and AB were each retrained with a second batch-order seed.

  | Metric | BASE s0 | BASE s1 | AB s0 | AB s1 | Δ (means) | Verdict |
  |---|---|---|---|---|---|---|
  | Probe aggregate | .454 | .478 | .563 | .519 | +.075 | replicated |
  | Distractor | .588 | .630 | .835 | .812 | +.215 | replicated |
  | Distractor with bait | .437 | .474 | .752 | .708 | +.274 | replicated |
  | Missing info | .405 | .405 | .566 | .491 | +.123 | replicated |
  | Boundary | .477 | .504 | .538 | .519 | +.038 | small; both AB > both BASE |
  | CF relevant both-correct | .038 | .025 | .225 | .050 | +.106 | **not replicated** |
  | CF irrelevant stable | .763 | .825 | .650 | .725 | −.106 | **replicated regression** |
  | Real benchmarks mean | .4981 | .4980 | .5018 | .4998 | +.003 | no change |
  | Procedural / OOD | .964 / .900 | .962 / .899 | .965 / .900 | .962 / .900 | 0 | no change |
  | Artifact-G accuracy | .4325 | .4573 | .4545 | .4270 | −.004 | noise (McNemar p = 0.24 per seed pair) |
  | Artifact-G silent high-stakes | 20 | 36 | 17 | 30 | — | too noisy to use |
  | Invariance TV max | .017 | .016 | .009 | .010 | — | intact |

### D.1 Environment RLCD re-applied to AB (`runs/fix/AB_rc`, 24–25 Sep, 1.5 h, one seed)

| Metric | AB (before RLCD) | AB_rc (after) | Δ | Verdict |
|---|---|---|---|---|
| graph_route optimal-action rate | 0.700 | **0.945** | +0.245 | **regression repaired** (release was 0.84) |
| grid_nav optimal-action rate | 0.472 | 0.559 | +0.087 | improved |
| tictactoe / tetris / browser | .982 / .918 / 1.0 | .996 / .917 / 1.0 | ≈0 | unchanged |
| Real benchmarks mean (16) | 0.5018 | 0.5065 | +0.005 | no change (each within its CI) |
| proc_test / proc_test_ood | .9645 / .9000 | .9595 / .9095 | −.005 / +.010 | no change |
| Probe aggregate | 0.563 | 0.530 | −0.033 | see bait row |
| Distractor with bait | 0.752 | 0.696 | −0.056 | **PLAUSIBLE regression**: 7/9 cells down, 0 up (sign p ≈ 0.016), unpaired z = −1.45 |
| Missing info | 0.566 | 0.563 | 0 | unchanged |
| CF relevant both-correct | 0.225 | 0.125 | −0.10 | 40 pairs; within the seed spread above |
| Artifact-G accuracy / template-macro | .4545 / .4287 | .4573 / .4136 | +.003 / −.015 | noise at this benchmark's resolution |
| Jev-parity suite (7,840 cases, chance 0.216) | — | 0.208 | — | **at chance**; T1 release 0.206. Neither model saw v3 domains or phrasing |

**An apparent RLCD failure that was not one.** During RLCD, training-stream decision accuracy fell 0.96 → 0.90 and the objective's own expected reward fell 0.92 → 0.83.
- The hypothesis that environment gradients were swamping the decision objective was **refuted**: the *frozen pre-RLCD* model also scores 0.912 on the same later batches, so batch order explains the decline.
- On those batches the post-RLCD model scores 0.979.
- **Fix:** RLCD now logs a fixed held-out val slice (`rlcd_val`), so this confound cannot recur.
- For T2, the release is chosen by a pre-registered val rule that includes a per-variant check covering bait (`production/select_release.py`).

## E. Remaining problems

1. **Boundary/threshold abstraction is not fixed.** Held-out phrasings reach 0.54 (chance ≈ 0.5). Fix A raised in-generator accuracy to 72%, not 100%, and did not transfer. **HYPOTHESIS:** 85M encoder parameters and about 60K rule examples are not enough to induce a phrasing-independent comparison. A number-aware input (digit-level tokens or explicit numeric features) or far more template diversity is needed.
2. **Benchmark volatility.** FAM-F is 15 near-identical cases around one near-tie (p ≈ 0.45 vs 0.50). A fine-tune without new data moved it 0.00 → 0.76, and AB/B moved it again. Per-family deltas on Artifact-G are not evidence without paired controls. B's FAM-F failure reveals a real coverage gap in Fix B: a "scheduled future announcement" role. **This gap was found by reading locked-test cases, so it was deliberately not patched.** Fixing it and re-scoring on the same test would be test-set tuning. It needs a fresh evaluation set.
3. **Risk control cannot be certified on 7/13 real domains** (selective AUROC 0.57–0.65). The learned selective head is no better than max-prob off-distribution.
4. **CF-irrelevant stability dropped** by 10 pt, replicated across seeds. The irrelevant-change variants in Fix A (name and filler swaps) were not enough to counter the extra sensitivity the counterfactual pairs taught. **HYPOTHESIS:** an explicit paired consistency loss (a KL penalty between irrelevant twins) is needed, not just more data.
5. **No measurable Artifact-G effect.** Two control seeds differ by 2.5 pt, which is more than any fix moved it. With 81 templates, Artifact-G cannot resolve effects this size. The held-out probes (≥ 570 items per axis, templates independent of training) can.
6. ~~Environment RLCD not re-applied~~ **Resolved (D.1):** graph-route 0.70 → 0.945. RLCD may cost some bait-distractor robustness (PLAUSIBLE, one seed); T2 guards it with the per-variant release rule.

## F. Recommended next experiment

**Numeric grounding ablation plus a fresh, independently written evaluation set.**

1. Add digit-split number tokenization (each digit a token, with place-value position ids) as arm N, versus the current BPE. Both arms get AB data plus 3× more rule-template diversity, generated from 30+ independently written frames.
2. Hold out *frames*, not just values: train on 70% of frames and test on the rest plus the audit probes.
3. Validate on a **new** decision benchmark written by someone who has not seen the generators, with ≥ 10 templates per family and paired counterfactuals by construction. Report cluster-bootstrap CIs and paired tests against a matched control, with ≥ 3 seeds.
4. Re-run environment RLCD on the chosen model and replace the selective head's escalation score with max-prob if it still wins worst-group AUROC on calibration.
