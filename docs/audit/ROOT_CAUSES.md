# VerQen-T1: root causes behind the Artifact-G failures

This was written on 24 Sep 2026, before any model code was changed.

Every number here comes from the release checkpoint `runs/verqen-t1-v2/release` (sha in `runs/audit/baseline/config.json`).
- The Artifact-G benchmark was reproduced exactly with the unchanged harness (`scripts/audit/repro_benchmark.py` → `runs/audit/baseline/`).
- Diagnostic probes come from my own templates plus random FineWeb-Edu sentences (`scripts/audit/probes.py`, `run_probes.py`, `confirm_root_causes.py`, `rerun_cf_missing.py` → `runs/audit/probes_baseline/`). No benchmark text, label or template is used.

Each item carries one of these labels:
- **OBSERVED**: measured.
- **ROOT CAUSE**: supported by a controlled test that isolates it.
- **HYPOTHESIS**: plausible but not yet isolated.

---

## RC-0: evaluation defects (they change the report's conclusions, not the model)

| # | Defect | Evidence (OBSERVED) | Consequence |
|---|---|---|---|
| E1 | Baselines "General-purpose AI" and "Human expert" are built from `case["ground_truth"]` plus `hash()` noise | `harness/baselines.py` | They are oracles, not baselines. "VerQen vs GPT-class" comparisons are invalid. |
| E2 | ECE defaults to 0.0 for systems that output no probabilities | `metrics.py` | Heuristic and oracle baselines get a perfect ECE by construction. |
| E3 | Counterfactual sensitivity counts only `pair_cf` pairs where both answers are correct and different: n = 2. The syllogism pairs have a one-sided `pair_id`. | dataset + `metrics.py` | The CF headline is a 2-sample statistic. |
| E4 | The "15 catastrophic errors on FAM-N" figure is `high_stakes_catastrophic_error` = wrong, not escalated and cost ≥ 20 across **all** families | `error_analyzer.py` | Misattributed to FAM-N. |
| E5 | Cases are heavily templated, so the effective n is far below 363 | `builder.py` | The per-case binomial CIs in the report are too narrow. |
| E6 | The report says VerQen "mean-pools a state embedding" | model code | False. DECIDE attends to state tokens through the branch mask. The mechanism story in the report is a hypothesis at best. |
| E7 | The harness calls the default (eager) engine | `evaluator.py` | The 34.9 ms latency is the eager path. The compiled path measures 4.8 ms. |

Fix policy: the original metrics stay untouched. A separate corrected analysis is added next to them (see `scripts/audit/corrected_eval.py`).

---

## RC-1: lexical bait. The model picks options that are *mentioned* in the state, whatever role the mention plays.

- **OBSERVED (benchmark).**
  - FAM-F: 15/15 wrong answers chose the action named in a distractor sentence.
  - FAM-D, FAM-M, FAM-N: the wrong picks concentrate on one option that is named in the state.
- **OBSERVED (probe, `distractor`).** The directive sentence names the right action, while k FineWeb sentences add noise. In the "named" condition the distractors also mention wrong actions as past suggestions.

  | condition | accuracy |
  |---|---|
  | k=1, no bait | 0.87 to 0.97 |
  | k=1, bait | 0.47 to 0.70 |
  | k=3, no bait | 0.60 to 0.83 |
  | k=3, bait | 0.43 to 0.50 |
  | k=6, no bait | 0.57 to 0.80 |
  | k=6, bait | 0.40 to 0.63 |

  Mean over cells: no-bait 0.78 vs bait 0.50 (−28 pt).
- **ROOT CAUSE (isolation).** With k=1 only one wrong option is mentioned. 64% of errors land on it, against 33% expected by chance (`confirm.json` B).
- **ROOT CAUSE (why it was learned).** In the real multiple-choice training data, an option whose content words appear in the passage is correct twice as often (P(mention | correct) = 0.14 vs P(mention | wrong) = 0.07; 8% sample of train.jsonl). The procedural data is neutral on this. **No training family contains an option that is mentioned but wrong because of its discourse role**: a rejected suggestion, history, negation, a hypothetical, or a superseded instruction. Lexical overlap is therefore a rewarded feature, and nothing ever penalises it.
- **HYPOTHESIS H1 (recency) refuted.** The directive in the *last* position is not favoured; it is the worst position (no bait, k=6: first 0.80, last 0.57). The benchmark's "last-sentence action" pattern is explained by bait, not recency.
- **OBSERVED (attention, weak evidence).** Averaged over the last 4 layers, the DECIDE token puts roughly the directive's token share of attention on the directive: ratio ≈ 1.0 to 2.2. It does not concentrate on the authoritative clause. Attention is not causal, so this is corroboration only.

## RC-2: rule application is bound to training templates. There is no abstract threshold, equality or information-sufficiency skill.

- **OBSERVED (probe, `boundary`).** Four new threshold phrasings, v = T−5 … T+5. Accuracy is at chance at *every* distance from the threshold (0.29 to 0.71, no trend), and the first option is chosen 62 to 88% of the time.
- **ROOT CAUSE (isolation).** In the training template (`refund_policy`), the same model is 90 to 100% accurate at *every* delta, including ±1 (`confirm.json` A). The skill exists but is tied to that template.
- **H3 (number tokenization) refuted as the primary cause.** 29/30/31 and 89/90/91 are single tokens, and the in-template accuracy at ±1 is perfect. The inconsistent BPE splits (649 → "6"+"49", 651 → "65"+"1") are a secondary risk, not the explanation.
- **OBSERVED (probe, counterfactual, confound fixed).**
  - SLA pairs: the model responds to the minutes value *systematically with the wrong sign*. As the value crosses above the target, it moved "breached → within" in 19/40 pairs and the correct way in 1/40.
  - Role pairs: when the requester's role does not match the only permitted role, it still grants access in 34/40 and is right in 0/40. It does not check role equality.
  - Irrelevant changes (name, a synonym) flip 10 to 20%. That is about the size of the relevant-change flips, so the model is barely more sensitive to the decisive variable than to irrelevant ones.
- **ROOT CAUSE (data).** Each rule type exists in about one fixed template:
  - refund thresholds only in `refund_policy`;
  - no role or permission family;
  - no rule family with paraphrased operators ("at least", "exceeds", "no more than", "within", "under");
  - no family that pairs a case with its counterfactual twin.

  The cheapest solution to the training loss is template-specific features, and the model found it.

## RC-3: no information-sufficiency signal

- **OBSERVED (two probe sets, 8 rule cases).** The "request/verify info" option is chosen about as often when the information **is present** (complete: 58% and 50%) as when it is missing, unknown or contradictory (33 to 67%, 40%). Accuracy on complete cases is only 12 to 30%, because the info option attracts picks through lexical overlap ("weigh_parcel_first" ↔ "weighs").
- **ROOT CAUSE (data).** Info-request or "cannot be determined" answers appear only in `logic` ("cannot be determined") and `intent`. No family asks for a decision where the correct action is to obtain the missing or conflicting variable, and no family contrasts that with the complete case. **Caveat:** n = 8 cases, so the effect size is uncertain. The direction agrees with RC-1.

## RC-4: the risk-controlled escalation guarantee is marginal over an easy mixture, and the selective head adds nothing off-distribution

- **OBSERVED (existing eval report).** At target risk 0.10, accuracy on the kept cases is far below 0.90 on most real benchmarks: MMLU 0.34, AQuA 0.33, MedMCQA 0.40, ARC-C 0.46, ARC-E 0.60, CSQA 0.61. Procedural data sits at 0.99.
- **ROOT CAUSE.** `calibration/fit.py` fits one global Clopper-Pearson threshold on `calib.jsonl`, which is 26% procedural and 74% real across 16 sources. The guarantee is *marginal over that mixture*: easy procedural and MNLI cases subsidise hard sources. Per-source risk is not controlled.
- **OBSERVED (benchmark predictions, analysis only, nothing tuned).** The AUROC for separating right from wrong is p_correct 0.635, max-prob 0.651 and margin 0.661. The learned selective head is *worse* than the raw max-probability.
- **HYPOTHESIS.** Selective-head labels are dominated by easy procedural positives, so it learned "looks procedural → confident". Untested.

## RC-5: high-stakes and OOD

- **OBSERVED.** The high-stakes failures (FAM-M/N/D) are RC-1 bait errors that carry a high cost. Nothing in the model or API uses cost. Escalation depends only on p_correct.
- **ROOT CAUSE.** This is a product and API gap: decisions are cost-blind. A cost-aware escalation rule belongs in the decision layer as an explicit, documented policy. It must not be a benchmark-specific special case.
- **OOD (FAM-L).** The OOD failures overlap with RC-2 (new phrasings). No separate mechanism has been isolated. **HYPOTHESIS:** it will improve with RC-2's fix. That will be verified, not assumed.

---

## Fix design (principled, data and objective level, plus ablations)

| Fix | Targets | What | Not allowed / guarded against |
|---|---|---|---|
| **A: compositional rule curriculum** | RC-2, RC-3 | A generator composes predicates (numeric thresholds with 10+ operator phrasings and many units and nouns; role/set membership; conjunction/exception) with case facts. Every example is emitted as a **counterfactual family**: the base case, the decisive variable flipped across the rule (label flips), an irrelevant variable changed (label fixed), and the decisive variable missing, unknown or conflicting (label = information option). The info option is present in all four, so its presence is no cue. | The vocabulary is disjoint from the probe templates. A 13-gram contamination check runs against the benchmark and the probes. |
| **B: role-aware distractors** | RC-1 | The authoritative directive names the correct action. Other sentences mention wrong actions in non-authoritative roles: rejected suggestion, history, negation, hypothetical, superseded instruction, customer demand. The wrong actions are often mentioned *more* times than the right one. The directive position is uniform, and there is random FineWeb filler. Contrast twins are included: the same sentences with the roles swapped. | The action vocabulary is disjoint from the probe ACTIONS and the benchmark options (checked). |
| **C: worst-group risk control** | RC-4 | The threshold is chosen so that the Clopper-Pearson upper bound on risk is ≤ α for **every** calibration source with n ≥ 150 (Bonferroni over groups), instead of for the pooled mixture. The confidence score (selective head vs max-prob vs margin) is chosen by worst-group AUROC **on the calibration split**. | Nothing is fitted on the benchmark. |
| **D: cost-aware escalation (API)** | RC-5 | An optional `cost` argument escalates when the expected loss of acting, (1 − p_correct) · error_cost, exceeds escalation_cost. It is off by default. When the harness is run with it, the result is reported as a separate configuration. | Not enabled silently for the benchmark. |

**Ablation protocol (pre-registered).**
- Arms: BASE, +A, +B, +A+B. Each is fine-tuned from `release` for the same fixed number of steps on the same replay subset of the original training data. The arms differ only in the added families.
- Every arm then gets the same calibration.
- **Selection rule, fixed before any results:** pick the arm with the best mean of held-out probe accuracy and real-benchmark mean accuracy. It must not regress by more than 1 pt on the procedural test or the real-benchmark mean, and invariance must stay at 100%. The Artifact-G locked test is reported for **all** arms and is never used for selection.
