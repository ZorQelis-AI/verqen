# Targets for the next VerQen run (from the T2 run, 24–25 Sep 2026)

This is a running list. Each item says what was OBSERVED and what to change. Items from the T2-vs-Laya head-to-head get added when those results land.

## 1. Multi-step (agentic) decision data. Highest priority.

**OBSERVED.** Before RLCD, T2 trailed T1 on the multi-step environments (optimal-action rate):

| Environment | T2 | T1 |
|---|---|---|
| grid_nav | 0.105 | 0.47 |
| graph_route | 0.54 | 0.70 |
| browser | 0.48 | 1.0 |

At the same time, T2 reached 88.2% on single-step validation.

**Cause.** In v3, T1's procedural families (games, grids, routes) shrank to about 13% of 1.5M decisions. The rest are single-step "case → action" decisions. The model sees few examples of carrying state across steps.

**Change.**
- Add a trajectory generator: roll each environment out with its oracle policy (plus perturbed, recoverable detours) and emit every step as a decision example. The state shows the task, the history of earlier actions and observations, and the current observation; the target is the next oracle action.
- Include episodes where an earlier step was wrong, so the model learns recovery.
- Add browser/tool-use trajectories built from the `browser` and `tools` domain specs: multi-step forms, search, argument gathering.
- Target about 10–15% of the decision mix, and track it as its own capability in the Jev-parity suite (a `multi_step` column).
- Give RLCD more of the budget if these skills matter (T2: 2 h, single GPU).

## 2. Preflight must probe every GPU stage, RLCD included

**OBSERVED.** RLCD ran out of memory 30 s into the stage (31.30 of 31.36 GiB). Preflight had measured pretraining, context extension and decisions, but never RLCD, which holds a second (reference) model and runs an environment-gradient pass on top.

**Change.** Add an RLCD probe (policy + reference + one decision step + one environment chunk at `max_seq_len`) that sets `decision_rows` / `env_chunk_rows`. The T2 fix was a hand-set third of the decisions micro-batch (8 rows) and 4 rows per environment chunk. The rule is the same as before: *a stage that has never run has never been tested.*

## 3. Every evaluation set in the pretraining blocklist from the start

**OBSERVED.** Four test splits (emotion, banking77, sst5, massive_en) joined the evaluation after the corpus was tokenized, so the pretraining text was never 13-gram filtered against them. This is disclosed in the build report and `evals/HEAD_TO_HEAD.md`.

**Change.** Fix the full evaluation list, including any competitor's benchmarks, before `prepare_data.sh` runs.

## 4. Run the whole evaluation suite on a toy checkpoint before renting

**OBSERVED.** Three evaluation paths broke only on the box, on real data:
- the suite looked for T1's `proc_test.jsonl`;
- the fast engine capped options at 64 and inputs at 512 tokens.

**Change.** The smoke chain ends with `production/finalize.py` on the smoke checkpoint over the v3 data layout, with a 77-option and a 1,500-token case included.

The smoke data also needs an `eval/` directory. `data/decision/v3_smoke` has none, so the suite dies with `StopIteration` looking for `proc_v3_test` (found 25 Sep while smoke-testing the agentic top-up; the box data is complete). The suite should also degrade instead of crashing when a split is missing.

## 5. Knowledge and languages (the Laya gap)

- **Knowledge.** Laya's ModernBERT-large backbone saw about 2T tokens; T2 saw 7B. Closing that needs compute, not tricks. 2× H100 for 25 h fits about 360M parameters on about 12B tokens.
- **Languages.** Laya-Multilingual covers 100+ languages; T2 is English-only. A multilingual variant needs a multilingual tokenizer and corpus (for example FineWeb-2) and multilingual decision data (MASSIVE covers 51 languages).

## 6. Conflicting and unknown facts: the model learned sentences, not the concept

**OBSERVED** (T2 release, 25 Sep). Jev-parity renders these cases with the engine's held-out frame 0 wording.

| Case | Jev-parity |
|---|---|
| "Two sources disagree on …" (conflict) | 2.4% |
| "… is not stated anywhere in the file" (unknown) | 1.4% |
| Fact simply absent | 97.8% |
| Training-wording val rows (both kinds) | 100% |

A second set of unseen paraphrases (`data/decision/info_para/val_B`) gives the same picture: conflict 4.7%, unknown 24.6%. The model confidently decides from one of the two values.

**Cause.** The engine has four phrasings per case (`CONFLICT` / `MISSING` in `verqen/data/v3/engine.py`). Frame 0 is held out, so training saw three phrasings, about 45K rows each. That was enough to memorise the sentences.

**Fix used in the T2 top-up.** `scripts/v3/augment_info.py` rewrites 50K training rows with 18 conflict and 14 unknown phrasings (pool A). The frame 0 wording is excluded, and pool B is held out for measurement. This fix was chosen AFTER seeing the Jev-parity numbers, so T2's post-fix conflicting / missing_info Jev-parity scores must be reported with that caveat. `val_B` is the clean transfer measure.

**Change for the next run.**
- Give the engine 20+ phrasings per case from the start. The same applies to every other fact-reporting frame.
- Add "two sources AGREE" and "conflict on a non-decisive fact" counter-cases, so the model cannot learn "two values mentioned → gather".
- `long_state` (41%) is the next-weakest capability and needs its own look.

## 7. The T2 decision data had no distractor prose. Silent path bug.

**OBSERVED** (26 Sep). The decision data was rebuilt on a new box with the FineWeb shards at `data/pretrain/raw/sample/10BT/`. It did not reproduce the original:

| | Original | Rebuilt |
|---|---|---|
| Release val accuracy | 87.41% | 86.17% |
| info_missing | 100% | 65.6% |

- Synthetic val rows that now contain a prose sentence score 90.1%; with the prose lines removed, the same rows score 98.6%. The labels are fine.
- `build_data.first_shard_prose` looked only in the PC layout. On the rented box, `prepare_pretrain.py` had stored FineWeb under `data/pretrain/v3/raw/fineweb_edu/...`, so it got `[]`.
- Every one of the 1.5M original decisions was therefore built WITHOUT the distractor prose the engine was designed to add. T2 never learned to ignore unrelated text inside a case. This is part of why long_state scored 24% when the filler sits after the case.

**Fixed.** `first_shard_prose` searches both layouts and exits if it finds none. The agentic top-up trains on the rebuilt (prose) data.

**Change for next time.** Every optional input a builder can silently skip must be a hard failure. Add a gate: the share of rows containing distractor prose must be > 0 before training starts.

## 8. Smaller items

- `finalize.py --upload-stages` failed silently for the first stage. Test every CLI mode in the smoke chain.
- Log readers: a stage log can lag behind the GPU by minutes. Judge progress by step *times* between two readings, never by one snapshot.
