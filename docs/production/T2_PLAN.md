# VerQen-T2: production run plan (2× RTX 5090, 25 h, from scratch)

**Goal:** beat TypeSafe AI's **Jev** on the things a decision model is used for. **Constraint:** from scratch, one rented 2× RTX 5090 box, 25 hours.

## 1. What we are competing against (public facts, Sep 2026)

| | Jev (TypeSafe, launched 15 Sep 2026) | Source |
|---|---|---|
| Output | Typed decisions (choice / score / noul) plus calibrated confidence; no text | TypeSafe blog; Wikipedia |
| Training | RLCD; "trained exclusively on synthetic data"; size and architecture undisclosed (observers suspect an open-weight LLM base) | Wikipedia |
| Accuracy | 67.8% aggregate on TypeSafe's *internal* workflow dashboard (best comparator 74.1%); per task 61.7–76.0% | layer3labs |
| Independent test | 60-case tool-risk benchmark: 91.7%, the same as Claude Sonnet 5; ECE 0.05–0.07; p50 380–420 ms | github.com/themsquared/jev-benchmark |
| Latency / price | 70–500 ms end to end (API); $0.042 per M input tokens | TypeSafe blog |
| Stated weaknesses | **Cannot do arithmetic, counting, date comparisons or multi-step reasoning**; about 32K-token input cap; vulnerable to adversarial prompting | layer3labs |

There is no independent head-to-head benchmark. "Beat Jev" is measured on our **Jev-parity suite** (§6), which has an adapter slot for Jev's API if early access is available.

## 2. Where T2 can win, and where it honestly cannot

| Axis | Plan | Expected |
|---|---|---|
| Latency | Local GPU, CUDA-graph engine (T1: 4.6 ms) | **Win**: ~6–10 ms vs 70–500 ms |
| Cost | Self-hosted, ~1,000 decisions/s per GPU | **Win**: ~$0.1–0.5 per M decisions vs ~$13 per M (300-token inputs at $0.042/MTok) |
| Arithmetic, counting, dates, thresholds | Digit-level tokenizer plus computed-label curricula | **Target win**: Jev states it cannot |
| Grounded rule application (policy + case → action) | Scenario engine: counterfactual families, role-aware distractors, info sufficiency | **Target win** on held-out phrasings |
| Calibration and risk control | RLCD plus per-group certified escalation (Fix C) plus cost-aware escalation (Fix D) | **Target win**: Jev publishes no ECE or risk guarantee |
| Adversarial / injection | Escaped control tokens, isolated branches, adversarial curricula | **Target win** |
| Missing information / conflicting evidence | Info-request and escalation options trained explicitly | Target win |
| World knowledge (MMLU-style) | 200M from-scratch encoder, ~4B tokens | **Likely loss** if Jev sits on an LLM base: a from-scratch model at 7.5e18 FLOPs cannot reach frontier knowledge; T1 MMLU is 29% |
| Context length | Native 2,048 tokens, plus chunked evidence scoring for longer states | Loss vs 32K natively; mitigated |

"Frontier-model intelligence" is not reachable from scratch on ~50 GPU-hours: frontier pretraining is ~10^25 FLOPs, and this run is ~10^19. This run concentrates its compute where decisions need it — reading the state, applying rules, numbers, calibration — instead of memorizing world knowledge it cannot afford.

## 3. Compute budget (measured basis, not guessed)

- A 5090 gives **31.4 GB usable** (LESSONS #27). DDP replicates, it does not pool.
- Prior measurement on this exact box type: SoreQen S1 reached 163K tok/s at 121M ≈ 1.2e14 FLOPs/s total (~52% MFU). Planning assumes **~1.1e14 FLOPs/s** for both GPUs.
- The **throughput probe runs first on the box** (`production/probe_throughput.py`: median of ≥ 3 timed steps after warmup). The token budget is wall-clock based, so it adapts automatically.

**Model: VerQen-T2-base, from scratch** (size fixed 25 Sep 2026). **27 layers** × 768 hidden, 12 heads × 64, SwiGLU 2048, RMSNorm, QK-norm, RoPE, branch mask, 7-layer RTD generator.
- **232.2M parameters**: 207.0M non-embedding, 25.2M embedding. Counted on the built model.
  - The earlier "195M" for 24 layers was an under-count: that shape is 210.9M.
  - A 216M option would therefore have been nearly the same model.
- Deeper than T1 (12 layers): deep-narrow is better for small models, and depth is what multi-step rule application needs.
- **Measured:** 15,916 tok/s on the 5060 Ti (compiled, micro 16, 12.4 GB peak). That is 0.82× the 24-layer model's 19,387.
- On the box that gives **3.2B–6.7B pretraining tokens in 14.5 h**, 15–32 per non-embedding parameter. At the low end of the throughput range, 232M is the compute-optimal size.

| Stage | Hours | Notes |
|---|---|---|
| 0. Pre-flight, env, data pull, tokenize | 0.75 | Parallel downloads; the box pulls, never pushed to (LESSONS #34) |
| 1. Pretrain (RTD), 512 ctx | 13.0 | DDP ×2, time-budgeted WSD, save every 20 min |
| 1b. Context extension to 2,048 | 1.5 | RoPE base ×8; the same objective continues |
| 2. Decision training (hybrid v3 data) | 4.5 | DDP ×2, resumable; the v3 data is built on the CPU *during* stage 1 |
| 3. RLCD (decision + environments) | 2.0 | Single GPU; GPU 1 evaluates the stage-2 checkpoint in parallel |
| 4. Calibrate (marginal + worst-group), eval, probes, Jev-parity, export, upload | 1.25 | Each stage in its own process |
| Buffer | 2.0 | Crash and resume, slow downloads |
| **Total** | **25.0** | A watchdog stops the instance after the upload marker |

### 3.1 Model size is measured on the box, not fixed in advance (added 25 Sep)

The proposal was a 360M model on 12–18B tokens. **VRAM decides what fits; compute decides how many tokens.** DDP keeps a full copy on each card, so 2 × 32 GB is not one 64 GB pool, and memory does not make a step faster.

The measured basis:
- Today, on the 5060 Ti, the T2 RTD setup (24 × 768 + 6-layer generator) ran at 19,387 tok/s, about 3.1e13 FLOP/s.
- A 5090 has about 4.4× the dense BF16 throughput and 4× the bandwidth, so the box is **~8× this PC at best**. SoreQen measured about 1.2e14 FLOP/s on this box type, which is ~4×.

| Candidate | Non-embedding params | Pretraining tokens in 14.5 h (1.2e14 → 2.5e14 FLOP/s) | Tokens per param |
|---|---|---|---|
| t2-195m (24 × 768; really 210.9M total) | 186M | 3.9B → 8.2B | 21 → 44 |
| **t2-232m (27 × 768), chosen** | 207M | **3.2B → 6.7B** (measured 0.82× of 24 × 768) | 15 → 32 |
| t2-360m (24 × 1024) | 357M | 2.0B → 4.2B | 6 → 12 |
| 360M with 12B / 18B tokens | 357M | would need **41–86 h / 62–130 h** of pretraining | — |

**Rule (`size_rule`; now forced to `t2-232m` by decision, and `auto` remains available).**
- Preflight measures each candidate's throughput and picks the one whose projected tokens are closest (in log) to 20 per parameter, the compute-optimal ratio. Around that optimum, quality is flat to within about ±50% of N, so the rule avoids the two real failure modes: badly under-trained or under-sized.
- At the SoreQen throughput it picks 195M. At the optimistic throughput it picks 360M.
- `size_rule: t2-360m` forces the larger model. The cost is about 1.9× inference compute per decision, which is still well under Jev's 70–500 ms.

**The corpus follows the budget.** `pretrain.data_scale` = projected tokens ÷ 3.95B planned, clipped to [1, 2.5]. Download and tokenization scale with it, so a fast box does not spend its compute repeating the corpus.

## 4. Training-pipeline audit (T1 → T2 fixes)

| # | Defect (OBSERVED in T1) | Fix in T2 |
|---|---|---|
| P1 | No multi-GPU anywhere | `verqen/training/distributed.py`: torchrun/NCCL DDP, rank-sharded deterministic data order, rank-0 I/O, traceback printed before teardown (DDP forensics #1), `destroy_process_group` |
| P2 | **Decision stage has no crash-resume**: the 24 Sep PC crash lost a run | Resumable decision and RLCD stages (optimizer + data cursor + RNG), atomic saves |
| P3 | Saves every N *steps* (up to 2,000 steps lost on a crash) | **Wall-clock saves** every 20 min, keep 2, atomic rename |
| P4 | MLM trains on only 30% of positions | **RTD** (ELECTRA / DeBERTa-v3 style) with gradient-disentangled embedding sharing: every token trains the discriminator. Local A/B vs MLM before the box (§7) |
| P5 | BPE merges digits ("650", "6"+"49") | **Digit-split pre-tokenizer**; each digit is a token |
| P6 | Pretraining corpus: FineWeb-Edu only | Mix: FineWeb-Edu 60%, Cosmopedia-v2 15%, FineMath-4+ 8%, Wikipedia-en 8%, structured program state (JSON/logs/tables/forms) 5%, decision-format mid-training 4%. 13-gram decontamination against every eval |
| P7 | 512-token context | Context-extension stage to 2,048 |
| P8 | Checkpoint chosen by val loss (misleads after RLCD sharpening) | Choose by val **proper score + accuracy**, pre-registered; always keep `last` |
| P9 | Risk threshold is marginal over an easy calibration mixture | Per-group certified thresholds plus `risk_domain` (Fix C, done) |
| P10 | Selective head below max-prob off-distribution | Escalation score selected on calibration by worst-group AUROC (done for worst-group mode; now the default too) |
| P11 | One process ran calibration + suite + probes: CUDA-graph memory → OOM | One process per GPU stage (done in the ablation driver; production uses the same) |
| P12 | Single-seed conclusions; benchmarks with 3 templates per family | Paired tests, cluster bootstrap, frame-held-out tests, 2 seeds for final comparisons |
| P13 | Throughput guessed | Probe on the box, median of ≥ 3 steps after warmup (LESSONS #13, #26) |
| P14 | Cross-stage loads could silently mismatch | `load_model` already raises on a mismatch; prefixes (`_orig_mod.`, `module.`) stripped on both ends (LESSONS #1) |

### 4.1 Second audit round (24–25 Sep 2026): defects found by running the pipeline locally

Each row was found by *running* a stage, not by reading code. The first three would each have broken or silently corrupted the rented run.

| # | OBSERVED | ROOT CAUSE | FIX | VERIFIED |
|---|---|---|---|---|
| P15 | Pretraining decontamination dropped **0** documents across all 5 sources | `ngrams()` used built-in `hash()`, which is salted per process; the blocklist was built in the parent and checked in spawn workers | Stable 64-bit blake2b n-gram hash (`verqen/data/audit.py`) | Spawn-worker hits 0/6 → 6/6. Linux re-tokenize now drops fineweb 781, wikipedia 524 (BoolQ passages are Wikipedia), finemath 169, cosmopedia 110 docs |
| P16 | On the box, tokenization would start while the decision build was still writing (and first *deleting*) `eval/*.jsonl`, so the blocklist would have been built from Artifact-G alone | Stage ordering in `run_all.sh` | New `build_data.py --evals-only` stage writes `data/decision/v3_evals` before tokenization. Tokenize refuses to run with < 20 real eval splits. The full build must reproduce identical eval files (gate) | Evals-only: 29 splits; full build gate reports `identical` |
| P17 | The decisions stage reused the pretraining micro-batch (sized at 512 tokens) at 1,024 tokens (≈1.6× activations): an OOM at about hour 15 | Only pretraining was probed | Preflight measures the decisions stage on real packed v3 rows and writes `training.decision_micro_rows`; RLCD rows are capped at the same value | Local dry run, full T2 model on the 5060 Ti: pretraining fits micro 16 at 512; decisions **OOM at 16** at 1,024 and fit at 12. The old code would have used 16 |
| P18 | The real-data build failed its zero-row gate on MMLU | MMLU is eval-only and QASC train-only *by design*; the gate counted train rows only | The gate counts per **declared** role | 31 sources pass at 5% scale |
| P19 | Real-data train/val/calib splits changed between builds | Shuffle seeded with `hash(name)` | `zlib.crc32(name)`; eval rows drawn from their own RNG stream | Evals-only and full-build eval files are byte-identical |
| P20 | Tokenization `MemoryError` | `Executor.map` submits the entire corpus at once | Bounded, ordered submission window (4 × workers) | Linux (WSL) re-tokenize of about 1.4B tokens completes. A Windows spawn run hit `WinError 6` in a worker (Windows-only; the box is Linux) |
| P21 | A crash mid-tokenization would leave a partial `.bin` that the resume accepts as finished | In-place writes | Write to `.tmp`, then `os.replace` | — |
| P22 | The pipeline released the RLCD checkpoint unconditionally | — | `production/select_release.py`: a pre-registered rule on **val** (overall −0.5 pt, per-domain −3 pt, per-variant −3 pt); a failed RLCD falls back to the decision checkpoint | Rule unit-checked (4 cases) |
| P23 | RLCD training-stream accuracy fell 0.96 → 0.90; this looked like RLCD damaging decisions | **Batch order**: the frozen pre-RLCD model also scores 0.912 on the same later batches. Held-out: real mean +0.47 pt; graph_route 0.70 → 0.945 | RLCD now logs a *fixed* held-out val slice (`rlcd_val`). No loss change: the gradient-domination hypothesis was refuted | Smoke run logs `rlcd_val` |
| P24 | RLCD lowered bait-distractor probes 0.752 → 0.696 (7/9 cells down, 0 up; z = −1.45, not significant) | Not established | Per-variant rule in P22 guards it on val | PLAUSIBLE, watch in T2 |
| P25 | The local preflight hung at 100% GPU for 20+ min and starved other jobs of RAM | The Windows driver spills VRAM to system RAM instead of raising OOM | `set_per_process_memory_fraction(headroom)`: "does not fit" is a clean OOM | Local dry run completes; every non-fitting size is a clean OOM |
| P26 | A locally written `configs/v3/t2.box.yaml` would ship in the package; the box would then skip preflight (it is the stage marker) and train with 5060 Ti settings | `package.py` shipped all of `configs/` | `*.box.yaml` is excluded from the tarball | — |

## 5. Data v3: curated hybrid, 24 domains

### 5.1 Training-data audit (T1 → v3)

| # | Defect (OBSERVED) | Fix |
|---|---|---|
| D1 | Each rule type in ~1 template → template-bound skills (RC-2) | **Scenario engine**: typed variables × composable predicates × ≥ 6 independently written frames per subdomain; **frames held out** for val/test |
| D2 | No mentioned-but-wrong options (lexical bait, RC-1) | Role-aware distractor layer on every scenario; the mention rate of correct and wrong options is **measured and balanced** (audit gate) |
| D3 | No information-sufficiency cases (RC-3) | Every scenario family emits missing / unknown / conflicting variants → gather-info or escalate |
| D4 | No counterfactual structure | Every scenario emits base / decisive-flip / irrelevant-change twins |
| D5 | Numbers in one format | Formats: integers, decimals, thousands separators, percentages, currencies, units, dates (ISO / written / relative), times |
| D6 | Real data dominated by MNLI/RACE | Per-domain quotas and per-source caps |
| D7 | Tiny vocabularies (T1 probes caught it) | Entity pools, 24 domain lexicons, FineWeb filler from a held-out shard |
| D8 | Only choice-type answers are well covered | Choice, score (ordinal) and noul for every domain; multi-question states |

### 5.2 The 24 domains

Each domain lists its subdomains; `spec` subdomains are generated by the engine, and real sources are mapped in (see `verqen/data/v3/taxonomy.py`):

1. Customer support: refunds, returns, warranty, account access, complaint routing, SLA
2. Finance and accounting: invoices, expense policy, payments, reconciliation, approvals
3. Banking and lending: credit rules, KYC, limits, overdraft, fraud signals
4. Insurance: claims eligibility, coverage limits, deductibles, waiting periods, fraud flags
5. Healthcare operations (admin/triage rules, not diagnosis): scheduling, triage protocol, dosing-review rules, discharge criteria, PubMedQA/MedMCQA
6. Legal and compliance: contract clauses, precedence, retention, jurisdiction, CaseHOLD/LexGLUE
7. Cybersecurity: access control, alert triage, log anomalies, phishing, vulnerability priority
8. IT and DevOps: incident severity, on-call routing, deploy gates, capacity thresholds, CI triage
9. Software engineering: code execution, review gates, dependency policy, test outcomes
10. Data labeling and moderation: sentiment, intent (Banking77/CLINC), toxicity (Civil Comments), topic, emotion
11. Web and browser agents: next action on page state, form filling, navigation (Mind2Web)
12. Tool use and API routing: tool selection, argument sufficiency, destructive-call risk
13. Games and planning: tic-tac-toe, nim, tetris, grid, graph routes, puzzles
14. Robotics and spatial: relative position, path blocking, pick/place feasibility
15. Logistics and supply chain: carrier choice, weight/volume, reorder points, routing
16. E-commerce and retail: promotions, pricing rules, stock, fulfilment
17. HR and workplace policy (job-related criteria only): leave, overtime, expense, eligibility
18. Manufacturing and industrial: sensor voting, maintenance thresholds, QC tolerance
19. Energy and utilities / IoT: load limits, alarms, tariff rules, outage triage
20. Transportation and travel: delays, rebooking, connections, baggage, visas (rule-given)
21. Education: rubric grading, answer checking, prerequisite rules
22. Math and quantitative: arithmetic, units, rates, percentages, dates/times, counting (GSM8K/AQuA/MathQA as choice)
23. Reading, commonsense and NLI: RACE, BoolQ, SciQ, ARC, OBQA, CSQA, SIQA, PIQA, HellaSwag, WinoGrande, MNLI/SNLI/WANLI, FEVER/VitaminC, TabFact
24. Safety, ethics and adversarial robustness: injection resistance, ETHICS/Moral Stories, escalation of harmful requests

Sources are **license-tagged**. Build with `--commercial-only` to drop CC-BY-NC sources (ANLI, SciQ, SciFact, ContractNLI).

### 5.3 Quality gates (a build fails if any gate fails)

1. Every procedural label is recomputed from the rendered text by an independent parser (a metamorphic check) on a sample; 100% agreement is required.
2. Exact plus near-duplicate removal (content key + MinHash on states).
3. Option-mention balance: |P(mention|correct) − P(mention|wrong)| ≤ 0.03 per domain.
4. Label balance: no option position or type > 1.5× uniform; noul yes-rate between 0.4 and 0.6.
5. Decontamination: 13-grams and option ids against every eval (Artifact-G, audit probes, all benchmark eval splits).
6. Frame hold-out: val/test frames never appear in train.
7. Per-source accept counts printed; **a source with 0 rows fails the build** (LESSONS #24).
8. Length and token statistics with the real tokenizer (LESSONS #14).

## 6. Evaluation and "beat Jev" protocol

- **Jev-parity suite** (`evals/jev_parity/`): the 24 domains × {grounded rules, numeric/date, counting, multi-step, missing info, conflicting evidence, distractors, injection, long state} on **held-out frames**. Metrics per cell: accuracy, ECE, AURC, risk at coverage, p50/p95 latency, and cost. Adapters: VerQen (local); Jev (plug in API key and endpoint); any LLM.
- **Plus:** the T1 audit probes, the 16 held-out real benchmarks, Artifact-G with the corrected analysis, and the environments.
- **Comparisons:** paired, cluster-bootstrapped by frame; claims only when the 95% CI excludes 0.

## 7. Before renting: local gates

1. Unit and distributed tests pass (gloo, 2 CPU ranks).
2. RTD-vs-MLM mini A/B on the local 5060 Ti (same compute, then a 15-minute decision fine-tune): pick the objective by val decision accuracy.
   **Result (25 Sep 2026): MLM.** 8L × 512 proxy, 40 min pretraining each, identical fine-tune:

   | Arm | Pretraining tokens | Val decision accuracy | Jev-parity (chance 21.6%) |
   |---|---|---|---|
   | RTD | 223M | 0.431 | 29.0% |
   | **MLM** | 277M | **0.622** | **48.8%** |

   RTD's discriminator reached 92.35% accuracy against a 91.7% predict-original baseline: at this compute it barely learned. The box was switched from RTD to MLM a few minutes into pretraining, and the partial RTD stage was discarded.
3. Smoke run of every stage end to end with `configs/v3/smoke.yaml`.
4. v3 data built locally once: gates pass, stats read by a human.
5. `production/preflight.sh` dry-run.
