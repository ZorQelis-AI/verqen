# VerQen-T2 production runbook (2x RTX 5090, 25 h)

The plan and its reasoning live in `T2_PLAN.md`. This file lists the steps, in order.

## 0. Before renting (local, about 10 minutes)

1. Create a Hugging Face **write** token (<https://huggingface.co/settings/tokens>).
2. Package the code and publish it for the box to pull. Never push data over the vast SSH proxy (LESSONS #34).

   ```bash
   set HF_TOKEN=hf_xxx
   python production/package.py --repo <you>/verqen-t2-code
   ```

   It prints four `export` lines (CODE_URL, CODE_MD5, SETUP_URL, HF_TOKEN) plus the setup command. The signed URLs expire after a while, so re-run `package.py` shortly before you rent.

3. Rent the instance. Requirements:
   - 2x RTX 5090;
   - a driver that supports CUDA 12.8 or newer;
   - at least 150 GB of disk on `/workspace`;
   - at least 32 CPU cores (tokenizing and the decision-data build run on the CPU while the GPUs train);
   - at least 64 GB of RAM;
   - good inbound bandwidth.

   Choose a "stop"-able (on-demand) instance so the watchdog can stop it.

## 1. Set up the box (about 15 minutes)

```bash
cat > ~/.verqen_env <<'EOF'
export CODE_URL='...'  CODE_MD5='...'  SETUP_URL='...'
export HF_TOKEN='hf_xxx'  VERQEN_HF_REPO='<you>/verqen-t2-results'
EOF
chmod 600 ~/.verqen_env && source ~/.verqen_env
curl -fsSL "$SETUP_URL" -o setup.sh && bash setup.sh
```

`setup.sh` does the following:
- probes the model hub and falls back to hf-mirror;
- checks that 2 GPUs and the disk space are present;
- pulls the package and verifies its md5;
- creates the venv and forces IPv4;
- installs the cu128 torch wheels;
- runs the unit tests (they must end in `passed`).

## 2. Launch (detached, both commands)

```bash
cd /workspace/verqen && source ~/.verqen_env
setsid nohup bash production/run_all.sh    > logs/run_all.log  2>&1 < /dev/null &
setsid nohup bash production/watchdog.sh 25 > logs/watchdog.log 2>&1 < /dev/null &
```

Do not wrap either command in `timeout ssh` (it leaves orphans), and do not keep an SSH session open for monitoring.

## 3. What runs, and when

| Stage | Marker | Wall-clock | GPUs |
|---|---|---|---|
| preflight: NCCL all-reduce; measures the model size(s) in `size_candidates` (now only **t2-232m**, forced by `size_rule`; `auto` picks by the 20-tokens-per-param rule); measures micro-batches for pretraining, context extension and decisions separately; sets `data_scale`; writes `configs/v3/t2.box.yaml` | `configs/v3/t2.box.yaml` | about 15 min | 2 |
| real eval splits (the pretraining decontamination blocklist) | `data/decision/v3_evals/EVALS_OK` | about 5 min | 0 |
| download (fineweb-edu, cosmopedia-v2, finemath-4+, wikipedia) × `data_scale`, plus 400M synthetic-text tokens | `data/pretrain/v3/.download_done` | about 20–40 min | 0 |
| decision data v3 (105 specs × 22 domains + real data + v2 + fixes), **in the background** | `data/decision/v3/build_report.json` | about 1–2 h (CPU) | 0 |
| tokenize (13-gram decontamination at tokenization time; the log line `N docs dropped` must be > 0 for the web sources) | `data/pretrain/v3/manifest.json` | about 20–55 min (scales with `data_scale`) | 0 |
| pretrain: RTD + GDES **232M (27L × 768)**, context 512, about 3.2–6.7B tokens depending on measured speed | `runs/verqen-t2/pretrain/final` | 13 h | 2 (DDP) |
| pretrain_ext: context 2048, RoPE base 80k | `runs/verqen-t2/pretrain_ext/final` | 1.5 h | 2 |
| decisions (typed heads, sharded deterministic stream) | `runs/verqen-t2/decisions/final` | 4.5 h | 2 |
| RLCD (GPU0), while GPU1 evaluates the pre-RLCD checkpoint | `runs/verqen-t2/rlcd/final` | 2 h | 1 + 1 |
| **select**: RLCD vs pre-RLCD on the val split, by a fixed rule | `runs/verqen-t2/release_choice.json` | about 5 min | 1 |
| finalize: calibration (marginal + per-domain worst-group), suite, probes, Artifact-G, Jev-parity, model card, upload | `runs/verqen-t2/release/FINAL_OK` | about 30 min | 1 |

The planned total is about 23 h. The last 2 h are margin, and they belong to the watchdog.

**Check after preflight** (`cat runs/verqen-t2/preflight.json`):
- `size_selection.chosen` and each candidate's `projected_tokens_B`;
- `choices.decisions.micro` must be ≤ `choices.pretrain.micro`;
- `nccl.ok` must be true.

## 4. Monitoring (short connections only)

```bash
cat /workspace/verqen/runs/verqen-t2/progress.json          # the stage table, rewritten atomically
tail -2 /workspace/verqen/logs/pretrain.log                  # loss, tokens/s, ETA
tail -3 /workspace/verqen/logs/watchdog.log                  # GPU state, stall warnings
```

Healthy signs:
- `pretrain.log` prints a line every few minutes, with both ranks at the same step;
- the RTD discriminator loss falls within the first hour;
- the generator MLM accuracy rises past 0.3.

The watchdog warns when a log is silent for more than 30 min. When GPU utilisation is at 99% but no log line appears, the cause is allocator thrash (SoreQen): restart the run (section 5).

## 5. If something breaks

- **Anything crashed, or the box rebooted.** Re-source `~/.verqen_env`, then launch again exactly as in section 2. Finished stages are skipped, and unfinished ones resume from their last wall-clock checkpoint (saved every 20 min, atomically).
- **A stage failed.** `run_all.log` prints the last 30 lines of that stage's log. Under DDP, the real exception is printed by `guarded` *before* the teardown barrier, so read the whole log, not only the tail.
- **RLCD failed.** Nothing to do: the release falls back to the decision checkpoint.
- **Out of time.** At hour 25 the watchdog:
  - kills the stage;
  - finalizes the best existing checkpoint (in this order: the validated choice, `decisions/final`, `rlcd/final`, `decisions/resume`, `pretrain_ext/final`, `pretrain/final`);
  - uploads it;
  - stops the instance.

## 6. Retrieve the results

Everything lands in the private HF model repo `$VERQEN_HF_REPO`:
- `release/release/`: the model, the calibration, `model_card.json`, `jev_parity.json` and the audit outputs;
- `run/`: per-stage `metrics.jsonl`, `progress.json` and `preflight.json`.

```bash
huggingface-cli download <you>/verqen-t2-results --local-dir runs/verqen-t2-remote
```

Once the download is confirmed, **destroy** the instance: a stopped instance still bills for its disk.

## 7. After the run: the Jev comparison

```bash
python evals/jev_parity/run.py --checkpoint runs/verqen-t2-remote/release/release --out evals/jev_parity/results/t2.json
python evals/jev_parity/run.py --adapter jev --out evals/jev_parity/results/jev.json      # needs JEV_URL / JEV_API_KEY
```

The suite has 7,840 cases:
- 10 capabilities;
- 22 domains;
- 105 subdomains;
- only held-out phrasing frames.

Report paired differences with CIs per capability. Claim a win only where the CI excludes 0.
