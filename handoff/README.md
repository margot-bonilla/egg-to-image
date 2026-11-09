# Teammate baseline handoff

## What to run

A **single-seed development replica of the established two-stage encoder + REVE recipe**, not the full submitted five-encoder/two-REVE ensemble. Run folds 1 and 2 serially on one GPU. This provides a clean reference for the newer experiments.

Each fold runs:

1. Encoder: Alljoined source people + THINGS-EEG2, then Alljoined only.
2. REVE: frozen-token attention head, then top-four-layer fine-tuning from that fitted head.
3. Raw-window prediction export and evaluation of each member and a fixed 0.75 encoder / 0.25 REVE mixture.

The committed `frozen_split.json` preserves the existing 1,500 development images, the same per-person trial identities and full galleries. Development, confirmation and public test image exclusions apply to **both datasets and every stage**, including exact-DINO aliases with different names. Alljoined held-out people are excluded entirely. The 1,500 confirmation images are exclusion identities only; they are never evaluated.

**Not included:** the other three seen/unseen diagnostic cells, new trial splits, new model ideas, hyperparameter grids, full-seed retraining, reserved confirmation, packaging or competition upload. A separate prospective specification is needed for those.

## Clone an isolated branch

Do not edit or switch the checkout running someone else's training.

```bash
git clone --filter=blob:none --sparse --branch teammate-baseline-handoff \
  https://github.com/margot-bonilla/egg-to-image.git egg-baseline
cd egg-baseline
git sparse-checkout set handoff scripts src submission configs
```

This avoids fetching all historical submission weights. Use the branch's own committed code, not copied uncommitted scripts from a training machine.

## Environment

Use **Python 3.12 on macOS or Linux**. Windows is not supported by this POSIX-lock runner. Prefer the team's already working competition environment. Otherwise:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r handoff/requirements.txt
mkdir -p external
git clone https://github.com/neural-interfaces26/2026-competition.git external/2026-competition
git -C external/2026-competition checkout a66bc23d1a0e686b895c093a642b45c74970758b
```

Install the appropriate mutually compatible torch/torchvision/torchaudio wheels for the host's CPU/MPS/CUDA platform; do not mix incompatible builds. The reference environment is recorded in `reference_environment.json`, but this handoff does not promise bit-identical results across accelerator types. Each run records Python and execution dependency versions, the source commit, device and worker count, and rejects a changed environment on resume.

The CPU tests were run with the reference environment, not a newly installed virtualenv. Full real-data training was deliberately not launched during preparation.

## Data and cache prerequisites

Set `CACHE_ROOT` to a private, prepared cache root with this layout:

```text
CACHE_ROOT/
  windows/
    xu2025alljoined/{meta.json,train/,val/,test/}
    things_eeg2/{meta.json,train/,val/,test/}
  reve_tokens/
    xu2025alljoined/{meta.json,train/,val/}       # final-layer tokens
    xu2025alljoined_L18/{meta.json,train/,val/}   # input to top-four layers
  huggingface/hub/                              # pinned public REVE snapshot
  mne_data/                                    # REVE channel-position cache, if required
```

**Recommended:** transfer the team's existing official caches privately. They are large; check their actual sizes and available disk space first. If copying token caches, dereference metadata symlinks (for example, `rsync -aL`) so they do not point back to the original machine. Never commit EEG, embeddings, model weights, cache contents or licence passwords to GitHub.

The raw-cache metadata and target fingerprints are intentionally strict. Rebuilding caches may produce different byte identities or floating-point target coordinates; a mismatch stops the run. Do not edit the freeze or weaken validation to bypass it. Resolve the cache difference or transfer the matching cache. `X.npy` uses header and host-stat checks, not a full EEG payload hash; this is documented rather than presented as stronger evidence.

If rebuilding raw caches is necessary, use the repository's **official** pipeline (`scripts/cache_windows.py`) under the pinned competition stack, following the dataset/licence setup in the project README. Do not compute DINO targets with a custom extractor.

Download the pinned public REVE snapshot during setup, before the offline training run:

```bash
export CACHE_ROOT=/absolute/path/to/cache
export HF_HOME="$CACHE_ROOT/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export REVE_POSITIONS_PATH="$CACHE_ROOT/mne_data"
python -c 'from huggingface_hub import snapshot_download; snapshot_download("brain-bzh/reve-base", revision="33841986f6cc1438d9dcdbca5a2137d60f6007d9")'
```

If the two token caches are absent, prepare them serially with the **pinned** helper rather than a moving model revision:

```bash
python -B -m handoff.cache_tokens --cache-root "$CACHE_ROOT" --layer 22
python -B -m handoff.cache_tokens --cache-root "$CACHE_ROOT" --layer 18
```

These are frozen-feature extraction jobs, not baseline training. They can be expensive. They never compute DINO targets or fit REVE, and refuse to overwrite an existing token directory. An interrupted extraction retains its partial file; preserve/rename the incomplete directory before an explicitly planned retry. Existing complete official token caches can be reused after metadata and actual public-model source-sample compatibility checks.

## Check before spending GPU time

```bash
python -B -m unittest handoff.test_handoff -v
python -B -m handoff.run --smoke
python -B -m handoff.run --preflight --cache-root "$CACHE_ROOT"
```

- Unit tests and `--smoke` use disposable synthetic CPU data. They require no real EEG, REVE weights, network, GPU or competition checkout.
- `--preflight` is read-only: verifies the pinned competition commit, raw-cache metadata/target identities, frozen query addresses, source exclusions and token row/layer identities. It does not instantiate REVE or create output files.
- The real run also checks source-only cached token samples against the pinned public REVE before training.

On the matching reference caches, preflight reports **874,633 Alljoined source windows in fold 1**, **874,918 in fold 2**, **547,203 extra THINGS windows**, and **6,000 development queries per fold**.

## Preparation evidence

Before handoff: 30 CPU regression tests and the synthetic end-to-end smoke passed. Tests include worker-backed interrupted resume, published-loader shuffle/RNG parity, metadata staying off MPS, frozen exclusions, prefix/raw routing, token-output isolation and returned-gallery order. A read-only preflight against the reference caches passed with the counts above. Independent source review has no remaining blocker. Actual accelerator training, full real-REVE extraction and a fresh virtualenv installation were not performed during preparation.

Metric values in JSON are **fractions**, not percentages: `0.001` means `0.1%`.

## One command for the baseline

```bash
python -B -m handoff.run --run --cache-root "$CACHE_ROOT" \
  --out "$PWD/handoff_runs/baseline" --device auto
```

On a Mac plugged into AC, command-scoped sleep protection is recommended:

```bash
caffeinate -i -s python -B -m handoff.run --run --cache-root "$CACHE_ROOT" \
  --out "$PWD/handoff_runs/baseline" --device mps
```

Do not run another training process on the same GPU. `auto` chooses MPS, CUDA, then CPU. CPU ranking uses one thread by default and preserves the official float64 subject/image averaging and average-tie rank semantics. Training uses the selected accelerator. No target-person calibration or gallery fitting occurs.

The recipe retains the published maximum epochs and smoothed-development early stopping: 40/40 encoder stages, 40 head epochs and 15 top-four epochs. Checkpoint selection now uses the frozen fresh-image cap-1 development score instead of the old 200-image test proxy. This is an explicit validation change, not a promise to reproduce the old leaderboard score. Training batches use the established shuffled baseline loader, not the newer paired regulariser sampler.

## Interruptions and resume

Re-run the **same command, same output directory and same environment**. The immutable run manifest checks executable source hashes, cache identities/stats, source selections, device and versions. Stages checkpoint model, optimiser, scheduler, temperature, best state and torch RNG at completed epoch boundaries. Completed stages are verified and reused. An interrupted epoch is rerun from the previous complete epoch.

Changing code, data, stage ancestry or environment rejects resume. Keep the branch immutable during execution. Do not point this command at another experiment's output directory. The handoff uses its own lock and never controls another process or alters global power settings.

## What to send back

Privately send the output directory, excluding heavyweight checkpoints unless requested:

- `manifest.json`: source/data/environment identity.
- `summary.json`: per-fold/member/cap metrics, hit counts and query counts.
- `fold*/evaluation.json`: per-person metrics and exact correct-image ranks.
- `fold*/query_identity.json`: gallery stems, raw trial row identities and repetition groups.
- `fold*/gallery.npy` and its receipt: the ordered original-DINO gallery matrix and hashes, allowing standalone CPU re-ranking of returned vectors without a separate cache dependency.
- `fold*/encoder.npy`, `fold*/reve.npy` and receipts: CPU float32, 1536-D **per-trial prediction vectors**, enabling later fixed-mixture analysis without model inference.
- Stage `curve.json` and `complete.json` files: training history, ancestry and weight hashes.

All reports are **public-development, single-seed evidence**, selected on reused development images—not independent confirmation and not a sealed-test gain. Send the branch/commit and the exact command too. Do not upload to Codabench from this task.
