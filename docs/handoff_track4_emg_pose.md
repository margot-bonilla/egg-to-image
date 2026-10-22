# Handoff: starting Track 4 (EMG-to-Pose) from scratch

Written Oct 22, 2026 from the Track 1 (EEG-to-Image) project, `margot-bonilla/egg-to-image`, to
kick off Track 4 in a **new repo and a new Claude Code session**.

**How to use this file**
1. Create the new repo (e.g. `emg-to-pose`) and copy this file into it as `HANDOFF.md`.
2. Start a new Claude Code session in that repo and paste the **kickoff prompt** (§10).
3. Keep the Track 1 repo checked out next to it (`../egg-to-image`) so files can be copied from it.

---

## 1. The competition (all tracks)

Neural Interfaces 2026 (NeurIPS 2026 Brain & Body workshop), on Codabench. Team: **fireflies**
(Margot, Enrique, maybe more). Rules as of the organisers' Oct 22 announcement:

| | |
|---|---|
| Warm-up **and registration** close | **Sun Nov 1, 23:59 CET**. Register for **Track 4 separately** (each person), and be in the team's Codabench organisation |
| No submissions | Nov 2–4 |
| **Sealed phase** | **Nov 5, 00:01 → Nov 21, 23:59 CET** |
| Sealed uploads | **5 per participant per track, in total** (not per day). A team's best appears on the public leaderboard; you see all your own scores |
| Warm-up uploads | free (daily quota), don't count for the prize: use them to check mechanics |
| Top-3 | **reproducibility audit**: organisers retrain from the committed code + config and must land within ±2σ |
| Failed uploads | still use a slot: always run the contract check locally first |

Codabench had a multi-day outage around Oct 22: a submitted file isn't necessarily scored; check.

## 2. Track 4 in one page

Official sources (in the competition repo, `neural-interfaces26/2026-competition`):
`tracks/emg_pose/README.md`, `tracks/emg_pose/objective.py`, `codabench/pages/competition_emg.md`,
`codabench/pages/scientific_faq_emg.md`, the NeuralBench guide
(facebookresearch.github.io/neuroai/neuralbench/auto_examples/biosignal_challenge_2026/plot_track4_emg_to_pose.html).

- **Task:** from **16-channel wrist surface EMG**, predict continuous trajectories of **20 UmeTrack
  hand-joint angles**.
- **Contract:** `predict(X)` with `X: (B, 16, T)` → `(B, n_joints, T')` **in radians**
  (`n_joints = meta["n_joints"] = 20`). A coarser `T'` is nearest-resampled to the target length.
  Read sizes from `meta` (`n_chans`, `n_times`, `sfreq`, `n_joints`); don't hard-code a batch size.
- **Input:** contiguous, non-overlapping **5 s windows at 2 kHz** (10,000 samples), **raw**: no
  filtering, notch, baseline, scaling or clipping; only volts → microvolts (×10⁶). Windows are
  independent: **no state across `predict` calls**, batches of any size and order.
- **Metric (ranking):** **mean absolute angular error in degrees**, over all examples × joints ×
  time points; lower is better. Also reported: per-subject MAE mean/std, RMSE, R².
- **Data:** public **EMG2Pose** (Salter et al. 2024, arXiv 2412.02725; Meta Reality Labs),
  NeuralFetch study `Salter2024Emg2pose`, NEMAR `NM000281`, licence **CC-BY-NC-SA-4.0**, **193
  participants**, 29 movement stages. **Check its download size on NEMAR before downloading** (it
  may be hundreds of GB; Margot's Mac had ~540 GB free on Oct 22, shared with Track 1's caches).
- **Warm-up evaluation:** EMG2Pose's **`user_stage` test split** (unseen users × unseen stages),
  the NeuralBench start-kit test set. Public labels: warm-up leakage is possible, so warm-up
  scores of other teams mean little.
- **Sealed evaluation:** a **private 2026 Meta Reality Labs cohort**: **new users, new movement
  stages** (some outside the 29 public ones), unseen user-stage combinations. Same contract.
- **Reference:** NeuroPose ≈ **17.5° ± 1.5** MAE (public EMG2Pose, NeuralBench). Official solvers:
  `MeanPose` (floor), `Ridge`, `Torch-Linear` (per-time-step readouts), `EEGNet` (dense
  braindecode); submission template `neuralbench_vemg2pose_submission.py` for NeuralBench's VEMG2Pose.
- **NeuralBench start kit:**
  ```bash
  neuralbench emg pose -m vemg2pose --download
  neuralbench emg pose -m vemg2pose --prepare
  neuralbench emg pose -m vemg2pose --debug
  ```

## 3. Hardware and the GPU-sharing rule

Margot's MacBook (Apple Silicon, 64 GB, PyTorch on `mps`) runs Track 1's queue. **Track 1 keeps
priority on this Mac's GPU, especially at night**, until its sealed candidates are ready.
Track 4 trains in daytime gaps or, better, on **Enrique's Mac** (same kind of machine). Before
starting a GPU job here, check `make training-status` in the Track 1 repo; never run two trainings
on the GPU at once (both slow down ~2×).

## 4. Setup from scratch

1. **Python 3.12 only** (the Codabench worker runs 3.12; newer Pythons lack wheels). Copy Track 1's
   `Makefile` `env` targets and `scripts/neuralbench_config.py`.
2. **Pin the stack to the worker's versions:** `neuralbench`, `neuralset`, `neuralfetch`,
   `neuraltrain` **0.4.0** and `exca==0.5.29` (see Track 1's `requirements.txt`; confirm against
   the competition repo's `requirements.txt`). Different versions = different preprocessing than
   Codabench.
3. **Official repo in `external/`:** `git clone --depth 1 https://github.com/neural-interfaces26/2026-competition external/2026-competition`.
4. **NeuralBench config** (`~/.neuralbench/config.json`) pointing at this repo's `data/`, `cache/`,
   `save/` (all gitignored). Track 1's `make env` writes it; on a shared Mac it may already point
   at Track 1's folders: decide whether to share `data/` (saves disk) or keep them apart.
5. **Mac GPU:** the official pipelines only auto-detect CUDA. Track 1 has `scripts/neuralbench_mps.py`
   (+ `scripts/mps_patch.py`, written for DINOv2 targets): adapt it so Lightning trains on `mps`.
6. **Contract check first:** the simulated `starter.yml` (`MeanPose`) runs without any download.

## 5. What to copy from the Track 1 repo

Copy and adapt (don't link the repos). All are in `../egg-to-image`:

| File | What it gives you | Adapt for Track 4 |
|---|---|---|
| `scripts/train_solver.py` | train any solver file on local data, export a ready-to-upload folder with `run_config.json` (command, commit, settings, curve) | objective = `EMG-pose`; early-stopping score = **lower** MAE (flip the sign or add a "minimise" flag); drop image-specific options |
| `scripts/eval_local.py` | score submission folders exactly like Codabench, plus ensembles | metric keys (`angular_mae`); ensemble = average of predicted angles |
| `src/data/window_cache.py` + `scripts/cache_windows.py` | windows extracted once to memory-mapped `.npy`, batches bit-identical to `load_task`, ~4× faster loading, no per-worker copies | targets are per-time-step trajectories `(20, T)`, not one embedding per image: store them per window (no `targets`/`image` indirection); consider float16 (5 s × 2 kHz × 16 ch = 640 KB per window in float32) |
| `src/data/heldout.py` (pattern) | a committed held-out split + multi-gallery scoring | hold out **users and stages** (EMG2Pose has `user`, `stage`, `user_stage` generalisation splits): mirror the sealed shift |
| `submission/submission.py` → `Solver.fit` | training loop with learnable settings via `--set`, lr schedule + warm-up, early stopping on held-out with **smoothing** (`smooth`, `patience`), **per-epoch checkpoints + resume**, layer sizes read back from the weights | loss = angular MAE/L1 (or smooth L1) in radians instead of the CLIP loss |
| `scripts/training.sh`, `make pause-training / resume-training / training-status` | freeze/resume a run (SIGSTOP/CONT) to free the GPU and screen | none |
| `scripts/battery_guard.sh` | pause training below 25% battery, resume at 60% | none |
| `logs/round*.sh` (gitignored) | queue pattern: chained bash scripts, `--resume`, one status file per round | paths |
| `scripts/docs_to_latex.py` + `make docs-latex` | Markdown docs → LaTeX/PDF, Mermaid → TikZ (pandoc from RStudio's bundle) | none |
| `RESULTS.md` format | one row per upload: date, commit, model, command, local score, Codabench score | Track 4 columns |

## 6. Lessons learned on Track 1 (do these from day 1)

**Science / evaluation**
1. **Local evaluation = Codabench.** Track 1's local test reproduced every Codabench score to two
   decimals (7 uploads). Build `eval_local.py` first; then you rarely need to spend uploads.
2. **Never select on the warm-up test set** (it's the public leaderboard data): build a committed
   held-out split that mirrors the sealed shift (here: **unseen users AND unseen stages**).
3. **The sealed shift dominates everything.** On Track 1, the same model scored 0.64 on seen people
   and 0.23 on unseen people. Measure on held-out users from the start, not after tuning.
4. **More participants was the biggest lever** for unseen people (Track 1: about +3.6 points per
   doubling). EMG2Pose has 193 users: use them; measure the curve (train on 25 / 50 / 100 / all).
5. **Per-window normalisation** helped unseen people a lot on one dataset (+4 points) and not on
   another: test it early. EMG arrives raw in µV and amplitude varies with skin and band placement,
   so normalisation is probably more important here.
6. **Single-seed validation scores are noisy:** select and early-stop on a **smoothed** score
   (mean of the last 3 epochs, patience ~10); with patience 5, runs stopped on noise spikes.
7. **Train/validation epoch counts:** models peak much earlier on unseen people; final retrains
   without a validation set need a fixed epoch count taken from the smoothed curves.
8. **Ensembles of seeds** reliably added 2–3 points, and they survive the reproducibility audit
   better than a lucky single seed.
9. **Foundation models: fine-tune, don't just probe.** Frozen REVE was weak on Track 1; NeuralBench's
   fully fine-tuned REVE is the strongest Track 1 baseline. Check NeuralBench's Track 4 baselines
   and wrappers (`neuralbench/defaults/downstream_wrappers.yaml`) for what they fine-tune.

**Engineering (each cost us hours)**
10. **Cache windows before training** (~4× faster, less memory; workers don't duplicate data).
11. **Checkpoint every epoch + `--resume`.** The Mac restarted twice in one day; runs without
    checkpoints were lost.
12. **The screen blinking = GPU at ~97%.** Don't restart: `make pause-training`.
13. **Power:** a 60 W USB-C link drained the battery ~1%/min *while plugged in* during training.
    Use the 140 W adapter (or a 100 W / 5 A cable); keep `battery_guard.sh` running.
14. **Lid closed = macOS sleeps despite `caffeinate`:** keep the lid open for overnight runs.
15. **NeuralBench `--prepare` can hang** in its final one-batch sanity check after the cache is
    built: when the log only repeats warnings and CPU is 0%, kill it.
16. **Never edit a running bash queue script** (bash reads it as it goes): stop it, edit, re-launch.
    Edit Python modules that queued runs import atomically (write a temp file, then `os.replace`).
17. **macOS spawns DataLoader workers** by re-importing the main script: any script with workers
    needs `if __name__ == "__main__":`.
18. **zsh doesn't word-split `$var`:** `for p in $PIDS` and `cmd $ARGS` pass one argument. Use
    `bash -c`, arrays, or `${=var}`.
19. **Downloads and web search are blocked in Claude's sandbox** (proxy at localhost:10054,
    "internet mode" off): Margot runs downloads in her own terminal. pandoc comes from RStudio's
    bundle (`/Applications/RStudio.app/.../quarto/bin/tools/aarch64/pandoc`).
20. **Merge PRs with "Create a merge commit"**, never squash/rebase: `RESULTS.md` and
    `run_config.json` cite exact commit hashes for the audit. Watch stacked PRs: one merged into
    its base branch instead of `main`.
21. **Every upload:** `make test-submission` (read-only, like Codabench) first; the ZIP holds
    `submission.py` + weights at its root; `run_config*` stays out of the ZIP.

## 7. A first plan (Oct 23 → Nov 4)

| Step | Goal | Done when |
|---|---|---|
| 1 | Repo + env + official repo + simulated contract check | `starter.yml` passes |
| 2 | Data: check size, download, `--prepare` | EMG2Pose cached |
| 3 | `eval_local.py` + a baseline (NeuralBench VEMG2Pose or `Torch-Linear`) | local score = Codabench warm-up score |
| 4 | Held-out **users × stages** split, committed | a validation number for every run |
| 5 | Window cache, `train_solver.py`, checkpoints, smoothed early stopping | fast, restartable runs |
| 6 | Experiments on held-out users: per-window normalisation, model size, participant curve, foundation-model fine-tuning | a recipe that beats the baseline on held-out users |
| 7 | Final candidates: retrain on all users, seed ensembles, contract check | 2–3 candidates ready by Nov 4 |

Sealed phase (Nov 5–21): with 5 uploads each, upload 2–3 very different candidates early, refine
what wins, and finish with the best.

## 8. Track 1 status for context (Oct 22)

Warm-up 0.29 → 0.78 uploaded (3-seed ensemble), 0.80 locally with wider convolutions (upload
pending the Codabench outage). On the final's format (Alljoined-1.6M, 32-channel Emotiv, unseen
people) ≈ 0.27; current work: fine-tuning the REVE foundation model. Details:
`../egg-to-image/docs/how_it_works.md`, `docs/architecture_deep_dive.md`, `RESULTS.md`.

## 9. Open questions to settle early

- EMG2Pose download size vs free disk; share `data/` with Track 1 or not.
- Which NeuralBench model is the start kit's best (`vemg2pose`?) and its exact training cost on a Mac.
- Is anything frozen/pretrained in the Track 4 baselines (fine-tuning lesson, §6.9)?
- Who uploads what: 5 sealed uploads per person on Track 4, separate from Track 1's.

## 10. Kickoff prompt (paste into the new session)

> I'm starting Track 4 (EMG-to-Pose) of the Neural Interfaces 2026 competition in this new repo.
> Read `HANDOFF.md` first: it has the rules, the Track 4 contract, what to copy from my Track 1
> repo at `../egg-to-image`, and the lessons learned there. Then: (1) set up the environment
> (Python 3.12, pinned neuralbench 0.4.0 stack, official repo in `external/`), (2) run the
> simulated contract check, (3) tell me the EMG2Pose download size before I download anything,
> and (4) propose the first week's plan. Important: the Mac's GPU is shared with Track 1, which has
> priority (check `make training-status` in `../egg-to-image` before any GPU job), and
> downloads have to run in my own terminal. Save the key facts to memory as you go.
