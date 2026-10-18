# Neural Interfaces 2026 - Track 1: EEG-to-Image

Codabench: [Track 1: EEG-to-Image](https://www.codabench.org/competitions/17974/) ·
Challenge site: [neural-interfaces26.github.io](https://neural-interfaces26.github.io/tracks.html#track-1) ·
Official repo: [neural-interfaces26/2026-competition](https://github.com/neural-interfaces26/2026-competition) ·
Workshop: NeurIPS 2026 Brain & Body Workshop (Sydney)

---

## 🎯 The task

Given one EEG epoch recorded while a person views a natural image, predict that image's **1536-d `facebook/dinov2-giant` embedding**.

- **Input:** `X` of shape `(B, C, T)`: 1 s epochs from -0.2 to +0.8 s around image onset, **resampled to 120 Hz** (`T = 120`), filtered, baseline-corrected, robust-scaled and clipped by the NeuralBench image-task pipeline.
- **Output:** `(B, 1536)` float embeddings.
- **Metric:** **top-5 retrieval accuracy**. Each prediction is ranked by **cosine similarity** against the unique image embeddings of the test set.
- **Generalization:** the final test set has **unseen participants and unseen images**.

| | Warm-up phase | Sealed final phase |
|---|---|---|
| Evaluation data | public THINGS-EEG2 test split | private 2026 Alljoined cohort (11 new subjects) |
| Input shape | **63 channels** × 120 | **32 channels** (Emotiv) × 120 |
| Ranking metric | pooled `top5_acc` | `top5_acc_subject_agg` (repeats of the same image averaged within each subject first) |
| Counts for the prize? | no, indicative only | yes |

> The two phases have **different channel counts**. Build models from `meta["n_chans"]` / `meta["n_times"]`. A fixed-size model needs separate weights for each phase.

## 📅 Timeline

| Date | What |
|---|---|
| **Oct 24, 2026** | **Registration closes.** Every teammate must be registered *and* added to our Codabench organization by then |
| Oct 24, 21:00 BST | Warm-up ends (Codabench page; the official repo says Oct 25, so plan for the earlier date). 5 submissions/person/day |
| **Oct 28 - Nov 21** | **Sealed final.** 1 submission/person/day. Ranking = **best of the team's last 5** submissions |
| After Nov 21 | Reproducibility audit for the top 3: organizers **rerun our training from the committed config** and must land within ±2σ |
| Dec 11-12 | Winners announced at NeurIPS |

## 🚀 Setup (≈10 min)

```bash
git clone https://github.com/margot-bonilla/egg-to-image.git && cd egg-to-image
brew install wget python@3.12   # macOS: wget for some dataset downloads; Python 3.12 like the Codabench worker
make env                   # Python 3.12 .venv + requirements + official repo in external/ + NeuralBench config
source .venv/bin/activate
make test                  # unit smoke tests
make test-submission       # builds dist/submission.zip and runs it read-only, the way Codabench does
```

- **Python 3.12 only.** The Codabench worker runs 3.12, and newer Pythons (3.13+) may lack wheels for the torch/neuroai stack. `make env` refuses anything else.
- `requirements.txt` pins `neuralbench`/`neuralset`/`neuralfetch`/`neuraltrain==0.4.0` and `exca==0.5.29`, the same versions as the Codabench worker, so preprocessing matches the platform. Don't change these pins.
- `make env` writes `~/.neuralbench/config.json`, pointing NeuralBench at this repo's `data/`, `cache/` and `save/` (all gitignored). An existing config is never overwritten (`python scripts/neuralbench_config.py --force` to reset it).
- Datasets are large: THINGS-EEG2 alone is ~117 GB to download and ~200 GB once extracted. Only download what your task needs.
- THINGS-EEG2 also needs the THINGS image database, which is licence-gated: read the [LICENSE](https://osf.io/52wrx), get the password from [`password_images.txt`](https://osf.io/j6a3m), and `export NEURALFETCH_THINGS_PASSWORD=...` before downloading. Research and non-commercial use only. Never commit the password.

### Training on a Mac (Apple GPU)

The official pipelines only detect CUDA, so on a Mac they silently fall back to CPU. Use the wrapper instead of `neuralbench`:

```bash
python scripts/neuralbench_mps.py eeg image            # same arguments as the neuralbench CLI
```

- Runs the DINOv2 target embeddings on the Apple GPU (`scripts/mps_patch.py`): ~4× faster than CPU, with identical embeddings (cosine 1.000000 vs CPU) and **unchanged cache keys**, so the cached targets are reusable on any machine.
- Lightning trains on `mps` automatically.
- Runs in a single process: it regenerates `configs/neuralbench.local.json` (gitignored, don't edit it) from your `~/.neuralbench/config.json` with `CLUSTER: null`.
- Keeps each run's best checkpoint as `save/.../best_kept.ckpt` (NeuralBench normally deletes it), ready for `submission_templates/neuralbench_eegnet_submission.py`.
- For Benchopt training, set `COMPET_DEVICE=mps` (its datasets also only auto-detect CUDA).

**Joining the team on Codabench** (needed before you can submit):
1. Create a Codabench account and register for the competition, then fill in the [registration form](https://forms.gle/p3t2V25nuQtVXyj9A) with your Codabench email.
2. Ask Margot to add you to the team's Codabench organization.

## 📦 How a submission works

Codabench is **inference-only**. You upload a ZIP with `submission.py` and `weights.pt` **at the root** of the archive. The platform calls:

1. `Solver.load_model(meta)`: rebuilds the model from `meta` (`n_chans`, `n_times`, `n_outputs`, `device`, ...) and loads weights from `meta["submission_dir"]`, a **read-only** folder.
2. `model.predict(X)`: returns `(B, n_outputs)` for each batch.

Rules that cause failed uploads (and a failed upload still uses up a daily slot):
- `submission.py` may only import what the worker already has: `torch` plus the competition stack. No extra `.py` files.
- No training, downloading or writing files inside `load_model` / `predict`.
- No state carried between `predict` calls. Batches may mix subjects and change size.
- The full test pass must finish in under 60 minutes on one GPU.

**Before every upload:**
```bash
make test-submission                       # must end with "EEGToImageBaseline: done"
# then open notebooks/submission_check.ipynb for contract checks and retrieval plots
```
Then upload `dist/submission.zip` under **My Submissions** for the active phase, and **log it in [RESULTS.md](RESULTS.md)**.

Full guide: [`external/2026-competition/codabench/pages/participate.md`](https://github.com/neural-interfaces26/2026-competition/blob/main/codabench/pages/participate.md).

## 🏗️ Repo map

```
egg-to-image/
├── submission/
│   ├── submission.py        # ✅ Codabench entrypoint: Solver + model, torch-only, has fit/save_model
│   └── weights.pt           # ⚠️ debug weights (63ch, synthetic data): pipeline test, ~chance score
├── src/
│   ├── models/eeg_encoder.py   # ✅ EEGNet-style encoder (same state_dict keys as submission.py)
│   ├── models/tinyFM.py        # 🧪 channel-position-conditioned encoder (works for any montage)
│   ├── losses/retrieval_loss.py# ✅ InfoNCE · ⏳ CosineMSEHybridLoss TODO
│   ├── metrics.py              # ✅ top-k retrieval
│   └── train.py                # ❌ broken: imports src/data/dataset.py, which doesn't exist (workstream A)
├── notebooks/
│   ├── quickstart.ipynb        # NeuralBench basics
│   └── submission_check.ipynb  # loads dist/unzipped like Codabench, checks contract, plots retrieval
├── configs/default.yaml
├── scripts/
│   ├── smoke_test.py           # `make test`
│   ├── neuralbench_mps.py      # neuralbench CLI on the Apple GPU, keeps best checkpoints
│   ├── mps_patch.py            # DINOv2 embeddings on MPS (cache keys unchanged)
│   └── neuralbench_config.py   # writes ~/.neuralbench/config.json (`make env`)
├── external/2026-competition/  # official repo (gitignored, created by `make env`)
└── RESULTS.md                  # log of every upload
```

`make help` lists every command. `make train-debug` reproduces the current debug weights.

## 🧩 Workstreams

Each workstream has a **GitHub Issue** (label `workstream`). Assign yourself before you start. One branch per workstream. We're a team of 2–3 (Margot included), so take one workstream at a time.

**Compute: Apple Silicon only (no CUDA).** Train with `scripts/neuralbench_mps.py` (see [Training on a Mac](#training-on-a-mac-apple-gpu)). Never compute DINOv2 targets with your own script: they must match the platform's extraction exactly. A, E and F don't need a GPU.

**Suggested order:** **B** first (a real warm-up score before Oct 24), with **A** in parallel → **C** and **D** for the final phase, where 32-channel data is what counts → **E** and **F** whenever someone is waiting on compute.

| | Workstream | Needs GPU | Done when |
|---|---|---|---|
| **G** | ✅ **Compute:** embeddings and training on the Mac GPU (`scripts/neuralbench_mps.py`) | – | done |
| **A** | **Data loader:** rebuild `src/data/dataset.py` on top of the official `Image` dataset (`external/.../benchmark_utils/nb_task.py`, `datasets/image_studies.py`) | no* | `make train` and `make test` pass on a small real study |
| **B** | **First real warm-up model:** train on THINGS-EEG2 (`training.yml`, or `neuralbench eeg image`) and upload | yes | a real warm-up score in RESULTS.md, clearly above chance (~2%) |
| **C** | **Final-phase data:** Alljoined-1.6M (32ch, `--dataset xu2025alljoined`), with a validation split holding out **both subjects and images** | yes | validation `top5_acc_subject_agg` reported for a baseline |
| **D** | **REVE probe:** port the official `solvers/reve_probe.py` (84.75% top-5 on THINGS-EEG2, vs 28% for EEGNet) into our submission | yes | passes `make test-submission`, score in RESULTS.md |
| **E** | **Loss and model experiments:** `CosineMSEHybridLoss`, learnable temperature, augmentations, channel-agnostic models (`tinyFM.py`) | no | improvement shown on a fixed validation set |
| **F** | **Evaluation and tracking:** local validation that matches the sealed metric, comparison notebook, RESULTS.md discipline | no | anyone can score a checkpoint locally with one command |

\* A needs DINOv2 embeddings for real data. Use a small study or the `Simulated` dataset while developing.

Reference scores on THINGS-EEG2 (not Codabench scores): chance 2.22 · EEGNet 28.13 · REVE frozen probe 84.75 (top-5 %).

## 🤝 How we work

- **Branches and PRs:** branch per workstream (`ws-a-data-loader`, …), PR into `main`. `make test` and `make test-submission` must pass before merging.
- **Never commit** data, caches or checkpoints. The only exception is `submission/weights.pt`, and only when it's the version we're submitting.
- **Reproducibility (audit):** every uploaded model must be traceable to a **commit + config + command**. Fix seeds, and don't train with uncommitted changes.
- **Uploads:** everyone uploads (quotas are per person and all count for the team). Always run `make test-submission` first, then add a row to RESULTS.md.
  - **Warm-up:** iterate freely, 5/day each.
  - **Final phase:** only the **team's last 5** count, so announce an upload in the team chat first, and only upload a model that beats our best on local validation. One weak upload pushes a good one out of the window.
