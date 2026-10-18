# Results log

One row per Codabench upload. Needed to pick the best model, and for the reproducibility audit (commit + config + command must reproduce the weights).

The leaderboard ranks on **top-5 subject-aggregated** (also the sealed metric). NeuralBench's `test/full_retrieval/top5_acc_subject-agg` on THINGS-EEG2 reproduces the warm-up score exactly, so compare models locally before spending a submission.

| Date | Who | Phase | Commit | Model / config | Command to reproduce | Local val | Codabench top-5 | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-10-18 | Margot | Warm-up | bf5bf17 | `EEGToImageBaseline`, synthetic 63ch debug weights | `make train-debug` | 100% (synthetic, seen images) | _pending_ | Pipeline test only; expect ~chance |
| 2026-10-18 | Margot | Warm-up | _fill in after merge_ | NeuralBench EEGNet, THINGS-EEG2, seed 34: `submissions/neuralbench_eegnet_things_eeg2_seed34/` (official template, config in `run_config.yaml`) | `python scripts/neuralbench_mps.py eeg image` (grid seeds 33–35; this is seed 34) | THINGS-EEG2 test: pooled top-5 **11.9%**, subject-agg top-5 **28.8%** (reference EEGNet 28.13) | subject-agg **0.29** · pooled 0.12 (top-1: 0.09 · 0.03) | First real model. Submission 1001528, rank 67, 266 s. Local NeuralBench test metrics match Codabench exactly |
