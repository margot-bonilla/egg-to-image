# Results log

One row per Codabench upload. Needed to pick the best model, and for the reproducibility audit (commit + config + command must reproduce the weights).

| Date | Who | Phase | Commit | Model / config | Command to reproduce | Local val | Codabench top-5 | Notes |
|---|---|---|---|---|---|---|---|---|
| 2026-10-18 | Margot | Warm-up | _fill in_ | `EEGToImageBaseline`, synthetic 63ch debug weights | `make train-debug` | 100% (synthetic, seen images) | _pending_ | Pipeline test only; expect ~chance |
