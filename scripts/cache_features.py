"""Cache frozen-encoder features for the train/val/test splits, for fast head experiments.

    python scripts/cache_features.py [--encoder reve] [--dataset xu2025alljoined] [--preprocessing nb-reve]

Runs the official REVEEncoder (tracks/image_decoding/solvers/reve_probe.py, mean-pooled
to (N, 512)) over the competition's data pipeline, i.e. exactly the input Codabench feeds
a submission, and saves per split to cache/features/<encoder>_<dataset>/<split>.npz:
features (N, D) float32, targets (N, 1536) float16, subject (N,), image (N,) indices
into the split's unique targets. Heads trained on these files can be scored with the
official metrics (benchmark_utils.metrics) without touching the EEG again.
"""
import argparse
import importlib.util
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
TRACK = REPO / "external/2026-competition/tracks/image_decoding"

# EEG preprocessing variants. "competition" = what Codabench feeds a submission (120 Hz).
# "nb-reve" = NeuralBench's models/reve.yaml (REVE's native 200 Hz): a diagnostic only,
# since Codabench never sends this input.
PREPROCESSING = {
    "competition": {},
    "nb-reve": {
        "neuro.frequency": 200.0,
        "neuro.filter": [0.5, 99.5],
        "neuro.notch_filter": None,
        "neuro.baseline": None,
        "neuro.scaler": "StandardScaler",
        "neuro.scale_factor": None,
        "neuro.clamp": 15,
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--encoder", default="reve", choices=["reve"])
    parser.add_argument("--dataset", default=None, help="image-task overlay (default: THINGS-EEG2)")
    parser.add_argument("--preprocessing", default="competition", choices=list(PREPROCESSING))
    parser.add_argument("--splits", nargs="+", default=["val", "test", "train"])
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    sys.path.insert(0, str(TRACK))
    from benchmark_utils.data import subject_ids, to_numpy
    from benchmark_utils.nb_task import load_task

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    cache = str(REPO / "cache")
    loaders, meta = load_task(
        "eeg", "image", data_dir=REPO / "data", dataset=args.dataset, device=device,
        batch_size=args.batch_size, overrides={
            "study.source.infra.folder": cache,
            "neuro.infra.folder": cache,
            "target.infra.folder": cache,
            "target.infra.cluster": None,
            "target.infra.permissions": None,
            **PREPROCESSING[args.preprocessing],
        },
    )
    meta = {k: v for k, v in meta.items() if k not in ("target_shape", "raw_target_shape")}
    print(f"input: {meta['n_chans']} channels x {meta['n_times']} samples at {meta['sfreq']} Hz", flush=True)

    spec = importlib.util.spec_from_file_location("reve_probe", TRACK / "solvers/reve_probe.py")
    reve_probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reve_probe)
    encoder = reve_probe.REVEEncoder({**meta, "device": device})

    suffix = "" if args.preprocessing == "competition" else f"_{args.preprocessing}"
    out_dir = REPO / "cache" / "features" / f"{args.encoder}_{args.dataset or 'gifford2022large'}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        start = time.time()
        feats, targets, subjects = [], [], []
        for X, y, info in loaders[split]:
            feats.append(to_numpy(encoder.encode(X).mean(dim=1).float()))
            targets.append(to_numpy(y).astype(np.float16))
            subjects.append(subject_ids(info, len(X)))
        targets = np.concatenate(targets)
        _, image = np.unique(targets, axis=0, return_inverse=True)
        np.savez(out_dir / f"{split}.npz", features=np.concatenate(feats), targets=targets,
                 subject=np.concatenate(subjects), image=image.reshape(-1))
        print(f"{split}: {len(targets)} windows in {(time.time() - start) / 60:.1f} min -> {out_dir / split}.npz",
              flush=True)


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", str(REPO / "cache" / "huggingface"))
    os.environ.setdefault("REVE_POSITIONS_PATH", str(REPO / "cache" / "mne_data"))
    main()
