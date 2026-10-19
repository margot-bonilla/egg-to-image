"""Test post-hoc recalibrations of a submission's predictions (issue #23).

    python scripts/recalibration.py submissions/<name> [--n-train 100000]

The submission must have been trained with --holdout. Its predictions on a sample of
*training* windows (held-out concepts excluded) are used to fit each recalibration, which
is then applied to the held-out split's predictions and scored with the official metrics:
- center: subtract the mean training prediction
- center+target-mean: subtract the mean prediction, add the mean training target
- ridge (alpha grid): linear map from predictions to the true embeddings
Never fitted on held-out or test data.
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
TRACK = REPO / "external/2026-competition/tracks/image_decoding"


def predict_split(model, loader, limit=None):
    from benchmark_utils.data import subject_ids, to_numpy

    preds, targets, subjects, n = [], [], [], 0
    for X, y, info in loader:
        preds.append(to_numpy(model.predict(X)).astype(np.float32))
        targets.append(to_numpy(y).astype(np.float32))
        subjects.append(subject_ids(info, len(X)))
        n += len(X)
        if limit and n >= limit:
            break
    return {"pred": np.concatenate(preds), "targets": np.concatenate(targets), "subject": np.concatenate(subjects)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("submission", type=Path)
    parser.add_argument("--n-train", type=int, default=100_000)
    args = parser.parse_args()
    sys.path.insert(0, str(TRACK))
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts"))
    from benchmark_utils.nb_task import load_task
    from src.data.heldout import heldout_split
    from train_head import score

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cache = str(REPO / "cache")
    loaders, meta = load_task("eeg", "image", data_dir=REPO / "data", device=device, batch_size=512, overrides={
        "study.source.infra.folder": cache, "neuro.infra.folder": cache, "target.infra.folder": cache,
        "target.infra.cluster": None, "target.infra.permissions": None})
    n_outputs = int(meta.pop("target_shape")[-1])
    meta.pop("raw_target_shape", None)
    split = heldout_split(loaders, device, batch_size=512, num_workers=4)

    spec = importlib.util.spec_from_file_location("sub", args.submission / "submission.py")
    sub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sub)
    solver = sub.Solver.get_instance()
    solver.train_loader = None
    model = solver.load_model({**meta, "n_outputs": n_outputs, "device": device,
                               "submission_dir": args.submission.resolve()})

    train = predict_split(model, split["train"], limit=args.n_train)  # shuffled loader: a random sample
    held = predict_split(model, split["heldout"])
    print(f"fit on {len(train['pred'])} training windows, score on {len(held['pred'])} held-out windows\n", flush=True)

    mu_p, mu_t = train["pred"].mean(0), train["targets"].mean(0)
    variants = {"none": lambda p: p, "center": lambda p: p - mu_p, "center+target-mean": lambda p: p - mu_p + mu_t}
    from sklearn.linear_model import Ridge
    for alpha in (1.0, 10.0, 100.0):
        ridge = Ridge(alpha=alpha).fit(train["pred"], train["targets"])
        variants[f"ridge alpha={alpha:g}"] = lambda p, r=ridge: r.predict(p)
        ridge_c = Ridge(alpha=alpha).fit(train["pred"] - mu_p, train["targets"])
        variants[f"center+ridge alpha={alpha:g}"] = lambda p, r=ridge_c: r.predict(p - mu_p)

    print(f"{'variant':<26}{'top5_sa':>9}{'top1_sa':>9}{'top5':>8}")
    for name, fn in variants.items():
        s = score(fn(held["pred"]).astype(np.float32), {"targets": held["targets"], "subject": held["subject"]})
        print(f"{name:<26}{s['top5_sa']:>9.4f}{s['top1_sa']:>9.4f}{s['top5']:>8.4f}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", str(REPO / "cache" / "huggingface"))
    main()
