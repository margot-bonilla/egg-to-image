"""Score submission folders locally, exactly like Codabench, on the real test split.

    python scripts/eval_local.py submissions/a [submissions/b ...] [--dataset xu2025alljoined]

Loads each folder's submission.py + weights the way the platform does
(``Solver.load_model(meta)``), feeds it the competition's own data pipeline
(benchmark_utils.nb_task.load_task) and scores it with the official objective
(tracks/image_decoding/objective.py). With several folders it also scores their
ensemble (mean of L2-normalized predictions).

Default data = THINGS-EEG2 (the warm-up evaluation set); local scores matched
the warm-up leaderboard exactly. Reuses data/ and the cache/ built by NeuralBench.
"""
import argparse
import importlib.util
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
TRACK = REPO / "external/2026-competition/tracks/image_decoding"
METRICS = ["top5_acc_subject_agg", "top1_acc_subject_agg", "top5_acc", "top1_acc", "n_candidates"]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Ensemble:
    """Average of the members' L2-normalized predictions."""

    def __init__(self, models):
        self.models = models

    def predict(self, X):
        preds = [torch.nn.functional.normalize(m.predict(X).float(), dim=-1) for m in self.models]
        return torch.stack(preds).mean(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("submissions", nargs="+", type=Path, help="folders with submission.py + weights")
    parser.add_argument("--dataset", default=None, help="image-task overlay, e.g. xu2025alljoined (default: THINGS-EEG2)")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    sys.path.insert(0, str(TRACK))
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
        },
    )
    n_outputs = int(meta.pop("target_shape")[-1])
    meta.pop("raw_target_shape", None)

    objective = load_module("objective", TRACK / "objective.py").Objective.get_instance()
    objective.set_data(train_loader=None, test_loader=loaders["test"], n_outputs=n_outputs, **meta)

    models = {}
    for i, folder in enumerate(args.submissions):
        submission = load_module(f"submission_{i}", folder / "submission.py")
        solver = submission.Solver.get_instance()
        solver.train_loader = None  # inference-only, like the platform
        models[folder.name] = solver.load_model(
            {**meta, "n_outputs": n_outputs, "device": device, "submission_dir": folder.resolve()}
        )
    if len(models) > 1:
        models[f"ensemble of {len(models)}"] = Ensemble(list(models.values()))

    print(f"\nTest split: {len(loaders['test'].dataset)} windows, "
          f"{meta['n_chans']} channels x {meta['n_times']} samples, device={device}\n")
    width = max(len(name) for name in models) + 2
    print(f"{'model':<{width}}" + "".join(f"{m:>22}" for m in METRICS))
    for name, model in models.items():
        result = objective.evaluate_result(model)
        print(f"{name:<{width}}" + "".join(f"{result[m]:>22.4f}" if m != "n_candidates"
                                          else f"{result[m]:>22d}" for m in METRICS), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", str(REPO / "cache" / "huggingface"))
    main()
