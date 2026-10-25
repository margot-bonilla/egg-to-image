"""Score every combination of submission folders as an ensemble, computing each model's predictions once.

    python scripts/ensemble_search.py FOLDER [FOLDER ...] --dataset xu2025alljoined --split heldout_subjects
        [--cache DIR] [--max-size 4]

Each folder is loaded like Codabench (Solver.load_model) and predicts the evaluation windows once;
every combination is then scored with the official objective (tracks/image_decoding/objective.py)
on the mean of its members' L2-normalised predictions (as EmbeddingEnsemble / eval_local's
Ensemble), averaged over galleries. Prints combinations sorted by top-5 subject-aggregated.
"""
import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
TRACK = REPO / "external/2026-competition/tracks/image_decoding"


class Replay:
    """A 'model' replaying stored per-batch predictions in loader order (the evaluation loaders don't shuffle)."""

    def __init__(self, batches):
        self.batches, self.i = batches, 0

    def predict(self, X):
        pred = self.batches[self.i % len(self.batches)]
        self.i += 1
        return pred


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default="heldout_subjects", choices=["test", "heldout", "heldout_subjects"])
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--fold", type=int, default=None, help="cross-validation fold (configs/cv_folds_<dataset>.json)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-size", type=int, default=4, help="largest ensemble to score")
    args = parser.parse_args()

    sys.path.insert(0, str(TRACK))
    sys.path.insert(0, str(REPO))
    from benchmark_utils.nb_task import load_task
    from src.data.heldout import heldout_split, heldout_subjects_split, score_galleries

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    cache = str(REPO / "cache")
    loaders, meta = load_task("eeg", "image", data_dir=REPO / "data", dataset=args.dataset, device=device,
                              batch_size=args.batch_size, overrides={
                                  "study.source.infra.folder": cache, "neuro.infra.folder": cache,
                                  "target.infra.folder": cache, "target.infra.cluster": None,
                                  "target.infra.permissions": None})
    n_outputs = int(meta.pop("target_shape")[-1])
    meta.pop("raw_target_shape", None)
    if args.cache:
        from src.data.window_cache import cached_loaders
        splits = ("test",) if args.split == "test" else ("train", "val", "test")
        loaders = cached_loaders(loaders, args.cache, device, args.batch_size, splits=splits)
    if args.split == "heldout_subjects":
        galleries = heldout_subjects_split(loaders, device, args.batch_size, dataset=args.dataset,
                                           fold=args.fold)["galleries"]
    elif args.split == "heldout":
        galleries = [heldout_split(loaders, device, args.batch_size)["heldout"]]
    else:
        galleries = [loaders["test"]]

    spec = __import__("importlib.util").util
    preds = {}  # folder name -> per gallery, list of per-batch L2-normalised predictions (CPU)
    for i, folder in enumerate(args.folders):
        s = spec.spec_from_file_location(f"sub{i}", folder / "submission.py")
        module = spec.module_from_spec(s)
        s.loader.exec_module(module)
        solver = module.Solver.get_instance()
        solver.train_loader = None
        model = solver.load_model({**meta, "n_outputs": n_outputs, "device": device, "submission_dir": folder.resolve()})
        preds[folder.name] = [[torch.nn.functional.normalize(model.predict(X).float(), dim=-1).cpu()
                               for X, _, _ in g] for g in galleries]
        print(f"predicted {folder.name}", flush=True)
        del model

    s = spec.spec_from_file_location("objective", TRACK / "objective.py")
    objective_module = spec.module_from_spec(s)
    s.loader.exec_module(objective_module)
    objective = objective_module.Objective.get_instance()

    results = []
    names = list(preds)
    for size in range(1, min(args.max_size, len(names)) + 1):
        for combo in itertools.combinations(names, size):
            replays = []
            for g in range(len(galleries)):
                batches = [torch.stack([preds[n][g][b] for n in combo]).mean(0) for b in range(len(preds[combo[0]][g]))]
                replays.append(Replay(batches))
            scores = []
            for g, loader in enumerate(galleries):
                objective.set_data(train_loader=None, test_loader=loader, n_outputs=n_outputs, **meta)
                scores.append(objective.evaluate_result(replays[g]))
            top5 = float(np.mean([r["top5_acc_subject_agg"] for r in scores]))
            top1 = float(np.mean([r["top1_acc_subject_agg"] for r in scores]))
            results.append((top5, top1, combo))
    for top5, top1, combo in sorted(results, reverse=True):
        print(f"top-5 {top5:.4f}  top-1 {top1:.4f}  {' + '.join(combo)}")


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", str(REPO / "cache" / "huggingface"))
    os.environ.setdefault("REVE_POSITIONS_PATH", str(REPO / "cache" / "mne_data"))
    main()
