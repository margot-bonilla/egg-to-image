"""Train a Codabench solver file on local data and export a ready-to-upload folder.

    python scripts/train_solver.py SOLVER.py OUT_FOLDER [--holdout] [--seed N] [--set name=value ...]

Same steps as a Benchopt ``training=True`` run (CompetSolver.run +
_export_submission), but on the data/ and cache/ we already have instead of
Benchopt's own data path: the competition's data pipeline
(benchmark_utils.nb_task.load_task) builds the train split, then
``load_model`` -> ``fit`` -> ``save_model`` into OUT_FOLDER, with SOLVER.py
copied as OUT_FOLDER/submission.py. Score the result with scripts/eval_local.py.
"""
import argparse
import ast
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
TRACK = REPO / "external/2026-competition/tracks/image_decoding"


def _curve_summary(history):
    """Epochs run and, with held-out scoring, the best epoch and its score (for run_config.json)."""
    if not history:
        return {}
    out = {"epochs_run": len(history)}
    scored = [r for r in history if "heldout" in r]
    if scored:
        best = max(scored, key=lambda r: r["heldout"])
        out.update(best_epoch=best["epoch"], best_heldout_top5_sa=round(best["heldout"], 4))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("solver", type=Path, help="solver file defining class Solver(CompetSolver)")
    parser.add_argument("out", type=Path, help="submission folder to create")
    parser.add_argument("--dataset", default=None, help="image-task overlay, e.g. xu2025alljoined (default: THINGS-EEG2)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=None, help="sets Solver.seed, if the solver has one")
    parser.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE",
                        help="override Solver class attributes, e.g. --set dedup=True n_epochs=30")
    parser.add_argument("--num-workers", type=int, default=8, help="data-loading processes (0 = in-process)")
    parser.add_argument("--holdout", action="store_true",
                        help="train without the 200 held-out concepts (configs/heldout_concepts.json), "
                             "so the model can be selected with eval_local.py --split heldout")
    parser.add_argument("--no-early-stopping", action="store_true",
                        help="with --holdout, don't score the held-out split each epoch (fixed n_epochs)")
    parser.add_argument("--use-val", action="store_true",
                        help="also train on NeuralBench's val split (+25%% data on THINGS-EEG2; with --holdout, "
                             "minus the held-out concepts)")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty: pick a new folder (load_model would reuse its weights)")

    sys.path.insert(0, str(TRACK))
    from benchmark_utils.nb_task import load_task

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    cache = str(REPO / "cache")
    loaders, meta = load_task(
        "eeg", "image", data_dir=REPO / "data", dataset=args.dataset, device=device,
        batch_size=args.batch_size, num_workers=args.num_workers, overrides={
            "study.source.infra.folder": cache,
            "neuro.infra.folder": cache,
            "target.infra.folder": cache,
            "target.infra.cluster": None,
            "target.infra.permissions": None,
        },
    )
    sys.path.insert(0, str(REPO))
    heldout_loader = None
    if args.holdout:
        from src.data.heldout import heldout_split

        split = heldout_split(loaders, device, args.batch_size, args.num_workers, include_val=args.use_val)
        loaders["train"], heldout_loader = split["train"], split["heldout"]
    elif args.use_val:
        from src.data.heldout import train_plus_val

        loaders["train"] = train_plus_val(loaders, device, args.batch_size, args.num_workers)
    meta["n_outputs"] = int(meta.pop("target_shape")[-1])
    meta.pop("raw_target_shape", None)
    print(f"Train split: {len(loaders['train'].dataset)} windows, "
          f"{meta['n_chans']} channels x {meta['n_times']} samples, device={device}", flush=True)

    solver_source = args.solver.read_text()  # snapshot now: the file may be edited while we train
    spec = importlib.util.spec_from_file_location("solver", args.solver)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    solver = module.Solver.get_instance()
    solver.train_loader = loaders["train"]  # a training run, like objective training=True
    solver.device = device
    if args.seed is not None:
        solver.seed = args.seed
    overrides = {}
    for item in args.set:
        name, _, value = item.partition("=")
        if not hasattr(solver, name):
            raise SystemExit(f"{type(solver).__name__} has no attribute {name!r}")
        try:
            overrides[name] = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            overrides[name] = value
        setattr(solver, name, overrides[name])

    if heldout_loader is not None and hasattr(solver, "epoch_callback") and not args.no_early_stopping:
        # Score the held-out split with the official objective after each epoch: the solver
        # keeps the best epoch and stops early (patience). Only used for local training.
        spec_obj = importlib.util.spec_from_file_location("objective", TRACK / "objective.py")
        objective_module = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(objective_module)
        objective = objective_module.Objective.get_instance()
        objective.set_data(train_loader=None, test_loader=heldout_loader, **meta)

        def heldout_score(model, epoch, train_loss, temperature):
            return objective.evaluate_result(model)["top5_acc_subject_agg"]

        solver.epoch_callback = heldout_score

    args.out.mkdir(parents=True, exist_ok=True)
    model = solver.load_model({**meta, "device": device, "submission_dir": args.out.resolve()})
    start = time.time()
    solver.fit(model, loaders["train"])
    print(f"fit done in {(time.time() - start) / 60:.1f} min", flush=True)

    (args.out / "submission.py").write_text(solver_source)
    solver.save_model(model, args.out.resolve())
    # Audit record (left out of the ZIP by `make zip-submission`, like other run_config* files).
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, cwd=REPO).stdout.strip())
    (args.out / "run_config.json").write_text(json.dumps({
        "command": "python " + " ".join(sys.argv),
        "commit": commit + ("+uncommitted-changes" if dirty else ""),
        "solver": str(args.solver), "holdout": args.holdout, "use_val": args.use_val, "dataset": args.dataset,
        "seed": getattr(solver, "seed", None), "overrides": overrides,
        "train_windows": len(loaders["train"].dataset), "fit_minutes": round((time.time() - start) / 60, 1),
        **_curve_summary(getattr(solver, "history", None)),
    }, indent=2))
    if getattr(solver, "history", None):
        (args.out / "run_config_curve.json").write_text(json.dumps(solver.history, indent=1))
    print(f"Submission ready in {args.out}: {sorted(p.name for p in args.out.iterdir())}")


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", str(REPO / "cache" / "huggingface"))
    os.environ.setdefault("REVE_POSITIONS_PATH", str(REPO / "cache" / "mne_data"))
    main()
