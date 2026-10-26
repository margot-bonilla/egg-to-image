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

CODE_PATHS = ["submission", "scripts", "src", "configs", "Makefile", "requirements.txt"]


def code_state():
    """(commit, uncommitted code files) at the start of a run, for run_config.json (the audit
    reruns training from this commit). Only code paths count: e.g. an edited notebook doesn't."""
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=REPO).stdout
    dirty = [line[3:] for line in git("status", "--porcelain", "--untracked-files=no", "--", *CODE_PATHS).splitlines()]
    return git("rev-parse", "HEAD").strip() + ("+uncommitted-changes" if dirty else ""), dirty



def _curve_summary(history):
    """Epochs run and, with held-out scoring, the best epoch and its score (for run_config.json)."""
    if not history:
        return {}
    out = {"epochs_run": len(history)}
    scored = [r for r in history if "heldout" in r]
    if scored:
        # the epoch fit kept: by the smoothed score when Solver.smooth > 1, else the raw one
        key = "heldout_smooth" if any("heldout_smooth" in r for r in scored) else "heldout"
        best = max((r for r in scored if key in r), key=lambda r: r[key])
        out.update(best_epoch=best["epoch"], best_heldout_top5_sa=round(best["heldout"], 4))
        if key == "heldout_smooth":
            out["best_heldout_smooth"] = round(best["heldout_smooth"], 4)
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
    parser.add_argument("--num-workers", type=int, default=None,
                        help="data-loading processes (0 = in-process; default 8, or 4 with --cache)")
    parser.add_argument("--extra-cache", type=Path, default=None, metavar="DIR",
                        help="also train on another dataset's window cache (its train + val splits, all its "
                             "participants), reduced to this dataset's channels by name, e.g. THINGS-EEG2 "
                             "(63 ch) added to Alljoined-1.6M (32 ch). Never its test split")
    parser.add_argument("--cache", type=Path, default=None, metavar="DIR",
                        help="read windows from a scripts/cache_windows.py cache (#14): ~4x faster loading, "
                             "and workers share the memory-mapped files instead of each loading the dataset")
    parser.add_argument("--holdout", action="store_true",
                        help="train without the 200 held-out concepts (configs/heldout_concepts.json), "
                             "so the model can be selected with eval_local.py --split heldout")
    parser.add_argument("--init-from", type=Path, default=None, metavar="WEIGHTS",
                        help="start from these weights (two-stage training, e.g. a joint THINGS-EEG2 + "
                             "Alljoined model fine-tuned on Alljoined); same encoder settings needed")
    parser.add_argument("--resume", action="store_true",
                        help="continue an interrupted run from OUT_FOLDER/checkpoint.pt (saved after every epoch); "
                             "starts from scratch if there is none, and skips a run that already finished")
    parser.add_argument("--holdout-subjects", action="store_true",
                        help="--holdout plus 2 held-out participants (configs/heldout_subjects.json): select on "
                             "unseen people and unseen concepts, like the sealed final (10 galleries, averaged)")
    parser.add_argument("--fold", type=int, default=None, metavar="K",
                        help="with --holdout-subjects, hold out fold K of configs/cv_folds_<dataset>.json")
    parser.add_argument("--train-subjects", type=int, default=None, metavar="N",
                        help="with --holdout-subjects, train on only the first N remaining participants")
    parser.add_argument("--no-early-stopping", action="store_true",
                        help="with --holdout, don't score the held-out split each epoch (fixed n_epochs)")
    parser.add_argument("--avg-reps", type=int, default=0, metavar="K",
                        help="train on the mean of K repetitions of each (subject, image) (#17); 0 = off")
    parser.add_argument("--avg-reps-random", action="store_true",
                        help="with --avg-reps K, draw the number of repetitions uniformly in 1..K per item")
    parser.add_argument("--use-val", action="store_true",
                        help="also train on NeuralBench's val split (+25%% data on THINGS-EEG2; with --holdout, "
                             "minus the held-out concepts)")
    args = parser.parse_args()
    commit, dirty_files = code_state()  # now: the code this run actually uses
    args.holdout = args.holdout or args.holdout_subjects
    if args.extra_cache and not args.cache:
        raise SystemExit("--extra-cache needs --cache (the main dataset's window cache)")
    if args.num_workers is None:
        args.num_workers = 4 if args.cache else 8
    checkpoint = args.out / "checkpoint.pt"
    if args.resume and (args.out / "run_config.json").exists():
        print(f"{args.out} already holds a finished run: nothing to resume")
        return
    leftovers = {p.name for p in args.out.iterdir()} - {"checkpoint.pt", "checkpoint.pt.tmp"} if args.out.exists() else set()
    if leftovers or (checkpoint.exists() and not args.resume):
        raise SystemExit(f"{args.out} is not empty: pick a new folder (load_model would reuse its weights)"
                         + ("" if leftovers else ", or pass --resume to continue the interrupted run"))

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
    if args.cache:
        from src.data.window_cache import cached_loaders

        # test too: on Alljoined the held-out participants are scored on the test split every epoch
        loaders = cached_loaders(loaders, args.cache, device, args.batch_size, splits=("train", "val", "test"),
                                 num_workers=args.num_workers)
    galleries = None  # held-out loaders, scored after each epoch (mean over galleries)
    if args.holdout_subjects:
        from src.data.heldout import heldout_subjects_split

        split = heldout_subjects_split(loaders, device, args.batch_size, args.num_workers, include_val=args.use_val,
                                       n_train_subjects=args.train_subjects, dataset=args.dataset, fold=args.fold)
        loaders["train"], galleries = split["train"], split["galleries"]
    elif args.holdout:
        from src.data.heldout import heldout_split

        split = heldout_split(loaders, device, args.batch_size, args.num_workers, include_val=args.use_val)
        loaders["train"], galleries = split["train"], [split["heldout"]]
    elif args.use_val:
        from src.data.heldout import train_plus_val

        loaders["train"] = train_plus_val(loaders, device, args.batch_size, args.num_workers)
    if args.extra_cache:
        from src.data.heldout import _loader
        from src.data.window_cache import CachedWindows

        extra_names = json.loads((args.extra_cache / "meta.json").read_text())["ch_names"]
        missing = [n for n in meta["ch_names"] if n not in extra_names]
        if missing:
            raise SystemExit(f"{args.extra_cache} lacks channels {missing}")
        channels = [extra_names.index(n) for n in meta["ch_names"]]
        # train + val only: THINGS-EEG2's test images are Alljoined's test images (our validation)
        extra = CachedWindows([args.extra_cache / "train", args.extra_cache / "val"], channels=[channels] * 2,
                              subject_offset=[1000] * 2)
        loaders["train"] = _loader([loaders["train"].dataset, extra], device, True, args.batch_size, args.num_workers)
        print(f"Extra training data: {len(extra)} windows from {args.extra_cache} "
              f"({len(extra_names)} -> {len(channels)} channels)", flush=True)
    if args.avg_reps:
        from src.data.heldout import _loader
        from src.data.repetitions import RepetitionAveraged

        averaged = RepetitionAveraged(loaders["train"].dataset, k=args.avg_reps, random_k=args.avg_reps_random,
                                      seed=args.seed or 0)
        loaders["train"] = _loader([averaged], device, True, args.batch_size, args.num_workers)
        print(f"Repetition averaging: {len(averaged)} (subject, image) groups, k={args.avg_reps}"
              f"{' (random 1..k)' if args.avg_reps_random else ''}", flush=True)
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
    if hasattr(solver, "checkpoint"):
        solver.checkpoint = checkpoint
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

    if galleries is not None and hasattr(solver, "epoch_callback") and not args.no_early_stopping:
        # Score the held-out split with the official objective after each epoch: the solver
        # keeps the best epoch and stops early (patience). Only used for local training.
        spec_obj = importlib.util.spec_from_file_location("objective", TRACK / "objective.py")
        objective_module = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(objective_module)
        objective = objective_module.Objective.get_instance()
        from src.data.heldout import score_galleries

        def heldout_score(model, epoch, train_loss, temperature):
            return score_galleries(objective, galleries, model, meta)["top5_acc_subject_agg"]

        solver.epoch_callback = heldout_score

    args.out.mkdir(parents=True, exist_ok=True)
    model = solver.load_model({**meta, "device": device, "submission_dir": args.out.resolve()})
    if args.init_from:  # a resumed run's checkpoint overrides these weights in fit
        model.load_state_dict(torch.load(args.init_from, map_location=device, weights_only=True))
        print(f"Initialised from {args.init_from}", flush=True)
    start = time.time()
    solver.fit(model, loaders["train"])
    print(f"fit done in {(time.time() - start) / 60:.1f} min", flush=True)

    (args.out / "submission.py").write_text(solver_source)
    solver.save_model(model, args.out.resolve())
    if meta.get("ch_names"):  # the training montage: lets inference pick these channels by name
        (args.out / "channels.json").write_text(json.dumps({"ch_names": [str(n) for n in meta["ch_names"]]}))
    # Audit record (left out of the ZIP by `make zip-submission`, like other run_config* files).
    (args.out / "run_config.json").write_text(json.dumps({
        "command": "python " + " ".join(sys.argv),
        "commit": commit, "uncommitted_code": dirty_files,
        "solver": str(args.solver), "holdout": args.holdout, "holdout_subjects": args.holdout_subjects, "fold": args.fold, "train_subjects": args.train_subjects, "use_val": args.use_val, "dataset": args.dataset,
        "batch_size": args.batch_size, "cache": str(args.cache) if args.cache else None,
        "extra_cache": str(args.extra_cache) if args.extra_cache else None,
        "init_from": str(args.init_from) if args.init_from else None, "avg_reps": args.avg_reps, "avg_reps_random": args.avg_reps_random,
        "seed": getattr(solver, "seed", None), "overrides": overrides,
        "train_windows": len(loaders["train"].dataset), "fit_minutes": round((time.time() - start) / 60, 1),
        "resumed_after_epoch": getattr(solver, "resumed_from", None),  # fit_minutes then counts from there
        **_curve_summary(getattr(solver, "history", None)),
    }, indent=2))
    if getattr(solver, "history", None):
        (args.out / "run_config_curve.json").write_text(json.dumps(solver.history, indent=1))
    checkpoint.unlink(missing_ok=True)  # never part of a submission ZIP
    print(f"Submission ready in {args.out}: {sorted(p.name for p in args.out.iterdir())}")


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", str(REPO / "cache" / "huggingface"))
    os.environ.setdefault("REVE_POSITIONS_PATH", str(REPO / "cache" / "mne_data"))
    main()
