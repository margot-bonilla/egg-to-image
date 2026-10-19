"""Train a Codabench solver file on local data and export a ready-to-upload folder.

    python scripts/train_solver.py SOLVER.py OUT_FOLDER [--dataset xu2025alljoined]

Same steps as a Benchopt ``training=True`` run (CompetSolver.run +
_export_submission), but on the data/ and cache/ we already have instead of
Benchopt's own data path: the competition's data pipeline
(benchmark_utils.nb_task.load_task) builds the train split, then
``load_model`` -> ``fit`` -> ``save_model`` into OUT_FOLDER, with SOLVER.py
copied as OUT_FOLDER/submission.py. Score the result with scripts/eval_local.py.
"""
import argparse
import importlib.util
import os
import shutil
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
TRACK = REPO / "external/2026-competition/tracks/image_decoding"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("solver", type=Path, help="solver file defining class Solver(CompetSolver)")
    parser.add_argument("out", type=Path, help="submission folder to create")
    parser.add_argument("--dataset", default=None, help="image-task overlay, e.g. xu2025alljoined (default: THINGS-EEG2)")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty: pick a new folder (load_model would reuse its weights)")

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
    meta["n_outputs"] = int(meta.pop("target_shape")[-1])
    meta.pop("raw_target_shape", None)
    print(f"Train split: {len(loaders['train'].dataset)} windows, "
          f"{meta['n_chans']} channels x {meta['n_times']} samples, device={device}", flush=True)

    spec = importlib.util.spec_from_file_location("solver", args.solver)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    solver = module.Solver.get_instance()
    solver.train_loader = loaders["train"]  # a training run, like objective training=True
    solver.device = device

    args.out.mkdir(parents=True, exist_ok=True)
    model = solver.load_model({**meta, "device": device, "submission_dir": args.out.resolve()})
    start = time.time()
    solver.fit(model, loaders["train"])
    print(f"fit done in {(time.time() - start) / 60:.1f} min", flush=True)

    shutil.copyfile(args.solver, args.out / "submission.py")
    solver.save_model(model, args.out.resolve())
    print(f"Submission ready in {args.out}: {sorted(p.name for p in args.out.iterdir())}")


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", str(REPO / "cache" / "huggingface"))
    os.environ.setdefault("REVE_POSITIONS_PATH", str(REPO / "cache" / "mne_data"))
    main()
