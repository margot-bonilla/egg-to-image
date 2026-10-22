"""Train the REVE attention head (submission/reve_submission.py) on cached REVE tokens.

    python scripts/train_on_tokens.py TOKENS OUT_FOLDER [--dataset xu2025alljoined] [--holdout-subjects]
        [--use-val] [--seed N] [--set name=value ...] [--head name=value ...]

TOKENS = a scripts/cache_reve_tokens.py folder (e.g. cache/reve_tokens/xu2025alljoined). Training
reuses submission/submission.py's Solver.fit unchanged (CLIP loss, learnable temperature, lr
schedule, early stopping on the held-out split, per-epoch checkpoints, --resume): the model is the
head and the batches are token tensors instead of windows. --set overrides Solver settings,
--head AttentionPoolHead arguments (n_queries, hidden, dropout, n_heads).

OUT_FOLDER gets submission.py (= reve_submission.py) + weights.pt: a Codabench submission that
runs frozen REVE on raw windows (score it end to end with scripts/eval_local.py).
"""
import argparse
import ast
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
TRACK = REPO / "external/2026-competition/tracks/image_decoding"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def literal(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("tokens", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--dataset", default=None, help="held-out participants config to use (default: THINGS-EEG2)")
    parser.add_argument("--holdout-subjects", action="store_true", help="select on held-out participants")
    parser.add_argument("--use-val", action="store_true", help="also train on the val split")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE", help="Solver settings")
    parser.add_argument("--head", nargs="*", default=[], metavar="NAME=VALUE", help="AttentionPoolHead arguments")
    parser.add_argument("--top-layers", type=int, default=0, metavar="K",
                        help="stage 2: also fine-tune REVE's top K layers (TOKENS must be cached after layer 22-K)")
    parser.add_argument("--init-head", type=Path, default=None, metavar="WEIGHTS",
                        help="start the head from a stage-1 run's weights.pt (LP-FT)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and (args.out / "run_config.json").exists():
        print(f"{args.out} already holds a finished run")
        return
    if args.out.exists() and any(args.out.iterdir()) and not args.resume:
        raise SystemExit(f"{args.out} is not empty: pick a new folder, or pass --resume")

    sys.path.insert(0, str(TRACK))
    sys.path.insert(0, str(REPO))
    from src.data.heldout import _loader, heldout_subjects_split, score_galleries
    from src.data.window_cache import CachedWindows

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    meta = json.loads((args.tokens / "meta.json").read_text())
    loaders = {split: type("Loader", (), {"dataset": CachedWindows([args.tokens / split]), "batch_size": args.batch_size})()
               for split in ("train", "val", "test") if (args.tokens / split / "X.npy").exists()}
    galleries = None
    if args.holdout_subjects:
        split = heldout_subjects_split(loaders, device, args.batch_size, args.num_workers, include_val=args.use_val,
                                       dataset=args.dataset)
        train_loader, galleries = split["train"], split["galleries"]
    else:
        parts = [loaders["train"].dataset] + ([loaders["val"].dataset] if args.use_val else [])
        train_loader = _loader(parts, device, True, args.batch_size, args.num_workers)
    print(f"Train: {len(train_loader.dataset)} token sets of shape {tuple(loaders['train'].dataset.parts[0]['X'].shape[1:])}"
          f", device={device}", flush=True)

    solver_module = load_module("solver", REPO / "submission/submission.py")
    reve_module = load_module("reve_submission", REPO / "submission/reve_submission.py")
    solver = solver_module.Solver.get_instance()
    solver.seed = args.seed
    overrides = {name: literal(value) for name, _, value in (s.partition("=") for s in args.set)}
    for name, value in overrides.items():
        if not hasattr(solver, name):
            raise SystemExit(f"Solver has no attribute {name!r}")
        setattr(solver, name, value)
    head_args = {name: literal(value) for name, _, value in (s.partition("=") for s in args.head)}
    args.out.mkdir(parents=True, exist_ok=True)
    solver.checkpoint = args.out / "checkpoint.pt"
    if galleries is not None:
        objective = load_module("objective", TRACK / "objective.py").Objective.get_instance()
        eval_meta = {k: v for k, v in meta.items() if k != "dataset"}
        solver.epoch_callback = lambda model, *_: score_galleries(objective, galleries, model, eval_meta)["top5_acc_subject_agg"]

    torch.manual_seed(args.seed)
    head = reve_module.AttentionPoolHead(n_outputs=meta["n_outputs"], **head_args).to(device)
    if args.init_head:
        head.load_state_dict(torch.load(args.init_head, map_location=device, weights_only=True))
        print(f"Head initialised from {args.init_head}", flush=True)
    if args.top_layers:
        reve = reve_module.FrozenREVE(meta["ch_names"], meta["sfreq"], meta["n_times"], device)
        blocks = reve.model.transformer.layers
        start = len(blocks) - args.top_layers
        if meta.get("layer") != start:
            raise SystemExit(f"--top-layers {args.top_layers} needs tokens cached after layer {start}, "
                             f"{args.tokens} has layer {meta.get('layer')}")
        import copy
        head = reve_module.REVETopFinetune(copy.deepcopy(blocks[start:]), head).to(device).requires_grad_(True)
        del reve
        print(f"Fine-tuning REVE layers {start + 1}-{len(blocks)} "
              f"({sum(p.numel() for p in head.layers.parameters()) / 1e6:.1f}M parameters) + head", flush=True)
    start = time.time()
    solver.fit(head, train_loader)
    print(f"fit done in {(time.time() - start) / 60:.1f} min", flush=True)

    shutil.copy(REPO / "submission/reve_submission.py", args.out / "submission.py")
    torch.save(head.state_dict(), args.out / "weights.pt")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    history = getattr(solver, "history", [])
    scored = [r for r in history if "heldout" in r]
    key = "heldout_smooth" if any("heldout_smooth" in r for r in scored) else "heldout"  # the epoch fit kept
    best = max((r for r in scored if key in r), key=lambda r: r[key]) if scored else {}
    (args.out / "run_config.json").write_text(json.dumps({
        "command": "python " + " ".join(sys.argv), "commit": commit, "tokens": str(args.tokens),
        "dataset": args.dataset, "holdout_subjects": args.holdout_subjects, "use_val": args.use_val,
        "seed": args.seed, "overrides": overrides, "head": head_args, "top_layers": args.top_layers,
        "init_head": str(args.init_head) if args.init_head else None, "train_items": len(train_loader.dataset),
        "fit_minutes": round((time.time() - start) / 60, 1), "epochs_run": len(history),
        "best_epoch": best.get("epoch"), "best_heldout_top5_sa": round(best["heldout"], 4) if best else None,
        "best_heldout_smooth": round(best["heldout_smooth"], 4) if "heldout_smooth" in best else None,
        "resumed_after_epoch": getattr(solver, "resumed_from", None),
    }, indent=2))
    (args.out / "run_config_curve.json").write_text(json.dumps(history, indent=1))
    solver.checkpoint.unlink(missing_ok=True)
    print(f"Submission ready in {args.out}: {sorted(p.name for p in args.out.iterdir())}")


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", str(REPO / "cache" / "huggingface"))
    os.environ.setdefault("REVE_POSITIONS_PATH", str(REPO / "cache" / "mne_data"))
    main()
