"""Assemble a weighted multi-family ensemble into one Codabench submission folder.

    python scripts/package_ensemble.py OUT --encoders FOLDER [FOLDER ...] --reve FOLDER [FOLDER ...]
        [--reve-weight 0.25] [--channels-from FOLDER]

Encoder folders hold a weights.pt (EEGToImageEncoder); REVE folders a weights.pt from
scripts/train_on_tokens.py. OUT gets submission/ensemble_submission.py as submission.py, the encoder
and REVE solver code (encoder_model.py, reve_model.py), the weights renamed weights_<i>.pt / reve_<i>.pt,
channels.json (the training montage) and ensemble.json (groups and weights; the encoder group gets
1 - reve-weight). Zip it with `make zip-submission SUBMISSION=OUT`.
"""
import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--encoders", nargs="+", type=Path, required=True)
    parser.add_argument("--reve", nargs="*", type=Path, default=[])
    parser.add_argument("--reve-weight", type=float, default=0.25)
    parser.add_argument("--channels-from", type=Path, default=None,
                        help="folder or window cache with the training montage (channels.json or meta.json); "
                             "default: the first encoder folder's channels.json, else Alljoined's window cache")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"{args.out} is not empty")
    args.out.mkdir(parents=True, exist_ok=True)

    sources = [args.channels_from] if args.channels_from else [*args.encoders, REPO / "cache/windows/xu2025alljoined"]
    names = None
    for src in sources:
        for f in ("channels.json", "meta.json"):
            if (src / f).exists():
                names = json.loads((src / f).read_text())["ch_names"]
                break
        if names:
            break
    if not names:
        raise SystemExit("no training montage found (channels.json / meta.json)")
    (args.out / "channels.json").write_text(json.dumps({"ch_names": names}))

    shutil.copy(REPO / "submission/ensemble_submission.py", args.out / "submission.py")
    shutil.copy(REPO / "submission/submission.py", args.out / "encoder_model.py")
    groups = []
    enc_files = []
    for i, folder in enumerate(args.encoders, 1):
        shutil.copy(folder / "weights.pt", args.out / f"weights_{i}.pt")
        enc_files.append(f"weights_{i}.pt")
    groups.append({"kind": "encoder", "files": enc_files, "weight": 1 - args.reve_weight if args.reve else 1.0,
                   "sources": [str(f) for f in args.encoders]})
    if args.reve:
        shutil.copy(REPO / "submission/reve_submission.py", args.out / "reve_model.py")
        reve_files = []
        for i, folder in enumerate(args.reve, 1):
            shutil.copy(folder / "weights.pt", args.out / f"reve_{i}.pt")
            reve_files.append(f"reve_{i}.pt")
        groups.append({"kind": "reve", "files": reve_files, "weight": args.reve_weight,
                       "sources": [str(f) for f in args.reve]})
    (args.out / "ensemble.json").write_text(json.dumps({"groups": groups}, indent=1))
    size = sum(f.stat().st_size for f in args.out.iterdir()) / 1e6
    print(f"Ensemble ready in {args.out} ({size:.0f} MB): {sorted(p.name for p in args.out.iterdir())}")


if __name__ == "__main__":
    main()
