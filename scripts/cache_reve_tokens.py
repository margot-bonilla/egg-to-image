"""Run frozen REVE once over a window cache and store its per-channel tokens (stage 1 of REVE transfer).

    python scripts/cache_reve_tokens.py [--dataset xu2025alljoined] [--splits train val test] [--batch-size 256]

REVE (brain-bzh/reve-base, braindecode; pretrained on EEG from many people) maps each 1 s window to
one 512-d token per channel. The official probe (solvers/reve_probe.py) averaged them over channels,
which discards where on the scalp the signal is; here every channel's token is kept, so a head can
pool them with attention (submission/reve_submission.py, scripts/train_on_tokens.py).

Input preprocessing is the official REVEEncoder's: resample to 200 Hz (linear interpolation),
per-window channel z-scoring, clamp at +-15. Output: cache/reve_tokens/<dataset>/<split>/X.npy
(N, C, 512) float16, in the window cache's order; its image/target/info/fields files are linked,
so CachedWindows and the held-out splits read the token cache like a window cache.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
TRACK = REPO / "external/2026-competition/tracks/image_decoding"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", default=None, help="image-task overlay, e.g. xu2025alljoined (default: THINGS-EEG2)")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="only the first N windows per split (smoke test)")
    parser.add_argument("--layer", type=int, default=None,
                        help="tokens after this many of REVE's 22 layers (stage 2 fine-tunes the layers above); "
                             "default: final tokens")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    name = args.dataset or "things_eeg2"
    windows = REPO / "cache" / "windows" / name
    out = args.out or REPO / "cache" / "reve_tokens" / (name if args.layer is None else f"{name}_L{args.layer}")

    sys.path.insert(0, str(TRACK))
    sys.path.insert(0, str(REPO))
    from benchmark_utils.nb_task import load_task
    from src.data.window_cache import CachedWindows

    cache = str(REPO / "cache")
    _, meta = load_task("eeg", "image", data_dir=REPO / "data", dataset=args.dataset, device="cpu", overrides={
        "study.source.infra.folder": cache, "neuro.infra.folder": cache, "target.infra.folder": cache,
        "target.infra.cluster": None, "target.infra.permissions": None})
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    spec = __import__("importlib.util").util.spec_from_file_location("reve_submission", REPO / "submission/reve_submission.py")
    module = __import__("importlib.util").util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reve = module.FrozenREVE(meta["ch_names"], meta["sfreq"], meta["n_times"], device)

    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps({"ch_names": list(meta["ch_names"]), "sfreq": meta["sfreq"],
                                               "n_times": meta["n_times"], "n_outputs": int(meta["target_shape"][-1]),
                                               "dataset": args.dataset, "layer": args.layer}, indent=1))
    for split in args.splits:
        src = CachedWindows([windows / split])
        n = len(src) if args.limit is None else min(args.limit, len(src))
        d = out / split
        d.mkdir(exist_ok=True)
        for f in (windows / split).iterdir():  # targets, image ids, info, fields: shared with the window cache
            if f.name != "X.npy" and not (d / f.name).exists():
                (d / f.name).symlink_to(f.resolve())
        tokens = None
        start = time.time()
        for lo in range(0, n, args.batch_size):
            X, _, _ = src.__getitems__(list(range(lo, min(lo + args.batch_size, n))))
            t = reve.tokens(X.to(device), args.layer).to(torch.float16).cpu().numpy()
            if tokens is None:
                tokens = np.lib.format.open_memmap(d / "X.npy.tmp", mode="w+", dtype=np.float16, shape=(n, *t.shape[1:]))
            tokens[lo:lo + len(t)] = t
            print(f"\r{split}: {lo + len(t)}/{n} windows ({(time.time() - start) / 60:.1f} min)", end="", flush=True)
        print()
        tokens.flush()
        del tokens
        (d / "X.npy.tmp").rename(d / "X.npy")
        if args.limit is not None:  # a truncated split: keep the shared files consistent with it
            for f in d.iterdir():
                if f.is_symlink() and f.suffix == ".npy" and f.name != "targets.npy":
                    arr = np.load(f)
                    f.unlink()
                    np.save(f, arr[:n])
            fields = json.loads((d / "fields.json").read_text())
            for key in fields:
                fields[key]["index"] = fields[key]["index"][:n]
            (d / "fields.json").unlink()
            (d / "fields.json").write_text(json.dumps(fields))
    print(f"REVE tokens written to {out}")


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", str(REPO / "cache" / "huggingface"))
    os.environ.setdefault("REVE_POSITIONS_PATH", str(REPO / "cache" / "mne_data"))
    main()
