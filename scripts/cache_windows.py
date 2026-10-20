"""Extract the competition's windows once into a memory-mapped cache (issue #14).

    python scripts/cache_windows.py [--out cache/windows/things_eeg2] [--splits train val test]
                                    [--dataset xu2025alljoined] [--dtype float32|float16]

Then pass ``--cache DIR`` to scripts/train_solver.py and scripts/eval_local.py. The cache
holds exactly what benchmark_utils.nb_task.load_task yields (float32 by default, about
20 GB for THINGS-EEG2 train + val); see src/data/window_cache.py for the layout. Rebuild it
if the data pipeline changes: both scripts check window counts and the first/last window.
"""
import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TRACK = REPO / "external/2026-competition/tracks/image_decoding"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=None, help="default: cache/windows/<dataset>")
    parser.add_argument("--dataset", default=None, help="image-task overlay, e.g. xu2025alljoined (default: THINGS-EEG2)")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    parser.add_argument("--num-workers", type=int, default=8)
    args = parser.parse_args()
    out = args.out or REPO / "cache" / "windows" / (args.dataset or "things_eeg2")

    sys.path.insert(0, str(TRACK))
    sys.path.insert(0, str(REPO))
    from benchmark_utils.nb_task import load_task
    from src.data.window_cache import build_cache

    cache = str(REPO / "cache")
    loaders, _meta = load_task(
        "eeg", "image", data_dir=REPO / "data", dataset=args.dataset, device="cpu", overrides={
            "study.source.infra.folder": cache,
            "neuro.infra.folder": cache,
            "target.infra.folder": cache,
            "target.infra.cluster": None,
            "target.infra.permissions": None,
        },
    )
    start = time.time()
    build_cache(loaders, out, args.splits, args.dtype, args.num_workers)
    print(f"Cache written to {out} in {(time.time() - start) / 60:.1f} min")


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", str(REPO / "cache" / "huggingface"))
    os.environ.setdefault("REVE_POSITIONS_PATH", str(REPO / "cache" / "mne_data"))
    main()
