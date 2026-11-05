"""Check that an unzipped submission survives montage surprises, and time its inference.

    python scripts/check_robustness.py dist/replay/<name> [--device mps]

Loads the submission like Codabench (Solver.load_model with the test montage's ch_names) on 64
Alljoined-1.6M test windows, then compares its predictions (mean cosine) with the reference ones
when the channel names are upper-case, the channels shuffled, 2 or 4 channels missing, or 6
unknown channels added. Name matching should make the first two and the last exactly 1.0;
missing channels are zero-filled, so they only degrade the predictions.
"""
import argparse
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(REPO / "cache/huggingface"))
os.environ.setdefault("REVE_POSITIONS_PATH", str(REPO / "cache/mne_data"))
sys.path.insert(0, str(REPO / "external/2026-competition/tracks/image_decoding"))
sys.path.insert(0, str(REPO))
from src.data.window_cache import CachedWindows  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()
    sub, device = args.submission_dir.resolve(), args.device
    spec = importlib.util.spec_from_file_location("candidate", sub / "submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cache = REPO / "cache/windows/xu2025alljoined"
    names = json.loads((cache / "meta.json").read_text())["ch_names"]
    X, _, _ = CachedWindows([cache / "test"]).__getitems__(list(range(0, 64000, 1000)))  # 64 windows
    base_meta = {"n_times": 120, "sfreq": 120.0, "n_outputs": 1536, "device": device, "submission_dir": sub}

    def load(ch_names):
        solver = module.Solver.get_instance()
        solver.train_loader = None
        return solver.load_model({**base_meta, "n_chans": len(ch_names), "ch_names": ch_names})

    def run(ch_names, Xin):
        return load(ch_names).predict(Xin.to(device)).cpu()

    ref = run(names, X)

    def cos(pred):
        return float(torch.nn.functional.cosine_similarity(pred, ref).mean())

    rng = random.Random(0)
    perm = list(range(len(names)))
    rng.shuffle(perm)
    print(f"upper-case names:         cosine {cos(run([n.upper() for n in names], X)):.4f}")
    print(f"shuffled channel order:   cosine {cos(run([names[i] for i in perm], X[:, perm])):.4f}")
    for k in (2, 4):
        drop = set(rng.sample(range(len(names)), k))
        keep = [i for i in range(len(names)) if i not in drop]
        print(f"{k} channels missing:       cosine {cos(run([names[i] for i in keep], X[:, keep])):.4f}")
    extra = torch.randn(len(X), 6, X.shape[-1])
    extra_names = names + [f"X{i}" for i in range(6)]
    print(f"6 extra unknown channels: cosine {cos(run(extra_names, torch.cat([X, extra], 1))):.4f}")
    big = X.repeat(16, 1, 1)  # 1,024 windows
    model = load(names)
    start = time.time()
    for i in range(0, len(big), 256):
        model.predict(big[i:i + 256].to(device))
    per_window = (time.time() - start) / len(big)
    print(f"inference: {per_window * 1000:.2f} ms/window ({device}) -> 300k windows ~ {per_window * 3e5 / 60:.0f} min")


if __name__ == "__main__":
    main()
