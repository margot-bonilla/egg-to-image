"""Held-out-concept validation split for the image task, mirroring the test set.

THINGS-EEG2's test set uses 200 concepts never seen in training, one image each
(a 200-image gallery). Selecting models on that test set means selecting on the
warm-up leaderboard data, so we hold out 200 *training* concepts the same way:

- one image per held-out concept forms the 200-image validation gallery;
- every window of a held-out concept is removed from training (no concept leaks).

The held-out concepts and images live in configs/heldout_concepts.json (committed),
so every teammate and script uses the identical split. Windows come from the
competition loaders (benchmark_utils.nb_task.load_task): training = its "train"
split minus held-out concepts; validation = held-out images in its "train" and
"val" splits (NeuralBench's own timeline-based val split).
"""
import json
from pathlib import Path

import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Subset

from src.data.window_cache import CachedLoader, CachedWindows, _batch

REPO = Path(__file__).resolve().parents[2]
HELDOUT_FILE = REPO / "configs" / "heldout_concepts.json"


class _OnDevice:
    """Move (X, y, info) batches to ``device`` in the main process (like nb_task's loaders)."""

    def __init__(self, loader, device):
        self.loader, self.device = loader, device

    def __iter__(self):
        for X, y, info in self.loader:
            yield X.to(self.device), y.to(self.device), info

    def __len__(self):
        return len(self.loader)

    def __getattr__(self, name):
        return getattr(self.loader, name)


def window_triggers(loader, keys=("category", "stem")) -> list[dict]:
    """Image fields (concept ``category``, image ``stem``) of every window, in loader order.

    Read from the trigger event's ``extra`` dict: ``to_dict()`` would rebuild the whole
    event for each of the ~500k windows.
    """
    ds = loader.dataset
    if hasattr(ds, "fields"):  # src.data.window_cache.CachedWindows
        return ds.fields(keys)
    segments = ds.seg_ds.segments
    return [{k: segments[int(i)].trigger.extra[k] for k in keys} for i in ds.index]


def make_heldout_file(train_loader, n_concepts: int = 200, seed: int = 0) -> dict:
    """Pick the held-out concepts (and one image each) and write HELDOUT_FILE."""
    triggers = window_triggers(train_loader)
    stems = {}
    for t in triggers:
        stems.setdefault(t["category"], set()).add(t["stem"])
    rng = np.random.default_rng(seed)
    concepts = sorted(rng.choice(sorted(stems), size=n_concepts, replace=False))
    heldout = {"seed": seed, "n_concepts": n_concepts,
               "images": {c: str(rng.choice(sorted(stems[c]))) for c in concepts}}
    HELDOUT_FILE.write_text(json.dumps(heldout, indent=1, sort_keys=True))
    return heldout


def _subset(dataset, indices):
    return dataset.take(indices) if isinstance(dataset, CachedWindows) else Subset(dataset, indices)


def _loader(datasets: list, device: str, shuffle: bool, batch_size: int, num_workers: int) -> _OnDevice:
    cached = all(isinstance(d, CachedWindows) for d in datasets)
    if len(datasets) == 1:
        dataset = datasets[0]
    else:
        dataset = CachedWindows.concat(datasets) if cached else ConcatDataset(datasets)
    if cached and num_workers == 0:  # memory-mapped batched reads in the main process
        return _OnDevice(CachedLoader(dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=_batch), device)
    # Cached workers share the memory-mapped files (no copy of the dataset per worker).
    return _OnDevice(DataLoader(dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                persistent_workers=num_workers > 0, collate_fn=_batch if cached else None), device)


def heldout_split(loaders: dict, device: str, batch_size: int | None = None, num_workers: int = 0,
                  include_val: bool = False) -> dict:
    """Return {"train": loader without held-out concepts, "heldout": held-out image windows}.

    include_val: also train on NeuralBench's "val" split (minus held-out concepts), which
    otherwise only contributes held-out windows: +25% training data on THINGS-EEG2.
    """
    if not HELDOUT_FILE.exists():
        raise FileNotFoundError(f"{HELDOUT_FILE} missing: create it once with make_heldout_file()")
    images = json.loads(HELDOUT_FILE.read_text())["images"]
    concepts, gallery = set(images), set(images.values())

    train_parts, heldout_parts = [], []
    for split in ("train", "val"):
        trig = window_triggers(loaders[split])
        if split == "train" or include_val:
            idx = [i for i, t in enumerate(trig) if t["category"] not in concepts]
            train_parts.append(_subset(loaders[split].dataset, idx))
        heldout_parts.append(_subset(loaders[split].dataset, [i for i, t in enumerate(trig) if t["stem"] in gallery]))

    bs = batch_size or loaders["train"].batch_size
    return {
        "train": _loader(train_parts, device, True, bs, num_workers),
        "heldout": _loader(heldout_parts, device, False, bs, num_workers),
    }


def train_plus_val(loaders: dict, device: str, batch_size: int | None = None, num_workers: int = 0) -> _OnDevice:
    """All of NeuralBench's train + val windows (for final models trained on every concept)."""
    bs = batch_size or loaders["train"].batch_size
    return _loader([loaders["train"].dataset, loaders["val"].dataset], device, True, bs, num_workers)
