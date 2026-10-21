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

``heldout_subjects_split`` also holds out 2 of the 10 participants (configs/heldout_subjects.json),
like the sealed final (new people *and* new images): training drops their windows too, and they
are scored on 10 galleries of 200 held-out images (one image per held-out concept each).
On Alljoined-1.6M (``dataset="xu2025alljoined"``: 20 participants, no concept labels, THINGS' images)
it holds out 4 participants (configs/heldout_subjects_xu2025alljoined.json) and scores them on the
test split: the 200 test images, never trained on, ~80 repetitions each, as in the final.
"""
import json
import re
from pathlib import Path

import numpy as np
from torch.utils.data import ConcatDataset, DataLoader, Subset

from src.data.window_cache import CachedLoader, CachedWindows, _batch

REPO = Path(__file__).resolve().parents[2]
HELDOUT_FILE = REPO / "configs" / "heldout_concepts.json"
HELDOUT_SUBJECTS_FILE = REPO / "configs" / "heldout_subjects.json"


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


def _field(extra: dict, key: str):
    """A trigger field, with the image named ``stim_file`` on Alljoined (``stem`` on THINGS-EEG2)
    and no concept (``category`` = "") there."""
    if key in extra:
        return extra[key]
    if key == "stem":
        return extra["stim_file"]
    if key == "category":
        return ""
    return extra[key]


def window_triggers(loader, keys=("category", "stem")) -> list[dict]:
    """Image fields (concept ``category``, image ``stem``) of every window, in loader order.

    Read from the trigger event's ``extra`` dict: ``to_dict()`` would rebuild the whole
    event for each of the ~500k windows.
    """
    ds = loader.dataset
    if hasattr(ds, "fields"):  # src.data.window_cache.CachedWindows
        return ds.fields(keys)
    segments = ds.seg_ds.segments
    return [{k: _field(segments[int(i)].trigger.extra, k) for k in keys} for i in ds.index]


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


def subjects_file(dataset: str | None = None) -> Path:
    """Held-out participants of a dataset (THINGS-EEG2 by default)."""
    return HELDOUT_SUBJECTS_FILE if dataset is None else REPO / "configs" / f"heldout_subjects_{dataset}.json"


def make_heldout_subjects_file(train_loader, n_subjects: int = 2, seed: int = 0, dataset: str | None = None) -> dict:
    """Pick the held-out participants and write subjects_file(dataset)."""
    subjects = sorted({t["subject"] for t in window_triggers(train_loader, ("subject",))})
    rng = np.random.default_rng(seed)
    heldout = {"seed": seed, "subjects": sorted(str(s) for s in rng.choice(subjects, size=n_subjects, replace=False))}
    subjects_file(dataset).write_text(json.dumps(heldout, indent=1))
    return heldout


def _natural(name: str):
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", name)]


def heldout_subjects_split(loaders: dict, device: str, batch_size: int | None = None, num_workers: int = 0,
                           include_val: bool = False, n_train_subjects: int | None = None,
                           dataset: str | None = None) -> dict:
    """Return {"train": loader without held-out concepts or participants,
    "galleries": [loader per gallery]}: the held-out participants' windows of the held-out
    concepts, gallery g holding the g-th image (sorted by name) of every concept.

    Score = mean over galleries of the official metric (``score_galleries``): 10x more queries
    than one gallery, which matters with 2 participants and ~4 repetitions per image.
    n_train_subjects: train on only the first N remaining participants (natural order: 1, 2, ..., 10),
    to measure how the unseen-participant score grows with the number of training participants.
    """
    path = subjects_file(dataset)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: create it once with make_heldout_subjects_file()")
    subjects = set(json.loads(path.read_text())["subjects"])
    triggers = {split: window_triggers(loaders[split], ("category", "stem", "subject")) for split in ("train", "val")}
    train_subjects = sorted({str(t["subject"]) for t in triggers["train"]} - subjects, key=_natural)
    train_subjects = set(train_subjects[:n_train_subjects])
    bs = batch_size or loaders["train"].batch_size

    if not any(t["category"] for t in triggers["train"][:1000]):  # no concepts (Alljoined): score on the test split
        train_parts = [_subset(loaders[split].dataset, [i for i, t in enumerate(triggers[split])
                                                        if str(t["subject"]) in train_subjects])
                       for split in (("train", "val") if include_val else ("train",))]
        test = window_triggers(loaders["test"], ("subject",))
        gallery = _subset(loaders["test"].dataset, [i for i, t in enumerate(test) if str(t["subject"]) in subjects])
        return {"train": _loader(train_parts, device, True, bs, num_workers),
                "galleries": [_loader([gallery], device, False, bs, num_workers)]}

    concepts = set(json.loads(HELDOUT_FILE.read_text())["images"])
    train_parts, held = [], []
    for split in ("train", "val"):
        trig = triggers[split]
        if split == "train" or include_val:
            idx = [i for i, t in enumerate(trig) if t["category"] not in concepts and str(t["subject"]) in train_subjects]
            train_parts.append(_subset(loaders[split].dataset, idx))
        held.append((split, [(i, t) for i, t in enumerate(trig) if t["category"] in concepts and str(t["subject"]) in subjects]))

    images = {}
    for _, windows in held:
        for _, t in windows:
            images.setdefault(t["category"], set()).add(t["stem"])
    images = {c: sorted(v) for c, v in images.items()}
    n_galleries = min(len(v) for v in images.values())
    galleries = []
    for g in range(n_galleries):
        gallery = {v[g] for v in images.values()}
        parts = [_subset(loaders[split].dataset, [i for i, t in windows if t["stem"] in gallery]) for split, windows in held]
        galleries.append(_loader(parts, device, False, bs, num_workers))
    return {"train": _loader(train_parts, device, True, bs, num_workers), "galleries": galleries}


def score_galleries(objective, galleries, model, meta: dict) -> dict:
    """Official metrics on each gallery (objective = tracks/image_decoding Objective), then their
    mean, plus the standard error of top5_acc_subject_agg across galleries."""
    results = []
    for loader in galleries:
        objective.set_data(train_loader=None, test_loader=loader, **meta)
        results.append(objective.evaluate_result(model))
    mean = {k: float(np.mean([r[k] for r in results])) for k in results[0]}
    top5 = [r["top5_acc_subject_agg"] for r in results]
    mean["top5_acc_subject_agg_se"] = float(np.std(top5, ddof=1) / np.sqrt(len(top5))) if len(top5) > 1 else 0.0
    return mean
