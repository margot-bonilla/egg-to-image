"""Repetition averaging for training (issue #17).

Each THINGS-EEG2 training image is shown ~4 times to each subject. Averaging k repetitions
of the same (subject, image) gives a cleaner EEG response; the subject-aggregated metric
does the same at test time. ``RepetitionAveraged`` turns a training dataset of windows
into one item per (subject, image) group: the mean of k of its windows (k fixed, or
random in 1..k), with the group's shared target. Training-time only: ``predict`` still
receives single windows.
"""
import random

import torch
from torch.utils.data import ConcatDataset, Dataset, Subset


def dataset_fields(dataset, keys=("subject", "stem")) -> list[dict]:
    """Trigger-event fields of every item of a (nested Subset/ConcatDataset of) competition
    window dataset(s) (``benchmark_utils.nb_task._NBWindows``), in dataset order."""
    if hasattr(dataset, "fields"):  # src.data.window_cache.CachedWindows
        return dataset.fields(keys)
    if isinstance(dataset, ConcatDataset):
        return [f for d in dataset.datasets for f in dataset_fields(d, keys)]
    if isinstance(dataset, Subset):
        inner = dataset_fields(dataset.dataset, keys)
        return [inner[i] for i in dataset.indices]
    segments = dataset.seg_ds.segments
    return [{k: segments[int(i)].trigger.extra[k] for k in keys} for i in dataset.index]


class RepetitionAveraged(Dataset):
    """One item per (subject, image) group: the mean of k of its windows.

    k: number of repetitions averaged (capped by the group size).
    random_k: draw k uniformly in 1..k for every item (mixes clean and noisy examples).
    """

    def __init__(self, dataset, k: int = 2, random_k: bool = False, seed: int = 0):
        self.dataset, self.k, self.random_k = dataset, k, random_k
        groups = {}
        for i, f in enumerate(dataset_fields(dataset)):
            groups.setdefault((f["subject"], f["stem"]), []).append(i)
        self.groups = list(groups.values())
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, g: int):
        members = self.groups[g]
        k = min(self.rng.randint(1, self.k) if self.random_k else self.k, len(members))
        picks = self.rng.sample(members, k)
        items = [self.dataset[i] for i in picks]
        X = torch.stack([x for x, _, _ in items]).mean(0)
        _, y, info = items[0]
        return X, y, info
