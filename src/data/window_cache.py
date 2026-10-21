"""Window cache (issue #14): the competition's windows, extracted once to memory-mapped .npy.

Every run used to re-extract ~670k windows through 8 DataLoader workers, each holding part
of the dataset in memory (two parallel runs swapped on Oct 19). ``build_cache`` writes each
split of ``benchmark_utils.nb_task.load_task`` once, in dataset order:

    <cache>/<split>/X.npy            (N, C, T) windows exactly as load_task yields them
    <cache>/<split>/targets.npy      (M, D) the split's distinct image embeddings (float32)
    <cache>/<split>/image.npy        (N,) row of each window's target in targets.npy
    <cache>/<split>/<info key>.npy   (N,) subject_id, record_id, onset, stream_id[, context_id]
    <cache>/<split>/fields.json      category / stem / subject of every window, as indices into
                                     name lists (for the held-out split and repetition averaging)

``CachedWindows`` reads a split back as a map-style dataset with batched reads
(``__getitems__``): a DataLoader over it (persistent workers, or a ``CachedLoader`` without
workers) yields the same ``(X, y, info)`` batches as load_task's loaders and shuffles with the
same RNG draws. Workers and parallel runs share the files' pages through the OS cache instead
of each holding part of the dataset; 4 workers load ~4x faster than load_task's 8.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

INFO_KEYS = ("subject_id", "record_id", "onset", "stream_id", "context_id")
FIELD_KEYS = ("category", "stem", "subject")


def build_cache(loaders: dict, out: Path, splits=("train", "val", "test"), dtype="float32",
                num_workers: int = 8, batch_size: int = 256) -> None:
    """Write ``loaders[split]``'s windows (load_task loaders) to ``out/<split>/``."""
    from src.data.heldout import window_triggers

    for split in splits:
        ds = loaders[split].dataset
        d = Path(out) / split
        d.mkdir(parents=True, exist_ok=True)
        triggers = window_triggers(loaders[split], FIELD_KEYS)
        fields = {}
        for key in FIELD_KEYS:
            names = sorted({str(t[key]) for t in triggers})
            lookup = {n: i for i, n in enumerate(names)}
            fields[key] = {"names": names, "index": [lookup[str(t[key])] for t in triggers]}
        stems = np.array(fields["stem"]["index"])
        X0, y0, info0 = ds[0]
        n = len(ds)
        X = np.lib.format.open_memmap(d / "X.npy.tmp", mode="w+", dtype=dtype, shape=(n, *X0.shape))
        targets = np.full((len(fields["stem"]["names"]), *y0.shape), np.nan, dtype=np.float32)
        info = {k: np.empty(n, dtype=np.float64 if k == "onset" else np.int64) for k in INFO_KEYS if k in info0}
        start = 0
        # In dataset order (no shuffling), so row i is window i of the split.
        for Xb, yb, infob in DataLoader(ds, batch_size=batch_size, num_workers=num_workers):
            stop = start + len(Xb)
            X[start:stop] = Xb.numpy()
            for k in info:
                info[k][start:stop] = infob[k].numpy()
            rows = stems[start:stop]
            new = np.isnan(targets[rows, 0])
            targets[rows[new]] = yb[new].numpy()
            # Repeated presentations of an image must share its embedding (the objective's gallery).
            if not np.array_equal(targets[rows], yb.numpy()):
                raise ValueError(f"{split}: windows of the same image stem have different targets")
            start = stop
            print(f"\r{split}: {stop}/{n} windows", end="", flush=True)
        print()
        X.flush()
        del X
        (d / "X.npy.tmp").rename(d / "X.npy")
        np.save(d / "targets.npy", targets)
        np.save(d / "image.npy", stems)
        for k, v in info.items():
            np.save(d / f"{k}.npy", v)
        (d / "fields.json").write_text(json.dumps(fields))


def _batch(batch):
    """collate_fn for CachedWindows: ``__getitems__`` already returns the collated batch."""
    return batch


class CachedLoader(DataLoader):
    """A worker-less DataLoader that draws from torch's global RNG like a persistent-worker one.

    Every new DataLoader iterator draws a base seed from the global RNG; with persistent
    workers (the uncached loaders) only the first epoch creates one. Undoing the draw on later
    epochs keeps shuffling and dropout identical to an uncached run with workers.
    """

    _iterations = 0

    def __iter__(self):
        state = torch.get_rng_state() if self._iterations else None
        iterator = super().__iter__()  # draws the base seed; the sampler draws lazily, later
        if state is not None:
            torch.set_rng_state(state)
        self._iterations += 1
        return iterator


class CachedWindows(Dataset):
    """A cached split (or a view/concatenation of cached splits) as an (X, y, info) dataset.

    Use ``loader(...)`` (or a DataLoader with ``collate_fn=_batch``): reads are batched.
    ``take(indices)`` / ``concat(parts)`` mirror ``Subset`` / ``ConcatDataset``.
    channels: per split directory, the channel indices to keep (None = all), e.g. THINGS-EEG2's
        63 channels reduced to Alljoined's 32, in Alljoined's order, to train on both datasets.
    subject_offset: per split directory, added to info["subject_id"] (keeps datasets' ids apart).
    """

    def __init__(self, split_dirs, index=None, channels=None, subject_offset=None):
        self.split_dirs = [Path(d) for d in split_dirs]
        self.channels = list(channels) if channels is not None else [None] * len(self.split_dirs)
        self.subject_offset = list(subject_offset) if subject_offset is not None else [0] * len(self.split_dirs)
        self._parts = None
        sizes = [int(np.load(d / "image.npy", mmap_mode="r").shape[0]) for d in self.split_dirs]
        self.offsets = np.cumsum([0] + sizes)
        self.index = np.arange(self.offsets[-1]) if index is None else np.asarray(index, dtype=np.int64)

    @property
    def parts(self):
        """Per split: memory-mapped arrays, opened lazily (and again in each worker process)."""
        if self._parts is None:
            self._parts = []
            for d in self.split_dirs:
                part = {"X": np.load(d / "X.npy", mmap_mode="r"), "targets": np.load(d / "targets.npy"),
                        "image": np.load(d / "image.npy")}
                part.update({k: np.load(d / f"{k}.npy") for k in INFO_KEYS if (d / f"{k}.npy").exists()})
                self._parts.append(part)
        return self._parts

    def __getstate__(self):
        return {**self.__dict__, "_parts": None}

    def __len__(self):
        return len(self.index)

    def take(self, indices) -> "CachedWindows":
        return CachedWindows(self.split_dirs, self.index[np.asarray(indices, dtype=np.int64)], self.channels,
                             self.subject_offset)

    @staticmethod
    def concat(parts) -> "CachedWindows":
        dirs, index, channels, offsets = [], [], [], []
        for p in parts:
            shift = sum(int(np.load(d / "image.npy", mmap_mode="r").shape[0]) for d in dirs)
            dirs += p.split_dirs
            channels += p.channels
            offsets += p.subject_offset
            index.append(p.index + shift)
        return CachedWindows(dirs, np.concatenate(index), channels, offsets)

    def fields(self, keys=FIELD_KEYS) -> list[dict]:
        """category / stem / subject of every window, in dataset order (like window_triggers)."""
        cols = {k: [] for k in keys}
        for d in self.split_dirs:
            f = json.loads((d / "fields.json").read_text())
            for k in keys:
                cols[k] += [f[k]["names"][i] for i in f[k]["index"]]
        return [{k: cols[k][int(i)] for k in keys} for i in self.index]

    def __getitems__(self, indices):
        rows = self.index[np.asarray(indices, dtype=np.int64)]
        which = np.searchsorted(self.offsets, rows, side="right") - 1
        n = len(rows)
        X = y = None
        info = {}
        for p in np.unique(which):
            sel = np.flatnonzero(which == p)
            local = rows[sel] - self.offsets[p]
            part = self.parts[p]
            order = np.argsort(local)  # sorted reads from the memory map, then back to batch order
            Xp = np.empty((len(local), *part["X"].shape[1:]), dtype=np.float32)
            Xp[order] = part["X"][local[order]]
            if self.channels[p] is not None:
                Xp = Xp[:, self.channels[p]]
            yp = part["targets"][part["image"][local]]
            if X is None:
                X = np.empty((n, *Xp.shape[1:]), dtype=np.float32)
                y = np.empty((n, *yp.shape[1:]), dtype=np.float32)
            X[sel], y[sel] = Xp, yp
            for k in INFO_KEYS:
                if k in part:
                    values = part[k][local] + (self.subject_offset[p] if k == "subject_id" else 0)
                    info.setdefault(k, np.empty(n, dtype=part[k].dtype))[sel] = values
        return torch.from_numpy(X), torch.from_numpy(y), {k: torch.from_numpy(v) for k, v in info.items()}

    def __getitem__(self, i):
        X, y, info = self.__getitems__([i])
        return X[0], y[0], {k: v[0].item() for k, v in info.items()}


def check_cache(loaders: dict, cache: Path, splits=("train", "val", "test")) -> None:
    """Fail fast if the cache doesn't match load_task's splits (window counts, first/last window)."""
    for split in splits:
        ds = loaders[split].dataset
        cached = CachedWindows([Path(cache) / split])
        if len(cached) != len(ds):
            raise SystemExit(f"cache {cache}/{split} has {len(cached)} windows, load_task has {len(ds)}: rebuild it")
        dtype = cached.parts[0]["X"].dtype  # float16 caches hold rounded windows
        for i in (0, len(ds) - 1):
            X, y, _ = ds[i]
            Xc, yc, _ = cached[i]
            if not (torch.equal(Xc, torch.from_numpy(X.numpy().astype(dtype)).float()) and torch.equal(yc, y)):
                raise SystemExit(f"cache {cache}/{split} window {i} differs from load_task: rebuild it")


def cached_loaders(loaders: dict, cache: Path, device: str, batch_size: int, splits=("train", "val", "test"),
                   num_workers: int = 0) -> dict:
    """load_task-style loaders reading from the cache (train shuffled, others in order)."""
    from src.data.heldout import _loader

    check_cache(loaders, cache, splits)
    out = dict(loaders)
    for split in splits:
        out[split] = _loader([CachedWindows([Path(cache) / split])], device, split == "train", batch_size, num_workers)
    return out
