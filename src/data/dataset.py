"""Synthetic EEG/embedding data for local development (used by src/train.py and the smoke test).

Real data goes through the competition pipeline instead (benchmark_utils.nb_task.load_task,
see scripts/train_solver.py and scripts/eval_local.py).
"""
import torch
from torch.utils.data import Dataset


def standardize_channel_wise(epoch: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Zero mean, unit standard deviation per channel of a (C, T) epoch."""
    mean = epoch.mean(dim=-1, keepdim=True)
    std = epoch.std(dim=-1, keepdim=True).clamp_min(eps)
    return (epoch - mean) / std


class EEGImageDataset(Dataset):
    """(C, T) EEG epochs paired with (D,) image embeddings and subject ids."""

    def __init__(self, eeg: torch.Tensor, targets: torch.Tensor, subject_ids: torch.Tensor,
                 standardize: bool = True):
        self.eeg, self.targets, self.subject_ids = eeg, targets, subject_ids
        self.standardize = standardize

    def __len__(self) -> int:
        return len(self.eeg)

    def __getitem__(self, i: int) -> dict:
        eeg = self.eeg[i]
        if self.standardize:
            eeg = standardize_channel_wise(eeg)
        return {"eeg": eeg, "target": self.targets[i], "subject_id": int(self.subject_ids[i])}


def create_synthetic_dataset(n_samples: int = 256, n_chans: int = 32, n_times: int = 256,
                             embedding_dim: int = 1536, planted_signal_strength: float = 0.5,
                             n_subjects: int = 4, seed: int = 0) -> EEGImageDataset:
    """Noise EEG with each target embedding linearly planted in it, so retrieval is learnable."""
    gen = torch.Generator().manual_seed(seed)
    targets = torch.randn(n_samples, embedding_dim, generator=gen)
    projection = torch.randn(embedding_dim, n_chans * n_times, generator=gen) / embedding_dim ** 0.5
    signal = (targets @ projection).reshape(n_samples, n_chans, n_times)
    noise = torch.randn(n_samples, n_chans, n_times, generator=gen)
    eeg = planted_signal_strength * signal + (1 - planted_signal_strength) * noise
    subject_ids = torch.randint(0, n_subjects, (n_samples,), generator=gen)
    return EEGImageDataset(eeg, targets, subject_ids)
