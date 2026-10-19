"""
Codabench / Benchopt Submission Entrypoint for Track 1: EEG-to-Image.

Implements the CompetSolver contract from the official Submission Guide:
  - ``Solver.load_model(meta)`` rebuilds the architecture from ``meta`` and loads
    ``weights.pt`` from ``meta["submission_dir"]``.
  - The returned model exposes ``predict(X)``: (B, C, T) -> (B, meta["n_outputs"]).
  - ``fit`` / ``save_model`` are optional local-training hooks; Codabench never calls them.

To submit, ZIP ``submission.py`` and ``weights.pt`` at the ROOT of the archive
(``make zip-submission``), never inside a folder.

Only ``torch`` is imported: nothing is installed on the evaluation worker.
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark_utils.base_solver import CompetSolver


class EEGToImageEncoder(nn.Module):
    """Self-contained model architecture (state_dict keys match src/models/eeg_encoder.py)."""

    def __init__(
        self,
        n_chans: int = 32,
        n_times: int = 256,
        output_dim: int = 1536,
        hidden_dim: int = 512,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(1, 31), stride=1, padding=(0, 15), bias=False),
            nn.BatchNorm2d(32),
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(n_chans, 1), groups=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(dropout),
        )
        self.sep_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 15), padding=(0, 7), groups=64, bias=False),
            nn.Conv2d(64, 128, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_chans, n_times)
            x = self.sep_conv(self.spatial_conv(self.temporal_conv(dummy)))
            flattened_dim = x.numel()

        self.projection_head = nn.Sequential(
            nn.Linear(flattened_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.sep_conv(x)
        x = torch.flatten(x, start_dim=1)
        emb = self.projection_head(x)
        return F.normalize(emb, p=2, dim=-1)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x.float())


def info_nce_loss(pred: torch.Tensor, target: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Asymmetric CLIP / InfoNCE loss: classify each window among the batch targets."""
    logits = pred @ F.normalize(target, dim=-1).T / temperature
    return F.cross_entropy(logits, torch.arange(len(pred), device=pred.device))


class Solver(CompetSolver):
    name = "EEGToImageBaseline"

    # Local-training hyperparameters (only used by ``fit``).
    n_epochs = 20
    lr = 1e-3

    def load_model(self, meta: dict):
        """
        meta (built by the platform) provides:
          - n_chans, n_times, sfreq, ch_names, chs_info: input description
          - n_outputs: embedding dimension D (1536 for DINOv2-giant)
          - device: where the input batches live
          - submission_dir: read-only folder holding the files extracted from the ZIP
        """
        device = meta["device"]
        model = EEGToImageEncoder(
            n_chans=meta["n_chans"],
            n_times=meta["n_times"],
            output_dim=meta["n_outputs"],
        )

        # A local training run (train_loader set) starts from scratch; every
        # inference-only run, including Codabench's, must load the shipped weights.
        if getattr(self, "train_loader", None) is None:
            weights_path = Path(meta["submission_dir"]) / "weights.pt"
            state_dict = torch.load(weights_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)

        return model.to(device).eval()

    def fit(self, model, train_loader):
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        model.train()
        for _ in range(self.n_epochs):
            for X, y, _info in train_loader:
                if len(X) < 2:  # BatchNorm1d needs more than one sample per batch
                    continue
                optimizer.zero_grad()
                loss = info_nce_loss(model(X.float()), y.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        model.eval()

    def save_model(self, model, path):
        torch.save(model.state_dict(), Path(path) / "weights.pt")
