"""
Codabench submission for Track 1: frozen REVE + attention-pooling head (REVE transfer, stage 1).

REVE (El Ouahidi et al., 2025; braindecode, "brain-bzh/reve-base", pretrained on EEG from many
people) turns each 1 s window into one 512-d token per channel, placing channels by name, so it
reads any montage (63 or 32 channels). Unlike the official probe (solvers/reve_probe.py), which
averages the tokens over channels, a small head attends over them with learned queries, keeping
where on the scalp the signal is, and projects to the image embedding.

Only the head is trained (scripts/train_on_tokens.py, on tokens cached by
scripts/cache_reve_tokens.py) and shipped (weights.pt or weights_*.pt for an ensemble). REVE's
weights come from the worker's staged Hugging Face cache; braindecode is installed on the worker
(external/2026-competition/tools/Dockerfile, staged_hf_models.txt).
"""
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark_utils.base_solver import CompetSolver


class FrozenREVE:
    """Frozen pretrained REVE -> (B, C, 512) tokens, with the official REVEEncoder's preprocessing:
    resample to 200 Hz (linear), per-window channel z-score, clamp at +-15."""

    SFREQ, PATCH = 200, 200

    def __init__(self, ch_names, sfreq: float, n_times: int, device):
        from braindecode.models import REVE

        self.device = device
        self.n_times = max(round(n_times * self.SFREQ / float(sfreq)), self.PATCH)
        self.model = REVE.from_pretrained("brain-bzh/reve-base", n_outputs=1, n_chans=len(ch_names),
                                          n_times=self.n_times, sfreq=self.SFREQ, chs_info=None).to(device)
        names = [str(n).removeprefix("EEG ").split("-", 1)[0] for n in ch_names]
        unknown = [n for n in names if n not in self.model._position_bank.mapping]
        if unknown:
            raise ValueError("REVE cannot locate these EEG channels: " + ", ".join(unknown))
        self.model.default_pos = self.model.get_positions(names)
        self.model.eval().requires_grad_(False)

    @torch.inference_mode()
    def tokens(self, X: torch.Tensor) -> torch.Tensor:
        X = X.float().to(self.device)
        if X.shape[-1] != self.n_times:
            X = F.interpolate(X, size=self.n_times, mode="linear", align_corners=False)
        X = (X - X.mean(-1, keepdim=True)) / X.std(-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        out = self.model(X.clamp(-15.0, 15.0), return_features=True)
        feats = out["features"] if isinstance(out, dict) else out  # (B, C, T', 512)
        return feats.mean(dim=2) if feats.ndim == 4 else feats  # (B, C, 512): one token per channel


class AttentionPoolHead(nn.Module):
    """(B, C, D) channel tokens -> L2-normalised (B, n_outputs). ``n_queries`` learned queries attend
    over the channels (any C), their outputs are concatenated and projected by an MLP."""

    def __init__(self, n_outputs: int = 1536, dim: int = 512, n_queries: int = 8, n_heads: int = 8,
                 hidden: int = 1024, dropout: float = 0.25):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.queries = nn.Parameter(torch.randn(n_queries, dim) / math.sqrt(dim))
        self.attention = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.projection_head = nn.Sequential(  # named like EEGToImageEncoder's (Solver.fit's adversary reads it)
            nn.Linear(n_queries * dim, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, n_outputs))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.norm(tokens.float())
        pooled, _ = self.attention(self.queries.expand(len(tokens), -1, -1), tokens, tokens, need_weights=False)
        return F.normalize(self.projection_head(pooled.flatten(1)), dim=-1)

    @torch.inference_mode()
    def predict(self, tokens):  # on cached tokens (scripts/train_on_tokens.py)
        self.eval()
        return self.forward(tokens)


def head_sizes(state_dict):
    """AttentionPoolHead arguments read back from its weights."""
    n_queries, dim = state_dict["queries"].shape
    return {"n_outputs": state_dict["projection_head.3.weight"].shape[0], "dim": dim, "n_queries": n_queries,
            "hidden": state_dict["projection_head.0.weight"].shape[0]}


class REVEImageModel:
    """Raw EEG windows -> image embeddings: frozen REVE tokens -> head(s) (mean of an ensemble's
    L2-normalised predictions)."""

    def __init__(self, reve: FrozenREVE, heads):
        self.reve, self.heads = reve, heads

    @torch.inference_mode()
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        tokens = self.reve.tokens(X)
        return torch.stack([h.predict(tokens) for h in self.heads]).mean(0)


class Solver(CompetSolver):
    name = "REVEAttentionHead"

    def load_model(self, meta: dict):
        device = meta["device"]
        reve = FrozenREVE(meta["ch_names"], meta["sfreq"], meta["n_times"], device)
        submission_dir = Path(meta["submission_dir"])
        heads = []
        for path in sorted(submission_dir.glob("weights_*.pt")) or [submission_dir / "weights.pt"]:
            state = torch.load(path, map_location=device, weights_only=True)
            head = AttentionPoolHead(**head_sizes(state)).to(device)
            head.load_state_dict(state)
            heads.append(head.eval())
        return REVEImageModel(reve, heads)
