"""
Codabench submission for Track 1: frozen REVE + attention-pooling head (REVE transfer, stage 1).

REVE (El Ouahidi et al., 2025; braindecode, "brain-bzh/reve-base", pretrained on EEG from many
people) turns each 1 s window into one 512-d token per channel, placing channels by name, so it
reads any montage (63 or 32 channels). Unlike the official probe (solvers/reve_probe.py), which
averages the tokens over channels, a small head attends over them with learned queries, keeping
where on the scalp the signal is, and projects to the image embedding.

Stage 1 trains only the head (scripts/train_on_tokens.py, on tokens cached by
scripts/cache_reve_tokens.py). Stage 2 also fine-tunes REVE's top K transformer layers (LP-FT:
Kumar et al. 2022, as NeuralBench's REVE recipe fine-tunes the backbone), trained on tokens cached
after layer 22-K. Only the trained parts are shipped (weights.pt or weights_*.pt for an ensemble);
K is read back from the weights. REVE's
weights come from the worker's staged Hugging Face cache; braindecode is installed on the worker
(external/2026-competition/tools/Dockerfile, staged_hf_models.txt).
"""
import copy
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
    def tokens(self, X: torch.Tensor, layer: int | None = None) -> torch.Tensor:
        """Final-layer tokens (B, C, 512), or with ``layer`` the tokens after that many transformer
        layers (B, C x T', 512), the input of a fine-tuned top part (REVETopFinetune)."""
        X = X.float().to(self.device)
        if X.shape[-1] != self.n_times:
            X = F.interpolate(X, size=self.n_times, mode="linear", align_corners=False)
        X = (X - X.mean(-1, keepdim=True)) / X.std(-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        X = X.clamp(-15.0, 15.0)
        if layer is not None:
            return self.model(X, return_output=True)[layer]  # [input, after layer 1, ..., after 22]
        out = self.model(X, return_features=True)
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


class REVETopFinetune(nn.Module):
    """REVE's top transformer layers (copies of the pretrained ones, trainable) + an attention head:
    (B, N, 512) tokens after layer 22-K -> L2-normalised (B, n_outputs). At initialisation it computes
    the frozen model's final tokens, so a head trained on those (stage 1) is the natural start."""

    def __init__(self, layers: nn.ModuleList, head: AttentionPoolHead):
        super().__init__()
        self.layers, self.head = layers, head
        self.projection_head = head.projection_head  # for Solver.fit's optional adversary

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens.float()
        for attn, ff in self.layers:  # REVE's residual blocks (braindecode TransformerBackbone)
            x = attn(x) + x
            x = ff(x) + x
        return self.head(x)

    @torch.inference_mode()
    def predict(self, tokens):
        self.eval()
        return self.forward(tokens)


def top_layers(state_dict) -> int:
    """Number K of fine-tuned REVE layers in a stage-2 state_dict (0 for a stage-1 head)."""
    return len({k.split(".")[1] for k in state_dict if k.startswith("layers.")})


def head_sizes(state_dict):
    """AttentionPoolHead arguments read back from its weights."""
    if top_layers(state_dict):
        state_dict = {k[len("head."):]: v for k, v in state_dict.items() if k.startswith("head.")}
    n_queries, dim = state_dict["queries"].shape
    return {"n_outputs": state_dict["projection_head.3.weight"].shape[0], "dim": dim, "n_queries": n_queries,
            "hidden": state_dict["projection_head.0.weight"].shape[0]}


class REVEImageModel:
    """Raw EEG windows -> image embeddings: frozen REVE tokens -> head(s) (mean of an ensemble's
    L2-normalised predictions). ``layer``: members fine-tune REVE's top layers from that layer on."""

    def __init__(self, reve: FrozenREVE, heads, layer: int | None = None):
        self.reve, self.heads, self.layer = reve, heads, layer

    @torch.inference_mode()
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        tokens = self.reve.tokens(X, self.layer)
        return torch.stack([h.predict(tokens) for h in self.heads]).mean(0)


class Solver(CompetSolver):
    name = "REVEAttentionHead"

    def load_model(self, meta: dict):
        device = meta["device"]
        reve = FrozenREVE(meta["ch_names"], meta["sfreq"], meta["n_times"], device)
        submission_dir = Path(meta["submission_dir"])
        heads, layers = [], set()
        for path in sorted(submission_dir.glob("weights_*.pt")) or [submission_dir / "weights.pt"]:
            state = torch.load(path, map_location=device, weights_only=True)
            head = AttentionPoolHead(**head_sizes(state)).to(device)
            k = top_layers(state)
            if k:  # stage 2: copies of REVE's top k layers, overwritten by the fine-tuned weights
                blocks = reve.model.transformer.layers
                head = REVETopFinetune(copy.deepcopy(blocks[len(blocks) - k:]), head).to(device)
                layers.add(len(blocks) - k)
            head.load_state_dict(state)
            heads.append(head.eval())
        if len(layers) > 1:
            raise ValueError("ensemble members must fine-tune the same number of REVE layers")
        return REVEImageModel(reve, heads, layers.pop() if layers else None)
