"""
Codabench / Benchopt Submission Entrypoint for Track 1: EEG-to-Image.

This script implements the CompetSolver contract expected by the platform.
To submit:
1. Train your model and save weights to `weights.pt`.
2. Place `submission.py` and `weights.pt` inside a folder.
3. Zip the folder and upload to Codabench under 'My Submissions'.
"""
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import benchmark_utils  # noqa: F401
    from compet_core.base_solver import CompetSolver
except ImportError:
    # Fallback dummy class for local testing without compet_core installed
    class CompetSolver:
        requirements = []


class EEGToImageEncoder(nn.Module):
    """Self-contained model architecture for inference submission."""
    def __init__(
        self,
        n_chans: int = 32,
        n_times: int = 256,
        output_dim: int = 1536,
        hidden_dim: int = 512,
        dropout: float = 0.0,
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
        )
        self.sep_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 15), padding=(0, 7), groups=64, bias=False),
            nn.Conv2d(64, 128, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_chans, n_times)
            x = self.sep_conv(self.spatial_conv(self.temporal_conv(dummy)))
            flattened_dim = x.numel()

        self.projection_head = nn.Sequential(
            nn.Linear(flattened_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
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
        return self.forward(x)


class Solver(CompetSolver):
    name = "EEGToImageBaseline"
    requirements = CompetSolver.requirements + []

    def load_model(self, meta: dict):
        """
        Build model and load pre-trained weights.
        meta dictionary provides:
          - n_chans: int (number of EEG channels, e.g. 32)
          - n_times: int (number of time points, e.g. 256)
          - n_outputs: int (1536 for DINOv2-giant)
          - device: str or torch.device
          - weights_dir: Path to model weights directory
        """
        device = meta.get("device", "cpu")
        weights_dir = Path(meta.get("weights_dir", Path(__file__).parent))
        weights_path = weights_dir / "weights.pt"

        model = EEGToImageEncoder(
            n_chans=meta["n_chans"],
            n_times=meta["n_times"],
            output_dim=meta.get("n_outputs", 1536),
        )

        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)

        model = model.to(device).eval()
        return model
