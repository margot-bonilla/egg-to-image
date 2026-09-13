"""
EEG Encoder architecture for Track 1: EEG-to-Image.
Maps multichannel EEG input (B, C, T) to a 1536-dimensional DINOv2-giant visual feature vector.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGToImageEncoder(nn.Module):
    """
    Spatiotemporal EEG Encoder based on EEGNet principles for cross-modal visual decoding.

    Neuroscience & Signal Processing Intuition:
    -------------------------------------------
    An EEG recording is a 2D matrix of shape (C, T) representing C electrodes across T timepoints.
    Unlike natural images where both dimensions (height, width) represent spatial pixel grids:
      1. Dimension T represents time (rhythmic oscillations in Delta, Theta, Alpha, Beta, Gamma bands).
      2. Dimension C represents discrete spatial electrode positions on the scalp.

    Therefore, processing EEG requires decoupled spatiotemporal filtering:
      - Stage 1 (Temporal Filtering): Learnable 1D filter across time that extracts frequency bands.
      - Stage 2 (Spatial Filtering): Depthwise filter across all C channels that acts as a spatial montage.
      - Stage 3 (Separable Temporal Filtering): Aggregates higher-level temporal patterns across time.
      - Stage 4 (Projection Head): Linear mapping with normalization into the 1536-d DINOv2-giant space.
    """
    def __init__(
        self,
        n_chans: int = 32,
        n_times: int = 256,
        output_dim: int = 1536,
        hidden_dim: int = 512,
        dropout: float = 0.25,
        normalize_output: bool = True,
    ):
        """
        Args:
            n_chans: Number of EEG electrode channels (e.g. 32 for Alljoined-1.6M / Emotiv).
            n_times: Number of time samples per epoch (e.g. 256 for 1 second at 256 Hz).
            output_dim: Target visual embedding dimension (1536 for DINOv2-giant).
            hidden_dim: Latent representation size in the projection head.
            dropout: Dropout probability for regularization.
            normalize_output: If True, L2-normalizes the final 1536-d output vector.
        """
        super().__init__()
        self.n_chans = n_chans
        self.n_times = n_times
        self.output_dim = output_dim
        self.normalize_output = normalize_output

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
            nn.AvgPool2d(kernel_size=(1,4), stride=(1,4)),
            nn.Dropout(dropout)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_chans, n_times)
            feat = self._forward_features(dummy)
            flattened_dim = feat.shape[1]

        self.projection_head = nn.Sequential(
            nn.Linear(flattened_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )


    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes input through temporal, spatial, and separable conv layers, then flattens.
        Input x: (B, 1, n_chans, n_times)
        Output: (B, flattened_features)
        """
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.sep_conv(x)

        return torch.flatten(x, start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, T) or (B, 1, C, T)
        Returns:
            Tensor of shape (B, 1536)
        """
        # Ensure 4D tensor (B, 1, C, T)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        features = self._forward_features(x)
        emb = self.projection_head(features)

        return F.normalize(emb, p=2, dim=-1) if self.normalize_output else emb

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Inference wrapper expected by the Codabench submission solver."""
        self.eval()
        return self.forward(x)
