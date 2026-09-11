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

        # ---------------------------------------------------------------------
        # TODO [Layer 1 - Temporal Convolution]:
        # Filter across time within each channel (learns frequency band features).
        # Expected input shape: (B, 1, n_chans, n_times)
        # We apply a 2D conv with kernel_size=(1, 31), stride=1, padding=(0, 15)
        # to preserve temporal length. Out channels: 32.
        # Follow with BatchNorm2d(32).
        # ---------------------------------------------------------------------
        # self.temporal_conv = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=(1, 31), stride=1, padding=(0, 15), bias=False),
        #     nn.BatchNorm2d(32),
        # )
        self.temporal_conv = None

        # ---------------------------------------------------------------------
        # TODO [Layer 2 - Spatial Depthwise Convolution]:
        # Filter across all EEG channels (learns spatial topographies / montages).
        # We apply a 2D conv with kernel_size=(n_chans, 1) to collapse channels from n_chans to 1.
        # Out channels: 64. Follow with BatchNorm2d, ELU activation, AvgPool2d(1, 4), and Dropout.
        # ---------------------------------------------------------------------
        # self.spatial_conv = nn.Sequential(
        #     nn.Conv2d(32, 64, kernel_size=(n_chans, 1), groups=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ELU(),
        #     nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
        #     nn.Dropout(dropout),
        # )
        self.spatial_conv = None

        # ---------------------------------------------------------------------
        # TODO [Layer 3 - Separable Temporal Convolution]:
        # Depthwise temporal conv (kernel=(1, 15), groups=64) followed by
        # Pointwise conv (kernel=(1, 1), out_channels=128), BatchNorm, ELU, AvgPool2d(1, 4), Dropout.
        # ---------------------------------------------------------------------
        # self.sep_conv = ...
        self.sep_conv = None

        # ---------------------------------------------------------------------
        # TODO [Layer 4 - Projection Head]:
        # 1. Determine the flattened feature dimension after convolutions.
        #    Hint: Pass a dummy tensor (1, 1, n_chans, n_times) through the conv blocks.
        # 2. Define an MLP projection head:
        #    Linear(flattened_dim, hidden_dim) -> BatchNorm1d(hidden_dim) -> GELU() -> Dropout -> Linear(hidden_dim, output_dim)
        # ---------------------------------------------------------------------
        self.projection_head = None

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes input through temporal, spatial, and separable conv layers, then flattens.
        Input x: (B, 1, n_chans, n_times)
        Output: (B, flattened_features)
        """
        # TODO: Implement feature extraction forward pass
        raise NotImplementedError("TODO: Implement _forward_features")

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

        # TODO [Step 1]: Extract features using self._forward_features(x)
        # TODO [Step 2]: Project features into visual space using self.projection_head(features)
        # TODO [Step 3]: If self.normalize_output is True, L2-normalize embeddings along dim=-1
        raise NotImplementedError("TODO: Implement forward pass")

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Inference wrapper expected by the Codabench submission solver."""
        self.eval()
        return self.forward(x)
