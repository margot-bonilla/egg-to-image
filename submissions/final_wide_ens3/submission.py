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
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark_utils.base_solver import CompetSolver


# 2D scalp positions of the standard 10-05 electrodes (MNE's standard_1005 montage, sphere-fitted
# and flattened by an azimuthal-equidistant projection from the vertex, as in EEG topographic maps;
# fixed scale, so every montage shares one coordinate frame). Generated once with MNE: the
# Codabench worker only has torch. Used by SpatialAttention to place channels by name.
ELECTRODE_XY = (
    "Fp1:0.402,0.824;Fpz:0.498,0.828;Fp2:0.594,0.826;AF9:0.277,0.859;AF7:0.310,0.790;AF5:0.356,0.777;"
    "AF3:0.402,0.763;AF1:0.449,0.753;AFz:0.498,0.752;AF2:0.550,0.753;AF4:0.597,0.762;AF6:0.642,0.779;"
    "AF8:0.687,0.792;AF10:0.723,0.860;F9:0.167,0.771;F7:0.230,0.723;F5:0.296,0.701;F3:0.364,0.685;F1:"
    "0.432,0.676;Fz:0.499,0.674;F2:0.569,0.678;F4:0.637,0.689;F6:0.706,0.703;F8:0.769,0.726;F10:0.833"
    ",0.772;FT9:0.095,0.647;FT7:0.171,0.622;FC5:0.257,0.609;FC3:0.341,0.602;FC1:0.421,0.595;FCz:0.499"
    ",0.594;FC2:0.578,0.598;FC4:0.659,0.604;FC6:0.743,0.612;FT8:0.827,0.628;FT10:0.906,0.649;T9:0.069"
    ",0.502;T7:0.153,0.501;C5:0.247,0.508;C3:0.335,0.511;C1:0.420,0.514;Cz:0.499,0.514;C2:0.581,0.515"
    ";C4:0.666,0.513;C6:0.754,0.511;T8:0.848,0.505;T10:0.933,0.499;TP9:0.102,0.361;TP7:0.178,0.388;CP"
    "5:0.264,0.411;CP3:0.346,0.426;CP1:0.424,0.435;CPz:0.499,0.438;CP2:0.578,0.436;CP4:0.656,0.428;CP"
    "6:0.738,0.414;TP8:0.822,0.388;TP10:0.898,0.357;P9:0.176,0.247;P7:0.240,0.297;P5:0.304,0.327;P3:0"
    ".369,0.348;P1:0.435,0.359;Pz:0.499,0.363;P2:0.567,0.361;P4:0.633,0.349;P6:0.695,0.326;P8:0.760,0"
    ".296;P10:0.822,0.244;PO9:0.277,0.172;PO7:0.320,0.237;PO5:0.358,0.260;PO3:0.401,0.275;PO1:0.450,0"
    ".282;POz:0.498,0.286;PO2:0.549,0.282;PO4:0.596,0.273;PO6:0.641,0.260;PO8:0.678,0.235;PO10:0.719,"
    "0.169;O1:0.408,0.207;Oz:0.498,0.206;O2:0.589,0.207;I1:0.386,0.135;Iz:0.497,0.129;I2:0.608,0.133;"
    "AFp9h:0.335,0.844;AFp7h:0.373,0.810;AFp5h:0.411,0.807;AFp3h:0.447,0.800;AFp1h:0.481,0.792;AFp2h:"
    "0.518,0.792;AFp4h:0.553,0.797;AFp6h:0.586,0.807;AFp8h:0.623,0.811;AFp10h:0.660,0.846;AFF9h:0.237"
    ",0.787;AFF7h:0.294,0.749;AFF5h:0.351,0.732;AFF3h:0.409,0.719;AFF1h:0.470,0.715;AFF2h:0.531,0.714"
    ";AFF4h:0.591,0.720;AFF6h:0.648,0.733;AFF8h:0.704,0.751;AFF10h:0.760,0.790;FFT9h:0.156,0.691;FFT7"
    "h:0.235,0.667;FFC5h:0.312,0.650;FFC3h:0.389,0.640;FFC1h:0.464,0.634;FFC2h:0.537,0.635;FFC4h:0.61"
    "1,0.643;FFC6h:0.689,0.654;FFT8h:0.765,0.669;FFT10h:0.843,0.694;FTT9h:0.111,0.566;FTT7h:0.203,0.5"
    "61;FCC5h:0.292,0.557;FCC3h:0.377,0.555;FCC1h:0.459,0.554;FCC2h:0.539,0.555;FCC4h:0.622,0.557;FCC"
    "6h:0.708,0.559;FTT8h:0.797,0.565;FTT10h:0.889,0.567;TTP9h:0.118,0.431;TTP7h:0.206,0.451;CCP5h:0."
    "295,0.464;CCP3h:0.379,0.472;CCP1h:0.460,0.475;CCP2h:0.540,0.475;CCP4h:0.622,0.473;CCP6h:0.706,0."
    "466;TTP8h:0.794,0.453;TTP10h:0.882,0.430;TPP9h:0.170,0.314;TPP7h:0.244,0.353;CPP5h:0.320,0.378;C"
    "PP3h:0.394,0.392;CPP1h:0.466,0.399;CPP2h:0.538,0.400;CPP4h:0.610,0.394;CPP6h:0.682,0.380;TPP8h:0"
    ".756,0.354;TPP10h:0.829,0.312;PPO9h:0.253,0.231;PPO7h:0.302,0.279;PPO5h:0.354,0.304;PPO3h:0.411,"
    "0.317;PPO1h:0.470,0.323;PPO2h:0.530,0.325;PPO4h:0.589,0.319;PPO6h:0.644,0.303;PPO8h:0.696,0.277;"
    "PPO10h:0.744,0.228;POO9h:0.349,0.182;POO7h:0.377,0.227;POO5h:0.407,0.239;POO3h:0.443,0.245;POO1h"
    ":0.479,0.246;POO2h:0.517,0.244;POO4h:0.554,0.242;POO6h:0.590,0.237;POO8h:0.621,0.226;POO10h:0.64"
    "8,0.181;OI1h:0.448,0.165;OI2h:0.547,0.164;Fp1h:0.450,0.830;Fp2h:0.545,0.831;AF9h:0.285,0.820;AF7"
    "h:0.332,0.782;AF5h:0.380,0.771;AF3h:0.424,0.756;AF1h:0.474,0.752;AF2h:0.525,0.751;AF4h:0.574,0.7"
    "56;AF6h:0.618,0.772;AF8h:0.665,0.784;AF10h:0.712,0.822;F9h:0.193,0.744;F7h:0.262,0.711;F5h:0.330"
    ",0.692;F3h:0.398,0.680;F1h:0.466,0.675;F2h:0.535,0.675;F4h:0.603,0.683;F6h:0.671,0.697;F8h:0.738"
    ",0.712;F10h:0.805,0.746;FT9h:0.128,0.631;FT7h:0.214,0.615;FC5h:0.298,0.604;FC3h:0.381,0.598;FC1h"
    ":0.461,0.594;FC2h:0.538,0.595;FC4h:0.618,0.600;FC6h:0.701,0.606;FT8h:0.785,0.620;FT10h:0.872,0.6"
    "34;T9h:0.108,0.496;T7h:0.200,0.505;C5h:0.292,0.510;C3h:0.378,0.513;C1h:0.460,0.514;C2h:0.540,0.5"
    "15;C4h:0.623,0.514;C6h:0.709,0.512;T8h:0.800,0.508;T10h:0.892,0.496;TP9h:0.140,0.370;TP7h:0.221,"
    "0.400;CP5h:0.305,0.420;CP3h:0.386,0.432;CP1h:0.463,0.437;CP2h:0.540,0.438;CP4h:0.617,0.433;CP6h:"
    "0.697,0.422;TP8h:0.780,0.402;TP10h:0.860,0.368;P9h:0.210,0.268;P7h:0.272,0.312;P5h:0.336,0.339;P"
    "3h:0.402,0.354;P1h:0.468,0.362;P2h:0.535,0.363;P4h:0.600,0.356;P6h:0.665,0.340;P8h:0.729,0.313;P"
    "10h:0.789,0.265;PO9h:0.300,0.202;PO7h:0.339,0.249;PO5h:0.378,0.269;PO3h:0.425,0.279;PO1h:0.474,0"
    ".285;PO2h:0.524,0.284;PO4h:0.573,0.278;PO6h:0.620,0.268;PO8h:0.660,0.247;PO10h:0.697,0.201;O1h:0"
    ".453,0.203;O2h:0.543,0.203;I1h:0.442,0.128;I2h:0.552,0.127;AFp9:0.339,0.886;AFp7:0.355,0.811;AFp"
    "5:0.392,0.809;AFp3:0.429,0.804;AFp1:0.462,0.794;AFpz:0.498,0.790;AFp2:0.536,0.792;AFp4:0.570,0.8"
    "02;AFp6:0.605,0.810;AFp8:0.641,0.813;AFp10:0.661,0.886;AFF9:0.219,0.822;AFF7:0.269,0.761;AFF5:0."
    "324,0.742;AFF3:0.380,0.725;AFF1:0.439,0.715;AFFz:0.499,0.713;AFF2:0.561,0.715;AFF4:0.620,0.726;A"
    "FF6:0.676,0.742;AFF8:0.729,0.764;AFF10:0.782,0.823;FFT9:0.125,0.713;FFT7:0.196,0.676;FFC5:0.273,"
    "0.657;FFC3:0.351,0.645;FFC1:0.427,0.637;FFCz:0.499,0.634;FFC2:0.574,0.638;FFC4:0.650,0.648;FFC6:"
    "0.727,0.660;FFT8:0.802,0.680;FFT10:0.875,0.714;FTT9:0.076,0.576;FTT7:0.156,0.563;FCC5:0.248,0.55"
    "9;FCC3:0.335,0.556;FCC1:0.419,0.555;FCCz:0.499,0.554;FCC2:0.580,0.556;FCC4:0.665,0.559;FCC6:0.75"
    "1,0.562;FTT8:0.843,0.568;FTT10:0.927,0.577;TTP9:0.079,0.428;TTP7:0.160,0.442;CCP5:0.251,0.458;CC"
    "P3:0.338,0.468;CCP1:0.420,0.474;CCPz:0.499,0.476;CCP2:0.581,0.474;CCP4:0.663,0.470;CCP6:0.749,0."
    "461;TTP8:0.840,0.444;TTP10:0.922,0.424;TPP9:0.134,0.301;TPP7:0.206,0.339;CPP5:0.283,0.367;CPP3:0"
    ".357,0.387;CPP1:0.430,0.397;CPPz:0.499,0.400;CPP2:0.573,0.398;CPP4:0.645,0.388;CPP6:0.719,0.369;"
    "TPP8:0.794,0.338;TPP10:0.865,0.297;PPO9:0.224,0.204;PPO7:0.278,0.263;PPO5:0.329,0.291;PPO3:0.382"
    ",0.312;PPO1:0.440,0.322;PPOz:0.499,0.325;PPO2:0.560,0.323;PPO4:0.617,0.312;PPO6:0.672,0.293;PPO8"
    ":0.720,0.261;PPO10:0.773,0.201;POO9:0.331,0.149;POO7:0.363,0.219;POO5:0.392,0.233;POO3:0.425,0.2"
    "42;POO1:0.461,0.248;POOz:0.498,0.246;POO2:0.535,0.244;POO4:0.572,0.241;POO6:0.607,0.234;POO8:0.6"
    "34,0.217;POO10:0.664,0.146;OI1:0.399,0.170;OIz:0.498,0.167;OI2:0.598,0.169;T3:0.153,0.501;T5:0.2"
    "40,0.297;T4:0.848,0.505;T6:0.760,0.296;M1:0.065,0.356;M2:0.936,0.352;A1:0.039,0.454;A2:0.963,0.4"
    "52"
)


def channel_positions(ch_names):
    """(C, 2) scalp positions in [0, 1]^2 and a (C,) mask of the names found in ELECTRODE_XY."""
    table = {}
    for entry in "".join(ELECTRODE_XY).split(";"):
        name, xy = entry.split(":")
        table[name.lower()] = [float(v) for v in xy.split(",")]
    known = torch.tensor([n.lower() in table for n in ch_names])
    positions = torch.tensor([table.get(n.lower(), [0.5, 0.5]) for n in ch_names])
    return positions, known


class SpatialAttention(nn.Module):
    """Montage-independent spatial layer (Defossez et al., 2023, "Decoding speech perception from
    non-invasive brain recordings", the "spatial attention" of their brain module).

    Each of ``n_out`` virtual channels is a softmax-weighted average of the input channels, with
    the weights given by a learned smooth function of each electrode's 2D scalp position (a Fourier
    series with ``n_freqs`` x ``n_freqs`` frequencies). The parameters don't depend on the number or
    order of channels, so one model reads 63-channel and 32-channel recordings alike.
    ``dropout_radius``: during training, drop every electrode within that distance of a random
    scalp point (robustness to missing or different electrodes).
    """

    def __init__(self, n_out: int, n_freqs: int = 16, dropout_radius: float = 0.0, margin: float = 0.2):
        super().__init__()
        self.n_freqs, self.dropout_radius, self.margin = n_freqs, dropout_radius, margin
        self.heads = nn.Parameter(torch.randn(n_out, 2 * n_freqs * n_freqs) / math.sqrt(2 * n_freqs * n_freqs))

    def fourier(self, positions):
        freqs = torch.arange(self.n_freqs, device=positions.device, dtype=positions.dtype)
        x, y = (positions[:, :1] + self.margin) / (1 + 2 * self.margin), (positions[:, 1:] + self.margin) / (1 + 2 * self.margin)
        phase = 2 * math.pi * (freqs[None, :, None] * x[:, :, None] + freqs[None, None, :] * y[:, :, None])
        return torch.cat([torch.cos(phase).flatten(1), torch.sin(phase).flatten(1)], dim=1)  # (C, 2 F^2)

    def forward(self, x, positions, known):
        scores = self.heads @ self.fourier(positions).T  # (n_out, C)
        drop = ~known
        if self.training and self.dropout_radius > 0:
            center = torch.rand(2, device=x.device)
            dropped = drop | ((positions - center).norm(dim=1) < self.dropout_radius)
            drop = drop if dropped.all() else dropped
        scores = scores.masked_fill(drop[None, :], float("-inf"))
        return torch.einsum("bct,oc->bot", x, torch.softmax(scores, dim=1))


class SubjectAdversary(nn.Module):
    """Training-only participant classifier on the embedding, for participant-invariant training
    by domain confusion (Tzeng et al., 2015): the classifier learns to recognise the participant
    from (detached) embeddings, while the encoder is trained to make its output uniform over the
    participants. (A gradient-reversal layer, as in DANN, let the encoder make the classifier
    confidently *wrong* instead, which keeps the participant information.) Not saved."""

    def __init__(self, dim: int, n_subjects: int = 128, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, n_subjects))

    def forward(self, emb):
        return self.net(emb)

    def losses(self, emb, subjects):
        """(classifier loss, confusion loss, number of correct guesses) for one batch."""
        logits = self.net(emb.detach())  # trains the classifier only
        frozen = {k: v.detach() for k, v in self.named_parameters()}
        confused = torch.func.functional_call(self, frozen, (emb,))  # trains the encoder only
        present = torch.unique(subjects)  # uniform over the participants actually in the batch
        confusion = -F.log_softmax(confused[:, present], dim=1).mean()
        return F.cross_entropy(logits, subjects), confusion, int((logits.argmax(1) == subjects).sum())


class ResidualBlock(nn.Module):
    """x + Dropout(Linear(GELU(x))), as in ATM's projection head."""

    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.linear(F.gelu(x)))


class EEGToImageEncoder(nn.Module):
    """Self-contained model architecture. With the default sizes the state_dict keys match
    src/models/eeg_encoder.py; ``encoder_sizes`` reads the sizes back from a state_dict."""

    def __init__(
        self,
        n_chans: int = 32,
        n_times: int = 256,
        output_dim: int = 1536,
        hidden_dim: int = 512,
        dropout: float = 0.25,
        temporal_filters: int = 32,
        spatial_filters: int = 64,
        sep_filters: int = 128,
        head_blocks: int = 0,
        virtual_channels: int = 0,
        fourier_freqs: int = 16,
        spatial_dropout: float = 0.0,
        ch_names=None,
        window_norm: bool = False,
    ):
        super().__init__()
        # Optional per-window normalisation: each channel standardised over time inside the window,
        # removing per-person/per-session amplitude and offset differences. A marker buffer is saved
        # in the weights so that inference (encoder_sizes) switches it back on.
        self.window_norm = window_norm
        if window_norm:
            self.register_buffer("window_norm_on", torch.ones(1))
        # Optional montage-independent input layer: the channels (placed by name) are first mapped
        # to `virtual_channels` virtual ones, so the rest of the network never sees n_chans.
        self.spatial_attention = None
        if virtual_channels:
            if ch_names is None:
                raise ValueError("virtual_channels needs meta['ch_names'] to place the electrodes")
            self.spatial_attention = SpatialAttention(virtual_channels, fourier_freqs, spatial_dropout)
            positions, known = channel_positions(ch_names)
            self.register_buffer("positions", positions, persistent=False)  # per montage, not saved
            self.register_buffer("known", known, persistent=False)
            n_chans = virtual_channels
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, temporal_filters, kernel_size=(1, 31), stride=1, padding=(0, 15), bias=False),
            nn.BatchNorm2d(temporal_filters),
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(temporal_filters, spatial_filters, kernel_size=(n_chans, 1), groups=1, bias=False),
            nn.BatchNorm2d(spatial_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(dropout),
        )
        self.sep_conv = nn.Sequential(
            nn.Conv2d(spatial_filters, spatial_filters, kernel_size=(1, 15), padding=(0, 7),
                      groups=spatial_filters, bias=False),
            nn.Conv2d(spatial_filters, sep_filters, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(sep_filters),
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
        # Optional residual blocks on the output embedding (no parameters when head_blocks=0,
        # so earlier weights files still load).
        self.head_blocks = nn.Sequential(*[ResidualBlock(output_dim, dropout) for _ in range(head_blocks)])
        self.head_norm = nn.LayerNorm(output_dim) if head_blocks else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.window_norm:
            x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-6)
        if self.spatial_attention is not None:
            x = self.spatial_attention(x, self.positions, self.known)  # (B, C, T) -> (B, virtual, T)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.sep_conv(x)
        x = torch.flatten(x, start_dim=1)
        emb = self.head_norm(self.head_blocks(self.projection_head(x)))
        return F.normalize(emb, p=2, dim=-1)

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(x.float())


def encoder_sizes(state_dict):
    """EEGToImageEncoder size arguments read back from a saved state_dict, so the shipped
    submission.py rebuilds whatever size was trained (its Solver settings may be the defaults)."""
    return {
        "temporal_filters": state_dict["temporal_conv.0.weight"].shape[0],
        "spatial_filters": state_dict["spatial_conv.0.weight"].shape[0],
        "sep_filters": state_dict["sep_conv.1.weight"].shape[0],
        "hidden_dim": state_dict["projection_head.0.weight"].shape[0],
        "head_blocks": sum(1 for k in state_dict if k.startswith("head_blocks.") and k.endswith(".linear.weight")),
        **({"window_norm": True} if "window_norm_on" in state_dict else {}),
        **({"virtual_channels": state_dict["spatial_attention.heads"].shape[0],
            "fourier_freqs": math.isqrt(state_dict["spatial_attention.heads"].shape[1] // 2)}
           if "spatial_attention.heads" in state_dict else {}),
    }


def clip_loss(pred, target, logit_scale, dedup=False, symmetric=False):
    """InfoNCE with the options used in CLIP-style brain decoding.

    pred: (B, D) L2-normalized predictions; logit_scale = 1 / temperature.
    dedup: windows of the same image share a target, so another copy of the right image
        in the batch would count as a wrong answer; mask those false negatives out.
    symmetric: also classify each target among the batch predictions (image -> EEG).
    """
    target = F.normalize(target, dim=-1)
    logits = logit_scale * pred @ target.T
    if dedup:
        same_image = (target @ target.T) > 0.9999
        same_image.fill_diagonal_(False)
        logits = logits.masked_fill(same_image, float("-inf"))
    labels = torch.arange(len(pred), device=pred.device)
    loss = F.cross_entropy(logits, labels)
    if symmetric:
        loss = (loss + F.cross_entropy(logits.T, labels)) / 2
    return loss


def augment(x, max_shift=0, channel_dropout=0.0):
    """Training-time augmentation of (B, C, T) windows: random circular time shift, channel dropout."""
    if max_shift:
        shifts = torch.randint(-max_shift, max_shift + 1, (len(x),))
        x = torch.stack([torch.roll(xi, int(s), dims=-1) for xi, s in zip(x, shifts)])
    if channel_dropout:
        x = x * (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > channel_dropout)
    return x


def _get_rng_state(device):
    """CPU and accelerator RNG states (shuffling uses the CPU one, dropout the device's)."""
    state = {"cpu": torch.get_rng_state()}
    if device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    elif device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    return state


def _set_rng_state(state, device):
    torch.set_rng_state(state["cpu"].cpu())
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"].cpu())
    elif "cuda" in state:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


class EmbeddingEnsemble:
    """Mean of the members' L2-normalized embeddings (one member per shipped weights_*.pt)."""

    def __init__(self, models):
        self.models = models

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([m.predict(x) for m in self.models]).mean(0)


class Solver(CompetSolver):
    name = "EEGToImageBaseline"

    # Local-training hyperparameters (only used by ``fit``). The defaults reproduce the
    # 0.54 run; scripts/train_solver.py --set name=value overrides them per experiment.
    n_epochs = 20
    lr = 1e-3  # scale it with train_solver.py --batch-size (default 256)
    lr_schedule = "constant"  # or "cosine": per-step decay to 0 over n_epochs, after the warm-up
    warmup_epochs = 0.0  # linear lr warm-up (fractions allowed), with either schedule
    weight_decay = 1e-4
    dropout = 0.25  # training only: inference runs in eval mode
    # Encoder size for training (#27); inference reads the size from the weights instead.
    temporal_filters = 32
    spatial_filters = 64
    sep_filters = 128
    hidden_dim = 512
    head_blocks = 0  # residual blocks + LayerNorm on the output embedding, as in ATM
    # Montage-independent spatial attention (Defossez et al., 2023): 0 = off; else the number of
    # virtual channels, placed by electrode name, so one model reads any headset (63 or 32 channels).
    virtual_channels = 0
    fourier_freqs = 16
    spatial_dropout = 0.0  # training: drop electrodes within this radius of a random scalp point
    # Towards unseen participants: per-window channel standardisation (also applied at inference),
    # and participant-adversarial training (domain confusion; weight of its loss, 0 = off; training only).
    window_norm = False
    adversarial_weight = 0.0
    seed = 0
    temperature = 0.07
    learn_temperature = False
    min_temperature = 0.01  # floor of the learned temperature (CLIP's); R6 sat on it from epoch ~15
    dedup = False
    symmetric = False
    max_shift = 0  # samples (120 Hz)
    channel_dropout = 0.0
    patience = 5  # epochs without held-out improvement, when an epoch_callback is set
    smooth = 1  # select/stop on the mean held-out score of the last `smooth` epochs (noisy splits: 3)
    # Optional hook set by scripts/train_solver.py --holdout: called after each epoch as
    # epoch_callback(model, epoch, train_loss, temperature) -> held-out score (higher is
    # better); fit then keeps the best epoch and stops early. Never set on Codabench.
    epoch_callback = None
    # Optional path set by scripts/train_solver.py: fit saves its full state there after every
    # epoch and, if the file exists when fit starts, resumes from it (after a crash or restart).
    checkpoint = None

    def load_model(self, meta: dict):
        """
        meta (built by the platform) provides:
          - n_chans, n_times, sfreq, ch_names, chs_info: input description
          - n_outputs: embedding dimension D (1536 for DINOv2-giant)
          - device: where the input batches live
          - submission_dir: read-only folder holding the files extracted from the ZIP

        Ships either one ``weights.pt`` or several ``weights_*.pt`` (an ensemble).
        """
        device = meta["device"]

        def build(**sizes):
            return EEGToImageEncoder(
                n_chans=meta["n_chans"],
                n_times=meta["n_times"],
                output_dim=meta["n_outputs"],
                dropout=self.dropout,
                spatial_dropout=self.spatial_dropout,
                ch_names=meta.get("ch_names"),
                **sizes,
            ).to(device)

        # A local training run (train_loader set) starts from scratch; every
        # inference-only run, including Codabench's, must load the shipped weights.
        if getattr(self, "train_loader", None) is not None:
            torch.manual_seed(self.seed)  # reproducible weight init
            return build(temporal_filters=self.temporal_filters, spatial_filters=self.spatial_filters,
                         sep_filters=self.sep_filters, hidden_dim=self.hidden_dim,
                         head_blocks=self.head_blocks, virtual_channels=self.virtual_channels,
                         fourier_freqs=self.fourier_freqs, window_norm=self.window_norm).eval()

        submission_dir = Path(meta["submission_dir"])
        paths = sorted(submission_dir.glob("weights_*.pt")) or [submission_dir / "weights.pt"]
        models = []
        for path in paths:
            state_dict = torch.load(path, map_location=device, weights_only=True)
            model = build(**encoder_sizes(state_dict))
            model.load_state_dict(state_dict)
            models.append(model.eval())
        return models[0] if len(models) == 1 else EmbeddingEnsemble(models)

    def fit(self, model, train_loader):
        torch.manual_seed(self.seed)  # reproducible shuffling and dropout
        device = next(model.parameters()).device
        # Training-only parameter (not saved: inference doesn't use the temperature).
        log_scale = torch.nn.Parameter(torch.tensor(math.log(1 / self.temperature), device=device),
                                       requires_grad=self.learn_temperature)
        max_log_scale = math.log(1 / self.min_temperature)
        params = list(model.parameters()) + ([log_scale] if self.learn_temperature else [])
        adversary = None
        if self.adversarial_weight:  # training-only participant classifier (not saved)
            adversary = SubjectAdversary(model.projection_head[-1].out_features).to(device)
            params += list(adversary.parameters())
        total_steps = max(self.n_epochs * len(train_loader), 1)
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self._lr_factor(len(train_loader)))
        best_score, best_state, bad = -math.inf, None, 0
        self.history = []
        start, done = 0, False
        if self.checkpoint is not None and Path(self.checkpoint).exists():
            state = torch.load(self.checkpoint, map_location=device, weights_only=True)
            model.load_state_dict(state["model"])
            if adversary is not None:
                adversary.load_state_dict(state["adversary"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            with torch.no_grad():
                log_scale.copy_(state["log_scale"])
            best_score, best_state, bad = state["best_score"], state["best_state"], state["bad"]
            self.history, start, done = state["history"], state["epoch"], state["done"]
            _set_rng_state(state["rng"], device)
            self.resumed_from = start
            print(f"Resumed from {self.checkpoint} after epoch {start}", flush=True)
        for epoch in range(start, 0 if done else self.n_epochs):
            model.train()
            total, n_batches, adv_hits, adv_n = 0.0, 0, 0, 0
            for X, y, info in train_loader:
                if len(X) < 2:  # BatchNorm1d needs more than one sample per batch
                    continue
                X = augment(X.float(), self.max_shift, self.channel_dropout)
                optimizer.zero_grad()
                scale = log_scale.clamp(max=max_log_scale).exp()  # temperature >= min_temperature
                emb = model(X)
                loss = clip_loss(emb, y.float(), scale, self.dedup, self.symmetric)
                if adversary is not None:
                    # the confusion weight ramps from 0 to adversarial_weight (DANN's schedule)
                    progress = (epoch * len(train_loader) + n_batches) / total_steps
                    ramp = 2 / (1 + math.exp(-10 * progress)) - 1
                    subjects = info["subject_id"].to(device).long()
                    classifier_loss, confusion, hits = adversary.losses(emb, subjects)
                    loss = loss + classifier_loss + self.adversarial_weight * ramp * confusion
                    adv_hits, adv_n = adv_hits + hits, adv_n + len(subjects)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total, n_batches = total + loss.item(), n_batches + 1
            model.eval()
            temperature = float(1 / log_scale.clamp(max=max_log_scale).exp())
            record = {"epoch": epoch + 1, "train_loss": total / max(n_batches, 1), "temperature": temperature,
                      "lr": optimizer.param_groups[0]["lr"]}
            if adversary is not None:  # how well the participant can still be recognised (lower = more invariant)
                record["subject_acc"] = adv_hits / max(adv_n, 1)
            if self.epoch_callback is not None:
                record["heldout"] = float(self.epoch_callback(model, epoch + 1, record["train_loss"], temperature))
                recent = [r["heldout"] for r in self.history[-(self.smooth - 1):]] if self.smooth > 1 else []
                score = sum(recent + [record["heldout"]]) / (len(recent) + 1)
                if self.smooth > 1 and len(recent) + 1 == self.smooth:
                    record["heldout_smooth"] = score
                if len(recent) + 1 < self.smooth:
                    pass  # not a full window yet: an early spike would dominate the mean
                elif score > best_score:
                    best_score, bad = score, 0
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                else:
                    bad += 1
            self.history.append(record)
            print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()),
                  flush=True)
            stop = self.epoch_callback is not None and bad >= self.patience
            if self.checkpoint is not None:
                tmp = Path(f"{self.checkpoint}.tmp")
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(), "log_scale": log_scale.detach(),
                            "best_score": best_score, "best_state": best_state, "bad": bad,
                            "history": self.history, "epoch": epoch + 1, "done": stop,
                            "rng": _get_rng_state(device),
                            **({"adversary": adversary.state_dict()} if adversary is not None else {})}, tmp)
                tmp.replace(self.checkpoint)  # atomic: a crash mid-save keeps the previous checkpoint
            if stop:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

    def _lr_factor(self, steps_per_epoch):
        """lr multiplier per optimizer step: linear warm-up, then constant or cosine decay to 0."""
        warmup = int(self.warmup_epochs * steps_per_epoch)
        total = self.n_epochs * steps_per_epoch
        if self.lr_schedule not in ("constant", "cosine"):
            raise ValueError(f"lr_schedule must be 'constant' or 'cosine', not {self.lr_schedule!r}")

        def factor(step):
            if step < warmup:
                return (step + 1) / warmup
            if self.lr_schedule == "constant":
                return 1.0
            progress = min((step - warmup) / max(total - warmup, 1), 1.0)
            return 0.5 * (1 + math.cos(math.pi * progress))

        return factor

    def save_model(self, model, path):
        torch.save(model.state_dict(), Path(path) / "weights.pt")
