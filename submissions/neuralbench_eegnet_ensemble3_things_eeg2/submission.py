"""Ensemble of EEGNet checkpoints trained by the NeuralBench image start kit.

Extends the official ``submission_templates/neuralbench_eegnet_submission.py``:
loads every ``weights_*.pt`` shipped next to this file (one NeuralBench EEGNet
per training seed) and predicts the mean of their L2-normalized embeddings.
The checkpoint loader is the template's, unchanged.
"""

import torch
from torch.nn import functional as F
from braindecode.models import EEGNet

from benchmark_utils.base_solver import CompetSolver


def _load_neuralbench_weights(module, path, device):
    """Load only the brain model from a NeuralBench or plain checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)
    expected = module.state_dict()

    # NeuralBench's Lightning module stores the network below ``model.`` and
    # may also store learned loss parameters such as ``loss.temperature``.
    if any(key.startswith("model.") for key in state):
        state = {
            key.removeprefix("model."): value
            for key, value in state.items()
            if key.startswith("model.")
        }
    state = {key: value for key, value in state.items() if key in expected}

    missing = sorted(set(expected) - set(state))
    if missing:
        raise RuntimeError(
            "The checkpoint does not match the NeuralBench EEGNet image "
            f"architecture. Missing keys: {missing}"
        )
    module.load_state_dict(state, strict=True)


class EEGNetEnsemble:
    def __init__(self, nets, device):
        self.nets = nets
        self.device = device

    @torch.inference_mode()
    def predict(self, X):
        X = X.to(self.device, dtype=torch.float32)
        preds = [F.normalize(net.eval()(X), dim=-1) for net in self.nets]
        return torch.stack(preds).mean(0)


class Solver(CompetSolver):

    name = "NeuralBench-EEGNet-Image-Ensemble"
    requirements = ["pip::braindecode"]

    def load_model(self, meta):
        paths = sorted(meta["submission_dir"].glob("weights_*.pt"))
        if not paths:
            raise FileNotFoundError(f"no weights_*.pt in {meta['submission_dir']}")
        nets = []
        for path in paths:
            net = EEGNet(
                n_chans=meta["n_chans"],
                n_times=meta["n_times"],
                n_outputs=meta["n_outputs"],
            ).to(meta["device"])
            _load_neuralbench_weights(net, path, meta["device"])
            nets.append(net.eval())
        return EEGNetEnsemble(nets, meta["device"])
