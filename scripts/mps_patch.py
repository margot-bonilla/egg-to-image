"""Run neuralset's HuggingFace extractors (e.g. DINOv2 image targets) on the Apple GPU.

neuralset only auto-selects ``cuda`` or ``cpu``. This moves the loaded model to
``mps`` instead and leaves the extractor's ``device`` field untouched, so configs
still validate and cache keys are unchanged (``device`` is excluded from them):
embeddings computed here are reused as-is by CPU/CUDA runs.

Only affects the current process: run neuralbench in-process (``CLUSTER: null``),
see scripts/neuralbench_mps.py.
"""
import os

import torch
from neuralset.extractors import hf

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

_load_model = hf.HuggingFaceMixin.load_model


def _load_model_on_mps(self) -> torch.nn.Module:
    model = _load_model(self)
    if self.device in ("auto", "cpu") and torch.backends.mps.is_available():
        model = model.to("mps")
    return model


def apply() -> None:
    hf.HuggingFaceMixin.load_model = _load_model_on_mps
