"""
Codabench submission for Track 1: a weighted ensemble of model families in one ZIP.

Files at the ZIP root (written by scripts/package_ensemble.py):
  submission.py      this file
  encoder_model.py   our encoder's solver code (submission/submission.py)
  reve_model.py      the REVE solver code (submission/reve_submission.py)
  weights_*.pt       encoder members (EEGToImageEncoder state_dicts)
  reve_*.pt          REVE members (attention head, optionally with fine-tuned top layers)
  channels.json      the montage the members were trained on
  ensemble.json      {"groups": [{"kind": "encoder"|"reve", "files": [...], "weight": w}, ...]}

Prediction = sum over groups of weight x (mean of the group's L2-normalised member predictions).
The input channels are picked once, by name, to the training montage: encoder members get zeros for a
missing channel; REVE members just use the channels present (REVE places electrodes by name).
"""
import importlib.util
import json
import warnings
from pathlib import Path

import torch

from benchmark_utils.base_solver import CompetSolver


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WeightedEnsemble:
    def __init__(self, groups, encoder_index, encoder_present, reve_index, device):
        self.groups = groups  # [(weight, [models]), ...] with each model's input selection
        self.encoder_index = torch.tensor(encoder_index, device=device)
        self.encoder_mask = torch.tensor(encoder_present, device=device, dtype=torch.float32)[None, :, None]
        self.reve_index = torch.tensor(reve_index, device=device)

    @torch.inference_mode()
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        X = X.float()
        inputs = {"encoder": X[:, self.encoder_index] * self.encoder_mask, "reve": X[:, self.reve_index]}
        total = 0
        for weight, kind, models in self.groups:
            preds = torch.stack([torch.nn.functional.normalize(m.predict(inputs[kind]).float(), dim=-1)
                                 for m in models])
            total = total + weight * preds.mean(0)
        return total


class Solver(CompetSolver):
    name = "WeightedEnsemble"

    def load_model(self, meta: dict):
        device = meta["device"]
        d = Path(meta["submission_dir"])
        config = json.loads((d / "ensemble.json").read_text())
        trained = json.loads((d / "channels.json").read_text())["ch_names"]
        available = meta.get("ch_names")
        if available:
            lookup = {str(n).lower(): i for i, n in enumerate(available)}
            present = [str(n).lower() in lookup for n in trained]
            encoder_index = [lookup.get(str(n).lower(), 0) for n in trained]
            reve_names = [n for n, ok in zip(trained, present) if ok]
            reve_index = [lookup[str(n).lower()] for n in reve_names]
            if not all(present):
                warnings.warn(f"input lacks training channels {[n for n, ok in zip(trained, present) if not ok]}")
        elif meta["n_chans"] >= len(trained):  # unnamed input (simulated data): first channels
            present, encoder_index = [True] * len(trained), list(range(len(trained)))
            reve_names, reve_index = trained, list(range(len(trained)))
        else:
            raise ValueError(f"model needs {len(trained)} channels, input has {meta['n_chans']} unnamed ones")

        groups = []
        for group in config["groups"]:
            if group["kind"] == "encoder":
                module = _load_module("encoder_model", d / "encoder_model.py")
                models = []
                for f in group["files"]:
                    state = torch.load(d / f, map_location=device, weights_only=True)
                    model = module.EEGToImageEncoder(n_chans=len(trained), n_times=meta["n_times"],
                                                     output_dim=meta["n_outputs"], ch_names=trained,
                                                     **module.encoder_sizes(state)).to(device)
                    model.load_state_dict(state)
                    models.append(model.eval())
            elif group["kind"] == "reve":
                module = _load_module("reve_model", d / "reve_model.py")
                reve = module.FrozenREVE(reve_names, meta["sfreq"], meta["n_times"], device)
                heads, layers = [], set()
                for f in group["files"]:
                    state = torch.load(d / f, map_location=device, weights_only=True)
                    head = module.AttentionPoolHead(**module.head_sizes(state)).to(device)
                    k = module.top_layers(state)
                    if k:
                        import copy
                        blocks = reve.model.transformer.layers
                        head = module.REVETopFinetune(copy.deepcopy(blocks[len(blocks) - k:]), head).to(device)
                        layers.add(len(blocks) - k)
                    head.load_state_dict(state)
                    heads.append(head.eval())
                if len(layers) > 1:
                    raise ValueError("REVE members must fine-tune the same number of layers")
                models = [module.REVEImageModel(reve, heads, layers.pop() if layers else None)]
            else:
                raise ValueError(f"unknown member kind {group['kind']!r}")
            groups.append((float(group["weight"]), group["kind"], models))
        return WeightedEnsemble(groups, encoder_index, present, reve_index, device)
