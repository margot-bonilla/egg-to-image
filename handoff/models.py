"""Reuse the published encoder/REVE modules; pin public REVE weights explicitly."""
import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]


def load_sources(competition_root=None):
    track = Path(competition_root or ROOT / 'external/2026-competition') / 'tracks/image_decoding'
    if not (track / 'benchmark_utils/base_solver.py').is_file():
        raise ValueError('Competition dependency missing: follow handoff/README.md setup first')
    sys.path.insert(0, str(track))
    result = []
    for name in ('submission', 'reve_submission'):
        spec = importlib.util.spec_from_file_location('handoff_' + name, ROOT / 'submission' / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result.append(module)
    return tuple(result)


def check_competition(root, revision):
    root = Path(root or ROOT / 'external/2026-competition').resolve()
    result = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                            capture_output=True, text=True, check=True)
    if result.stdout.strip() != revision:
        raise ValueError('Competition checkout is not at the frozen dependency commit')
    dirty = subprocess.run(['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=no'],
                           capture_output=True, text=True, check=True)
    if dirty.stdout.strip():
        raise ValueError('Competition dependency has modified tracked files')
    return root


def pinned_reve(module, meta, device, revision):
    from braindecode.models import REVE
    # Same input processing/montage as FrozenREVE; only loading gains an explicit revision.
    reve = module.FrozenREVE.__new__(module.FrozenREVE)
    reve.device = torch.device(device)
    reve.n_times = max(round(meta['n_times'] * reve.SFREQ / meta['sfreq']), reve.PATCH)
    reve.model = REVE.from_pretrained('brain-bzh/reve-base', revision=revision,
                                     n_outputs=1, n_chans=meta['n_chans'], n_times=reve.n_times,
                                     sfreq=reve.SFREQ, chs_info=None).to(device)
    names = [str(n).removeprefix('EEG ').split('-', 1)[0] for n in meta['ch_names']]
    if any(n not in reve.model._position_bank.mapping for n in names):
        raise ValueError('REVE does not recognise the frozen channel montage')
    reve.model.default_pos = reve.model.get_positions(names)
    reve.model.eval().requires_grad_(False)
    if len(reve.model.transformer.layers) != 22:
        raise ValueError('Pinned REVE does not have the expected 22 transformer blocks')
    return reve


def make_encoder(module, meta):
    torch.manual_seed(1)
    return module.EEGToImageEncoder(n_chans=meta['n_chans'], n_times=120, output_dim=1536,
                                   hidden_dim=512, dropout=.25, temporal_filters=32,
                                   spatial_filters=64, sep_filters=128, window_norm=True)


def make_head(module):
    torch.manual_seed(1)
    return module.AttentionPoolHead(n_outputs=1536, dim=512, n_queries=8,
                                    n_heads=8, hidden=1024, dropout=.25)


def make_top(module, reve, head):
    return module.REVETopFinetune(copy.deepcopy(reve.model.transformer.layers[-4:]), head).requires_grad_(True)


def verify_token_sample(reve, raw, final, prefix, device):
    from src.data.window_cache import CachedWindows
    # Source rows only. This is a sample compatibility check, not a full token-content hash.
    indices = list(range(min(16, raw.parts['train']['n'])))
    windows = CachedWindows([raw.root / 'train'])
    X, _, _ = windows.__getitems__(indices)
    for path, layer in [(final, None), (prefix, 18)]:
        cached, _, _ = CachedWindows([Path(path) / 'train']).__getitems__(indices)
        expected = reve.tokens(X.to(device), layer).cpu()
        if not torch.allclose(cached, expected, atol=.01, rtol=.002):
            raise ValueError('Frozen token cache does not reproduce the pinned public REVE on source samples')
