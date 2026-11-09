"""Prepare frozen, pinned-REVE token dependencies before baseline training.

Uses existing official raw-window/DINO caches. It never computes DINO targets,
fits REVE, scores data, or overwrites an existing token directory.
"""
import argparse
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from . import models, protocol


def validate_destination(destination, raw_root):
    original = Path(destination).absolute()
    if original.is_symlink():
        raise ValueError('Token output must not be a symlink, including a dangling one')
    destination = original.resolve()
    raw_root = Path(raw_root).resolve()
    if destination.exists():
        raise ValueError(f'{destination} exists; validate/reuse it, or choose a new cache root. Do not overwrite it.')
    if destination == raw_root or destination.is_relative_to(raw_root) or raw_root.is_relative_to(destination):
        raise ValueError('Token output must be separate from raw inputs')
    for parent in destination.parents:
        if parent != models.ROOT and (parent / '.git').exists():
            raise ValueError('Token output is inside another checkout, including an active training project')
    return destination


def extract(reve, raw, destination, layer, device, batch_size=256):
    from src.data.window_cache import CachedWindows
    destination = validate_destination(destination, raw.root)
    destination.mkdir(parents=True)
    meta = {**raw.meta, 'dataset': 'xu2025alljoined', 'layer': layer,
            'reve_revision': '33841986f6cc1438d9dcdbca5a2137d60f6007d9'}
    (destination / 'meta.json').write_text(json.dumps(meta, indent=2))
    for split in ('train', 'val'):
        out = destination / split
        out.mkdir()
        dataset = CachedWindows([raw.root / split])
        partial = out / 'X.npy.partial'
        array = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float16,
                                          shape=(len(dataset), raw.meta['n_chans'], 512))
        with torch.inference_mode():
            for lo in range(0, len(dataset), batch_size):
                x, _, _ = dataset.__getitems__(list(range(lo, min(lo + batch_size, len(dataset)))))
                token = reve.tokens(x.to(device), None if layer == 22 else layer).float()
                if token.shape != (len(x), raw.meta['n_chans'], 512) or not torch.isfinite(token).all():
                    raise ValueError('Pinned public model returned invalid tokens')
                array[lo:lo + len(x)] = token.to(torch.float16).cpu().numpy()
                print(f'layer {layer} {split}: {lo + len(x)}/{len(dataset)}', flush=True)
        array.flush()
        del array
        partial.replace(out / 'X.npy')
        for name in ('fields.json', 'image.npy', 'targets.npy', 'subject_id.npy', 'record_id.npy', 'onset.npy'):
            shutil.copyfile(raw.root / split / name, out / name)
    (destination / 'complete.json').write_text(json.dumps({'layer': layer, 'raw_meta_sha256': protocol.sha(raw.root / 'meta.json'),
                                                         'reve_revision': meta['reve_revision']}))


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--cache-root', type=Path, required=True)
    a.add_argument('--competition-root', type=Path, default=models.ROOT / 'external/2026-competition')
    a.add_argument('--layer', type=int, choices=[18, 22], required=True)
    a.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    args = a.parse_args()
    spec = protocol.load_frozen()
    cache = args.cache_root.resolve()
    raw_path = protocol.validate_cache(cache / 'windows/xu2025alljoined', spec['inputs']['alljoined'])
    suffix = 'xu2025alljoined' + ('_L18' if args.layer == 18 else '')
    destination = validate_destination(cache / 'reve_tokens' / suffix, raw_path)
    raw = protocol.CacheIndex(raw_path)
    competition = models.check_competition(args.competition_root, spec['competition_commit'])
    os.environ['HF_HOME'] = str(cache / 'huggingface')
    os.environ['HF_HUB_CACHE'] = str(cache / 'huggingface/hub')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['REVE_POSITIONS_PATH'] = str(cache / 'mne_data')
    device = args.device
    if device == 'auto':
        device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    _, module = models.load_sources(competition)
    reve = models.pinned_reve(module, raw.meta, device, spec['reve_revision'])
    extract(reve, raw, destination, args.layer, device)


if __name__ == '__main__':
    main()
