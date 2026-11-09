"""Disposable CPU fixtures. No real EEG, pretrained weights, network or GPU."""
import contextlib
import importlib.util
import json
from pathlib import Path
import tempfile
import types
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from . import export_freeze, models, protocol, run


class Scale(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(.001 + .0007 * torch.sin(torch.arange(512).float()))
    def forward(self, x):
        return x * self.scale


class FakeREVE:
    """Shape-correct deterministic fixture, deliberately not a scientific model."""
    def __init__(self):
        self.model = nn.Module()
        self.model.transformer = nn.Module()
        self.model.transformer.layers = nn.ModuleList([nn.ModuleList([Scale(), Scale()]) for _ in range(22)])
        self.model.requires_grad_(False)
    def tokens(self, x, layer=None):
        frequency = torch.arange(1, 513, device=x.device).float()[None, None, :]
        value = x.mean(-1, keepdim=True)
        result = torch.sin(value * frequency / 32) + torch.cos(frequency / 19)
        for attention, feedforward in self.model.transformer.layers[:22 if layer is None else layer]:
            result = result + attention(result)
            result = result + feedforward(result)
        return result


def load_fixture_sources():
    class CompetSolver:
        @classmethod
        def get_instance(cls):
            return cls()
    base = types.ModuleType('benchmark_utils.base_solver')
    base.CompetSolver = CompetSolver
    package = types.ModuleType('benchmark_utils')
    package.__path__ = []
    loaded = []
    with patch.dict('sys.modules', {'benchmark_utils': package, 'benchmark_utils.base_solver': base}):
        for name in ('submission', 'reve_submission'):
            spec = importlib.util.spec_from_file_location('fixture_' + name,
                                                         models.ROOT / 'submission' / (name + '.py'))
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            loaded.append(module)
    return tuple(loaded)


def write_cache(path, subjects, images, targets, dataset, channels=3):
    path.mkdir(parents=True)
    rng = np.random.default_rng(37)
    meta = {'ch_names': ['Cz', 'Pz', 'Oz'][:channels], 'sfreq': 120., 'n_chans': channels,
            'n_times': 120, 'n_outputs': 1536}
    (path / 'meta.json').write_text(json.dumps(meta))
    for split in ('train', 'val', 'test'):
        names = images if split != 'test' else [dataset + '_test']
        if names[0] not in targets:
            targets[names[0]] = np.eye(1, 1536, 10, dtype=np.float32).reshape(-1)
        records = [(s, im) for s in subjects for im in names]
        d = path / split
        d.mkdir()
        fields = {key: {'names': sorted({str(row[i]) for row in records}), 'index': []}
                  for key, i in [('subject', 0), ('stem', 1)]}
        for key, i in [('subject', 0), ('stem', 1)]:
            fields[key]['index'] = [fields[key]['names'].index(row[i]) for row in records]
        fields['category'] = {'names': ['fixture'], 'index': [0] * len(records)}
        (d / 'fields.json').write_text(json.dumps(fields))
        n = len(records)
        X = rng.standard_normal((n, channels, 120)).astype(np.float32)
        for i, (_, im) in enumerate(records):
            X[i] += float(np.argmax(targets[im])) / 20
        np.save(d / 'X.npy', X)
        np.save(d / 'targets.npy', np.stack([targets[im] for im in fields['stem']['names']]))
        np.save(d / 'image.npy', np.array(fields['stem']['index'], dtype=np.int64))
        np.save(d / 'subject_id.npy', np.array(fields['subject']['index'], dtype=np.int64))
        np.save(d / 'record_id.npy', np.arange(n, dtype=np.int64) + (1000 if split == 'val' else 0))
        np.save(d / 'onset.npy', np.arange(n, dtype=np.float64))
    return meta


def cache_identity(path):
    return {'sha256': {name: export_freeze.sha(path / name) for name in ['meta.json'] +
                       [f'{split}/{name}' for split in ('train', 'val', 'test')
                        for name in ('fields.json', 'image.npy', 'targets.npy', 'subject_id.npy', 'record_id.npy', 'onset.npy')]},
            'X_headers': {split: export_freeze.npy_header(path / split / 'X.npy') for split in ('train', 'val', 'test')}}


def make_fixture(directory):
    directory = Path(directory)
    cache = directory / 'cache'
    aj = cache / 'windows/xu2025alljoined'
    things = cache / 'windows/things_eeg2'
    targets = {name: np.eye(1, 1536, i, dtype=np.float32).reshape(-1)
               for i, name in enumerate(['train0', 'train1', 'dev0', 'dev1', 'confirm0', 'aj_test'])}
    targets['alias_of_dev0'] = targets['dev0'].copy()
    people = ['fixture/1', 'fixture/2', 'fixture/3', 'fixture/4']
    meta = write_cache(aj, people, ['train0', 'train1', 'dev0', 'dev1', 'confirm0'], targets, 'aj')
    write_cache(things, ['other/1'], ['train0', 'train1', 'dev0', 'dev1', 'confirm0', 'alias_of_dev0', 'aj_test'], targets, 'things')
    fake = FakeREVE()
    for suffix, layer in [('xu2025alljoined', 22), ('xu2025alljoined_L18', 18)]:
        destination = cache / 'reve_tokens' / suffix
        destination.mkdir(parents=True)
        (destination / 'meta.json').write_text(json.dumps({**meta, 'layer': layer}))
        for split in ('train', 'val'):
            d = destination / split
            d.mkdir()
            for name in ('fields.json', 'image.npy', 'targets.npy', 'subject_id.npy', 'record_id.npy', 'onset.npy'):
                (d / name).write_bytes((aj / split / name).read_bytes())
            X = torch.from_numpy(np.load(aj / split / 'X.npy'))
            np.save(d / 'X.npy', fake.tokens(X, None if layer == 22 else layer).numpy().astype(np.float16))
    spec = {
        'schema': 'eeg-baseline-handoff-v1', 'fixture': True, 'seed': 1,
        'alljoined_meta': meta, 'development': ['dev0', 'dev1'], 'confirmation_exclusions': ['confirm0'],
        'test_exclusions': ['aj_test'], 'inputs': {'alljoined': cache_identity(aj), 'things': cache_identity(things)},
        'mixture': {'encoder': .75, 'reve': .25}, 'batch_size': 8, 'caps': ['1', '2', 'all'],
        'scope': 'Synthetic CPU fixture; not a performance result', 'folds': {},
        'recipes': {name: {'n_epochs': 2, 'lr': .001, 'weight_decay': .0001, 'smooth': 1, 'patience': 3}
                    for name in ['encoder_joint', 'encoder_aj', 'reve_head', 'reve_top4']},
    }
    index = protocol.CacheIndex(aj)
    for fold, held in [('1', people[:2]), ('2', people[2:])]:
        rows, groups = [], []
        for subject in held:
            for image in spec['development']:
                group_rows = []
                for split in ('train', 'val'):
                    p = index.parts[split]
                    local = np.flatnonzero((p['stem'] == p['stem_names'].index(image)) &
                                           (p['subject'] == p['subject_names'].index(subject)))[0]
                    global_row = int(local + index.offsets[0 if split == 'train' else 1])
                    group_rows.append(len(rows))
                    rows.append(global_row)
                groups.append({'subject': subject, 'image': image, 'rows': group_rows})
        spec['folds'][fold] = {'subjects': held, 'gallery_stems': sorted(targets), 'rows': rows, 'groups': groups}
    # Alias belongs to THINGS only; the Alljoined gallery must not contain it.
    for fold in spec['folds'].values():
        fold['gallery_stems'] = sorted(index.by_image)
    return spec, cache, fake


def run_smoke():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='eeg-handoff-smoke-') as directory:
        spec, cache, fake = make_fixture(directory)
        prepared = run.prepare(spec, cache, fixture=True)
        source, reve_module = load_fixture_sources()
        models.verify_token_sample(fake, prepared[0], prepared[1], prepared[2], 'cpu')
        result = run.execute(spec, prepared, Path(directory) / 'output', 'cpu', 0,
                             source, reve_module, fake, fixture=True, max_stage_epochs=2)
        # Re-running completed work verifies and reuses weights; it must not retrain.
        repeated = run.execute(spec, prepared, Path(directory) / 'output', 'cpu', 0,
                               source, reve_module, fake, fixture=True, max_stage_epochs=2)
        assert result == repeated
        print('CPU smoke passed: both folds, all four stages, raw inference, fixed mixture, caps and completed-run resume.')
        print('Synthetic fixture only. No real EEG, public weights, GPU, network or competition upload used.')
        return result
