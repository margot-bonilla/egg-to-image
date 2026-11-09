"""Portable frozen identities and source-only cache views. No EEG reads on import."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .export_freeze import npy_header, sha

ROOT = Path(__file__).resolve().parents[1]
FROZEN = Path(__file__).with_name('frozen_split.json')
FROZEN_SHA = '21349fb193913400e7c4cf960fade110e95785156f2287cb41fc6df333e1e528'
SPLITS = ('train', 'val', 'test')


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def load_frozen():
    if sha(FROZEN) != FROZEN_SHA:
        raise ValueError('The committed scientific split changed; do not resample it')
    spec = json.loads(FROZEN.read_text())
    validate_spec(spec)
    return spec


def validate_spec(spec, fixture=False):
    if spec.get('schema') != 'eeg-baseline-handoff-v1' or bool(spec.get('fixture')) != fixture:
        raise ValueError('Wrong protocol or fixture passed to a real run')
    dev, confirmation = spec['development'], spec['confirmation_exclusions']
    if not dev or len(dev) != len(set(dev)) or not confirmation or len(confirmation) != len(set(confirmation)):
        raise ValueError('Empty or duplicated image identities')
    if set(dev) & set(confirmation) or set(dev + confirmation) & set(spec['test_exclusions']):
        raise ValueError('Development, confirmation and test images must be disjoint')
    if set(spec['folds']) != {'1', '2'} or spec['mixture'] != {'encoder': .75, 'reve': .25}:
        raise ValueError('Only the fixed two-fold, fixed-mixture development screen is allowed')
    subjects = []
    for fold, frozen in spec['folds'].items():
        subjects += frozen['subjects']
        if len(set(frozen['gallery_stems'])) != len(frozen['gallery_stems']):
            raise ValueError('Duplicate gallery image')
        groups = frozen['groups']
        keys = {(g['subject'], g['image']) for g in groups}
        expected = {(s, im) for s in frozen['subjects'] for im in dev}
        if keys != expected or len(keys) != len(groups) or not set(dev) <= set(frozen['gallery_stems']):
            raise ValueError('Development query identities or gallery are incomplete')
        used = []
        for g in groups:
            if len(g['rows']) < 2 or len(set(g['rows'])) != len(g['rows']):
                raise ValueError('Queries need distinct frozen repetitions')
            used += g['rows']
        if sorted(used) != list(range(len(frozen['rows']))) or len(set(frozen['rows'])) != len(frozen['rows']):
            raise ValueError('Frozen query rows must be a complete, unique partition')
    if len(subjects) != len(set(subjects)):
        raise ValueError('Fold participants overlap')
    meta = spec['alljoined_meta']
    if meta['n_outputs'] != 1536 or meta['n_times'] != 120 or meta['sfreq'] != 120:
        raise ValueError('Original 1536-D DINO / 120-Hz input contract changed')
    if not fixture and (len(dev) != 1500 or len(confirmation) != 1500 or meta['n_chans'] != 32):
        raise ValueError('Production split size/montage changed')


def validate_cache(cache, expected):
    cache = Path(cache).resolve(strict=True)
    for relative, digest in expected['sha256'].items():
        if sha(cache / relative) != digest:
            raise ValueError(f'Cache identity mismatch: {cache}/{relative}; transfer/rebuild the official cache')
    for split, header in expected['X_headers'].items():
        if npy_header(cache / split / 'X.npy') != header:
            raise ValueError(f'Cache X header differs: {cache}/{split}')
    return cache


def target_hash(target):
    a = np.array(target, dtype=np.float32, copy=True)
    a[a == 0] = 0  # signed zero is not a distinct image coordinate
    return hashlib.sha256(a.tobytes()).hexdigest()


class CacheIndex:
    """Metadata and mmap targets, not EEG; concatenation is train then val then test."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.meta = json.loads((self.root / 'meta.json').read_text())
        self.parts = {}
        self.offsets = [0]
        self.by_image = {}
        for split in SPLITS:
            directory = self.root / split
            fields = json.loads((directory / 'fields.json').read_text())
            n = len(fields['stem']['index'])
            image = np.load(directory / 'image.npy', mmap_mode='r', allow_pickle=False)
            if image.shape != (n,) or not np.array_equal(image, fields['stem']['index']):
                raise ValueError('Cache image/field row alignment differs')
            if len(fields['subject']['index']) != n:
                raise ValueError('Invalid participant field length')
            targets = np.load(directory / 'targets.npy', mmap_mode='r', allow_pickle=False)
            if targets.shape != (len(fields['stem']['names']), 1536) or targets.dtype != np.float32:
                raise ValueError('Not the original float32 DINO coordinate cache')
            stem = np.asarray(fields['stem']['index'], dtype=np.int64)
            subject = np.asarray(fields['subject']['index'], dtype=np.int64)
            subject_names, stem_names = fields['subject']['names'], fields['stem']['names']
            if len(set(subject_names)) != len(subject_names) or len(set(stem_names)) != len(stem_names):
                raise ValueError('Ambiguous canonical identities')
            if ((stem < 0).any() or (stem >= len(stem_names)).any() or
                    (subject < 0).any() or (subject >= len(subject_names)).any()):
                raise ValueError('Field index out of bounds')
            for i, name in enumerate(stem_names):
                target = targets[i]
                if not np.isfinite(target).all() or not np.linalg.norm(target):
                    raise ValueError('Invalid DINO target')
                if name in self.by_image and not np.array_equal(self.by_image[name], target):
                    raise ValueError('Image coordinates differ across cache splits')
                self.by_image[name] = target
            self.parts[split] = {'n': n, 'stem': stem, 'subject': subject,
                                 'stem_names': stem_names, 'subject_names': subject_names,
                                 'targets': targets, 'record': np.load(directory / 'record_id.npy', mmap_mode='r')}
            self.offsets.append(self.offsets[-1] + n)
        if set(self.parts['test']['stem_names']) & set(self.parts['train']['stem_names'] + self.parts['val']['stem_names']):
            raise ValueError('Public test image leakage into source splits')

    def selected(self, held, excluded, excluded_target_hashes):
        selected = []
        for j, split in enumerate(('train', 'val')):
            p = self.parts[split]
            allowed_images = np.array([name not in excluded and target_hash(p['targets'][i]) not in excluded_target_hashes
                                       for i, name in enumerate(p['stem_names'])])
            allowed_people = np.array([name not in held for name in p['subject_names']])
            selected.append(np.flatnonzero(allowed_images[p['stem']] & allowed_people[p['subject']]) + self.offsets[j])
        rows = np.concatenate(selected)
        if len(rows) < 2:
            raise ValueError('No source-only training windows remain')
        return rows

    def validate_queries(self, fold):
        n_trainval = self.offsets[2]
        for group in fold['groups']:
            ids = [fold['rows'][i] for i in group['rows']]
            records = []
            for row in ids:
                if not 0 <= row < n_trainval:
                    raise ValueError('Development query is outside train+val')
                j = 0 if row < self.offsets[1] else 1
                p = self.parts[('train', 'val')[j]]
                local = row - self.offsets[j]
                if (p['stem_names'][p['stem'][local]] != group['image'] or
                        p['subject_names'][p['subject'][local]] != group['subject']):
                    raise ValueError('Frozen query does not address its canonical image/person')
                records.append(int(p['record'][local]))
            if records[0] == records[1]:
                raise ValueError('Cap2 must use distinct within-person recording IDs')

    def dataset(self, rows, channel_names=None):
        from src.data.window_cache import CachedWindows
        channels = None
        if channel_names is not None:
            names = self.meta['ch_names']
            if not set(channel_names) <= set(names):
                raise ValueError('Extra dataset lacks Alljoined channels')
            channels = [[names.index(n) for n in channel_names]] * 2
        return CachedWindows([self.root / 'train', self.root / 'val'], index=rows, channels=channels)

    def gallery(self, stems):
        targets = np.stack([self.by_image[s] for s in stems]).astype(np.float32)
        if len({target_hash(y) for y in targets}) != len(stems):
            raise ValueError('Exact target aliases in the gallery')
        return targets


def validate_tokens(tokens, raw, layer):
    tokens = Path(tokens).resolve(strict=True)
    meta = json.loads((tokens / 'meta.json').read_text())
    actual_layer = 22 if meta.get('layer') is None else meta['layer']
    if meta['ch_names'] != raw.meta['ch_names'] or actual_layer != layer:
        raise ValueError('Wrong REVE token montage/layer')
    for split in ('train', 'val'):
        for name in ('fields.json', 'image.npy', 'targets.npy', 'subject_id.npy', 'record_id.npy', 'onset.npy'):
            if sha(tokens / split / name) != sha(raw.root / split / name):
                raise ValueError('REVE tokens do not share raw-cache row/target identities')
        header = npy_header(tokens / split / 'X.npy')
        if header['shape'] != [raw.parts[split]['n'], raw.meta['n_chans'], 512] or header['dtype'] != '<f2':
            raise ValueError('Incomplete/wrong-shaped float16 token cache')
    return tokens
