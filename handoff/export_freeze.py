"""Export the existing development identities, not a newly sampled split.

This maintainer utility reads metadata only. It never reads EEG payloads or scores
confirmation. Run it in an isolated checkout, not the active training checkout.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import struct


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def npy_header(path):
    with Path(path).open('rb') as f:
        if f.read(6) != b'\x93NUMPY':
            raise ValueError('Not an NPY array')
        version = tuple(f.read(2))
        width = 2 if version == (1, 0) else 4
        size = struct.unpack('<H' if width == 2 else '<I', f.read(width))[0]
        header = ast.literal_eval(f.read(size).decode('latin1').strip())
    return {'shape': list(header['shape']), 'dtype': header['descr'],
            'fortran_order': header['fortran_order'], 'bytes': Path(path).stat().st_size}


def export(source, things, output):
    source, things, output = map(lambda p: Path(p).resolve(), (source, things, output))
    if output.exists() or source == output:
        raise ValueError('Export must create a new isolated file')
    p = json.loads(source.read_text())
    c = p['config']
    aj = Path(p['cache']).resolve()
    if output.is_relative_to(aj) or output.is_relative_to(things) or output.is_relative_to(source.parent):
        raise ValueError('Do not write into data or active experiment directories')
    portable = {
        'schema': 'eeg-baseline-handoff-v1', 'fixture': False,
        'source_protocol_sha256': sha(source), 'source_commit': 'c11d8c9857834e818f64ea6f315f278a1760d2c7',
        'competition_commit': 'a66bc23d1a0e686b895c093a642b45c74970758b',
        'reve_revision': '33841986f6cc1438d9dcdbca5a2137d60f6007d9',
        'seed': 1, 'development': c['development'], 'confirmation_exclusions': c['confirmation'],
        'alljoined_meta': json.loads((aj / 'meta.json').read_text()),
        'inputs': {}, 'folds': {},
        'scope': 'Single-seed replica; unseen people AND unseen images; no confirmation or uploads',
        'recipes': {
            'encoder_joint': {'n_epochs': 40, 'lr': 0.001, 'weight_decay': 0.0001, 'smooth': 3, 'patience': 10},
            'encoder_aj': {'n_epochs': 40, 'lr': 0.0003, 'weight_decay': 0.0001, 'smooth': 3, 'patience': 10},
            'reve_head': {'n_epochs': 40, 'lr': 0.001, 'weight_decay': 0.0001, 'smooth': 3, 'patience': 5},
            'reve_top4': {'n_epochs': 15, 'lr': 0.00003, 'weight_decay': 0.05, 'smooth': 3, 'patience': 5},
        },
        'mixture': {'encoder': 0.75, 'reve': 0.25},
        'selection': 'development cap1, mean of last 3 complete epoch scores; earliest ties',
        'batch_size': 256, 'caps': ['1', '2', 'all'],
    }
    for name, cache in [('alljoined', aj), ('things', things)]:
        files, headers = {}, {}
        files['meta.json'] = sha(cache / 'meta.json')
        for split in ('train', 'val', 'test'):
            d = cache / split
            headers[split] = npy_header(d / 'X.npy')
            # Explicit files only; never recursively scan caches.
            for filename in ('fields.json', 'image.npy', 'targets.npy', 'subject_id.npy',
                             'record_id.npy', 'onset.npy', 'stream_id.npy', 'context_id.npy'):
                f = d / filename
                if f.exists():
                    files[f'{split}/{filename}'] = sha(f)
        portable['inputs'][name] = {'sha256': files, 'X_headers': headers}
    test = json.loads((aj / 'test/fields.json').read_text())
    portable['test_exclusions'] = test['stem']['names']
    for fold in ('1', '2'):
        original = p['folds'][fold]
        q = original['queries']['development']
        portable['folds'][fold] = {
            'subjects': original['subjects'], 'gallery_stems': original['gallery']['stems'],
            'rows': q['rows'], 'groups': [
                {k: group[k] for k in ('subject', 'image', 'rows')} for group in q['groups']],
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as f:
        json.dump(portable, f, separators=(',', ':'), sort_keys=True, allow_nan=False)
        f.write('\n')
    print(json.dumps({'file': str(output), 'sha256': sha(output),
                      'development_images': len(portable['development']),
                      'confirmation_images_excluded_only': len(portable['confirmation_exclusions']),
                      'queries_per_fold': {k: len(v['groups']) for k, v in portable['folds'].items()}}))


if __name__ == '__main__':
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--source-protocol', required=True)
    a.add_argument('--things-cache', required=True)
    a.add_argument('--out', required=True)
    args = a.parse_args()
    export(args.source_protocol, args.things_cache, args.out)
