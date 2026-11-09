"""One-command, serial development baseline. Import is inert; --run trains."""
import argparse
import copy
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch

from . import evaluate, models, protocol

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = ['handoff/__init__.py', 'handoff/export_freeze.py', 'handoff/protocol.py',
                'handoff/models.py', 'handoff/evaluate.py', 'handoff/run.py', 'handoff/frozen_split.json',
                'handoff/cache_tokens.py',
                'submission/submission.py', 'submission/reve_submission.py', 'src/data/window_cache.py']


def atomic_json(path, value):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix='.publish-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_weights(path, state):
    fd, temporary = tempfile.mkstemp(prefix='.weights-', dir=Path(path).parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save({k: v.detach().cpu() for k, v in state.items()}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def runtime_versions(fixture=False):
    packages = ('torch', 'numpy') if fixture else (
        'torch', 'numpy', 'braindecode', 'benchopt', 'transformers',
        'neuralbench', 'neuralset', 'neuralfetch', 'neuraltrain', 'exca')
    return {'python': '.'.join(map(str, sys.version_info[:3])),
            **{k: importlib.metadata.version(k) for k in packages}}


def source_hashes():
    return {p: protocol.sha(ROOT / p) for p in SOURCE_FILES}


def clean_commit():
    dirty = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=normal', '--', *SOURCE_FILES],
                           cwd=ROOT, text=True, capture_output=True, check=True)
    if dirty.stdout.strip():
        raise ValueError('Commit the executable handoff sources before a real run')
    return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True,
                          capture_output=True, check=True).stdout.strip()


def cache_stats(paths):
    # Host-specific change detection, not a claim of full payload hashing.
    result = {}
    for path in paths:
        path = Path(path).resolve()
        stat = path.stat()
        result[str(path)] = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'inode': stat.st_ino}
    return result


def safe_output(out, inputs):
    original = Path(out).absolute()
    out = original.resolve()
    if original.is_symlink():
        raise ValueError('Output directory itself must not be a symlink')
    if out == ROOT or ROOT.is_relative_to(out):
        raise ValueError('Output must not contain the source checkout')
    if out.is_relative_to(ROOT) and out.relative_to(ROOT).parts[0] != 'handoff_runs':
        raise ValueError('Inside this checkout, outputs must stay under handoff_runs/')
    for path in inputs:
        path = Path(path).resolve()
        if out == path or out.is_relative_to(path) or path.is_relative_to(out):
            raise ValueError('Output and input cache trees must be disjoint')
        # Input caches may be borrowed read-only from another Git checkout,
        # including the owner's active training project. Never write anywhere
        # in that checkout, not merely outside its cache subdirectory.
        for parent in path.parents:
            if parent != ROOT and (parent / '.git').exists() and out.is_relative_to(parent):
                raise ValueError('Output is inside another checkout providing read-only inputs')
    return out


def prepare(spec, cache_root, fixture=False):
    protocol.validate_spec(spec, fixture=fixture)
    cache_root = Path(cache_root).resolve()
    aj_path = protocol.validate_cache(cache_root / 'windows/xu2025alljoined', spec['inputs']['alljoined'])
    things_path = protocol.validate_cache(cache_root / 'windows/things_eeg2', spec['inputs']['things'])
    aj, things = protocol.CacheIndex(aj_path), protocol.CacheIndex(things_path)
    if aj.meta != spec['alljoined_meta']:
        raise ValueError('Main input metadata differs from the frozen montage')
    excluded = set(spec['development'] + spec['confirmation_exclusions'] + spec['test_exclusions'])
    excluded_hashes = {protocol.target_hash(aj.by_image[im]) for im in excluded}
    # If a canonical name is shared, the two target caches must agree exactly.
    for name in aj.by_image.keys() & things.by_image.keys():
        if not np.array_equal(aj.by_image[name], things.by_image[name]):
            raise ValueError('DINO coordinates differ across training datasets')
    extra_rows = things.selected(set(), excluded, excluded_hashes)
    folds = {}
    for fold, frozen in spec['folds'].items():
        aj.validate_queries(frozen)
        selected = aj.selected(set(frozen['subjects']), excluded, excluded_hashes)
        folds[fold] = {'train_rows': selected, 'raw': aj.dataset(selected),
                       'extra': things.dataset(extra_rows, aj.meta['ch_names']),
                       'dev': aj.dataset(np.asarray(frozen['rows'])),
                       'gallery': aj.gallery(frozen['gallery_stems'])}
    final = protocol.validate_tokens(cache_root / 'reve_tokens/xu2025alljoined', aj, 22)
    prefix = protocol.validate_tokens(cache_root / 'reve_tokens/xu2025alljoined_L18', aj, 18)
    from src.data.window_cache import CachedWindows
    for fold, data in folds.items():
        for key, cache in [('final_tokens', final), ('prefix_tokens', prefix)]:
            data[key] = CachedWindows([cache / 'train', cache / 'val'], index=data['train_rows'])
    files = [p / split / 'X.npy' for p in [aj_path, things_path, final, prefix] for split in ('train', 'val')]
    identity = {'protocol_sha256': protocol.canonical_sha(spec), 'sources': source_hashes(),
                'input_paths': [str(p) for p in [aj_path, things_path, final, prefix]],
                'input_payload_stat': cache_stats(files),
                'source_selections': {fold: protocol.sha_bytes(data['train_rows'].tobytes()) for fold, data in folds.items()},
                'extra_selection': protocol.sha_bytes(extra_rows.tobytes())}
    return aj, final, prefix, folds, identity


def loader(dataset, device, batch_size, workers, solver):
    from src.data.window_cache import _batch
    # Worker seeding must not perturb restored parent RNG on resume. The cache
    # dataset is deterministic: it does not use worker RNG or random transforms.
    # Preserve the published loader's one initial parent base-seed draw, then
    # use an independent worker generator and a global-RNG RandomSampler.
    worker_rng = torch.Generator().manual_seed(0)
    sampler = torch.utils.data.RandomSampler(dataset)
    data = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                                       generator=worker_rng, num_workers=workers, collate_fn=_batch,
                                       drop_last=False, persistent_workers=workers > 0)
    class DeviceLoader:
        first = True
        def __len__(self):
            return len(data)
        def __iter__(self):
            if self.first:
                self.first = False
                if not getattr(solver, 'resumed_from', 0):
                    seed = int(torch.empty((), dtype=torch.int64).random_())
                    worker_rng.manual_seed(seed)
            for x, y, info in data:
                # onset is float64; MPS cannot hold it. Metadata is not a model input.
                yield x.to(device), y.to(device), info
        def close(self):
            iterator = getattr(data, '_iterator', None)
            if iterator is not None:
                iterator._shutdown_workers()  # only this loader's own synthetic/source workers
    return DeviceLoader()


def fit_stage(model, source_module, dataset, name, recipe, folder, data, frozen,
              device, batch_size, workers, guard, parent_sha=None):
    folder.mkdir(exist_ok=True)
    receipt = folder / 'complete.json'
    identity = {'name': name, 'recipe': recipe, 'parent_weights_sha256': parent_sha}
    if receipt.exists():
        result = json.loads(receipt.read_text())
        if result['identity'] != identity or protocol.sha(folder / 'weights.pt') != result['weights_sha256']:
            raise ValueError('Completed stage identity/weights changed')
        model.load_state_dict(torch.load(folder / 'weights.pt', map_location=device, weights_only=True))
        return result
    stage_manifest = folder / 'identity.json'
    if stage_manifest.exists():
        if json.loads(stage_manifest.read_text()) != identity:
            raise ValueError('Checkpoint belongs to another stage/parent')
    else:
        if any(folder.iterdir()):
            raise ValueError('Unidentified pre-existing stage output')
        atomic_json(stage_manifest, identity)
    solver = source_module.Solver.get_instance()
    train = loader(dataset, device, batch_size, workers, solver)
    for key, value in recipe.items():
        setattr(solver, key, value)
    solver.seed = 1
    solver.learn_temperature = True
    solver.checkpoint = folder / 'checkpoint.pt'
    last_scored = None
    def callback(current, epoch, loss, temperature):
        nonlocal last_scored
        guard()
        predictions = []
        with torch.inference_mode():
            for lo in range(0, len(data['dev']), 256):
                x, _, _ = data['dev'].__getitems__(list(range(lo, min(lo + 256, len(data['dev'])))))
                if name.startswith('reve'):
                    # A trained token model must be evaluated through the pinned raw backbone.
                    predictor = data['reve_raw'](current, name)
                    p = predictor.predict(x.to(device))
                else:
                    p = current.predict(x.to(device))
                predictions.append(p.float().cpu().numpy())
                guard()
        last_scored = evaluate.score(np.concatenate(predictions), frozen['groups'], data['gallery'],
                                     frozen['gallery_stems'], '1', guard=guard)
        return last_scored['metrics']['top5']
    solver.epoch_callback = callback
    guard()
    try:
        solver.fit(model, train)
    finally:
        train.close()
    guard()
    atomic_weights(folder / 'weights.pt', model.state_dict())
    atomic_json(folder / 'curve.json', solver.history)
    atomic_json(folder / 'complete.json', {'identity': identity, 'epochs_run': len(solver.history),
                                         'weights_sha256': protocol.sha(folder / 'weights.pt'),
                                         'scope': 'fresh public development; not a sealed result'})
    return json.loads(receipt.read_text())


def save_gallery_once(gallery, stems, folder):
    folder = Path(folder)
    path, receipt = folder / 'gallery.npy', folder / 'gallery.receipt.json'
    array = np.asarray(gallery, dtype=np.float32)
    identity = {'stems_sha256': protocol.canonical_sha(stems), 'shape': list(array.shape),
                'matrix_sha256': protocol.sha_bytes(array.tobytes())}
    if path.exists() and receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved['identity'] != identity or saved['file_sha256'] != protocol.sha(path):
            raise ValueError('Returned gallery matrix/order changed')
        return
    if path.exists():
        path.rename(path.with_name(path.name + '.unsealed-' + str(time.time_ns())))
    fd, temporary = tempfile.mkstemp(prefix='.gallery-', dir=folder)
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    atomic_json(receipt, {'identity': identity, 'file_sha256': protocol.sha(path)})


def predict_once(model, dataset, path, device, guard):
    path = Path(path)
    receipt = path.with_suffix('.receipt.json')
    if path.exists() and receipt.exists():
        expected = json.loads(receipt.read_text())
        if protocol.sha(path) != expected['sha256'] or expected['rows'] != len(dataset):
            raise ValueError('Saved prediction artifact changed')
        return np.load(path, mmap_mode='r', allow_pickle=False)
    for leftover in [path, path.with_suffix(path.suffix + '.partial')]:
        if leftover.exists():
            leftover.rename(leftover.with_name(leftover.name + '.unsealed-' + str(time.time_ns())))
    predictions = evaluate.export_predictions(model, dataset, path, device, guard=guard)
    atomic_json(receipt, {'sha256': protocol.sha(path), 'rows': len(dataset), 'outputs': 1536})
    return predictions


def execute(spec, prepared, output, device, workers, source_module, reve_module, reve,
            fixture=False, max_stage_epochs=None):
    aj, final, prefix, folds, identity = prepared
    output = safe_output(output, identity['input_paths'])
    identity = {**identity, 'device': str(device), 'workers': workers,
                'environment': runtime_versions(fixture=fixture),
                'git_commit': 'synthetic-fixture' if fixture else clean_commit()}
    manifest = output / 'manifest.json'
    # Reject existing unrelated locations before creating a lock or any other file.
    if output.exists():
        if manifest.exists():
            if json.loads(manifest.read_text()) != identity:
                raise ValueError('Resume source/data/device/environment differs from the frozen run')
        elif any(output.iterdir()):
            raise ValueError('Use an empty output directory for a new run')
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'runner.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if manifest.exists():
            if json.loads(manifest.read_text()) != identity:
                raise ValueError('Resume source/data/device/environment differs from the frozen run')
        else:
            if set(p.name for p in output.iterdir()) != {'runner.lock'}:
                raise ValueError('Use an empty output directory for a new run')
            atomic_json(manifest, identity)
        def guard():
            if source_hashes() != identity['sources']:
                raise ValueError('Executable sources changed during training')
            if cache_stats([Path(p) for p in identity['input_payload_stat']]) != identity['input_payload_stat']:
                raise ValueError('Input payload stat changed during training')
        from src.data.window_cache import CachedWindows
        results = {}
        for fold in ('1', '2'):
            guard()
            data, frozen = folds[fold], spec['folds'][fold]
            folder = output / ('fold' + fold)
            folder.mkdir(exist_ok=True)
            def raw(current, name):
                return reve_module.REVEImageModel(reve, [current], 18 if name == 'reve_top4' else None)
            data['reve_raw'] = raw
            encoder = models.make_encoder(source_module, aj.meta).to(device)
            joint = CachedWindows.concat([data['raw'], data['extra']])
            stages = [('encoder_joint', encoder, joint), ('encoder_aj', encoder, data['raw'])]
            head = models.make_head(reve_module).to(device)
            stages.append(('reve_head', head, data['final_tokens']))
            parent_sha = None
            receipts = {}
            for name, current, dataset in stages:
                recipe = copy.deepcopy(spec['recipes'][name])
                if fixture and max_stage_epochs is not None:
                    recipe.update(n_epochs=max_stage_epochs, smooth=1, patience=max_stage_epochs + 1)
                receipts[name] = fit_stage(current, source_module, dataset, name, recipe, folder / name,
                                           data, frozen, device, spec['batch_size'], workers, guard,
                                           parent_sha if name == 'encoder_aj' else None)
                if name == 'encoder_joint':
                    parent_sha = receipts[name]['weights_sha256']
            top = models.make_top(reve_module, reve, head).to(device)
            recipe = copy.deepcopy(spec['recipes']['reve_top4'])
            if fixture and max_stage_epochs is not None:
                recipe.update(n_epochs=max_stage_epochs, smooth=1, patience=max_stage_epochs + 1)
            receipts['reve_top4'] = fit_stage(top, source_module, data['prefix_tokens'], 'reve_top4', recipe,
                                              folder / 'reve_top4', data, frozen, device,
                                              spec['batch_size'], workers, guard,
                                              receipts['reve_head']['weights_sha256'])
            encoder_vectors = predict_once(encoder, data['dev'], folder / 'encoder.npy', device, guard)
            reve_vectors = predict_once(raw(top, 'reve_top4'), data['dev'], folder / 'reve.npy', device, guard)
            # Store raw trial identities once: later CPU analysis can reconstruct every cap.
            atomic_json(folder / 'query_identity.json', frozen)
            save_gallery_once(data['gallery'], frozen['gallery_stems'], folder)
            blend = evaluate.fixed_mixture(encoder_vectors, reve_vectors)
            reports = {}
            for name, predictions in [('encoder', encoder_vectors), ('reve', reve_vectors), ('mixture', blend)]:
                reports[name] = {cap: evaluate.score(predictions, frozen['groups'], data['gallery'],
                                                    frozen['gallery_stems'], cap, guard=guard) for cap in spec['caps']}
            atomic_json(folder / 'evaluation.json', reports)
            results[fold] = {name: {cap: {'metrics': report['metrics'], 'hits': report['top5_hits'],
                                        'queries': report['query_count']} for cap, report in caps.items()}
                             for name, caps in reports.items()}
            del head, top, encoder, joint, blend
            if torch.device(device).type == 'mps':
                torch.mps.empty_cache()
        guard()
        atomic_json(output / 'summary.json', {'scope': spec['scope'], 'results': results,
                                             'limitations': 'Single seed, checkpoint-selected development; no confirmation/sealed claim.',
                                             'manifest_sha256': protocol.sha(manifest)})
        return results


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__)
    group = a.add_mutually_exclusive_group(required=True)
    group.add_argument('--preflight', action='store_true', help='Validate metadata/identities only; no models or output')
    group.add_argument('--run', action='store_true', help='Run serial baseline training and final development evaluation')
    group.add_argument('--smoke', action='store_true', help='CPU synthetic end-to-end test; no real data or downloads')
    a.add_argument('--cache-root', type=Path, default=ROOT / 'cache')
    a.add_argument('--out', type=Path, default=ROOT / 'handoff_runs/baseline')
    a.add_argument('--competition-root', type=Path, default=ROOT / 'external/2026-competition')
    a.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    a.add_argument('--workers', type=int, default=4)
    a.add_argument('--cpu-threads', type=int, default=1)
    args = a.parse_args(argv)
    if args.workers < 0 or args.cpu_threads < 1:
        a.error('Nonnegative workers and positive CPU thread count required')
    torch.set_num_threads(args.cpu_threads)
    if args.smoke:
        from .smoke import run_smoke
        return run_smoke()
    if sys.version_info[:2] != (3, 12):
        raise ValueError('Use Python 3.12, matching the competition stack')
    spec = protocol.load_frozen()
    competition = models.check_competition(args.competition_root, spec['competition_commit'])
    prepared = prepare(spec, args.cache_root)
    print(json.dumps({'protocol_sha256': protocol.canonical_sha(spec),
                      'source_windows': {f: len(d['raw']) for f, d in prepared[3].items()},
                      'extra_windows': len(prepared[3]['1']['extra']),
                      'queries_per_fold': {f: len(spec['folds'][f]['groups']) for f in spec['folds']},
                      'confirmation': 'excluded, never scored'}, indent=2), flush=True)
    if args.preflight:
        return
    device = args.device
    if device == 'auto':
        device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'mps' and not torch.backends.mps.is_available() or device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('Requested accelerator is unavailable')
    os.environ['HF_HOME'] = str(args.cache_root.resolve() / 'huggingface')
    os.environ['HF_HUB_CACHE'] = str(args.cache_root.resolve() / 'huggingface/hub')
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['REVE_POSITIONS_PATH'] = str(args.cache_root.resolve() / 'mne_data')
    source, reve_module = models.load_sources(competition)
    reve = models.pinned_reve(reve_module, prepared[0].meta, device, spec['reve_revision'])
    models.verify_token_sample(reve, prepared[0], prepared[1], prepared[2], device)
    execute(spec, prepared, args.out, device, args.workers, source, reve_module, reve)
    print('Baseline complete. Read summary.json; no confirmation, packaging or upload performed.')


if __name__ == '__main__':
    main()
