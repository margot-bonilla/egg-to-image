"""Stateless raw prediction export and fixed-gallery, development-only retrieval."""
import json
from pathlib import Path
import statistics

import numpy as np
import torch
import torch.nn.functional as F


def export_predictions(model, dataset, path, device, batch_size=256, guard=lambda: None):
    path = Path(path)
    if path.exists():
        raise ValueError('Refusing to overwrite prediction vectors')
    temporary = path.with_suffix(path.suffix + '.partial')
    array = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32, shape=(len(dataset), 1536))
    with torch.inference_mode():
        for start in range(0, len(dataset), batch_size):
            guard()
            x, _, _ = dataset.__getitems__(list(range(start, min(start + batch_size, len(dataset)))))
            p = model.predict(x.to(device)).float()
            if p.shape != (len(x), 1536) or not torch.isfinite(p).all() or (p.norm(dim=-1) == 0).any():
                raise ValueError('Invalid prediction coordinates')
            array[start:start + len(x)] = F.normalize(p, dim=-1).cpu().numpy()
    array.flush()
    del array
    temporary.replace(path)
    return np.load(path, mmap_mode='r', allow_pickle=False)


def score(predictions, groups, gallery, stems, cap, device='cpu', chunk=32, guard=lambda: None):
    if cap not in ('1', '2', 'all'):
        raise ValueError('Only frozen development repetition caps are allowed')
    if predictions.ndim != 2 or predictions.shape[1] != 1536 or gallery.shape != (len(stems), 1536):
        raise ValueError('Not original DINO coordinates')
    columns = {name: i for i, name in enumerate(stems)}
    if len(columns) != len(stems) or not np.isfinite(gallery).all():
        raise ValueError('Invalid fixed gallery')
    # The official objective normalises float32 candidates, then multiplies by
    # float64 subject/image means. Rank on CPU: MPS does not support float64.
    normal_gallery = gallery / np.maximum(np.linalg.norm(gallery, axis=1, keepdims=True), 1e-12)
    G = torch.as_tensor(normal_gallery.astype(np.float64), device='cpu')
    rows = []
    with torch.inference_mode():
        for lo in range(0, len(groups), chunk):
            guard()
            selected = groups[lo:lo + chunk]
            vectors, counts = [], []
            for group in selected:
                indices = group['rows'] if cap == 'all' else group['rows'][:int(cap)]
                vectors.append(np.asarray(predictions[indices], dtype=np.float64).mean(axis=0))
                counts.append(len(indices))
            vectors = np.stack(vectors)
            vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
            P = torch.as_tensor(vectors, device='cpu')
            if not torch.isfinite(P).all() or (P.norm(dim=-1) == 0).any():
                raise ValueError('Invalid grouped predictions')
            scores = P @ G.T
            labels = torch.tensor([columns[g['image']] for g in selected], device='cpu')
            true = scores.gather(1, labels[:, None])
            # Average ties, matching scipy.stats.rankdata(-cosine) - 1 in the official objective.
            ranks = ((scores > true).sum(1) + ((scores == true).sum(1) - 1) / 2).cpu().tolist()
            for g, rank, repetitions in zip(selected, ranks, counts):
                rows.append({'subject': g['subject'], 'image': g['image'], 'rank': rank,
                             'repetitions': repetitions, 'top1': rank < 1, 'top5': rank < 5,
                             'top50': rank < 50, 'unseen': True})
    by_subject = {}
    for row in rows:
        by_subject.setdefault(row['subject'], []).append(row)
    per_subject = {}
    for subject, values in by_subject.items():
        per_subject[subject] = {'queries': len(values), 'top1': statistics.mean(v['top1'] for v in values),
                                'top5': statistics.mean(v['top5'] for v in values),
                                'top50': statistics.mean(v['top50'] for v in values),
                                'MRR': statistics.mean(1 / (v['rank'] + 1) for v in values),
                                'median_rank_1based': statistics.median(v['rank'] + 1 for v in values)}
    if not per_subject:
        raise ValueError('Empty development evaluation')
    macro = {k: statistics.mean(s[k] for s in per_subject.values()) for k in ('top1', 'top5', 'top50', 'MRR')}
    return {'gallery_size': len(stems), 'queries': rows, 'subject_metrics': per_subject,
            'metrics': macro, 'top5_hits': sum(v['top5'] for v in rows),
            'query_count': len(rows), 'cap': cap, 'scope': 'public development only'}


def fixed_mixture(encoder, reve):
    if encoder.shape != reve.shape:
        raise ValueError('Members have different query rows')
    # Each member is L2-normalised per trial before the fixed weighted mixture.
    def normal(a):
        a = np.asarray(a, dtype=np.float32)
        norms = np.linalg.norm(a, axis=1, keepdims=True)
        if not np.isfinite(a).all() or (norms == 0).any():
            raise ValueError('Invalid member predictions')
        return a / norms
    return .75 * normal(encoder) + .25 * normal(reve)
