"""Sealed-phase proxy: held-out participants, ~15k-image gallery, scored separately for seen and unseen images.

    python scripts/large_gallery_eval.py FOLDER [FOLDER ...] --fold K [--weights 0.75 0.25] [--grid W ...] [--max-reps R]

The sealed phase ranks each (participant, image) query against the unique target embeddings of its
whole evaluation partition (Alljoined-sized: its baselines score ~0.0003 top-5, chance for ~16k
candidates), with new participants AND new images. In a cross-validation fold the held-out
participants' val windows show the 16,540 training images (which the model learnt from other
participants), their test windows the 200 test images (never trained on). Gallery = all of them;
top-5 is reported over all queries, seen-image queries and unseen-image queries: the last is the
honest proxy (new person, new image, Alljoined-sized gallery of distractors), but has only 800
queries per fold (~2 hits at 0.3%), so both also get top-50 and the median target rank.

Scoring follows objective.py's subject-aggregated metric: raw predictions averaged per
(participant, image), then L2-normalised and ranked by cosine similarity against the
L2-normalised gallery. With several folders, also their ensemble: members' L2-normalised
predictions averaged with --weights (default equal), as in submission/ensemble_submission.py.
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
TRACK = REPO / "external/2026-competition/tracks/image_decoding"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--weights", type=float, nargs="*", default=None)
    parser.add_argument("--grid", type=float, nargs="*", default=None, metavar="W",
                        help="also score (1 - W) x first folder + W x second folder, for each W")
    parser.add_argument("--cache", type=Path, default=REPO / "cache/windows/xu2025alljoined")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-reps", type=int, default=None, metavar="R",
                        help="average at most R presentations per (participant, image) query (the test images are "
                             "shown ~80 times per participant, the val images 1-2: compare like with like)")
    args = parser.parse_args()
    sys.path.insert(0, str(TRACK))
    sys.path.insert(0, str(REPO))
    from src.data.window_cache import CachedWindows

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    subjects = set(json.loads((REPO / "configs/cv_folds_xu2025alljoined.json").read_text())["folds"][args.fold])
    meta = json.loads((args.cache / "meta.json").read_text())
    base_meta = {"n_chans": len(meta["ch_names"]), "ch_names": meta["ch_names"], "n_times": 120, "sfreq": 120.0,
                 "n_outputs": 1536, "device": device}

    # Held-out participants' windows: val (seen images) + test (unseen images), with a (participant, image) key.
    data = []
    for split in ("val", "test"):
        ds = CachedWindows([args.cache / split])
        fields = ds.fields(("stem", "subject"))
        rows = [i for i, f in enumerate(fields) if str(f["subject"]) in subjects]
        data.append((split, ds.take(rows), [(fields[i]["subject"], fields[i]["stem"]) for i in rows]))
    stems = sorted({stem for _, _, keys in data for _, stem in keys})
    stem_index = {s: i for i, s in enumerate(stems)}
    gallery = torch.empty(len(stems), 1536)
    for _, ds, keys in data:
        targets = ds.parts[0]["targets"][ds.parts[0]["image"][ds.index]]
        for (_, stem), t in zip(keys, targets):
            gallery[stem_index[stem]] = torch.from_numpy(t)
    gallery = torch.nn.functional.normalize(gallery, dim=-1).to(device)
    groups = {}  # (participant, image) -> window positions in the concatenated query order
    offset = 0
    unseen = set()
    for split, _, keys in data:
        for j, key in enumerate(keys):
            group = groups.setdefault(key, [])
            if args.max_reps is None or len(group) < args.max_reps:
                group.append(offset + j)
            if split == "test":
                unseen.add(key[1])
        offset += len(keys)
    print(f"fold {args.fold}: {offset} windows of participants {sorted(subjects)}, {len(groups)} queries, "
          f"gallery {len(stems)} images ({len(unseen)} never trained on); chance top-5 {5 / len(stems):.5f}", flush=True)

    preds = {}
    for i, folder in enumerate(args.folders):
        spec = importlib.util.spec_from_file_location(f"sub{i}", folder / "submission.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        solver = module.Solver.get_instance()
        solver.train_loader = None
        model = solver.load_model({**base_meta, "submission_dir": folder.resolve()})
        out = []
        with torch.no_grad():
            for _, ds, _ in data:
                for lo in range(0, len(ds), args.batch_size):
                    X, _, _ = ds.__getitems__(list(range(lo, min(lo + args.batch_size, len(ds)))))
                    out.append(model.predict(X.to(device)).float().cpu())
        preds[folder.name] = torch.cat(out)
        del model

    def score(P):
        keys = list(groups)
        Q = torch.stack([P[groups[k]].mean(0) for k in keys])
        Q = torch.nn.functional.normalize(Q, dim=-1).to(device)
        target = torch.tensor([stem_index[k[1]] for k in keys], device=device)
        ranks = []
        for lo in range(0, len(Q), 4096):
            sims = Q[lo:lo + 4096] @ gallery.T
            mine = sims.gather(1, target[lo:lo + 4096, None])
            ranks.append((sims > mine).sum(-1).cpu())  # 0 = the target is the top candidate
        ranks = torch.cat(ranks).numpy()
        is_unseen = np.array([k[1] in unseen for k in keys])
        out = {"top5 all": (ranks < 5).mean()}
        for label, r in (("seen", ranks[~is_unseen]), ("UNSEEN", ranks[is_unseen])):
            out.update({f"top5 {label}": (r < 5).mean(), f"top50 {label}": (r < 50).mean(),
                        f"median rank {label}": np.median(r) + 1})
        return out

    rows = {name: score(P) for name, P in preds.items()}
    if len(preds) > 1:
        w = args.weights or [1.0] * len(preds)
        P = sum(wi * torch.nn.functional.normalize(p, dim=-1) for wi, p in zip(w, preds.values()))
        rows[f"ensemble (weights {w})"] = score(P)
    a, b = (torch.nn.functional.normalize(p, dim=-1) for p in list(preds.values())[:2])
    for w in args.grid or []:
        rows[f"{1 - w:.2f} x first + {w:.2f} x second"] = score((1 - w) * a + w * b)
    width = max(map(len, rows)) + 2
    cols = list(next(iter(rows.values())))
    print(f"{'model':<{width}}" + "".join(f"{c:>18}" for c in cols))
    for name, r in rows.items():
        print(f"{name:<{width}}" + "".join(f"{r[c]:>18.5f}" if "median" not in c else f"{r[c]:>18.0f}" for c in cols), flush=True)

if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", str(REPO / "cache" / "huggingface"))
    os.environ.setdefault("REVE_POSITIONS_PATH", str(REPO / "cache" / "mne_data"))
    main()
