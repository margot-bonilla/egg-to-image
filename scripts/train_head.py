"""Train retrieval heads on cached encoder features (scripts/cache_features.py) and compare them.

    python scripts/train_head.py cache/features/reve_gifford2022large [--save HEAD.pt]

Heads (all on standardized features): ridge regression (what the official REVE probe
fits), and linear / MLP heads trained with the CLIP loss used by NeuralBench and the
EEGNet baselines. Scored like the official objective: cosine retrieval against the
split's unique image embeddings, pooled and subject-aggregated top-k.
Select on val; test = the warm-up evaluation set.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "external/2026-competition/tracks/image_decoding"))
from benchmark_utils.metrics import group_means, topk_accuracy  # noqa: E402

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def load(folder, split):
    d = np.load(Path(folder) / f"{split}.npz")
    return {k: d[k] for k in d.files}


def normalize(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _ranks(queries, cand, target_idx, chunk=4096):
    """Rank of each query's true candidate by cosine score (0 = best), in GPU chunks:
    the val gallery (~16.5k images) makes the full score matrix too big."""
    c = torch.from_numpy(normalize(cand)).float().to(DEVICE)
    q_all = torch.from_numpy(normalize(queries)).float()
    t_all = torch.from_numpy(target_idx).long()
    ranks = []
    for i in range(0, len(q_all), chunk):
        s = q_all[i:i + chunk].to(DEVICE) @ c.T
        true = s.gather(1, t_all[i:i + chunk, None].to(DEVICE))
        ranks.append((s > true).sum(1).cpu())
    return torch.cat(ranks).numpy()


def score(pred, split):
    """Official Image-decoding metrics (tracks/image_decoding/objective.py), via ranks."""
    candidates, target_idx = np.unique(split["targets"].astype(np.float32), axis=0, return_inverse=True)
    target_idx = target_idx.reshape(-1)
    keys, pred_sa = group_means(pred, np.stack([split["subject"], target_idx], axis=1))
    r_sa = _ranks(pred_sa, candidates, keys[:, 1])
    r = _ranks(pred, candidates, target_idx)
    return {
        "top5_sa": float((r_sa < 5).mean()),
        "top1_sa": float((r_sa < 1).mean()),
        "top5": float((r < 5).mean()),
        "n_cand": len(candidates),
    }


def score_official(pred, split):
    """Same metrics with benchmark_utils.metrics.topk_accuracy (full score matrix): for checks."""
    candidates, target_idx = np.unique(split["targets"].astype(np.float32), axis=0, return_inverse=True)
    target_idx = target_idx.reshape(-1)
    cand = normalize(candidates)
    keys, pred_sa = group_means(pred, np.stack([split["subject"], target_idx], axis=1))
    return {
        "top5_sa": topk_accuracy(normalize(pred_sa) @ cand.T, keys[:, 1], k=5),
        "top1_sa": topk_accuracy(normalize(pred_sa) @ cand.T, keys[:, 1], k=1),
        "top5": topk_accuracy(normalize(pred) @ cand.T, target_idx, k=5),
    }


class Head(nn.Module):
    def __init__(self, mean, std, d_out, hidden=None, dropout=0.0):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float32))
        d_in = len(mean)
        if hidden:
            self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(hidden, d_out))
        else:
            self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_in, d_out))

    def forward(self, feats):
        return self.net((feats - self.mean) / self.std)


def clip_loss(pred, target, temperature):
    logits = F.normalize(pred, dim=-1) @ F.normalize(target, dim=-1).T / temperature
    return F.cross_entropy(logits, torch.arange(len(pred), device=pred.device))


def predict(head, feats, batch=8192):
    head.eval()
    with torch.no_grad():
        return np.concatenate([head(torch.from_numpy(feats[i:i + batch]).to(DEVICE)).cpu().numpy()
                               for i in range(0, len(feats), batch)])


def train_clip(train, val, hidden=None, dropout=0.0, epochs=30, batch=1024, lr=1e-3, wd=1e-2,
               temperature=0.07, patience=5):
    X = torch.from_numpy(train["features"]).float()
    Y = torch.from_numpy(train["targets"].astype(np.float32))
    head = Head(X.mean(0), X.std(0).clamp_min(1e-6), Y.shape[1], hidden, dropout).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * (len(X) // batch))
    best, best_state, bad = -1.0, None, 0
    for epoch in range(epochs):
        head.train()
        perm = torch.randperm(len(X))
        for i in range(len(X) // batch):
            idx = perm[i * batch:(i + 1) * batch]
            loss = clip_loss(head(X[idx].to(DEVICE)), Y[idx].to(DEVICE), temperature)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        val_score = score(predict(head, val["features"]), val)["top5_sa"]
        if val_score > best:
            best, best_state, bad = val_score, {k: v.clone() for k, v in head.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    head.load_state_dict(best_state)
    return head, epoch + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("features", type=Path, help="folder with train/val/test.npz")
    parser.add_argument("--save", type=Path, help="save the best CLIP head (by val top5_sa) here")
    args = parser.parse_args()
    train, val, test = (load(args.features, s) for s in ("train", "val", "test"))
    print(f"train {train['features'].shape}, val {val['features'].shape}, test {test['features'].shape}, "
          f"device={DEVICE}\n", flush=True)

    rows = []
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    for alpha in (1.0, 1e4):
        start = time.time()
        ridge = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(train["features"], train["targets"])
        rows.append((f"ridge alpha={alpha:g}", None, score(ridge.predict(val["features"]), val),
                     score(ridge.predict(test["features"]), test), time.time() - start, "-"))
    heads = {}
    for name, kwargs in [("linear+CLIP", {}),
                         ("MLP-1024+CLIP", {"hidden": 1024, "dropout": 0.3})]:
        start = time.time()
        head, epochs = train_clip(train, val, **kwargs)
        heads[name] = head
        rows.append((name, head, score(predict(head, val["features"]), val),
                     score(predict(head, test["features"]), test), time.time() - start, epochs))

    print(f"{'head':<18}{'val top5_sa':>12}{'test top5_sa':>14}{'test top1_sa':>14}{'test top5':>11}"
          f"{'epochs':>8}{'time':>8}")
    for name, _, v, t, secs, epochs in rows:
        print(f"{name:<18}{v['top5_sa']:>12.4f}{t['top5_sa']:>14.4f}{t['top1_sa']:>14.4f}{t['top5']:>11.4f}"
              f"{epochs:>8}{secs:>7.0f}s")
    print(f"\n(candidates: val {rows[0][2]['n_cand']}, test {rows[0][3]['n_cand']}; "
          f"chance top-5 on test = {5 / rows[0][3]['n_cand']:.3f})")

    if args.save:
        name, head = max(((r[0], r[1]) for r in rows if r[1] is not None),
                         key=lambda nh: score(predict(nh[1], val["features"]), val)["top5_sa"])
        torch.save({"name": name, "state_dict": head.state_dict(),
                    "hidden": head.net[0].out_features if isinstance(head.net[0], nn.Linear) else None},
                   args.save)
        print(f"saved best head ({name}) to {args.save}")


if __name__ == "__main__":
    main()
