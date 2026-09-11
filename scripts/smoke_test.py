"""
Smoke test suite for Track 1: EEG-to-Image.

Run this script to verify your implementations as you code each module:
    python scripts/smoke_test.py

It tests:
  1. Top-k retrieval accuracy metrics (src/metrics.py)
  2. InfoNCE contrastive loss (src/losses/retrieval_loss.py)
  3. EEGToImageEncoder architecture & forward pass (src/models/eeg_encoder.py)
  4. Codabench submission solver contract (submission/submission.py)
"""
import sys
from pathlib import Path
import torch

# Add repository root to python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def test_metrics():
    print("\n[1/4] Testing Retrieval Metrics (src/metrics.py)...")
    try:
        from src.metrics import compute_retrieval_accuracies

        # Create synthetic queries and gallery
        # Case A: Perfect retrieval (queries identical to first N gallery items)
        gallery = torch.randn(20, 1536)
        queries = gallery[:5].clone()  # 5 queries matching items 0..4

        metrics = compute_retrieval_accuracies(queries, gallery, top_k=(1, 5))
        assert "top1_acc" in metrics and "top5_acc" in metrics, "Missing top1_acc or top5_acc keys"
        assert abs(metrics["top1_acc"] - 1.0) < 1e-5, f"Expected top1_acc=1.0, got {metrics['top1_acc']}"
        assert abs(metrics["top5_acc"] - 1.0) < 1e-5, f"Expected top5_acc=1.0, got {metrics['top5_acc']}"

        # Case B: Arbitrary target indices
        targets = torch.tensor([4, 3, 2, 1, 0])
        queries_reversed = gallery[targets].clone()
        metrics_rev = compute_retrieval_accuracies(queries_reversed, gallery, target_indices=targets, top_k=(1, 5))
        assert abs(metrics_rev["top1_acc"] - 1.0) < 1e-5, "Top-1 retrieval failed with custom target_indices"

        print("  ✅ [PASS] Metrics implementation is correct!")
        return True
    except NotImplementedError as e:
        print(f"  ⏳ [PENDING] {e}")
        return False
    except Exception as e:
        print(f"  ❌ [FAIL] Error in metrics: {e}")
        return False


def test_loss():
    print("\n[2/4] Testing InfoNCE Loss (src/losses/retrieval_loss.py)...")
    try:
        from src.losses.retrieval_loss import InfoNCELoss

        criterion = InfoNCELoss(temperature=0.07)
        B, D = 8, 1536
        pred_emb = torch.randn(B, D, requires_grad=True)
        target_emb = torch.randn(B, D)

        loss = criterion(pred_emb, target_emb)

        assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"
        assert not torch.isnan(loss) and not torch.isinf(loss), "Loss produced NaN or Inf"
        assert loss.item() > 0, "InfoNCE loss should be positive"

        # Check backward pass
        loss.backward()
        assert pred_emb.grad is not None, "Gradients were not computed for pred_emb"

        print("  ✅ [PASS] InfoNCE loss implementation is correct!")
        return True
    except NotImplementedError as e:
        print(f"  ⏳ [PENDING] {e}")
        return False
    except Exception as e:
        print(f"  ❌ [FAIL] Error in loss: {e}")
        return False


def test_encoder():
    print("\n[3/4] Testing EEG Encoder (src/models/eeg_encoder.py)...")
    try:
        from src.models.eeg_encoder import EEGToImageEncoder

        B, C, T = 4, 32, 256
        D = 1536
        encoder = EEGToImageEncoder(n_chans=C, n_times=T, output_dim=D)

        dummy_eeg = torch.randn(B, C, T)
        out = encoder(dummy_eeg)

        assert out.shape == (B, D), f"Expected shape ({B}, {D}), got {out.shape}"

        # Verify L2 normalization
        norms = torch.norm(out, p=2, dim=-1)
        expected_norms = torch.ones(B)
        assert torch.allclose(norms, expected_norms, atol=1e-4), "Output embeddings are not L2-normalized"

        print("  ✅ [PASS] EEGToImageEncoder implementation is correct!")
        return True
    except NotImplementedError as e:
        print(f"  ⏳ [PENDING] {e}")
        return False
    except Exception as e:
        print(f"  ❌ [FAIL] Error in encoder: {e}")
        return False


def test_submission():
    print("\n[4/4] Testing Codabench Submission Contract (submission/submission.py)...")
    try:
        from submission.submission import Solver

        solver = Solver()
        meta = {
            "n_chans": 32,
            "n_times": 256,
            "n_outputs": 1536,
            "device": "cpu",
            "weights_dir": str(REPO_ROOT / "submission"),
        }
        model = solver.load_model(meta)
        x = torch.randn(2, 32, 256)
        preds = model.predict(x)
        assert preds.shape == (2, 1536), f"Expected (2, 1536), got {preds.shape}"
        print("  ✅ [PASS] Submission contract works!")
        return True
    except Exception as e:
        print(f"  ⏳ [PENDING / NOTE] Submission test: {e}")
        return False


def main():
    print("=" * 60)
    print("🧠 Neural Interfaces 2026: Track 1 (EEG-to-Image) Smoke Test")
    print("=" * 60)

    results = [
        test_metrics(),
        test_loss(),
        test_encoder(),
        test_submission(),
    ]

    print("\n" + "=" * 60)
    passed = sum(results)
    print(f"Summary: {passed}/4 components passed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
