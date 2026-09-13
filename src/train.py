"""
End-to-End Training and Validation Pipeline for Track 1: EEG-to-Image.

Trains the spatiotemporal EEG encoder using InfoNCE contrastive loss against
frozen DINOv2-giant visual targets, evaluates cross-stimulus retrieval (Top-1 & Top-5),
and exports the best model weights to submission/weights.pt.
"""
import sys
from pathlib import Path
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.eeg_encoder import EEGToImageEncoder
from src.losses.retrieval_loss import InfoNCELoss
from src.metrics import compute_retrieval_accuracies
from src.data.dataset import create_synthetic_dataset, EEGImageDataset


def get_device() -> torch.device:
    """Selects best available hardware accelerator (Apple Silicon MPS, CUDA, or CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> float:
    """
    Executes one full training epoch over the dataset.

    Args:
        model: The EEGToImageEncoder model.
        dataloader: DataLoader yielding batches {"eeg": (B, C, T), "target": (B, D)}.
        criterion: The InfoNCELoss instance.
        optimizer: The optimizer (e.g. AdamW).
        device: Hardware device (mps/cuda/cpu).
        max_grad_norm: Max gradient norm for clipping.

    Returns:
        float: Average training loss across all batches in this epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for batch in dataloader:
        eeg = batch["eeg"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad()
        pred_emb = model(eeg)
        loss = criterion(pred_emb, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    top_k: tuple = (1, 5),
) -> dict:
    """
    Evaluates cross-stimulus retrieval on the held-out validation set.

    Args:
        model: The EEGToImageEncoder model in eval mode.
        dataloader: Validation DataLoader.
        device: Hardware device.
        top_k: Ranks to evaluate (e.g. (1, 5)).

    Returns:
        dict: Retrieval metrics, e.g. {"top1_acc": 0.25, "top5_acc": 0.65}.
    """
    model.eval()
    all_preds = []
    all_targets = []

    for batch in dataloader:
        eeg = batch["eeg"].to(device)
        target = batch["target"].to(device)

        preds = model(eeg)
        all_preds.append(preds.cpu())
        all_targets.append(target.cpu())

    # Concatenate all batches across the validation set
    query_embeddings = torch.cat(all_preds, dim=0)
    gallery_embeddings = torch.cat(all_targets, dim=0)

    return compute_retrieval_accuracies(query_embeddings, gallery_embeddings, top_k=top_k)


def run_training(config_path: str = "configs/default.yaml", epochs: int = 15):
    """Full training pipeline with logging and model checkpointing."""
    repo_root = Path(__file__).resolve().parent.parent
    submission_dir = repo_root / "submission"
    submission_dir.mkdir(exist_ok=True)
    weights_path = submission_dir / "weights.pt"

    # Load configuration
    cfg = {}
    if Path(config_path).exists():
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)

    device = get_device()
    print(f"🚀 Training on device: {device}")

    # Hyperparameters
    n_chans = cfg.get("dataset", {}).get("n_channels", 32)
    n_times = cfg.get("dataset", {}).get("n_times", 256)
    dim = cfg.get("target_model", {}).get("embedding_dim", 1536)
    lr = float(cfg.get("training", {}).get("learning_rate", 1e-3))
    batch_size = int(cfg.get("training", {}).get("batch_size", 32))
    temperature = float(cfg.get("training", {}).get("temperature", 0.07))

    # 1. Create dataset (synthetic for local end-to-end verification)
    print("📊 Preparing dataset...")
    full_dataset = create_synthetic_dataset(
        n_samples=256,
        n_chans=n_chans,
        n_times=n_times,
        embedding_dim=dim,
        planted_signal_strength=0.6,
    )
    val_size = max(int(len(full_dataset) * 0.25), batch_size)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    # 2. Instantiate Model, Loss, Optimizer, and Scheduler
    model = EEGToImageEncoder(n_chans=n_chans, n_times=n_times, output_dim=dim).to(device)
    criterion = InfoNCELoss(temperature=temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_top5 = 0.0
    print(f"\n🧠 Starting training for {epochs} epochs...")
    print(f"{'Epoch':<8}{'Train Loss':<14}{'Val Top-1':<14}{'Val Top-5':<14}{'Status'}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, device, top_k=(1, 5))
        scheduler.step()

        top1 = val_metrics.get("top1_acc", 0.0)
        top5 = val_metrics.get("top5_acc", 0.0)

        status = ""
        # Checkpoint: Save best model
        if top5 > best_top5:
            best_top5 = top5
            torch.save(model.state_dict(), weights_path)
            status = "⭐ Best (saved weights.pt)"

        print(f"{epoch:<8}{train_loss:<14.4f}{top1:<14.2%}{top5:<14.2%}{status}")

    print("-" * 60)
    print(f"🎉 Training complete! Best Validation Top-5 Accuracy: {best_top5:.2%}")
    print(f"📁 Weights exported to: {weights_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config")
    args = parser.parse_args()
    run_training(config_path=args.config, epochs=args.epochs)

