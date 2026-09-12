"""
Evaluation metrics for cross-stimulus EEG-to-Image retrieval.

Official Competition Metric:
Top-5 Retrieval Accuracy against the complete held-out image candidate gallery.
A prediction is counted as correct if the true viewed image's embedding is among
the 5 nearest candidate embeddings in the visual space (measured by cosine similarity).
"""
from typing import Dict, Tuple, Optional
import torch
import torch.nn.functional as F


def compute_retrieval_accuracies(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    target_indices: Optional[torch.Tensor] = None,
    top_k: Tuple[int, ...] = (1, 5),
) -> Dict[str, float]:
    """
    Computes top-k retrieval accuracies for decoded EEG embeddings against a candidate gallery.

    Args:
        query_embeddings: Tensor of shape (N, D) - Predicted embeddings decoded from N EEG test epochs.
        gallery_embeddings: Tensor of shape (M, D) - Ground-truth visual embeddings for M candidate images.
        target_indices: Tensor of shape (N,) - The true gallery index for each of the N queries.
                        If None, assumes N == M and 1-to-1 diagonal matching:
                        target_indices[i] = i (query i corresponds to gallery item i).
        top_k: Tuple of ints specifying the k-ranks to evaluate (e.g. (1, 5)).

    Returns:
        Dict[str, float]: Dictionary mapping metric names to accuracy in [0.0, 1.0],
                          e.g. {"top1_acc": 0.12, "top5_acc": 0.38}.

    Learning Objectives:
        1. Cosine similarity between query vectors and gallery vectors via normalized matrix multiplication.
        2. Ranking gallery items using PyTorch's `topk` operation.
        3. Determining whether the ground-truth target index is within the top-k retrieved ranks.
    """
    if target_indices is None:
        assert query_embeddings.size(0) <= gallery_embeddings.size(0), (
            "When target_indices is None, query_embeddings and gallery_embeddings must have the same size."
        )
        target_indices = torch.arange(query_embeddings.size(0), device=query_embeddings.device)

    # L2-Normalize
    query_embeddings_norm = F.normalize(query_embeddings, p=2, dim=-1)
    gallery_embeddings_norm = F.normalize(gallery_embeddings, p=2, dim=-1)

    # Cosine Similarity
    similarity = query_embeddings_norm @ gallery_embeddings_norm.T

    max_k = max(top_k)
    _, top_indices = torch.topk(similarity, k=max_k, dim=-1, largest=True, sorted=True)

    correct = (top_indices == target_indices.unsqueeze(1))
    results:Dict[str,float] = dict()
    for k in top_k:
        hits = correct[:, :k].any(dim=-1)
        acc = hits.float().mean().item()
        results[f"top{k}_acc"] = acc

    return results
