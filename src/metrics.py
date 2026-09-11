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
        assert query_embeddings.size(0) == gallery_embeddings.size(0), (
            "When target_indices is None, query_embeddings and gallery_embeddings must have the same size."
        )
        target_indices = torch.arange(query_embeddings.size(0), device=query_embeddings.device)

    # TODO [Step 1]: L2-normalize both query_embeddings and gallery_embeddings along dimension -1.
    # Hint: Use F.normalize(..., p=2, dim=-1).
    # Why? Cosine similarity between vectors u and v is (u / ||u||) @ (v / ||v||).T.
    raise NotImplementedError("TODO: Implement Step 1 - L2-normalize query and gallery embeddings")

    # TODO [Step 2]: Compute the pairwise cosine similarity matrix between all queries and gallery items.
    # Hint: Matrix multiplication of normalized queries (N, D) and normalized gallery transpose (D, M).
    # What shape should the resulting similarity matrix have? (N, M)
    # similarity = ...

    # TODO [Step 3]: Find the indices of the top-k highest similarity candidates for each query.
    # Hint: Use torch.topk(similarity, k=max_k, dim=-1, largest=True, sorted=True).
    # The returned values will be (values, top_indices), where top_indices has shape (N, max_k).
    # _, top_indices = ...

    # TODO [Step 4]: Check for each query if target_indices appears in the top-k retrieved indices.
    # Hint: Compare top_indices against target_indices.unsqueeze(1).
    # For each k in top_k:
    #   Check if target appears in [:k] ranks -> boolean tensor of shape (N,)
    #   Compute the mean across N samples -> float accuracy
    # results = {}
    # return results
