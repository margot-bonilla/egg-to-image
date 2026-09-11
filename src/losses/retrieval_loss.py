"""
Contrastive Loss functions for aligning EEG embeddings with frozen DINOv2-giant visual targets.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE (CLIP-style) Contrastive Loss.

    In EEG-to-Image decoding, we want the model to map an EEG epoch to an embedding
    that has high cosine similarity with the true viewed image's DINOv2 vector,
    and low cosine similarity with all other images in the current batch.

    Mathematical formulation:
    Given a batch of size B:
        - pred_emb:   E in R^{B x D} (predicted from EEG)
        - target_emb: V in R^{B x D} (ground-truth DINOv2 image embeddings)

    1. Normalize embeddings:
       e_i = E_i / ||E_i||_2,  v_i = V_i / ||V_i||_2

    2. Compute similarity matrix scaled by temperature tau:
       S_{i, j} = (e_i . v_j) / tau   (shape: B x B)

    3. For each e_i, the correct matching target is v_i (diagonal labels: [0, 1, ..., B-1]).
       Compute symmetric cross-entropy:
       L_{eeg -> img} = CrossEntropy(S, labels)
       L_{img -> eeg} = CrossEntropy(S^T, labels)
       Total Loss = (L_{eeg -> img} + L_{img -> eeg}) / 2
    """
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Softmax temperature parameter tau.
                         Lower temperature (< 0.1) sharpens the distribution,
                         penalizing hard negatives more aggressively.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, pred_emb: torch.Tensor, target_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_emb: (B, D) predicted embeddings from EEG encoder.
            target_emb: (B, D) ground truth DINOv2 embeddings of viewed images.

        Returns:
            Scalar tensor: InfoNCE contrastive loss.
        """
        # TODO [Step 1]: L2-normalize both pred_emb and target_emb along dim=-1.
        # Hint: pred_norm = F.normalize(pred_emb, p=2, dim=-1)
        raise NotImplementedError("TODO: Implement Step 1 - L2-normalize pred_emb and target_emb")

        # TODO [Step 2]: Compute scaled cosine similarity matrix (logits).
        # Shape must be (B, B), where entry (i, j) is cosine similarity between
        # the i-th EEG prediction and the j-th image target divided by self.temperature.
        # Hint: torch.matmul(pred_norm, target_norm.T) / self.temperature
        # logits = ...

        # TODO [Step 3]: Create target labels [0, 1, ..., B-1] on the same device.
        # Hint: torch.arange(B, device=pred_emb.device)
        # labels = ...

        # TODO [Step 4]: Compute bidirectional cross-entropy loss and average them.
        # Hint:
        # loss_eeg_to_img = F.cross_entropy(logits, labels)
        # loss_img_to_eeg = F.cross_entropy(logits.T, labels)
        # return (loss_eeg_to_img + loss_img_to_eeg) / 2.0


class CosineMSEHybridLoss(nn.Module):
    """
    Hybrid Loss combining Cosine Distance with Mean Squared Error (MSE).
    Encourages both directional alignment and metric distance minimization.
    """
    def __init__(self, cosine_weight: float = 1.0, mse_weight: float = 0.5):
        super().__init__()
        self.cosine_weight = cosine_weight
        self.mse_weight = mse_weight

    def forward(self, pred_emb: torch.Tensor, target_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_emb: (B, D) predicted embeddings.
            target_emb: (B, D) ground truth embeddings.
        """
        # TODO [Optional / Alternative Loss]:
        # 1. Normalize embeddings.
        # 2. Compute cosine loss: 1.0 - mean(sum(pred_norm * target_norm, dim=-1)).
        # 3. Compute MSE loss: F.mse_loss(pred_norm, target_norm).
        # 4. Return weighted sum.
        raise NotImplementedError("TODO: Implement CosineMSEHybridLoss")
