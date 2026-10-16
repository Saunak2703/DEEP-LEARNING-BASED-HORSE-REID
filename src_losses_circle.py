# Circle Loss — Sun et al. CVPR 2020
# "Circle Loss: A Unified Perspective of Pair Similarity Optimisation"
# Adapted for Horse Re-Identification

import torch
import torch.nn as nn
import torch.nn.functional as F


class CircleLoss(nn.Module):
    """
    Circle Loss (Sun et al. 2020).

    A unified metric learning loss that assigns adaptive weighting
    to each similarity score based on its distance from the
    optimal value, focusing learning effort on hard pairs.

    Compared to Triplet Loss (fixed margin, local pair comparison),
    Circle Loss uses:
    - Adaptive per-pair weights (alpha_p, alpha_n)
    - Global optimisation over all positive and negative pairs
    - Simultaneous pull of all positives and push of all negatives

    This directly improves mAP by producing more structured
    embeddings with better global intra/inter class separation.

    Args:
        scale (float): Scale factor gamma (default: 256).
                       Higher = sharper decision boundary.
        margin (float): Margin m (default: 0.25).
                        Controls separation between classes.
    """

    def __init__(self, scale=256.0, margin=0.25):
        super(CircleLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.soft_plus = nn.Softplus()

        # Optimal similarity targets from the paper
        self.O_p = 1 + margin   # optimal positive similarity = 1 + m
        self.O_n = -margin       # optimal negative similarity = -m
        self.delta_p = 1 - margin  # positive decision boundary
        self.delta_n = margin       # negative decision boundary

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings (Tensor): shape (batch, embedding_size)
                L2-normalised feature embeddings.
            labels (Tensor): shape (batch,) — identity labels.

        Returns:
            loss (Tensor): scalar Circle Loss value.
        """
        # L2-normalise embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Pairwise cosine similarity matrix
        sim_mat = torch.matmul(embeddings, embeddings.t())  # (N, N)

        # Build positive and negative masks
        # Positive: same identity, exclude self-pairs
        labels = labels.unsqueeze(1)  # (N, 1)
        pos_mask = (labels == labels.t())  # (N, N) bool
        neg_mask = ~pos_mask               # (N, N) bool
        pos_mask.fill_diagonal_(False)     # exclude self-pairs

        # Adaptive weights — how far each similarity is from optimal
        # alpha_p: weight for positives (0 if already above O_p)
        # alpha_n: weight for negatives (0 if already below O_n)
        ap = torch.clamp_min(-sim_mat.detach() + self.O_p, min=0.0)  # (N, N)
        an = torch.clamp_min(sim_mat.detach() - self.O_n, min=0.0)   # (N, N)

        # Weighted logits
        logit_p = -ap * (sim_mat - self.delta_p) * self.scale   # push positives up
        logit_n = an * (sim_mat - self.delta_n) * self.scale    # push negatives down

        # Compute loss per anchor
        loss = 0.0
        num_valid = 0

        for i in range(embeddings.size(0)):
            pos_logits = logit_p[i][pos_mask[i]]
            neg_logits = logit_n[i][neg_mask[i]]

            if pos_logits.numel() == 0 or neg_logits.numel() == 0:
                continue

            # log-sum-exp of negatives + log-sum-exp of positives
            # Circle loss = log(1 + sum_neg * sum_pos) per anchor
            loss_i = self.soft_plus(
                torch.logsumexp(neg_logits, dim=0) +
                torch.logsumexp(pos_logits, dim=0)
            )
            loss += loss_i
            num_valid += 1

        if num_valid == 0:
            return torch.tensor(0.0, requires_grad=True,
                                device=embeddings.device)

        return loss / num_valid
