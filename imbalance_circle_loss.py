# =============================================================================
# Imbalance-Aware Circle Loss — Learnable Margins Version
# Novel contribution for MSc Dissertation:
# "Deep Learning Based Horse Re-Identification"
# Author: Saunak Ghosh (6967472) | University of Surrey | 2026
# Supervisor: Dr. Anjan Dutta
#
# References:
#   Sun et al. "Circle Loss" CVPR 2020.
#   Cao et al. "LDAM Loss" NeurIPS 2019.
#
# Design:
#   - Same gamma=256 as standard Circle Loss (Sun et al. 2020)
#   - Margins initialised from class frequency (LDAM idea)
#   - Margins are nn.Parameter — learned via backpropagation
#   - Margins constrained to [base_margin, max_margin] via sigmoid
#   - Fully vectorised using torch.logsumexp for numerical stability
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImbalanceAwareCircleLoss(nn.Module):
    """
    Imbalance-Aware Circle Loss with LEARNABLE margins.

    Margins are initialised based on class frequency but are then
    optimised by backpropagation during training. Each horse identity
    learns its own optimal margin automatically.

    Margins are constrained to [base_margin, 2*base_margin] via sigmoid
    to prevent instability.
    """

    def __init__(self, class_counts, base_margin=0.25, gamma=256, power=0.5):
        super(ImbalanceAwareCircleLoss, self).__init__()
        self.gamma       = gamma
        self.power       = power
        self.base_margin = base_margin
        self.max_margin  = base_margin * 2.0  # upper bound for stability

        max_count = max(class_counts.values())
        num_ids   = max(class_counts.keys()) + 1

        # Initialise margins from class frequency (same as fixed version)
        init_margins = torch.full((num_ids,), base_margin, dtype=torch.float32)
        for horse_id, count in class_counts.items():
            weight = (max_count / count) ** power
            init_margins[horse_id] = base_margin * weight

        # Convert initial margins to unconstrained logits for sigmoid
        # sigmoid(logit) maps to (0,1), then scale to (base_margin, max_margin)
        # inverse: logit = log(m / (max_margin - m))
        clamped = init_margins.clamp(base_margin + 1e-4, self.max_margin - 1e-4)
        init_logits = torch.log(
            (clamped - base_margin) / (self.max_margin - clamped)
        )

        # nn.Parameter — will be optimised by backpropagation
        self.margin_logits = nn.Parameter(init_logits)

        print("\n=== Imbalance-Aware Circle Loss (LEARNABLE margins) ===")
        print(f"  base_margin={base_margin}, max_margin={self.max_margin}")
        print(f"  gamma={gamma}, power={power}")
        print(f"  Initial margins (will be learned during training):")
        for horse_id in sorted(class_counts.keys()):
            count  = class_counts[horse_id]
            margin = init_margins[horse_id].item()
            print(f"  Horse {horse_id:2d}: {count:5d} images -> init margin={margin:.3f}")
        print("=======================================================\n")

    def get_margins(self):
        """
        Convert unconstrained logits to constrained margins.
        Output range: [base_margin, max_margin]
        """
        return self.base_margin + (self.max_margin - self.base_margin) * torch.sigmoid(self.margin_logits)

    def forward(self, features, labels):
        # L2 normalise embeddings
        features = F.normalize(features, p=2, dim=1)

        # Pairwise cosine similarity
        sim_mat = torch.matmul(features, features.t()).clamp(-1 + 1e-6, 1 - 1e-6)

        N = features.size(0)

        # Get current learned margins — on same device as features
        all_margins = self.get_margins().to(features.device)  # (num_ids,)

        # Per-sample margins based on identity
        margins = all_margins[labels.cpu()].to(features.device)  # (N,)

        # Positive / negative pair masks
        labels_2d = labels.view(N, 1)
        pos_mask = (labels_2d == labels_2d.t())
        neg_mask = ~pos_mask
        pos_mask.fill_diagonal_(False)

        # Circle Loss targets per anchor
        m       = margins.view(N, 1)
        Op      = 1 + m
        On      = -m
        delta_p = m
        delta_n = -m

        # Adaptive weights (detached from similarity — not from margins)
        alpha_p = torch.clamp_min(Op - sim_mat.detach(), min=0.0)
        alpha_n = torch.clamp_min(sim_mat.detach() - On, min=0.0)

        # Logits
        logit_p = -self.gamma * alpha_p * (sim_mat - delta_p)
        logit_n =  self.gamma * alpha_n * (sim_mat - delta_n)

        # Mask invalid pairs
        neg_inf = torch.finfo(logit_p.dtype).min
        logit_p = logit_p.masked_fill(~pos_mask, neg_inf)
        logit_n = logit_n.masked_fill(~neg_mask, neg_inf)

        # Numerically stable aggregation
        lse_p = torch.logsumexp(logit_p, dim=1)
        lse_n = torch.logsumexp(logit_n, dim=1)

        per_anchor = F.softplus(lse_p + lse_n)

        valid = (pos_mask.sum(dim=1) > 0) & (neg_mask.sum(dim=1) > 0)
        if valid.sum() == 0:
            return torch.tensor(0.0, device=features.device, requires_grad=True)

        return per_anchor[valid].mean()
