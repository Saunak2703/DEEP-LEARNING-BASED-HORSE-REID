# ArcFace Loss — Deng et al. CVPR 2019
# "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"
# Adapted for Horse Re-Identification

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceLoss(nn.Module):
    """
    ArcFace Loss (Deng et al. 2019).

    Adds an angular margin m to the angle between the embedding
    and its ground-truth class centre, enforcing a geodesic
    distance margin on a normalised hypersphere.

    Loss = CrossEntropy(s * [cos(theta_yi + m), cos(theta_j)])

    Args:
        num_classes (int): Number of training identities.
        embedding_size (int): Dimension of the feature embedding (default: 512).
        scale (float): Feature scale s (default: 64).
        margin (float): Angular margin m in radians (default: 0.5).
        use_gpu (bool): Whether to use GPU.
        label_smooth (bool): Apply label smoothing to the cross-entropy loss.
    """

    def __init__(
        self,
        num_classes,
        embedding_size=512,
        scale=64.0,
        margin=0.5,
        use_gpu=True,
        label_smooth=False,
    ):
        super(ArcFaceLoss, self).__init__()
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.scale = scale
        self.margin = margin
        self.label_smooth = label_smooth

        # Class weight matrix — each row is a class centre on the hypersphere
        self.weight = nn.Parameter(
            torch.FloatTensor(num_classes, embedding_size)
        )
        nn.init.xavier_uniform_(self.weight)

        # Pre-compute cos and sin of margin for efficiency
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        # Safety threshold: cos(pi - m)
        self.th = math.cos(math.pi - margin)
        # mm = sin(pi - m) * m = sin(m) * m
        self.mm = math.sin(math.pi - margin) * margin

        if use_gpu:
            self.cuda()

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings (Tensor): shape (batch, embedding_size)
                Feature embeddings — should already be from the
                fc512 bottleneck layer before classification head.
            labels (Tensor): shape (batch,) — ground truth identity labels.

        Returns:
            loss (Tensor): scalar ArcFace loss value.
        """
        # L2-normalise both embeddings and weight vectors
        embeddings_norm = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity: cos(theta_j) for all classes j
        cosine = F.linear(embeddings_norm, weight_norm)  # (batch, num_classes)
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)

        # Compute cos(theta + m) using angle addition formula:
        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        sine = torch.sqrt(1.0 - cosine.pow(2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)

        # For numerical stability when theta + m > pi,
        # fall back to cos(theta) - m*sin(m)
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Build one-hot mask for ground-truth classes
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Apply margin only to ground-truth class, keep cosine for others
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale  # Scale logits

        # Cross-entropy loss with optional label smoothing
        if self.label_smooth:
            eps = 0.1
            n_classes = output.size(1)
            log_probs = F.log_softmax(output, dim=1)
            loss = -(1 - eps) * log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
            loss = loss - (eps / n_classes) * log_probs.sum(dim=1)
            loss = loss.mean()
        else:
            loss = F.cross_entropy(output, labels.long())

        return loss
