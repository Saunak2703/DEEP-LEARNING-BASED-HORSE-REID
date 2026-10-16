# Swin Transformer Re-ID Backbone
# Based on: Liu et al. Swin Transformer (ICCV 2021 Best Paper)
# "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
# Uses timm library for pretrained Swin Transformer
# Adapted for Horse Re-Identification

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    raise ImportError(
        "timm is required for Swin Transformer backbone. "
        "Install with: pip install timm"
    )


class SwinTransformerReID(nn.Module):
    """
    Swin Transformer backbone for Re-Identification.

    Architecture:
        Swin-Base or Swin-Small (pretrained on ImageNet-1K)
        → Global Average Pooling (over patch tokens)
        → FC 512 bottleneck (BNNeck)
        → Identity classifier head (training)
        → 512-d embedding (inference)

    Input image size: 224x224 (standard for Swin Transformer)
    NOTE: Use --height 224 --width 224 when running Swin experiments.

    Swin-Base feature dimension: 1024
    Swin-Small feature dimension: 768
    Output embedding dimension: 512

    Rationale for Swin Transformer in Re-ID:
    - Shifted window attention captures BOTH local coat markings
      AND global body shape — relevant for horse Re-ID
    - Hierarchical feature maps analogous to CNN stages
    - Strong ImageNet pretrained representations
    - WildlifeDatasets 2023 found Swin-B + ArcFace best for wildlife Re-ID

    Following BNNeck design (Luo et al. 2019):
    - f_t (before BN) → metric loss (triplet/circle/arcface)
    - f_i (after BN)  → classification loss (xent/arcface)
    """

    def __init__(
        self,
        num_classes,
        loss,
        pretrained=True,
        model_name="swin_base_patch4_window7_224",
        **kwargs
    ):
        super(SwinTransformerReID, self).__init__()
        self.loss = loss

        # Swin Transformer backbone — remove classification head
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,      # remove classification head
            global_pool="avg",  # global average pooling over patch tokens
        )

        # Feature dimension from Swin backbone
        backbone_feat_dim = self.backbone.num_features
        # swin_base_patch4_window7_224: 1024
        # swin_small_patch4_window7_224: 768

        # BNNeck: FC bottleneck + BatchNorm
        self.bottleneck = nn.Linear(backbone_feat_dim, 512, bias=False)
        self.bn = nn.BatchNorm1d(512)
        self.bn.bias.requires_grad_(False)

        # Classification head
        self.classifier = nn.Linear(512, num_classes, bias=False)

        # Weight initialisation
        nn.init.kaiming_normal_(self.bottleneck.weight, mode="fan_out")
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x):
        # Swin Transformer forward pass
        # Input must be 224x224 for standard Swin configurations
        f = self.backbone(x)       # (B, 1024) for swin_base

        # BNNeck
        f_t = self.bottleneck(f)   # (B, 512) — before BN
        f_i = self.bn(f_t)         # (B, 512) — after BN

        if not self.training:
            return F.normalize(f_i, p=2, dim=1)

        y = self.classifier(f_i)   # (B, num_classes)
        return y, f_t


def swin_base_reid(num_classes, loss, pretrained=True, **kwargs):
    """Build Swin-Base Re-ID model (1024-d features)."""
    return SwinTransformerReID(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        model_name="swin_base_patch4_window7_224",
        **kwargs
    )


def swin_small_reid(num_classes, loss, pretrained=True, **kwargs):
    """Build Swin-Small Re-ID model (768-d features, lighter)."""
    return SwinTransformerReID(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        model_name="swin_small_patch4_window7_224",
        **kwargs
    )
