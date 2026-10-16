# =============================================================================
# EfficientNet Re-ID Backbone
# =============================================================================
# Paper 1: Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs"
#           ICML 2019 -- introduced EfficientNet-B0 to B7
# Paper 2: Tan & Le, "EfficientNetV2: Smaller Models and Faster Training"
#           ICML 2021 -- improved V2 family (S, M, L variants)
# Paper 3: Vesely et al. 2023 (WildlifeDatasets) -- EfficientNetV2 used
#           for wildlife Re-ID across 31 species
#
# Architecture:
#   EfficientNet backbone (ImageNet pretrained via timm)
#   -> Global Average Pooling
#   -> FC 512 Linear + BatchNorm (BNNeck bottleneck)
#   -> Classifier head (training only)
#   -> 512-d L2-normalised embedding (inference)
#
# Variants supported:
#   efficientnet_b3_reid   -- EfficientNet-B3  (12M params, Tan & Le 2019)
#   efficientnetv2_s_reid  -- EfficientNetV2-S (21.5M params, Tan & Le 2021)
#
# EfficientNetV2-S selected for dissertation experiments:
#   - Directly cited in Vesely et al. 2023 for animal Re-ID
#   - 21.5M parameters -- fits within 4GB VRAM on RTX 3050 at batch size 16
#   - Superior accuracy-per-parameter vs B3 due to improved compound scaling
#
# Adapted for Horse Re-Identification Dissertation
# Author: Saunak Ghosh (6967472) | MSc AI | University of Surrey | 2026
# =============================================================================

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError(
        "timm is required for EfficientNet backbone.\n"
        "Install with: pip install timm"
    )


class EfficientNetReID(nn.Module):
    """
    EfficientNet backbone for Re-Identification.

    Supports EfficientNet-B3 (Tan & Le 2019) and
    EfficientNetV2-S (Tan & Le 2021).

    Architecture follows the BNNeck design from Luo et al. 2019
    (Bag of Tricks for Re-ID):
        Backbone -> GAP -> FC512 -> BN -> [Classifier (train) | Embedding (test)]
    """

    def __init__(self, num_classes, loss, pretrained=True,
                 model_name='tf_efficientnetv2_s', **kwargs):
        super(EfficientNetReID, self).__init__()
        self.loss = loss

        # --- Backbone ---
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,       # remove timm classifier head
            global_pool='avg'    # global average pooling
        )
        feature_dim = self.backbone.num_features

        # --- BNNeck bottleneck (Luo et al. 2019) ---
        self.bottleneck = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
        )
        self.bottleneck[0].apply(self._init_weights)

        # --- Identity classifier head (training only) ---
        self.classifier = nn.Linear(512, num_classes)
        self.classifier.apply(self._init_weights)

    def _init_weights(self, m):
        """Kaiming normal initialisation for linear layers."""
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Feature extraction
        f = self.backbone(x)

        # BNNeck
        f = self.bottleneck(f)

        # Inference — return 512-d embedding only
        if not self.training:
            return f

        # Training — return (classifier logits, embedding)
        y = self.classifier(f)

        # Handle loss type passed as set {'xent','htri'} or string
        loss = self.loss
        if isinstance(loss, set):
            if 'htri' in loss or 'triplet' in loss or 'circle' in loss:
                return y, f
            else:
                return y
        elif isinstance(loss, str):
            if loss in ['triplet', 'htri', 'circle']:
                return y, f
            else:
                return y
        else:
            return y, f


# =============================================================================
# Factory functions — registered in main.py EXTRA_MODELS dict
# =============================================================================

def efficientnet_b3_reid(num_classes, loss, pretrained=True, **kwargs):
    """
    EfficientNet-B3 for Re-ID.
    Reference: Tan & Le, ICML 2019.
    Parameters: ~12M | Input: 256x128
    """
    return EfficientNetReID(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        model_name='efficientnet_b3'
    )


def efficientnetv2_s_reid(num_classes, loss, pretrained=True, **kwargs):
    """
    EfficientNetV2-S for Re-ID.
    References:
      - Tan & Le, ICML 2021 (EfficientNetV2 paper)
      - Vesely et al. 2023 (WildlifeDatasets) -- EfficientNetV2 for animal Re-ID
    Parameters: ~21.5M | Input: 256x128
    Selected over B3: stronger citation, better accuracy, fits 4GB VRAM at bs=16
    """
    return EfficientNetReID(
        num_classes=num_classes,
        loss=loss,
        pretrained=pretrained,
        model_name='tf_efficientnetv2_s'
    )
