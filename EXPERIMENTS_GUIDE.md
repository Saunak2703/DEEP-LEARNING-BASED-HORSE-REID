# Horse Re-ID Experiment Guide
## University of Surrey — EEEM004 MSc Dissertation

---

## Overview

This codebase supports a systematic ablation study comparing:
- **3 Backbone architectures**: ResNet50, EfficientNet-B3, Swin-Base
- **3 Loss functions**: Triplet, ArcFace, Circle Loss
- **Augmentation strategies**: No aug, CJ only, RE only, Both

Motivated by **WildlifeDatasets (Veselý et al. 2023)** which found
Swin-Base + ArcFace best for wildlife Re-ID across 31 species.

---

## New Files Added

```
main.py                        ← modified (supports new backbones/losses)
args.py                        ← modified (new arguments)
src_losses_arcface.py          ← NEW: ArcFace Loss (Deng et al. 2019)
src_losses_circle.py           ← NEW: Circle Loss (Sun et al. 2020)
src_models_efficientnet_reid.py ← NEW: EfficientNet-B3 backbone
src_models_swin_reid.py        ← NEW: Swin Transformer backbone
requirements.txt               ← NEW: all dependencies
experiments_all.ps1            ← NEW: run all 12 experiments
run_single_experiment.ps1      ← NEW: run one experiment easily
```

---

## Installation

```powershell
pip install -r requirements.txt
```

Verify new backbones work:
```powershell
python -c "from src_models_efficientnet_reid import efficientnet_b3_reid; print('EfficientNet OK')"
python -c "from src_models_swin_reid import swin_base_reid; print('Swin OK')"
python -c "from src_losses_arcface import ArcFaceLoss; print('ArcFace OK')"
python -c "from src_losses_circle import CircleLoss; print('Circle OK')"
```

---

## Experiment Plan

### Core Experiments (follow WildlifeDatasets standard)

| Exp | Backbone | Loss | Aug | Input | Notes |
|-----|----------|------|-----|-------|-------|
| 1 | ResNet50 | Triplet | Both | 256x128 | **BASELINE DONE** Rank-1=76.0% mAP=58.0% |
| 2 | ResNet50 | ArcFace | Both | 256x128 | Tests mAP improvement |
| 3 | ResNet50 | Circle | Both | 256x128 | Tests global ranking |
| 4 | EfficientNet-B3 | Triplet | Both | 256x128 | CNN comparison |
| 5 | EfficientNet-B3 | ArcFace | Both | 256x128 | EffNet + ArcFace |
| 6 | EfficientNet-B3 | Circle | Both | 256x128 | EffNet + Circle |
| 7 | Swin-Base | Triplet | Both | **224x224** | Transformer CNN gap |
| 8 | Swin-Base | ArcFace | Both | **224x224** | **WildlifeDatasets best** |
| 9 | Swin-Base | Circle | Both | **224x224** | Swin + Circle |

### Augmentation Ablation (bonus)

| Exp | Backbone | Loss | Aug | Input |
|-----|----------|------|-----|-------|
| 10 | ResNet50 | Triplet | **None** | 256x128 |
| 11 | ResNet50 | Triplet | CJ only | 256x128 |
| 12 | ResNet50 | Triplet | RE only | 256x128 |

---

## New Arguments

### Loss Type
```powershell
--loss-type triplet    # CrossEntropy + Triplet Loss (default)
--loss-type arcface    # ArcFace + Triplet Loss
--loss-type circle     # CrossEntropy + Circle Loss
```

### ArcFace Settings
```powershell
--arcface-scale 64     # feature scale s (default: 64)
--arcface-margin 0.5   # angular margin m (default: 0.5)
```

### Circle Loss Settings
```powershell
--circle-scale 256     # scale gamma (default: 256)
--circle-margin 0.25   # margin m (default: 0.25)
```

---

## Important Notes for Swin Transformer

Swin Transformer requires **224x224 input** (not 256x128):
```powershell
--height 224 --width 224   # MUST use for Swin
--train-batch-size 8        # reduce batch for GPU memory
--lr 0.0001                 # lower LR for fine-tuning transformer
```

---

## Running Experiments

### Run all experiments:
```powershell
.\experiments_all.ps1
```

### Run one experiment:
Edit `run_single_experiment.ps1` then:
```powershell
.\run_single_experiment.ps1
```

### Run manually (example — Exp 8: Swin + ArcFace):
```powershell
python main.py `
    -s horse -t horse `
    --root "C:/Users/LENOVO/OneDrive/Desktop/REID3/dataset" `
    --arch swin_base `
    --loss-type arcface `
    --arcface-scale 64 `
    --arcface-margin 0.5 `
    --max-epoch 30 `
    --train-batch-size 8 `
    --test-batch-size 32 `
    --optim adam `
    --lr 0.0001 `
    --stepsize 15 25 `
    --gamma 0.1 `
    --margin 0.3 `
    --num-instances 4 `
    --train-sampler RandomIdentitySampler `
    --label-smooth `
    --color-jitter `
    --random-erase `
    --height 224 --width 224 `
    --eval-freq 10 `
    --save-dir "C:/Users/LENOVO/OneDrive/Desktop/REID3/results/exp8-swin-arcface" `
    --gpu-devices 0 -j 0
```

---

## Expected Results Table (fill in as experiments run)

| Exp | Backbone | Loss | Aug | Rank-1 | Rank-5 | mAP |
|-----|----------|------|-----|--------|--------|-----|
| 1 | ResNet50 | Triplet | Both | 76.0% | 88.3% | 58.0% |
| 2 | ResNet50 | ArcFace | Both | ??% | ??% | ??% |
| 3 | ResNet50 | Circle | Both | ??% | ??% | ??% |
| 4 | EfficientNet | Triplet | Both | ??% | ??% | ??% |
| 5 | EfficientNet | ArcFace | Both | ??% | ??% | ??% |
| 6 | EfficientNet | Circle | Both | ??% | ??% | ??% |
| 7 | Swin-Base | Triplet | Both | ??% | ??% | ??% |
| 8 | Swin-Base | ArcFace | Both | ??% | ??% | ??% |
| 9 | Swin-Base | Circle | Both | ??% | ??% | ??% |
| 10 | ResNet50 | Triplet | None | ??% | ??% | ??% |
| 11 | ResNet50 | Triplet | CJ | ??% | ??% | ??% |
| 12 | ResNet50 | Triplet | RE | ??% | ??% | ??% |

---

## Papers Supporting Experiment Design

1. **Hermans et al. 2017** — Triplet Loss with hard mining (your baseline)
2. **Deng et al. 2019 (ArcFace)** — Angular margin improves mAP
3. **Sun et al. 2020 (Circle Loss)** — Adaptive pair weighting
4. **Liu et al. 2021 (Swin)** — Transformer backbone, ICCV Best Paper
5. **Veselý et al. 2023 (WildlifeDatasets)** — Swin+ArcFace best for wildlife
6. **Tan & Le 2019 (EfficientNet)** — Efficient CNN baseline
