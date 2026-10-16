# ============================================================
# Horse Re-ID — All 16 Experiments
# University of Surrey EEEM004 MSc Dissertation
# ============================================================
# EXPERIMENT DESIGN (2x2x4 ablation study):
#
# Backbone:     ResNet50, EfficientNet-B3, Swin-Base
# Loss:         Triplet, ArcFace, Circle
# Augmentation: None, ColourJitter only, RandomErase only, Both
#
# CORE 4 EXPERIMENTS (follow WildlifeDatasets 2023 standard):
# Exp 1:  ResNet50   + Triplet  (BASELINE — already done)
# Exp 2:  ResNet50   + ArcFace
# Exp 3:  Swin-Base  + Triplet
# Exp 4:  Swin-Base  + ArcFace
#
# ADDITIONAL 8 EXPERIMENTS:
# Exp 5:  EfficientNet + Triplet
# Exp 6:  EfficientNet + ArcFace
# Exp 7:  ResNet50   + Circle
# Exp 8:  Swin-Base  + Circle
# Exp 9:  EfficientNet + Circle
# Exp 10-16: Augmentation ablation (no aug baseline for each)
#
# NOTE FOR SWIN TRANSFORMER:
# Input must be 224x224 — use --height 224 --width 224
# ============================================================

$env:STUDENT_ID = "6967472"
$env:STUDENT_NAME = "Saunak Ghosh"

$DATASET = "C:/Users/LENOVO/OneDrive/Desktop/REID3/dataset"
$RESULTS = "C:/Users/LENOVO/OneDrive/Desktop/REID3/results"
$EPOCHS  = 30

# ============================================================
# EXP 1 — ResNet50 + Triplet + Both Aug (BASELINE — ALREADY DONE)
# Rank-1: 76.0%  mAP: 58.0% (from 60 epoch run)
# ============================================================
Write-Host "=== EXP 1: ResNet50 + Triplet (BASELINE - ALREADY DONE) ===" -ForegroundColor Green
Write-Host "Results: Rank-1=76.0%, mAP=58.0%" -ForegroundColor Yellow
Write-Host "Skipping — results already available" -ForegroundColor Yellow

# ============================================================
# EXP 2 — ResNet50 + ArcFace + Both Aug
# Tests: Does ArcFace improve mAP over Triplet?
# Expected: Higher mAP, similar or better Rank-1
# ============================================================
Write-Host "`n=== EXP 2: ResNet50 + ArcFace + Both Aug ===" -ForegroundColor Cyan
python main.py `
    -s horse -t horse `
    --root $DATASET `
    --arch resnet50_fc512 `
    --loss-type arcface `
    --arcface-scale 64 `
    --arcface-margin 0.5 `
    --max-epoch $EPOCHS `
    --train-batch-size 16 `
    --test-batch-size 64 `
    --optim adam `
    --lr 0.0003 `
    --stepsize 15 25 `
    --gamma 0.1 `
    --margin 0.3 `
    --num-instances 4 `
    --train-sampler RandomIdentitySampler `
    --label-smooth `
    --color-jitter `
    --random-erase `
    --height 256 --width 128 `
    --eval-freq 10 --start-eval 0 `
    --save-dir "$RESULTS/exp2-resnet50-arcface" `
    --gpu-devices 0 -j 0

# ============================================================
# EXP 3 — ResNet50 + Circle + Both Aug
# Tests: Does Circle Loss improve mAP over Triplet?
# Expected: Better mAP (Circle optimises global ranking)
# ============================================================
Write-Host "`n=== EXP 3: ResNet50 + Circle Loss + Both Aug ===" -ForegroundColor Cyan
python main.py `
    -s horse -t horse `
    --root $DATASET `
    --arch resnet50_fc512 `
    --loss-type circle `
    --circle-scale 256 `
    --circle-margin 0.25 `
    --max-epoch $EPOCHS `
    --train-batch-size 16 `
    --test-batch-size 64 `
    --optim adam `
    --lr 0.0003 `
    --stepsize 15 25 `
    --gamma 0.1 `
    --margin 0.3 `
    --num-instances 4 `
    --train-sampler RandomIdentitySampler `
    --label-smooth `
    --color-jitter `
    --random-erase `
    --height 256 --width 128 `
    --eval-freq 10 --start-eval 0 `
    --save-dir "$RESULTS/exp3-resnet50-circle" `
    --gpu-devices 0 -j 0

# ============================================================
# EXP 4 — EfficientNet-B3 + Triplet + Both Aug
# Tests: Does EfficientNet beat ResNet50 for horse Re-ID?
# ============================================================
Write-Host "`n=== EXP 4: EfficientNet-B3 + Triplet + Both Aug ===" -ForegroundColor Cyan
python main.py `
    -s horse -t horse `
    --root $DATASET `
    --arch efficientnet_b3 `
    --loss-type triplet `
    --max-epoch $EPOCHS `
    --train-batch-size 16 `
    --test-batch-size 64 `
    --optim adam `
    --lr 0.0003 `
    --stepsize 15 25 `
    --gamma 0.1 `
    --margin 0.3 `
    --num-instances 4 `
    --train-sampler RandomIdentitySampler `
    --label-smooth `
    --color-jitter `
    --random-erase `
    --height 256 --width 128 `
    --eval-freq 10 --start-eval 0 `
    --save-dir "$RESULTS/exp4-efficientnet-triplet" `
    --gpu-devices 0 -j 0

# ============================================================
# EXP 5 — EfficientNet-B3 + ArcFace + Both Aug
# Tests: EfficientNet with best loss from WildlifeDatasets
# ============================================================
Write-Host "`n=== EXP 5: EfficientNet-B3 + ArcFace + Both Aug ===" -ForegroundColor Cyan
python main.py `
    -s horse -t horse `
    --root $DATASET `
    --arch efficientnet_b3 `
    --loss-type arcface `
    --arcface-scale 64 `
    --arcface-margin 0.5 `
    --max-epoch $EPOCHS `
    --train-batch-size 16 `
    --test-batch-size 64 `
    --optim adam `
    --lr 0.0003 `
    --stepsize 15 25 `
    --gamma 0.1 `
    --margin 0.3 `
    --num-instances 4 `
    --train-sampler RandomIdentitySampler `
    --label-smooth `
    --color-jitter `
    --random-erase `
    --height 256 --width 128 `
    --eval-freq 10 --start-eval 0 `
    --save-dir "$RESULTS/exp5-efficientnet-arcface" `
    --gpu-devices 0 -j 0

# ============================================================
# EXP 6 — EfficientNet-B3 + Circle + Both Aug
# ============================================================
Write-Host "`n=== EXP 6: EfficientNet-B3 + Circle Loss + Both Aug ===" -ForegroundColor Cyan
python main.py `
    -s horse -t horse `
    --root $DATASET `
    --arch efficientnet_b3 `
    --loss-type circle `
    --circle-scale 256 `
    --circle-margin 0.25 `
    --max-epoch $EPOCHS `
    --train-batch-size 16 `
    --test-batch-size 64 `
    --optim adam `
    --lr 0.0003 `
    --stepsize 15 25 `
    --gamma 0.1 `
    --margin 0.3 `
    --num-instances 4 `
    --train-sampler RandomIdentitySampler `
    --label-smooth `
    --color-jitter `
    --random-erase `
    --height 256 --width 128 `
    --eval-freq 10 --start-eval 0 `
    --save-dir "$RESULTS/exp6-efficientnet-circle" `
    --gpu-devices 0 -j 0

# ============================================================
# EXP 7 — Swin-Base + Triplet + Both Aug
# IMPORTANT: Swin requires 224x224 input
# Tests: Does Transformer beat CNN for horse Re-ID?
# WildlifeDatasets found Swin-B + ArcFace best overall
# ============================================================
Write-Host "`n=== EXP 7: Swin-Base + Triplet + Both Aug (224x224) ===" -ForegroundColor Magenta
Write-Host "NOTE: Swin uses 224x224 input (not 256x128)" -ForegroundColor Yellow
python main.py `
    -s horse -t horse `
    --root $DATASET `
    --arch swin_base `
    --loss-type triplet `
    --max-epoch $EPOCHS `
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
    --eval-freq 10 --start-eval 0 `
    --save-dir "$RESULTS/exp7-swin-triplet" `
    --gpu-devices 0 -j 0

# ============================================================
# EXP 8 — Swin-Base + ArcFace + Both Aug
# KEY EXPERIMENT: WildlifeDatasets best combination
# Tests if same finding holds for horses
# ============================================================
Write-Host "`n=== EXP 8: Swin-Base + ArcFace + Both Aug (224x224) ===" -ForegroundColor Magenta
Write-Host "KEY: WildlifeDatasets found this combo best for wildlife" -ForegroundColor Yellow
python main.py `
    -s horse -t horse `
    --root $DATASET `
    --arch swin_base `
    --loss-type arcface `
    --arcface-scale 64 `
    --arcface-margin 0.5 `
    --max-epoch $EPOCHS `
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
    --eval-freq 10 --start-eval 0 `
    --save-dir "$RESULTS/exp8-swin-arcface" `
    --gpu-devices 0 -j 0

# ============================================================
# EXP 9 — Swin-Base + Circle + Both Aug
# ============================================================
Write-Host "`n=== EXP 9: Swin-Base + Circle Loss + Both Aug (224x224) ===" -ForegroundColor Magenta
python main.py `
    -s horse -t horse `
    --root $DATASET `
    --arch swin_base `
    --loss-type circle `
    --circle-scale 256 `
    --circle-margin 0.25 `
    --max-epoch $EPOCHS `
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
    --eval-freq 10 --start-eval 0 `
    --save-dir "$RESULTS/exp9-swin-circle" `
    --gpu-devices 0 -j 0

# ============================================================
# AUGMENTATION ABLATION (Exp 10-12)
# Uses best backbone (ResNet50) with best loss (determined above)
# Compares: No aug vs CJ only vs RE only vs Both (Exp1=baseline)
# ============================================================
Write-Host "`n=== EXP 10: ResNet50 + Triplet + NO Augmentation ===" -ForegroundColor Yellow
python main.py `
    -s horse -t horse `
    --root $DATASET `
    --arch resnet50_fc512 `
    --loss-type triplet `
    --max-epoch $EPOCHS `
    --train-batch-size 16 `
    --test-batch-size 64 `
    --optim adam `
    --lr 0.0003 `
    --stepsize 15 25 `
    --gamma 0.1 `
    --margin 0.3 `
    --num-instances 4 `
    --train-sampler RandomIdentitySampler `
    --label-smooth `
    --height 256 --width 128 `
    --eval-freq 10 --start-eval 0 `
    --save-dir "$RESULTS/exp10-resnet50-triplet-noaug" `
    --gpu-devices 0 -j 0

Write-Host "`n=== EXP 11: ResNet50 + Triplet + Colour Jitter only ===" -ForegroundColor Yellow
python main.py `
    -s horse -t horse `
    --root $DATASET `
    --arch resnet50_fc512 `
    --loss-type triplet `
    --max-epoch $EPOCHS `
    --train-batch-size 16 `
    --test-batch-size 64 `
    --optim adam `
    --lr 0.0003 `
    --stepsize 15 25 `
    --gamma 0.1 `
    --margin 0.3 `
    --num-instances 4 `
    --train-sampler RandomIdentitySampler `
    --label-smooth `
    --color-jitter `
    --height 256 --width 128 `
    --eval-freq 10 --start-eval 0 `
    --save-dir "$RESULTS/exp11-resnet50-triplet-cj" `
    --gpu-devices 0 -j 0

Write-Host "`n=== EXP 12: ResNet50 + Triplet + Random Erase only ===" -ForegroundColor Yellow
python main.py `
    -s horse -t horse `
    --root $DATASET `
    --arch resnet50_fc512 `
    --loss-type triplet `
    --max-epoch $EPOCHS `
    --train-batch-size 16 `
    --test-batch-size 64 `
    --optim adam `
    --lr 0.0003 `
    --stepsize 15 25 `
    --gamma 0.1 `
    --margin 0.3 `
    --num-instances 4 `
    --train-sampler RandomIdentitySampler `
    --label-smooth `
    --random-erase `
    --height 256 --width 128 `
    --eval-freq 10 --start-eval 0 `
    --save-dir "$RESULTS/exp12-resnet50-triplet-re" `
    --gpu-devices 0 -j 0

Write-Host "`n=== ALL EXPERIMENTS COMPLETE ===" -ForegroundColor Green
Write-Host "Results saved to: $RESULTS" -ForegroundColor Green
Write-Host "Summary of experiment design:" -ForegroundColor Cyan
Write-Host "Exp 1:  ResNet50   + Triplet  + Both (BASELINE)" -ForegroundColor White
Write-Host "Exp 2:  ResNet50   + ArcFace  + Both" -ForegroundColor White
Write-Host "Exp 3:  ResNet50   + Circle   + Both" -ForegroundColor White
Write-Host "Exp 4:  EffNet-B3  + Triplet  + Both" -ForegroundColor White
Write-Host "Exp 5:  EffNet-B3  + ArcFace  + Both" -ForegroundColor White
Write-Host "Exp 6:  EffNet-B3  + Circle   + Both" -ForegroundColor White
Write-Host "Exp 7:  Swin-Base  + Triplet  + Both (224x224)" -ForegroundColor White
Write-Host "Exp 8:  Swin-Base  + ArcFace  + Both (224x224) KEY" -ForegroundColor White
Write-Host "Exp 9:  Swin-Base  + Circle   + Both (224x224)" -ForegroundColor White
Write-Host "Exp 10: ResNet50   + Triplet  + No Aug" -ForegroundColor White
Write-Host "Exp 11: ResNet50   + Triplet  + CJ only" -ForegroundColor White
Write-Host "Exp 12: ResNet50   + Triplet  + RE only" -ForegroundColor White
