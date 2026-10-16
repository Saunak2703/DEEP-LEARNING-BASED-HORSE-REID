# ============================================================
# Horse Re-ID — Single Experiment Runner
# University of Surrey EEEM004
# ============================================================
# USAGE: Edit the variables below and run:
#   .\run_single_experiment.ps1
#
# BACKBONES:
#   resnet50_fc512   → 256x128 input
#   efficientnet_b3  → 256x128 input
#   swin_base        → 224x224 input (IMPORTANT)
#   swin_small       → 224x224 input (IMPORTANT)
#
# LOSS TYPES:
#   triplet   → CrossEntropy + Triplet Loss (standard)
#   arcface   → ArcFace + Triplet Loss
#   circle    → CrossEntropy + Circle Loss
# ============================================================

$env:STUDENT_ID = "6967472"
$env:STUDENT_NAME = "Saunak Ghosh"

# ---- EDIT THESE ----
$ARCH      = "resnet50_fc512"   # backbone
$LOSS      = "triplet"          # triplet | arcface | circle
$HEIGHT    = 256                # 224 for Swin, 256 for others
$WIDTH     = 128                # 224 for Swin, 128 for others
$BATCH     = 16                 # 8 for Swin (memory), 16 for others
$LR        = 0.0003             # 0.0001 for Swin, 0.0003 for others
$AUG_CJ    = $true              # colour jitter on/off
$AUG_RE    = $true              # random erasing on/off
$EPOCHS    = 30
$EXP_NAME  = "exp-custom"
# --------------------

$DATASET = "C:/Users/LENOVO/OneDrive/Desktop/REID3/dataset"
$RESULTS = "C:/Users/LENOVO/OneDrive/Desktop/REID3/results"

# Build augmentation flags
$aug_flags = @()
if ($AUG_CJ) { $aug_flags += "--color-jitter" }
if ($AUG_RE) { $aug_flags += "--random-erase" }

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "Experiment: $EXP_NAME" -ForegroundColor Cyan
Write-Host "Backbone:   $ARCH" -ForegroundColor Cyan
Write-Host "Loss:       $LOSS" -ForegroundColor Cyan
Write-Host "Input:      ${HEIGHT}x${WIDTH}" -ForegroundColor Cyan
Write-Host "Aug:        CJ=$AUG_CJ  RE=$AUG_RE" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan

python main.py `
    -s horse -t horse `
    --root $DATASET `
    --arch $ARCH `
    --loss-type $LOSS `
    --arcface-scale 64 `
    --arcface-margin 0.5 `
    --circle-scale 256 `
    --circle-margin 0.25 `
    --max-epoch $EPOCHS `
    --train-batch-size $BATCH `
    --test-batch-size 64 `
    --optim adam `
    --lr $LR `
    --stepsize 15 25 `
    --gamma 0.1 `
    --margin 0.3 `
    --num-instances 4 `
    --train-sampler RandomIdentitySampler `
    --label-smooth `
    @aug_flags `
    --height $HEIGHT --width $WIDTH `
    --eval-freq 10 --start-eval 0 `
    --save-dir "$RESULTS/$EXP_NAME" `
    --gpu-devices 0 -j 0
