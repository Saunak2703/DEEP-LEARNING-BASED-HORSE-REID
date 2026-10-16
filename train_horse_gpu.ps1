$env:STUDENT_ID = "6967472"
$env:STUDENT_NAME = "Saunak Ghosh"

python main.py `
    -s horse `
    -t horse `
    --root "C:/Users/LENOVO/OneDrive/Desktop/REID3/dataset" `
    --arch resnet50_fc512 `
    --max-epoch 60 `
    --train-batch-size 16 `
    --test-batch-size 64 `
    --optim adam `
    --lr 0.0003 `
    --stepsize 20 40 `
    --gamma 0.1 `
    --margin 0.3 `
    --num-instances 4 `
    --train-sampler RandomIdentitySampler `
    --label-smooth `
    --color-jitter `
    --random-erase `
    --height 256 `
    --width 128 `
    --eval-freq 10 `
    --start-eval 0 `
    --save-dir "C:/Users/LENOVO/OneDrive/Desktop/REID3/results/horse-resnet50-gpu" `
    --gpu-devices 0 `
    -j 0
