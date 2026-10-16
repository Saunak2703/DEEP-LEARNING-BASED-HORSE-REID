# Horse Re-ID code base

## Horse Re-ID GPU run

On Windows PowerShell:

```powershell
.\train_horse_gpu.ps1
```

The script expects the dataset at:

```text
C:/Users/LENOVO/OneDrive/Desktop/REID3/dataset/horse/train/HORSE*/
C:/Users/LENOVO/OneDrive/Desktop/REID3/dataset/horse/test/HORSE*/
```

Training logs and checkpoints are saved under:

```text
C:/Users/LENOVO/OneDrive/Desktop/REID3/results/horse-resnet50-gpu
```

For a quick evaluation-only pipeline check:

```powershell
python main.py -s horse -t horse --root "C:/Users/LENOVO/OneDrive/Desktop/REID3/dataset" --arch resnet50_fc512 --max-epoch 1 --train-batch-size 16 --test-batch-size 64 --optim adam --lr 0.0003 --stepsize 1 --num-instances 4 --train-sampler RandomIdentitySampler --height 256 --width 128 --evaluate --save-dir "C:/Users/LENOVO/OneDrive/Desktop/REID3/results/horse-gpu-eval" --gpu-devices 0 -j 0
```
