# Masking-Clusters-in-Vision-language-Pretraining

This is the official repository for "Masking Clusters in Vision-language Pretraining", presented at CVPR 2024.

## Getting Started

To begin pretraining with our method, ensure you have all the necessary dependencies installed, see `requirements.txt` for details. Please also have `torch` and `torchvision` installed.

### Pretraining Script

Below is a sample script for pretraining the model on CC12M with 8 GPUs on a single node. This script outlines the basic usage and required parameters to start the training process. Make sure to replace `"path/to/cc12m"` and `"path/to/imagenet/val"` with the actual paths to your training and validation datasets, respectively.

```bash 
OMP_NUM_THREADS=1 \
torchrun --nproc_per_node 8 -m training.main \
    --save-frequency 1 \
    --train-num-samples 10968539 \
    --train-data="path/to/cc12m" \
    --warmup 2000 \
    --imagenet-val="path/to/imagenet/val" \
    --batch-size=256 \
    --epochs=32 \
    --lr=5e-4 \
    --workers=8 \
    --model ViT-B-16 \
    --seed 0 \
    --force-patch-dropout 0.03 \
    --target_mask_ratio 0.5 \
    --min_mask_ratio 0.3 \
    --local-loss \
    --gather-with-grad

```