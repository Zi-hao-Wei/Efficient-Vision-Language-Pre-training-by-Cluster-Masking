# Masking Clusters in Vision-language Pretraining

This is the official repository for "Masking Clusters in Vision-language Pretraining", presented at CVPR 2024.

## Introduction

The quest for optimal vision-language pretraining strategies has led to  the exploration of masking techniques as way to enhance data efficiency. Previous approaches include random masking and semantic masking, the  latter requiring the retention or exclusion of patches in areas with similar semantics. Despite its effectiveness, semantic masking often  needs an additional, complex model for identifying semantically related patches, increasing computational demands. Our method utilizes naturally emerging clusters within images unlike other approaches using text  supervision. We employ random clusters of image patches for masking,  utilizing the raw RGB values of patches as the feature representation.  This method capitalizes on the observation that basic visual similarity  measures can effectively identify coherent visual structures, such as  parts of objects. Our approach, therefore, combines the computational efficiency of random patch dropping with the enhanced performance achieved through masking coherent visual structures.

## Getting Started

To begin pretraining with our method, ensure you have all the necessary dependencies installed, see `requirements.txt` for details. Please also have `torch` and `torchvision` installed.

### Data

Our model is pretrained using CC12M, which is downloaded the [img2dataset](https://github.com/rom1504/img2dataset) library, which directly downloaded the dataset in the webdataset format. The model can be theoretically trained on any image-language pair dataset, and please refer to the [open-clip](https://github.com/mlfoundations/open_clip) repository for more details. 


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



### Evaluation

The representation learnt by our method is evaluated thoroughly through the [clip-benchmark]( https://github.com/LAION-AI/CLIP_benchmark#how-to-use). Please refer to clip-benchmark for building `models.txt` and downloading the `webdatatsets.txt` . After building the required input txt, you could evaluate using the following scripts.

**Zero-shot image text retrieval**

```bash
clip_benchmark eval --pretrained_model models.txt \
    --dataset "webdatasets.txt" \
    --task zeroshot_retrieval \
    --recall_k 1 5 10 \
    --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
    --output "benchmark_{dataset}_{pretrained}_{model}_{language}_{task}.json"
```

**Zero-shot classification**

```bash
clip_benchmark eval --pretrained_model models.txt \
    --dataset "webdatasets.txt" \
    --task zeroshot_classification \
    --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
    --output "benchmark_{dataset}_{pretrained}_{model}_{language}_{task}.json"
```

**Linear-probing**

```bash
clip_benchmark eval --pretrained_model models.txt \
    --dataset "webdatasets.txt" \
    --task linear_probe \
    --dataset_root "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main" \
    --output "benchmark_{dataset}_{pretrained}_{model}_{language}_{task}.json"
```

For testing the learnt correspondence, we utilize [SUGAR-CREPE](https://github.com/RAIVNLab/sugar-crepe) to perform a language composition test.  After setting up the dataset following the instructions, you could evaluate by using this Script.

**Language composition**

```bash
python main_eval.py --model ViT-B-16 \ 
    --pretrained /path_to_your_checkpoints \
    --output ./output \ 
    --coco_image_root ./data/coco/images/val2017/ \
    --data_root ./data/
```

## Acknowledgments

Our code base is developed based on MAE and Open-clip's repository. We highly appreciate their excellent work.

### Citing

```
@inproceedings{clustermasking,
  title={Masking Clustersin Vision-language Pretraining},
  author={Wei, Zihao and Pan, Zixuan and Owens, Andrew},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```
