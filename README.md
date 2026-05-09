# Vision Transformer for CIFAR-10 Image Classification

This repository contains a CIFAR-10 image classification project built for Assignment 3.

The main goal is to train a **Vision Transformer from scratch** on CIFAR-10 and compare it against two baselines:

1. Hybrid CNN + MLP
2. ResNet18 transfer learning

The project includes training scripts, evaluation scripts, a local inference demo, plots, metrics, and a notebook version of the implementation.

## Project Overview

CIFAR-10 is a 10-class image classification dataset containing 32x32 color images.

This project compares three model families:

| Model | Description |
|---|---|
| Vision Transformer | Transformer-based image classifier implemented from scratch |
| Hybrid CNN + MLP | CNN feature extractor followed by an MLP classifier |
| ResNet18 Transfer Learning | ImageNet-pretrained ResNet18 adapted for CIFAR-10 |

## Dataset

Dataset used:

```text
CIFAR-10 Python Dataset
```

Official dataset link:

```text
https://www.cs.toronto.edu/~kriz/cifar.html
```

The project expects the extracted CIFAR-10 Python dataset folder to exist locally.

Expected local structure:

```text
cifar-10-batches-py/
├── batches.meta
├── data_batch_1
├── data_batch_2
├── data_batch_3
├── data_batch_4
├── data_batch_5
├── test_batch
└── readme.html
```

The dataset folder is **not included** in this repository because it is a standard external dataset.

In the original local setup, the dataset was stored at:

```text
D:\cifar-10-python\cifar-10-batches-py
```

The code uses:

```python
torchvision.datasets.CIFAR10(root=data_root, download=False)
```

So before running the project, download and extract CIFAR-10 manually, then place `cifar-10-batches-py/` in the project root or pass a custom path using `--data-root`.

## Repository Structure

```text
vit-cifar10-transformer-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── train_all.py
├── evaluate_all.py
├── local_demo.py
│
├── src/
│   ├── config.py
│   ├── data.py
│   ├── eval_utils.py
│   ├── train_utils.py
│   ├── visualization.py
│   └── models/
│       ├── vit.py
│       ├── hybrid_cnn_mlp.py
│       └── resnet_transfer.py
│
├── notebooks/
│   └── vit_cifar10_assignment.ipynb
│
└── outputs/
    ├── metrics/
    ├── plots/
    ├── report_assets/
    └── reports/
```

## Models

### 1. Vision Transformer from Scratch

The Vision Transformer implementation includes:

- Patch embedding using `Conv2d`
- Learnable class token
- Learnable positional embeddings
- Custom multi-head self-attention
- Transformer encoder blocks
- MLP classification head

### 2. Hybrid CNN + MLP

The Hybrid CNN + MLP baseline includes:

- Convolutional feature extraction
- Batch normalization
- ReLU activations
- Pooling layers
- Fully connected MLP classifier

This model does not use self-attention or Transformer blocks.

### 3. ResNet18 Transfer Learning

The ResNet18 model uses:

```python
torchvision.models.resnet18(weights="IMAGENET1K_V1")
```

The final fully connected layer is replaced for CIFAR-10 classification.

The workflow supports:

- Frozen backbone training
- Optional `layer4` fine-tuning
- CIFAR-10 evaluation with ImageNet normalization

## Results

Final reported results:

| Model | Test Accuracy | F1 Score |
|---|---:|---:|
| Vision Transformer | 0.7536 | 0.7513 |
| Hybrid CNN + MLP | 0.8992 | 0.8997 |
| ResNet18 Transfer Learning | 0.8598 | 0.8595 |

The Hybrid CNN + MLP achieved the strongest performance on CIFAR-10 in this experiment. The Vision Transformer was implemented from scratch and performed reasonably, but CNN-based inductive bias remained more effective on this small image dataset.

## Outputs Included

The repository keeps lightweight result evidence such as:

```text
outputs/metrics/
outputs/plots/
outputs/report_assets/
outputs/reports/
```

These folders may include:

- Accuracy curves
- Loss curves
- Confusion matrices
- Correct prediction grids
- Incorrect prediction grids
- Augmentation samples
- Local demo prediction image
- Metrics summary files

Large model checkpoints are not included.

## Installation

Create or activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

## Main Commands

### Smoke Test

Run a quick small-data test:

```bash
python train_all.py --smoke-test
```

### Fast ViT Hyperparameter Tuning

```bash
python train_all.py --tune-vit-fast --smoke-test
```

### Full Training

```bash
python train_all.py
```

### Evaluate Saved Checkpoints

```bash
python evaluate_all.py
```

### Local Demo

Example using the Hybrid CNN + MLP checkpoint:

```bash
python local_demo.py --model hybrid --checkpoint outputs/checkpoints/hybrid_best.pt --num-images 16
```

Example using ResNet18:

```bash
python local_demo.py --model resnet --checkpoint outputs/checkpoints/resnet_best.pt --num-images 16
```

Example using ViT:

```bash
python local_demo.py --model vit --checkpoint outputs/checkpoints/vit_best.pt --num-images 16
```

## Notes

The following files and folders are intentionally excluded from the repository:

```text
cifar-10-batches-py/
outputs/checkpoints/
*.pt
*.pth
*.ckpt
*.npz
*.npy
__pycache__/
.ipynb_checkpoints/
```

Reason: datasets, model checkpoints, raw prediction arrays, and cache files are generated or externally available.

## Author

Muhammad Atlas Malik  
Department of Data Science  
FAST NUCES Islamabad
