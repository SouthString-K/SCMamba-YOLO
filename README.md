# SCMamba-YOLO

SCMamba-YOLO is a practical training and evaluation codebase for object detection built on Ultralytics and selective scan operators. It is mainly designed for submarine cable detection in complex underwater scenes, and can also be adapted to other custom detection tasks with the same training pipeline. This repository supports training, validation, testing, resume training, warm-start loading, and optional layer freezing through a single entry script. In typical use, you prepare a dataset yaml file, select a model configuration from `ultralytics/cfg/models/scmamba-yolo/`, train with `train.py`, and export paper-style evaluation statistics with `val.py`.

## Repository Structure

```text
.
├── train.py
├── selective_scan/
├── ultralytics/
│   ├── cfg/
│   │   ├── datasets/
│   │   └── models/
│   │       └── scmamba-yolo/
│   └── nn/
├── asserts/
```

## Environment Setup

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd SCMamba-YOLO
```

### 2. Create a conda environment

```bash
conda create -n scmambayolo python=3.11 -y
conda activate scmambayolo
```

### 3. Install PyTorch

Please install the PyTorch version that matches your CUDA environment. For example:

```bash
pip install torch torchvision torchaudio
```

### 4. Install dependencies

```bash
pip install seaborn thop timm einops
cd selective_scan
pip install .
cd ..
pip install -v -e .
```

## Dataset Preparation

Training follows the standard Ultralytics dataset-yaml format. Prepare a dataset yaml file such as:

```yaml
path: /path/to/your/dataset
train: images/train
val: images/val
test: images/test

names:
  0: cable
```

A typical folder structure is:

```text
dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

## Available Model Configurations

Model yaml files are located in:

```text
ultralytics/cfg/models/scmamba-yolo/
```

Examples:

- `ultralytics/cfg/models/scmamba-yolo/SCMamba-YOLO-T.yaml`
## Quick Start

### Train from a yaml configuration

```bash
python train.py \
  --task train \
  --data /path/to/your_dataset.yaml \
  --config ultralytics/cfg/models/scmamba-yolo/SCMamba-YOLO-T.yaml \
  --imgsz 640 \
  --epochs 400 \
  --batch_size 16 \
  --device 0 \
  --project runs/train \
  --name scmambayolo_t
```

### Resume interrupted training

`train.py` will resume automatically if the checkpoint path given by `--resume` exists.

```bash
python train.py \
  --task train \
  --data /path/to/your_dataset.yaml \
  --config ultralytics/cfg/models/scmamba-yolo/SCMamba-YOLO-T.yaml \
  --resume runs/train/scmambayolo_t/weights/last.pt \
  --device 0
```

## Common Arguments

`train.py` supports the following frequently used options:

- `--data`: dataset yaml path
- `--config`: model yaml path
- `--weights`: optional checkpoint for warm-start loading
- `--task`: `train`, `val`, or `test`
- `--imgsz`: input image size
- `--epochs`: training epochs
- `--batch_size`: batch size
- `--device`: GPU id such as `0`, `0,1`, or `cpu`
- `--workers`: dataloader workers
- `--optimizer`: `SGD`, `Adam`, or `AdamW`
- `--freeze`: freeze the first `n` model layers during training
- `--amp`: enable automatic mixed precision
- `--project`: save directory
- `--name`: experiment name
- `--resume`: checkpoint path for resume training


### Validation with `val.py`

`val.py` is a standalone evaluation script for exporting statistics, including:

- GFLOPs
- parameter count
- preprocessing / inference / postprocessing time
- FPS
- class-wise Precision / Recall / F1 / mAP
- overall metrics summary

Run it with:

```bash
python val.py \
  --weights /path/to/best.pt \
  --data /path/to/your_dataset.yaml \
  --split test \
  --imgsz 640 \
  --batch 4 \
  --device 0 \
  --project runs/val \
  --name scmambayolo_val
```

The current script writes the final summary to:

```text
runs/val/<exp_name>/paper_data.txt
```

## Notes

- `--config` is used to define the model structure from a yaml file.
- `--weights` is optional and is used to load an existing `.pt` checkpoint before training.
- Freezing layers does not change the total parameter count of the model; it only reduces the number of trainable parameters during optimization.
- For Mamba-based configs, make sure `selective_scan` is installed correctly before training or evaluation.

## Acknowledgements

This project is built on top of:

- [Ultralytics](https://github.com/ultralytics/ultralytics)
- selective scan operators from the VMamba ecosystem
