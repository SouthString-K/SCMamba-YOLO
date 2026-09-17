<div align="center">

# SCMamba-YOLO

**[IcaMal2026] Edge-Guided Mamba Object Detection for Underwater Submarine Cable Perception**

![Python 3.11](https://img.shields.io/badge/Python-3.11-green)
![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-blue)
![CUDA 11.8+](https://img.shields.io/badge/CUDA-11.8%2B-orange)

</div>

**SCMamba-YOLO** is an object detection framework for **underwater submarine cable perception**, built on [Ultralytics](https://github.com/ultralytics/ultralytics) and the [Mamba-YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO) state-space detector. It replaces the standard backbone with **edge-guided visual state-space (VSS) blocks** (`EGVSSBlock`), which couple Sobel-based boundary-aware enhancement, local topology aggregation, and range-scanning SS2D with gated feature refinement — a design tailored to thin, elongated cable structures in cluttered underwater scenes. For low-visibility imagery, enhancement-enabled variants place a pretrained [CDF-UIE](Enhancement-main/) image-enhancement front-end ahead of the detector.

<div align="center">
  <img src="./Figure.png" width="1000px"/>
  <p><i>Overview of the edge-guided SS2D block (EGSSBlock) used in the SCMamba-YOLO backbone.</i></p>
</div>

## Highlights

- **Edge-guided VSS backbone** — `EGVSSBlock` explicitly reinforces cable boundaries and long-range topology via boundary-aware enhancement and range-scanning state-space modeling.
- **Optional enhancement front-end** — `*-Enhance` configs attach a pretrained CDF-UIE enhancer (`Enhancement-main/`) before the detector for degraded underwater imagery.
- **Three model scales** — T / B / L variants, each with a standard and an enhancement-enabled config, under `ultralytics/cfg/models/scmamba-yolo/`.
- **Ultralytics-native workflow** — standard dataset-yaml format with simple `train.py` / `val.py` entry points.

## Environment Setup

```bash
git clone https://github.com/SouthString-K/SCMamba-YOLO.git
cd SCMamba-YOLO

conda create -n scmambayolo python=3.11 -y
conda activate scmambayolo

pip install torch torchvision torchaudio
pip install seaborn thop timm einops

cd selective_scan
pip install .
cd ..

pip install -v -e .
```

## Dataset Format

Training follows the standard Ultralytics dataset-yaml format:

```yaml
path: /path/to/your/dataset
train: images/train
val: images/val
test: images/test

names:
  0: cable
```

Typical folder structure:

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

## Model Configs

All model yaml files are under:

```text
ultralytics/cfg/models/scmamba-yolo/
```

Available variants:

- `SCMamba-YOLO-T.yaml`
- `SCMamba-YOLO-B.yaml`
- `SCMamba-YOLO-L.yaml`
- `SCMamba-YOLO-T-Enhance.yaml`
- `SCMamba-YOLO-B-Enhance.yaml`
- `SCMamba-YOLO-L-Enhance.yaml`

## Quick Start

### Train the standard detector

```bash
python train.py \
  --task train \
  --data /path/to/your_dataset.yaml \
  --config ultralytics/cfg/models/scmamba-yolo/SCMamba-YOLO-T.yaml \
  --imgsz 640 \
  --epochs 400 \
  --batch_size 16 \
  --device 0
```

### Train with the enhancement front-end

```bash
python train.py \
  --task train \
  --data /path/to/your_dataset.yaml \
  --config ultralytics/cfg/models/scmamba-yolo/SCMamba-YOLO-T-Enhance.yaml \
  --enhance_weights /path/to/interactnet.pth \
  --enhance_freeze \
  --imgsz 640 \
  --epochs 400 \
  --batch_size 16 \
  --device 0
```

### Validate a trained checkpoint

```bash
python val.py \
  --weights /path/to/best.pt \
  --data /path/to/your_dataset.yaml \
  --split test \
  --imgsz 640 \
  --batch 4 \
  --device 0
```

## Notes

- Use `SCMamba-YOLO-*-Enhance.yaml` only when you have a pretrained InteractNet enhancement checkpoint.
- For the released `train.py`, enhancement-enabled configs require `--enhance_weights`. If it is missing, the script will stop and print instructions for obtaining enhancement weights through `Enhancement-main/src/train.py`.
- The enhancement module can still be instantiated without weights for architecture inspection or local debugging, but random initialization is not intended for official results.
- Freezing layers does not reduce the total parameter count; it only reduces the number of trainable parameters during optimization.
- `val.py` writes a summary file to `runs/val/<exp_name>/paper_data.txt`.
- For Mamba-based configs, install `selective_scan` correctly before training or evaluation.

## Acknowledgements

This project is built on top of:

- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [Mamba-YOLO](https://github.com/HZAI-ZJNU/Mamba-YOLO)
- selective scan operators from the VMamba ecosystem
