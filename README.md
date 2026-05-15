# SegSlum

Official repository of **SegSlum**, a semi-supervised learning framework for slum detection from satellite imagery.

> **Minimally supervised learning on sub-meter satellite imagery reveals slum expansion during the COVID-19 pandemic**  
> Jeasurk Yang, Sungwon Park, Hyoshin Kim, Sumin Lee, Jungwon Kim, Jihee Kim, Meeyoung Cha  
> *Communications Sustainability*, 2026 · [Paper](https://www.nature.com/articles/s44458-026-00054-6) · [Project Page](https://jeasurk-yang.github.io/SEGSLUM/)

---

## Overview

SegSlum detects informal settlements (slums) from 60 cm/pixel RGB satellite imagery using a semi-supervised self-training pipeline. It requires only a small number of labeled images (~3% of the dataset) to achieve mIoU > 0.9 across diverse cities and time periods.

Key features:
- Semi-supervised learning with adaptive pseudo-label selection
- GRAM-matrix texture consistency regularization for photometric robustness
- Multi-temporal and cross-national slum detection
- Applied to 12 cities across Asia, Africa, and Latin America (2014–2024)

This codebase builds on [ST++](https://github.com/LiheYoung/ST-PlusPlus) (Yang et al., CVPR 2022).

---

## Environment

Tested with Python 3.8+ and CUDA-enabled PyTorch.

```bash
pip install -r requirements.txt
```

Dependencies: `torch`, `torchvision`, `numpy`, `pandas`, `scipy`, `scikit-image`, `opencv-python`, `Pillow`, `tqdm`

---

## Data

### Directory layout

`--data-root` must point to a directory with this structure:

```
<DATA_ROOT>/
├── image/
│   └── <id>.png        # 256×256 RGB satellite tile (60 cm/pixel)
└── label/
    └── <id>.tif        # binary mask (0 = non-slum, 1 = slum)
```

Image IDs (one per line, without extension) are listed in split files under [`dataset/splits/`](dataset/splits/). Each subfolder corresponds to one city/region (e.g. `BRA_NEW_multiyear`, `KEN_NEW_multiyear`).

### Downloads

| Resource | Link |
|---|---|
| Label dataset | [Download](https://drive.google.com/drive/folders/1FaEUNF5DNl5omALe2HWnihziONOY6ySP?usp=drive_link) |
| Model checkpoints | [Download](https://drive.google.com/drive/folders/1fdRhi-YdDI5rSxklTdthPH657STGRQn1?usp=drive_link) |

### Pretrained backbones

Place backbone checkpoints in [`pretrained/`](pretrained/):

| File | Download |
|---|---|
| `resnet50.pth` | https://download.pytorch.org/models/resnet50-0676ba61.pth |
| `resnet101.pth` | https://download.pytorch.org/models/resnet101-63fe2227.pth |

---

## Training

```bash
CUDA_VISIBLE_DEVICES=0,1 python -W ignore main.py \
  --dataset <DATASET_NAME> \
  --data-root <PATH_TO_DATASET_ROOT> \
  --batch-size 16 \
  --backbone resnet50 \
  --model deeplabv3plusaux \
  --labeled-id-path   dataset/splits/<DATASET_NAME>/<SPLIT>/labeled.txt \
  --unlabeled-id-path dataset/splits/<DATASET_NAME>/<SPLIT>/unlabeled.txt \
  --pseudo-mask-path  outdir/pseudo_masks/<EXPERIMENT_NAME> \
  --save-path         outdir/models/<EXPERIMENT_NAME> \
  --reliable-id-path  outdir/reliable_ids/<EXPERIMENT_NAME> \
  --plus
```

- `--plus` enables ST++ selective re-training. Omit for plain ST.
- `--nogram` disables GRAM-matrix consistency (baseline).

### CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--data-root` | required | Root directory containing `image/` and `label/` |
| `--dataset` | required | Name of the split folder under `dataset/splits/` |
| `--labeled-id-path` | required | Text file listing labeled image IDs |
| `--unlabeled-id-path` | required | Text file listing unlabeled image IDs |
| `--pseudo-mask-path` | required | Output path for pseudo masks |
| `--save-path` | required | Output path for model checkpoints |
| `--reliable-id-path` | ST++ only | Output path for reliable/unreliable ID lists |
| `--batch-size` | 16 | Batch size |
| `--lr` | 0.01 | Initial learning rate |
| `--epochs` | 100 | Epochs per stage |
| `--crop-size` | 256 | Input crop size |
| `--backbone` | resnet50 | `resnet50` or `resnet50_whitening` |
| `--model` | deeplabv3plusaux | `deeplabv3plusaux`, `deeplabv3plus`, `deeplabv3plusauxrgl`, `deeplabv2`, `pspnet` |
| `--plus` | — | Enable ST++ selective re-training |
| `--nogram` | — | Disable GRAM-matrix consistency |

---

## Citation

```bibtex
@article{yang2026segslum,
  title     = {Minimally supervised learning on sub-meter satellite imagery reveals slum expansion during the {COVID}-19 pandemic},
  author    = {Yang, Jeasurk and Park, Sungwon and Kim, Hyoshin and Lee, Sumin and Kim, Jungwon and Kim, Jihee and Cha, Meeyoung},
  journal   = {Communications Sustainability},
  volume    = {1},
  pages     = {52},
  year      = {2026},
  publisher = {Nature Publishing Group},
  doi       = {10.1038/s44458-026-00054-6}
}
```

## Acknowledgement

This codebase is adapted from **ST++**:
> Lihe Yang, Wei Zhuo, Lei Qi, Yinghuan Shi, Yang Gao. *ST++: Make Self-training Work Better for Semi-supervised Semantic Segmentation.* CVPR 2022. [[paper]](https://arxiv.org/abs/2106.05095) [[code]](https://github.com/LiheYoung/ST-PlusPlus)

## License

[MIT](LICENSE)
