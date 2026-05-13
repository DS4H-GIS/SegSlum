# SegSlum

Semi-supervised semantic segmentation for slum detection from satellite imagery.

This repository builds on the [ST++](https://github.com/LiheYoung/ST-PlusPlus) framework
(Yang et al., CVPR 2022) and adapts it to binary slum / non-slum segmentation with
optional GRAM-matrix regularization between weakly and strongly augmented views.

## Environment

Tested with Python 3.8+ and CUDA-enabled PyTorch.

```bash
pip install -r requirements.txt
```

Dependencies:

- `torch`, `torchvision`
- `numpy`, `pandas`, `scipy`
- `scikit-image`, `opencv-python`, `Pillow`
- `tqdm`

## Data preparation

### Directory layout

`--data-root` must point to a directory with this structure:

```
<DATA_ROOT>/
├── image/
│   └── <id>.png        # input satellite tile
└── label/
    └── <id>.tif        # binary ground-truth mask (0 = non-slum, 1 = slum)
```

The `<id>` strings are listed (one per line, without extension) in the split
text files under [`dataset/splits/`](dataset/splits/). Each subfolder there
corresponds to one dataset / region (e.g. `BRA_NEW_multiyear`, `KEN_NEW_multiyear`).

### Pretrained backbones

Place the backbone checkpoints in [`pretrained/`](pretrained/):

| File | Download |
|---|---|
| `resnet50.pth` | https://download.pytorch.org/models/resnet50-0676ba61.pth |
| `resnet101.pth` | https://download.pytorch.org/models/resnet101-63fe2227.pth |
| `deeplabv2_resnet101_coco_pretrained.pth` | [link](https://drive.google.com/file/d/14be0R1544P5hBmpmtr8q5KeRAvGunc6i/view?usp=sharing) (only needed for `--model deeplabv2`) |

These files are not tracked in git (see [`.gitignore`](.gitignore)).

## Training

Replace the `<...>` placeholders with values for your run:

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

- `--plus` enables the ST++ selective re-training stages. Omit it to run plain ST.
- `--nogram` disables the GRAM-matrix consistency term and runs the baseline.

### CLI arguments

| Argument | Required | Description |
|---|---|---|
| `--data-root` | yes | Root directory containing `image/` and `label/` |
| `--dataset` | yes | Name of the split folder under `dataset/splits/` |
| `--labeled-id-path` | yes | Text file listing labeled image IDs |
| `--unlabeled-id-path` | yes | Text file listing unlabeled image IDs |
| `--pseudo-mask-path` | yes | Where pseudo masks are saved |
| `--save-path` | yes | Where model checkpoints are saved |
| `--reliable-id-path` | for ST++ | Where reliable / unreliable ID lists are written |
| `--batch-size` | no (16) | Batch size |
| `--lr` | no (0.01) | Initial learning rate |
| `--epochs` | no (100) | Number of epochs per stage |
| `--crop-size` | no (256) | Input crop size |
| `--backbone` | no | `resnet50` or `resnet50_whitening` |
| `--model` | no | `deeplabv3plusaux` (default), `deeplabv3plus`, `deeplabv3plusauxrgl`, `deeplabv2`, `pspnet` |
| `--plus` | no | Enable ST++ selective re-training |
| `--nogram` | no | Disable GRAM-matrix consistency |

## Notes on hardcoded assumptions

The code currently targets **binary** slum segmentation at **256x256** inputs.
The following are hardcoded; change them if you adapt to a different problem:

- `class_num = 2` ([main.py:151](main.py#L151))
- `nclass=2` in model construction ([main.py:287,289](main.py#L287))
- input size `[256, 256]` for train/val/strong augmentations ([main.py:143-145](main.py#L143-L145))
- `args.crop_size = 256` override ([main.py:481](main.py#L481))
- `torch.clamp(mask, max=1)` forces the GT mask to binary ([dataset/semi.py:75,89,97](dataset/semi.py#L75))

## Acknowledgement

This codebase is adapted from **ST++** by Yang et al.:

> Lihe Yang, Wei Zhuo, Lei Qi, Yinghuan Shi, Yang Gao.
> *ST++: Make Self-training Work Better for Semi-supervised Semantic Segmentation.* CVPR 2022.
> [[paper]](https://arxiv.org/abs/2106.05095) [[code]](https://github.com/LiheYoung/ST-PlusPlus)

```bibtex
@inproceedings{st++,
  title={ST++: Make Self-training Work Better for Semi-supervised Semantic Segmentation},
  author={Yang, Lihe and Zhuo, Wei and Qi, Lei and Shi, Yinghuan and Gao, Yang},
  booktitle={CVPR},
  year={2022}
}
```

## License

See [LICENSE](LICENSE).
