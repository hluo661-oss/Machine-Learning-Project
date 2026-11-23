# Surgical Video Segmentation with DINOv2-UNet

Semi-supervised 2D segmentation for surgical videos under limited annotations, illumination variation and instrument occlusion.

The model combines a DINOv2 ViT encoder with a lightweight CNN decoder, trained under a Semantic Mean Teacher framework with a freeze-thaw strategy and weak-to-strong consistency on unlabeled frames.

## Features

- DINOv2 ViT encoder (ViT-S/B/L supported, default ViT-L/14) + lightweight CNN decoder with shallow texture stem
- Semantic Mean Teacher: student trained by gradient, teacher updated by EMA (decay 0.99) for stable pseudo-labels
- Freeze-thaw: freeze encoder for the first 20 epochs, then fine-tune with a smaller encoder LR
- Weak-to-strong consistency: weak vs. strong photometric / geometric augmentation on unlabeled pairs
- Foreground-balanced sampling, AMP, gradient clipping, early stopping
- Sliding-window-free full-image inference with mirror TTA and threshold search

## Repository layout

```text
train.py                 # semi-supervised training entrypoint (Mean Teacher + EMA)
dataset.py               # labeled / unlabeled datasets, mask parsing, augmentation
model_factory.py         # DINOv2-UNet / UNet / UNet++ / FPN / DeepLabV3+ / nnUNet-2D, Dice+Focal loss
utils.py                 # seed, logger, device, metric weighting
generate_predictions.py  # inference + submission JSON generation
visualize_comparison.py  # qualitative comparison figure for multiple checkpoints
requirements.txt
README.md
```

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ and CUDA for training. CPU-only inference is supported via `--device cpu`.

## Data layout

```text
data/surgical_video/
  train/
    <VIDEO_ID>_<FRAMEID:06d>.png
    <VIDEO_ID>_<FRAMEID:06d>_label_bin.png        # or <VIDEO_ID>_<FRAMEID:06d>_png_Label.tar
  val_external/                                   # optional, same format as train
  unlabeled/images/
    *.png
```

Labeled / unlabeled roots can be overridden with `--labeled-root`, `--external-val-root`, `--unlabeled-root`.

## Train

```bash
python train.py --arch dinov2_unet --epochs 150
```

Common options:

```bash
python train.py \
  --arch dinov2_unet \
  --labeled-root ./data/surgical_video/train \
  --unlabeled-root ./data/surgical_video/unlabeled/images \
  --output-dir ./runs/semi_default \
  --freeze-dinov2-epochs 20 \
  --semi-warmup-epochs 20 --unsup-ramp-epochs 30 --unsup-weight 0.6 \
  --ema-decay 0.99
```

Checkpoints, `history.csv`, `config.json` and `final_metrics.json` are saved under `--output-dir`. Best model is `checkpoints/bestmodel.pth`.

## Inference

```bash
python generate_predictions.py \
  --checkpoint ./runs/semi_default/checkpoints/bestmodel.pth \
  --data-dir /path/to/test_images \
  --output-dir ./submission/surgical_video
```

This writes `*_label_bin.png` masks and `predictions.json` under `--output-dir`. Use `--no-tta` to disable mirror TTA, `--device cpu` for CPU inference.

## Visualization

```bash
python visualize_comparison.py \
  --runs-dir ./runs \
  --data-root ./data/surgical_video/train \
  --output ./figures/qualitative_comparison.png \
  --num-cases 4
```

## Loss and metrics

- Supervised: `0.7 * Dice + 0.3 * Focal (gamma=2.0, alpha=0.75)`
- Unsupervised: BCE on high-confidence pseudo-labels (`pos_thr=0.70`, `neg_thr=0.10`, `min_area=80`)
- Validation: Dice (foreground-only), Hausdorff Distance, Average Surface Distance, auto threshold search over `0.2-0.65` with 4-fold flip TTA
