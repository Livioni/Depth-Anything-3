<div align="center">

<h1>DA-Next: Metric-Scale Visual Geometry on Top of Depth Anything 3</h1>

<a href="../README.md"><img src="https://img.shields.io/badge/Parent-SpatialBench-blue" alt="SpatialBench"></a>
<a href="https://github.com/ByteDance-Seed/Depth-Anything-3"><img src="https://img.shields.io/badge/Built_on-Depth_Anything_3-orange" alt="DA3"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-CC--BY-green" alt="License"></a>

</div>

> **Note**: This directory is a **git submodule** of [SpatialBench](../README.md). Environment setup, dataset download, and the evaluation harness are documented in the parent README — they are **not duplicated here**. This file only covers what is specific to DA-Next: model architecture changes on top of DA3, training, and inference.

<div align="center">
  <img src="assets/danext.png" alt="SpatialBench Overview" width="800"/>
</div>


## 🔍 Overview

DA-Next is the variant of [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) used as the **`Ours`** entry in the SpatialBench leaderboard. Compared to vanilla DA3, DA-Next adds:

- A **scale head** that predicts metric scale directly from the backbone features (depth supervision must be in meters).
- A **camera encoder** (`CameraEnc`) that injects optional camera priors into the network.
- **Ray-based pose decoding** (`use_ray_pose = True`) instead of DA3's `CameraDec` head — `cam_dec`, `gs_head`, and `gs_adapter` are removed.
- Training configuration with mixed-precision (`bf16`), gradient checkpointing, dropout for the pose prior, and a multi-resolution schedule.

## 📁 Repository Layout

```
DA-Next/
├── src/depth_anything_3/        # model code (backbone, heads, configs)
│   ├── configs/                 # model YAMLs (e.g. da3-giant-metric.yaml)
│   └── model/                   # network modules: da3, dinov2, dualdpt, scale_head, cam_enc
├── src/datasets/                # SpatialBench-compatible dataset readers
├── configs/
│   ├── train/                   # training configs (Python)
│   └── test/                    # evaluation configs (Python)
├── train_da3.py                 # training entry point
├── train_utils.py               # training utilities (loaders, schedulers, losses)
├── infer.py                     # minimal inference example
├── demo.py                      # interactive demo with viser / GLB export
├── visual_util.py               # GLB / point-cloud / PCA visualization helpers
└── requirements.txt
```

## 🔧 Setup

The base SpatialBench environment already provides PyTorch and most dependencies. Install the few DA-Next-specific extras on top:

```bash
# from the DA-Next/ directory
pip install -r requirements.txt
```

Place DA3 pretrained weights at `checkpoints/da3-giant-1.1/model.safetensors` (or update `model_checkpoint_path` in your config).

## 🚀 Quick Start

### Python API

```python
import glob, os, torch
from depth_anything_3.api import DepthAnything3
from safetensors.torch import load_file

# 1) Build the DA-Next network with the matching backbone size
api = DepthAnything3(model_name="da3-giant")

# 2) Load fine-tuned weights
sd = load_file("outputs/DA3-Giant-test/checkpoint-0-5000/model.safetensors")
api.model.load_state_dict(sd, strict=False)
api = api.to("cuda").eval()

# 3) Run inference on a folder of images
images = sorted(glob.glob("datasets/test/1/*.png"))
prediction = api.inference(
    images,
    export_format="glb-depth_vis",
    export_dir="output_vis_ft",
    use_ray_pose=True,
)

# prediction.depth        : (N, H, W) float32 — metric meters
# prediction.conf         : (N, H, W) float32
# prediction.extrinsics   : (N, 3, 4) float32 — OpenCV w2c / COLMAP convention
# prediction.intrinsics   : (N, 3, 3) float32
```

A runnable version of the above lives in [infer.py](infer.py).

### Interactive demo

```bash
python demo.py --image_folder example/office/images --use_ray_pose
```

`demo.py` exports a viewable `.glb` and starts a viser server for interactive inspection.

## 🏋️ Training

### Model Configuration

Model architecture is described by a YAML file under [src/depth_anything_3/configs/](src/depth_anything_3/configs/). The reference config for DA-Next-Giant is [da3-giant-metric.yaml](src/depth_anything_3/configs/da3-giant-metric.yaml). Key differences from upstream DA3-Giant:

```yaml
# Backbone (unchanged — DA3-Giant defaults)
net:
  __object__:
    path: depth_anything_3.model.dinov2.dinov2
    name: DinoV2
  name: vitg
  out_layers: [19, 27, 33, 39]
  alt_start: 13
  qknorm_start: 13
  rope_start: 13
  cat_token: True
  scale_token: True

# Dual DPT depth head (unchanged)
head:
  __object__:
    path: depth_anything_3.model.dualdpt
    name: DualDPT
  dim_in: 3072
  output_dim: 2
  features: 256
  out_channels: [256, 512, 1024, 1024]

# NEW: scale head — predicts metric scale.
# Requires all training depth to be in meters.
scale_head:
  __object__:
    path: depth_anything_3.model.scale_head
    name: ScaleHead
  dim_in: 1536
  hidden_dim: 1024
  num_layers: 3
  out_dim: 1
  activation: relu
  final_activation: softplus
  reduce: mean

# NEW: camera encoder — injects optional camera priors.
cam_enc:
  __object__:
    path: depth_anything_3.model.cam_enc
    name: CameraEnc
  dim_out: 1536

# REMOVED in DA-Next:
# - cam_dec     (pose is decoded from rays instead → use_ray_pose=True)
# - gs_head     (no 3D Gaussian Splatting head)
# - gs_adapter
```

### Training Config

Training hyperparameters are defined in a Python file under [configs/train/](configs/train/). Reference: [da3-giant-train.py](configs/train/da3-giant-train.py).

| Group | Key parameters |
|-------|----------------|
| **Common** | `output_dir`, `exp_name`, `logging_dir` |
| **Logging** | `wandb`, `tensorboard`, `checkpointing_steps`, `num_save_visual` |
| **Model** | `model_config`, `model_checkpoint_path`, `use_gradient_checkpointing`, `use_ray_pose=True`, `use_gs_infer=False` |
| **Freeze** | `backbone_freeze`, `head_freeze`, `cam_enc_freeze`, `scale_head_freeze`, `gs_head_freeze=True`, `seg_head_freeze=True` |
| **LoRA** (optional) | `use_lora`, `lora_r`, `lora_alpha`, `lora_target_modules=["qkv", "out_proj"]`, `lora_lr` |
| **Training** | `mixed_precision="bf16"`, `num_train_epochs`, `gradient_accumulation_steps`, `drop_prob`, `pose_condition_prob=0.2` (DA3 default) |
| **Optimizer** | `optimizer_type="adamw"`, `adam_beta1=0.9`, `adam_beta2=0.95`, `adam_weight_decay=0.01` |
| **Learning rates** | `lr`, `lr_backbone`, `lr_head`, `lr_cam_enc`, `lr_cam_dec`, `lr_gs_head` |
| **Scheduler** | `lr_scheduler_type="cosine_with_warmup"`, `warmup_steps`, `eta_min_factor` |
| **Losses** | `ray_loss_weight`, `depth_loss_weight`, `scale_loss_weight`, `gaussian_loss_weight` (only active if `gs_head_freeze=False`) |
| **Dataset** | `train_batch_images=18` (fixed), `num_workers`, multi-resolution `resolution=[(504, H) ...]` |

> Keep `use_ray_pose=True` and `use_gs_infer=False` for DA-Next — the cam decoder and GS head have been removed.

Dataset string follows the SpatialBench / CUT3R mixing syntax, for example:

```python
train_dataset = (
    "  8_000 @ ADT(...) "
    "+ 10_000 @ Colosseum(...) "
    "+ 20_000 @ HOI4D(...) "
    "+ 10_000 @ RLBench(...) "
    "+ 40_000 @ RoboTwin(...)"
)
```

Each entry is `N @ DatasetClass(use_cache=True, top_k=32, z_far=50, resolution=..., transform=ColorJitter, seed=985)`. The dataset readers live under [src/datasets/](src/datasets/) and follow the [SpatialBench data layout](../README.md#-dataset-coverage).

### Launch

Single node, 8 GPUs:

```bash
accelerate launch --num_processes=8 train_da3.py \
    --config configs/train/da3-giant-train.py
```

Debug / smoke configs are provided in [configs/train/](configs/train/):
- `da3-debug.py`, `da3-giant-train-debug.py`, `da3-giant-large-debug.py` — small-batch overfit configs
- `da3-large-train.py`, `da3-large-seg-train.py` — DA3-Large variants

## 🧪 Evaluation

Reference test config: [configs/test/da3-giant-test-on-rlbench.py](configs/test/da3-giant-test-on-rlbench.py). Important flags:

```python
# Model
model_config           = "src/depth_anything_3/configs/da3-giant-metric.yaml"
model_checkpoint_path  = "outputs/DA3-Giant-adt-col-hoi-rlb-rob/checkpoint-3-40000/model.safetensors"
model_requires_grad    = False
use_gradient_checkpointing = False
use_lora               = False

# Inference toggles
use_ray_pose           = True          # always True — cam_dec was removed
use_gs_infer           = False
use_pose_condition     = False         # whether to feed GT pose as input

# Runtime
num_workers            = 8
vis_prediction_mode    = "Ray+Depth"   # do not change

# Dataset
resolution             = [(504, 280)]  # keep 504 fixed; second dim must be a multiple of 14
seq_len                = 10
test_dataset           = "100 @ RLBench(..., resolution=[(504, 280)], transform=ImgNorm, seed=985)"
```

Resolution rule: width is fixed to 504. Pick the height so that the aspect ratio matches the source frames and the value is divisible by 14 (e.g. 720/1280 ≈ 280/504).

For end-to-end leaderboard evaluation against all SpatialBench datasets, use the parent harness:

```bash
# from the SpatialBench root
python benchmark/evaluation/run_benchmark.py \
    --config benchmark/configs/end2end/danext_eval.yaml
```

This wires DA-Next into the unified scene index and produces per-scene depth / pose / point-cloud metrics — see the parent [README](../README.md#-quick-start) for details.

## 🙏 Acknowledgments

DA-Next is built on top of [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) by ByteDance Seed and follows the multi-modality training paradigm of [OmniVGGT](https://github.com/Livioni/OmniVGGT-official). Dataset preprocessing reuses the [CUT3R](https://github.com/CUT3R/CUT3R) layout. Visualization uses [viser](https://github.com/nerfstudio-project/viser).
