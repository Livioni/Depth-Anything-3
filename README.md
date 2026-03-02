# Training DA3
本篇Readme简要讲解了训练Depth Anything3 的过程。

## 模型文件

以[da3-giant-metric.yaml](src/depth_anything_3/configs/da3-giant-metric.yaml) 为例：

```yaml
__object__:
  path: depth_anything_3.model.da3
  name: DepthAnything3Net
  args: as_params

net:
  __object__:
    path: depth_anything_3.model.dinov2.dinov2
    name: DinoV2
    args: as_params

  name: vitg
  out_layers: [19, 27, 33, 39]
  alt_start: 13
  qknorm_start: 13
  rope_start: 13
  cat_token: True
  scale_token: True

head:
  __object__:
    path: depth_anything_3.model.dualdpt
    name: DualDPT
    args: as_params

  dim_in: &head_dim_in 3072
  output_dim: 2
  features: &head_features 256
  out_channels: &head_out_channels [256, 512, 1024, 1024]

## 以上不用修改，DA3-Giant版本原本的定义。

## 新增添了scale head，用于预测metric scale，目前要求训练的深度数据全部以m为单位，不然没有意义。

scale_head:
  __object__:
    path: depth_anything_3.model.scale_head
    name: ScaleHead
    args: as_params

  dim_in: 1536
  hidden_dim: 1024
  num_layers: 3
  out_dim: 1
  activation: relu
  dropout: 0.0
  reduce: mean
  final_activation: softplus


cam_enc:
  __object__:
    path: depth_anything_3.model.cam_enc
    name: CameraEnc
    args: as_params

  dim_out: 1536

# 下面的内容被删除，因为使用Ray 对camera pose进行解码， gaussian head不用。

# cam_dec:
#   __object__:
#     path: depth_anything_3.model.cam_dec
#     name: CameraDec
#     args: as_params

#   dim_in: 3072

# gs_head:
#   __object__:
#     path: depth_anything_3.model.gsdpt
#     name: GSDPT
#     args: as_params

#   dim_in: *head_dim_in
#   output_dim: 38  # should align with gs_adapter's setting, for gs params
#   features: *head_features
#   out_channels: *head_out_channels


# gs_adapter:
#   __object__:
#     path: depth_anything_3.model.gs_adapter
#     name: GaussianAdapter
#     args: as_params

#   sh_degree: 2
#   pred_color: false  # predict SH coefficient if false
#   pred_offset_depth: true
#   pred_offset_xy: true
#   gaussian_scale_min: 1e-5
#   gaussian_scale_max: 30.0

```


## 训练配置文件

以[da3-giant-train.py](configs/train/da3-giant-train.py) 为例。

```python 
# ======================================================
# DA3 Training Configuration
# ======================================================

# == Common Configuration ==
output_dir = "outputs"
exp_name = "DA3-Giant-adt-col-hoi-rlb-rob-v2"
logging_dir = "logs"

# == Logging Configuration ==
wandb = True
tensorboard = True
report_to = "tensorboard"
num_save_log = 10
num_save_visual = 500
checkpointing_steps = 5000
save_each_epoch = False

# == Model Configuration ==
model_config = "src/depth_anything_3/configs/da3-giant-metric.yaml"
model_checkpoint_path = "checkpoints/da3-giant-1.1/model.safetensors"
model_requires_grad = True
backbone_freeze = False
head_freeze = False
cam_enc_freeze = False
cam_dec_freeze = True #这里无所谓，因为我们已经把decoder 给注释掉了
use_gradient_checkpointing = True   # Enable gradient checkpointing to save memory
use_ray_pose = True  #这里需要为True，我们使用ray 解码
use_gs_infer = False #没用到需要设置为False

# Additional freeze options for memory optimization
gs_head_freeze = True         # Freeze GS head to save memory (if not using 3DGS)
seg_head_freeze = True        # Freeze segmentation head (if not using segmentation)
scale_head_freeze = False      # Freeze scale head if not using scale loss

# ======================================================
# LoRA Configuration (NEW)
# ======================================================
use_lora = False                     # Enable LoRA fine-tuning
lora_r = 32                          # LoRA rank (higher = more parameters, typically 4-32)
lora_alpha = 64                     # LoRA scaling factor (typically 2*lora_r)
lora_dropout = 0.0                  # LoRA dropout rate
lora_bias = "lora_only"             # Bias handling: "none", "all", "lora_only"
lora_target_modules = ["qkv", "out_proj"]       # Target modules in DinoV2 attention
lora_lr = 5e-5                      # Learning rate for LoRA parameters (typically higher than backbone)
# 以上设置用不到，lora微调相关配置


# == Training Configuration ==
mixed_precision = "bf16"  # Options: "no", "fp16", "bf16"
seed = 42
num_train_epochs = 10
gradient_accumulation_steps = 2
max_grad_norm = 1.0
drop_prob = 0.1
pose_condition_prob = 0.2 # 给序列camera gt的概率，按照DA3原文是20%

# == Dataset Configuration ==
train_batch_images = 18 #不要改
num_workers = 8

# == Optimizer Configuration ==
optimizer_type = "adamw"
adam_beta1 = 0.9
adam_beta2 = 0.95
adam_epsilon = 1e-8
adam_weight_decay = 0.01

# == Learning Rate Configuration ==
# Note: When using LoRA, backbone LR is replaced by lora_lr
lr = 2e-5
lr_backbone = 1e-5
lr_head = 2e-5
lr_cam_enc = 2e-5
lr_cam_dec = 2e-5
lr_gs_head = 2e-6

# == Learning Rate Scheduler Configuration ==
lr_scheduler_type = "cosine_with_warmup"
warmup_steps = 1000
eta_min_factor = 0.1  # Minimum learning rate factor for cosine decay

# == Loss Configuration == 
# Camera loss 注释掉，因为用ray loss就够了？
# camera_loss_weight = 0.5 # mainly use ray loss
# camera_loss_type = "l1"  # Options: "l1", "l2", "smooth_l1"

# Ray Loss
ray_loss_weight = 1.0
ray_loss_type = "l1"  # Options: "l1", "l2", "smooth_l1"

# Depth loss
depth_loss_weight = 1.0
depth_gradient_loss_fn = "grad"
depth_valid_range = 0.98

# Scale loss (only active when scale_head_freeze = False)
scale_loss_weight = 1.0
scale_loss_log_space = True

# Gaussian loss (only active when gs_head_freeze = False)
gaussian_loss_weight = 1.0
gaussian_use_conf = False      # Use confidence mask from depth
gaussian_use_mask = True       # Use valid mask from batch
gaussian_use_alpha = False     # Use alpha from gaussian output
gaussian_use_lpips = True     # Use LPIPS perceptual loss
gaussian_lpips_weight = 0.1    # Weight for LPIPS loss

# == Visualization Configuration ==
vis_conf_threshold = 0.2
vis_filter_by_frames = "All"
vis_mask_black_bg = False
vis_mask_white_bg = False
vis_show_cam = True
vis_mask_sky = False
vis_prediction_mode = "Ray+Depth"

# == Resume Configuration ==
resume_model_path = None

# == Dataset Configuration ==
resolution = [(504, 504), (504, 490), (504, 476),
              (504, 462), (504, 448), (504, 434),
              (504, 420), (504, 406), (504, 392),
              (504, 378), (504, 364), (504, 350),
              (504, 336), (504, 322), (504, 308),
              (504, 294), (504, 280) ]

train_dataset = f"  8_000 @ ADT(use_cache = True, quick = False, top_k = 32, dset='', z_far = 50, aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985) \
                 + 10_000 @ Colosseum(use_cache = True, verbose=False, quick = False, top_k = 32, dset='', z_far = 50, aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985) \
                 + 20_000 @ HOI4D(use_cache = True, quick = False, top_k = 32, dset='', z_far = 50, aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985) \
                 + 10_000 @ RLBench(use_cache = True, quick = False, top_k = 32, dset='', z_far = 50, aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985) \
                 + 40_000 @ RoboTwin(use_cache = True, quick = False, top_k = 32, dset='', z_far = 50, aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985)"
                 
test_dataset = None  # Set to None to use same as train_dataset

```

## 启动脚本

```bash
accelerate launch --num_processes=8 train_da3.py --config configs/train/da3-giant-train.py
```

# Evaluation
这部分描述如何测试

## 测试配置文件
以[da3-giant-test-on-rlbench.py](configs/test/da3-giant-test-on-rlbench.py) 为例：

```python
# ======================================================
# Depth Anything 3 Evaluation Configuration
# ======================================================

# == Common ==
output_dir = "eval_output"
exp_name = "DA3-Giant-40k-RayPose-RLBench-test"

# == Model (required by train_utils.load_model) ==
model_config = "src/depth_anything_3/configs/da3-giant-metric.yaml"
model_checkpoint_path = "outputs/DA3-Giant-adt-col-hoi-rlb-rob/checkpoint-3-40000/model.safetensors" # 这里放训练checkpoint

# Evaluation only: no gradients needed
model_requires_grad = False
# Keep LoRA disabled for evaluation
use_lora = False
# Gradient checkpointing is mainly for training; keep it off by default for eval
use_gradient_checkpointing = False 

# == Inference toggles (read by Evaluation/da3_main_table.py) ==
use_ray_pose = True # 注意这里始终为True， 因为我们已经把Cam_dec 删掉了
use_gs_infer = False

# Wether use pose gt
use_pose_condition = False # 这里是是否加pose gt作为输入

# == Runtime ==
num_workers = 8
vis_prediction_mode = "Ray+Depth"  # 不要修改

# == Dataset ==
resolution = [(504, 280)] # 根据数据集来定，504固定，分辨率比例与原图比例一样来确定高度。 例如 720/1280 约等于 280/504 保证280被14整除
seq_len = 10

test_dataset = f"100 @ RLBench(use_cache = False, quick = True, top_k = 32, dset='', specify = True, z_far = 50, aug_crop=16, resolution={resolution}, transform=ImgNorm, seed=985)"


```

## 启动脚本

阅读[da3_main_table.py](Evaluation/da3_main_table.py) 脚本，此脚本给出depth，camera pose和scale的指标。



```bash
python Evaluation/da3_main_table.py --config configs/test/da3-giant-test-on-rlbench.py
```
