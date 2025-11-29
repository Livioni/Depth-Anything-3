# ======================================================
# OmniVGGT Training Configuration
# ======================================================

# == Common Configuration ==
output_dir = "outputs"
exp_name = "DA3-Large-Ray"
logging_dir = "logs"

# == Logging Configuration ==
wandb = False
tensorboard = True
report_to = "tensorboard"
num_save_log = 10
num_save_visual = 1000
checkpointing_steps = 10000
save_each_epoch = False

# == Model Configuration ==
model_config = "src/depth_anything_3/configs/da3-large.yaml"
model_checkpoint_path = "checkpoints/da3-large/model.safetensors"
model_requires_grad = True
backbone_freeze = False
head_freeze = False
cam_enc_freeze = False
cam_dec_freeze = False
use_gradient_checkpointing = True   # Enable gradient checkpointing to save memory

# == Training Configuration ==
mixed_precision = "no"  # Options: "no", "fp16", "bf16"
seed = 42
num_train_epochs = 5
gradient_accumulation_steps = 2
max_grad_norm = 1.0
drop_prob = 0.1
check_unused_parameters = True  # Set to True to check for unused parameters during training
pose_condition_prob = 0.2

# == Dataset Configuration ==
train_batch_images = 18
num_workers = 8

# == Optimizer Configuration ==
optimizer_type = "adamw"
adam_beta1 = 0.9
adam_beta2 = 0.95
adam_epsilon = 1e-8
adam_weight_decay = 0.01

# == Learning Rate Configuration ==
lr = 2e-5
lr_backbone = 1e-5
lr_head = 2e-5
lr_cam_enc = 2e-5
lr_cam_dec = 2e-5

# == Learning Rate Scheduler Configuration ==
lr_scheduler_type = "cosine_with_warmup"
warmup_steps = 1000
eta_min_factor = 0.1  # Minimum learning rate factor for cosine decay

# == Loss Configuration ==
# Camera loss
camera_loss_weight = 5.0
camera_loss_type = "l1"  # Options: "l1", "l2", "smooth_l1"

# Ray Loss
ray_loss_weight = 1.0
ray_loss_type = "l1"  # Options: "l1", "l2", "smooth_l1"

# Depth loss
depth_loss_weight = 1.0
depth_gradient_loss_fn = "grad"
depth_valid_range = 0.98

# == Visualization Configuration ==
vis_conf_threshold = 0.2
vis_filter_by_frames = "All"
vis_mask_black_bg = False
vis_mask_white_bg = False
vis_show_cam = True
vis_mask_sky = False
vis_prediction_mode = "Predicted Depth"

# == Resume Configuration ==
resume_model_path = None

# == Dataset Configuration ==
resolution = [(504, 504), (504, 490), (504, 476),
              (504, 462), (504, 448), (504, 434),
              (504, 420), (504, 406), (504, 392),
              (504, 378), (504, 364), (504, 350),
              (504, 336), (504, 322), (504, 308),
              (504, 294), (504, 280) ]

train_dataset = f"20000 @ Scannetppv2(use_cache = True, quick = False, top_k = 64, dset='', z_far = 50, aug_crop=16, resolution={resolution}, transform=ColorJitter, seed=985)"
test_dataset = None  # Set to None to use same as train_dataset

