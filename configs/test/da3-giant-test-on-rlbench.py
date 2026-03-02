# ======================================================
# Depth Anything 3 Evaluation Configuration
# ======================================================

output_dir = "eval_output"
exp_name = "DA3-Giant-40k-RayPose-RLBench-test"

# == Model (required by train_utils.load_model) ==
model_config = "src/depth_anything_3/configs/da3-giant-metric.yaml"
model_checkpoint_path = "outputs/DA3-Giant-adt-col-hoi-rlb-rob/checkpoint-3-40000/model.safetensors"

# Evaluation only: no gradients needed
model_requires_grad = False
# Keep LoRA disabled for evaluation
use_lora = False
# Gradient checkpointing is mainly for training; keep it off by default for eval
use_gradient_checkpointing = False

# == Inference toggles (read by Evaluation/da3_main_table.py) ==
use_ray_pose = True
use_gs_infer = False

# Wether use pose gt
use_pose_condition = False

# == Runtime ==
num_workers = 8
vis_prediction_mode = "Ray+Depth"

# == Dataset ==
resolution = [(504, 280)]
seq_len = 10

test_dataset = f"100 @ RLBench(use_cache = False, quick = True, top_k = 32, dset='', specify = True, z_far = 50, aug_crop=16, resolution={resolution}, transform=ImgNorm, seed=985)"

