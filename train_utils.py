"""
Training utility functions for OmniVGGT

This module contains helper functions for training setup, including:
- Dataset building
- Model loading
- Optimizer and scheduler setup
- Loss criterion setup
- Logging configuration

License: MIT
"""

import os
import math
import logging
from typing import Any, Optional, Tuple

import torch
import wandb
import numpy as np
import accelerate
import transformers
from safetensors.torch import load_file
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from accelerate.logging import get_logger
from depth_anything_3.cfg import create_object, load_config
from src.loss import MultitaskLoss
from src.datasets import get_data_loader

# LoRA imports
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger = get_logger(__name__, log_level="INFO")
    logger.warning("peft library not available. LoRA training will be disabled.")

logger = get_logger(__name__, log_level="INFO")


def build_dataset(
    dataset: str,
    batch_size: int,
    num_workers: int,
    test: bool = False
) -> torch.utils.data.DataLoader:
    """
    Build data loader for training or testing.
    
    Args:
        dataset: Dataset configuration string
        batch_size: Batch size
        num_workers: Number of data loading workers
        test: Whether this is a test dataset
        
    Returns:
        DataLoader instance
    """
    split = 'Test' if test else 'Train'
    logger.info(f'Building {split} DataLoader for dataset: {dataset}')
    
    loader = get_data_loader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=True,
        shuffle=not test,
        drop_last=not test
    )
    
    logger.info(f"{split} dataset length: {len(loader)}")
    return loader


def build_cosine_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    eta_min_factor: float = 0.05
) -> LambdaLR:
    """
    Build a learning rate scheduler with linear warmup and cosine decay.
    
    Args:
        optimizer: Optimizer instance
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        base_lr: Base learning rate
        eta_min_factor: Minimum learning rate factor (eta_min = eta_min_factor * base_lr)
        
    Returns:
        LambdaLR scheduler instance
    """
    def lr_lambda(current_step: int) -> float:
        # Linear warmup
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # Cosine decay
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_factor + (1.0 - eta_min_factor) * cosine_decay
    
    return LambdaLR(optimizer, lr_lambda)


def setup_logging(accelerator: accelerate.Accelerator) -> None:
    """Setup logging configuration for all processes."""
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()


def setup_directories(cfg: Any) -> Tuple[str, str]:
    """
    Setup output and logging directories.
    
    Args:
        cfg: Configuration object
        
    Returns:
        Tuple of (save_dir, logging_dir)
    """
    save_dir = os.path.join(cfg.get("output_dir"), cfg.get("exp_name"))
    logging_dir = os.path.join(save_dir, cfg.get("logging_dir"))
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    
    return save_dir, logging_dir


def setup_wandb(cfg: Any, save_dir: str) -> None:
    """Setup Weights & Biases logging."""
    if cfg.get("wandb", False):
        wandb_dir = os.path.join(save_dir, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        
        wandb.init(
            project="DA3",
            name=cfg.get("exp_name"),
            config=cfg.to_dict(),
            dir=wandb_dir,
            settings=wandb.Settings(code_dir=".")
        )
        wandb.run.log_code(".")
        logger.info("WandB logging initialized")


def setup_tensorboard(cfg: Any, save_dir: str) -> Optional[SummaryWriter]:
    """
    Setup TensorBoard logging.
    
    Args:
        cfg: Configuration object
        save_dir: Output directory
        
    Returns:
        SummaryWriter instance or None
    """
    if cfg.get("tensorboard", True):
        tensorboard_log_dir = os.path.join(save_dir, "tensorboard")
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_log_dir)
        logger.info(f"TensorBoard logging initialized at {tensorboard_log_dir}")
        return writer
    return None


def apply_lora_to_backbone(model, cfg):
    """
    Apply LoRA to the backbone of the model.
    
    Args:
        model: DA3 model instance
        cfg: Configuration object
        
    Returns:
        Modified model with LoRA applied to backbone
    """
    if not PEFT_AVAILABLE:
        logger.error("peft library is not installed. Please install it: pip install peft")
        raise ImportError("peft library required for LoRA training")
    
    # Get backbone model(s)
    backbones_to_lora = []
    backbone_names = []
    
    # For DepthAnything3Net
    if hasattr(model, 'backbone'):
        backbones_to_lora.append(('backbone', model.backbone))
        backbone_names.append('backbone')
    
    # For NestedDepthAnything3Net
    if hasattr(model, 'da3') and hasattr(model.da3, 'backbone'):
        backbones_to_lora.append(('da3.backbone', model.da3.backbone))
        backbone_names.append('da3.backbone')
    
    if hasattr(model, 'da3_metric') and hasattr(model.da3_metric, 'backbone'):
        backbones_to_lora.append(('da3_metric.backbone', model.da3_metric.backbone))
        backbone_names.append('da3_metric.backbone')
    
    if not backbones_to_lora:
        logger.warning("No backbone found in model. LoRA will not be applied.")
        return model
    
    logger.info(f"Applying LoRA to backbones: {backbone_names}")
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=cfg.get("lora_r", 8),  # Rank of LoRA matrices
        lora_alpha=cfg.get("lora_alpha", 16),  # Scaling factor
        target_modules=cfg.get("lora_target_modules", ["qkv"]),  # Target attention modules
        lora_dropout=cfg.get("lora_dropout", 0.0),
        bias=cfg.get("lora_bias", "none"),
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    
    logger.info(f"LoRA Configuration:")
    logger.info(f"  r (rank): {lora_config.r}")
    logger.info(f"  alpha: {lora_config.lora_alpha}")
    logger.info(f"  target_modules: {lora_config.target_modules}")
    logger.info(f"  dropout: {lora_config.lora_dropout}")
    
    # Apply LoRA to each backbone
    for name, backbone in backbones_to_lora:
        # First freeze all backbone parameters
        for param in backbone.parameters():
            param.requires_grad = False
        logger.info(f"✓ Froze all parameters in {name}")
        
        # Apply LoRA (this will unfreeze LoRA parameters)
        try:
            # Get the pretrained model from backbone
            if hasattr(backbone, 'pretrained'):
                target_model = backbone.pretrained
            else:
                target_model = backbone
            
            # Apply LoRA using get_peft_model
            lora_model = get_peft_model(target_model, lora_config)
            
            # Replace the pretrained model with LoRA version
            if hasattr(backbone, 'pretrained'):
                backbone.pretrained = lora_model
            else:
                # Replace the backbone directly
                if name == 'backbone':
                    model.backbone = lora_model
                elif name == 'da3.backbone':
                    model.da3.backbone = lora_model
                elif name == 'da3_metric.backbone':
                    model.da3_metric.backbone = lora_model
            
            logger.info(f"✓ Applied LoRA to {name}")
            
            # Print trainable parameters info
            trainable = sum(p.numel() for p in target_model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in target_model.parameters())
            logger.info(f"  {name} trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
            
        except Exception as e:
            logger.error(f"Failed to apply LoRA to {name}: {e}")
            raise
    
    return model


def load_model(cfg: Any, device: torch.device):
    """
    Load and initialize the DA3 model.
    
    Args:
        cfg: Configuration object
        device: Target device
        
    Returns:
        Tuple of (model, weight_dtype)
    """
    logger.info("Initializing Da3 model...")
    model = create_object(load_config(cfg.get("model_config", "src/depth_anything_3/configs/da3-giant.yaml")))

    # Load pretrained weights
    try:
        state_dict = load_file(cfg.get("model_checkpoint_path","checkpoints/da3-giant/model.safetensors"))
        for k in list(state_dict.keys()):
            if k.startswith('model.'):
                state_dict[k[6:]] = state_dict.pop(k)
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        logger.warning(f"Failed to load pretrained weights: {e}")
        logger.warning("Training from scratch...")
    
    # Apply LoRA if configured (before gradient checkpointing)
    if cfg.get("use_lora", False):
        logger.info("=" * 60)
        logger.info("Enabling LoRA for backbone fine-tuning")
        logger.info("=" * 60)
        model = apply_lora_to_backbone(model, cfg)
        
        # Set non-backbone modules trainability based on freeze config
        logger.info("Configuring non-backbone modules based on freeze settings...")
        trainable_modules = []
        frozen_modules = []
        
        # Head
        if hasattr(model, 'head'):
            if not cfg.get("head_freeze", False):
                model.head.requires_grad_(True)
                trainable_modules.append('head')
            else:
                model.head.requires_grad_(False)
                frozen_modules.append('head')
        
        # Camera Decoder
        if hasattr(model, 'cam_dec'):
            if not cfg.get("cam_dec_freeze", False):
                model.cam_dec.requires_grad_(True)
                trainable_modules.append('cam_dec')
            else:
                model.cam_dec.requires_grad_(False)
                frozen_modules.append('cam_dec')
        
        # Camera Encoder
        if hasattr(model, 'cam_enc') and model.cam_enc is not None:
            if not cfg.get("cam_enc_freeze", False):
                model.cam_enc.requires_grad_(True)
                trainable_modules.append('cam_enc')
            else:
                model.cam_enc.requires_grad_(False)
                frozen_modules.append('cam_enc')
        
        # GS Head
        if hasattr(model, 'gs_head') and model.gs_head is not None:
            if not cfg.get("gs_head_freeze", False):
                model.gs_head.requires_grad_(True)
                trainable_modules.append('gs_head')
            else:
                model.gs_head.requires_grad_(False)
                frozen_modules.append('gs_head')
        
        # GS Adapter
        if hasattr(model, 'gs_adapter') and model.gs_adapter is not None:
            if not cfg.get("gs_adapter_freeze", False):
                model.gs_adapter.requires_grad_(True)
                trainable_modules.append('gs_adapter')
            else:
                model.gs_adapter.requires_grad_(False)
                frozen_modules.append('gs_adapter')
        
        # Seg Head
        if hasattr(model, 'seg_head') and model.seg_head is not None:
            if not cfg.get("seg_head_freeze", False):
                model.seg_head.requires_grad_(True)
                trainable_modules.append('seg_head')
            else:
                model.seg_head.requires_grad_(False)
                frozen_modules.append('seg_head')
        
        # For NestedDepthAnything3Net
        if hasattr(model, 'da3'):
            if hasattr(model.da3, 'head') and not cfg.get("head_freeze", False):
                model.da3.head.requires_grad_(True)
                trainable_modules.append('da3.head')
            if hasattr(model.da3, 'cam_dec') and not cfg.get("cam_dec_freeze", False):
                model.da3.cam_dec.requires_grad_(True)
                trainable_modules.append('da3.cam_dec')
            if hasattr(model.da3, 'cam_enc') and not cfg.get("cam_enc_freeze", False):
                model.da3.cam_enc.requires_grad_(True)
                trainable_modules.append('da3.cam_enc')
        
        if hasattr(model, 'da3_metric'):
            if hasattr(model.da3_metric, 'head') and not cfg.get("head_freeze", False):
                model.da3_metric.head.requires_grad_(True)
                trainable_modules.append('da3_metric.head')
        
        if trainable_modules:
            logger.info(f"✓ Trainable modules: {trainable_modules}")
        if frozen_modules:
            logger.info(f"✓ Frozen modules: {frozen_modules}")
        if not trainable_modules and not frozen_modules:
            logger.warning("No non-backbone modules found")
    
    # Enable gradient checkpointing if configured
    if cfg.get("use_gradient_checkpointing", False):
        logger.info("Enabling gradient checkpointing for backbone to save memory...")
        # Enable gradient checkpointing for backbone (DinoV2)
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'pretrained'):
            if hasattr(model.backbone.pretrained, 'enable_gradient_checkpointing'):
                model.backbone.pretrained.enable_gradient_checkpointing()
                logger.info("✓ Gradient checkpointing enabled for backbone")
            else:
                logger.warning("Backbone does not support gradient checkpointing")
        # For NestedDepthAnything3Net
        elif hasattr(model, 'da3') and hasattr(model.da3, 'backbone'):
            if hasattr(model.da3.backbone.pretrained, 'enable_gradient_checkpointing'):
                model.da3.backbone.pretrained.enable_gradient_checkpointing()
                logger.info("✓ Gradient checkpointing enabled for da3 backbone")
            if hasattr(model, 'da3_metric') and hasattr(model.da3_metric, 'backbone'):
                if hasattr(model.da3_metric.backbone.pretrained, 'enable_gradient_checkpointing'):
                    model.da3_metric.backbone.pretrained.enable_gradient_checkpointing()
                    logger.info("✓ Gradient checkpointing enabled for da3_metric backbone")
        else:
            logger.warning("Could not find backbone to enable gradient checkpointing")
    
    # Set requires_grad (skip if using LoRA as it manages this)
    if not cfg.get("use_lora", False):
        model.requires_grad_(cfg.get("model_requires_grad", True))
    
    # Print learnable parameters with detailed breakdown
    logger.info("=" * 60)
    logger.info("Model Parameter Statistics")
    logger.info("=" * 60)
    
    # Overall statistics
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Detailed breakdown by module (only if using LoRA)
    if cfg.get("use_lora", False):
        logger.info("\nDetailed breakdown:")
        
        # Backbone (LoRA) parameters
        if hasattr(model, 'backbone'):
            backbone_params = sum(p.numel() for p in model.backbone.parameters())
            backbone_trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
            logger.info(f"  Backbone: {backbone_trainable:,} / {backbone_params:,} trainable (LoRA)")
        
        # Head parameters
        if hasattr(model, 'head'):
            head_params = sum(p.numel() for p in model.head.parameters())
            head_trainable = sum(p.numel() for p in model.head.parameters() if p.requires_grad)
            logger.info(f"  Head: {head_trainable:,} / {head_params:,} trainable")
        
        # Camera decoder parameters
        if hasattr(model, 'cam_dec'):
            if model.cam_dec:
                cam_dec_params = sum(p.numel() for p in model.cam_dec.parameters())
                cam_dec_trainable = sum(p.numel() for p in model.cam_dec.parameters() if p.requires_grad)
                logger.info(f"  Cam Decoder: {cam_dec_trainable:,} / {cam_dec_params:,} trainable")
        
        # Camera encoder parameters
        if hasattr(model, 'cam_enc'):
            if model.cam_enc:
                cam_enc_params = sum(p.numel() for p in model.cam_enc.parameters())
                cam_enc_trainable = sum(p.numel() for p in model.cam_enc.parameters() if p.requires_grad)
                logger.info(f"  Cam Encoder: {cam_enc_trainable:,} / {cam_enc_params:,} trainable")
        
        # GS head parameters
        if hasattr(model, 'gs_head'):
            if model.gs_head:
                gs_head_params = sum(p.numel() for p in model.gs_head.parameters())
                gs_head_trainable = sum(p.numel() for p in model.gs_head.parameters() if p.requires_grad)
                logger.info(f"  GS Head: {gs_head_trainable:,} / {gs_head_params:,} trainable")
        
        # Seg head parameters
        if hasattr(model, 'seg_head'):
            if model.seg_head:
                seg_head_params = sum(p.numel() for p in model.seg_head.parameters())
                seg_head_trainable = sum(p.numel() for p in model.seg_head.parameters() if p.requires_grad)
                logger.info(f"  Seg Head: {seg_head_trainable:,} / {seg_head_params:,} trainable")
    
    logger.info("=" * 60)
    print(f"[Model Params] Trainable: {trainable_params:,} | Total: {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    # Determine weight dtype
    weight_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    logger.info(f"Using weight dtype: {weight_dtype}")
    
    model.to(device)
    return model, weight_dtype


def build_optimizer(model: torch.nn.Module, cfg: Any) -> torch.optim.Optimizer:
    """
    Build optimizer with parameter groups for different learning rates.
    
    Args:
        model: Model instance
        cfg: Configuration object
        
    Returns:
        Optimizer instance
    """
    use_lora = cfg.get("use_lora", False)
    param_groups = []
    
    if use_lora:
        # When using LoRA, separate LoRA parameters from other trainable modules
        logger.info("Using LoRA mode: Backbone uses LoRA, other modules train normally")
        
        # Collect LoRA parameters (from backbone)
        lora_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                # LoRA parameters typically have 'lora' in their name
                if 'lora' in name.lower() or 'backbone' in name:
                    lora_params.append(param)
                else:
                    other_params.append(param)
        
        # Add LoRA parameter group (backbone with LoRA)
        if lora_params:
            param_groups.append({
                "params": lora_params,
                "lr": cfg.get("lora_lr", cfg.get("lr_backbone", 1e-5)),
                "name": "lora_backbone"
            })
            logger.info(f"LoRA backbone group: {len(lora_params)} parameter tensors")
        
        # Add other modules (head, cam_dec, etc.) with potentially different learning rates
        if other_params:
            # Try to separate different modules
            head_params = []
            cam_dec_params = []
            other_module_params = []
            
            for name, param in model.named_parameters():
                if param.requires_grad and 'lora' not in name.lower() and 'backbone' not in name:
                    if 'head' in name:
                        head_params.append(param)
                    elif 'cam_dec' in name:
                        cam_dec_params.append(param)
                    else:
                        other_module_params.append(param)
            
            if head_params:
                param_groups.append({
                    "params": head_params,
                    "lr": cfg.get("lr_head", cfg.get("lr", 2e-5)),
                    "name": "head"
                })
                logger.info(f"Head group: {len(head_params)} parameter tensors")
            
            if cam_dec_params:
                param_groups.append({
                    "params": cam_dec_params,
                    "lr": cfg.get("lr_cam_dec", cfg.get("lr", 2e-5)),
                    "name": "cam_dec"
                })
                logger.info(f"Camera decoder group: {len(cam_dec_params)} parameter tensors")
            
            if other_module_params:
                param_groups.append({
                    "params": other_module_params,
                    "lr": cfg.get("lr", 2e-5),
                    "name": "other"
                })
                logger.info(f"Other modules group: {len(other_module_params)} parameter tensors")
        
        if not param_groups:
            logger.warning("No trainable parameters found!")
    else:
        # Standard training mode
        if cfg.get("backbone_freeze", False):
            for param in model.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone parameters are frozen.")
        else:
            param_groups.append({
                "params": model.backbone.parameters(),
                "lr": cfg.get("lr_backbone"),
                "name": "backbone"
            })
            
        if cfg.get("head_freeze", False):
            for param in model.head.parameters():
                param.requires_grad = False
            logger.info("Head parameters are frozen.")
        else:
            param_groups.append({
                "params": model.head.parameters(),
                "lr": cfg.get("lr_head"),
                "name": "head"
            })
            
        if cfg.get("cam_enc_freeze", False):
            for param in model.cam_enc.parameters():
                param.requires_grad = False
            logger.info("Camera encoder parameters are frozen.")
        else:
            param_groups.append({
                "params": model.cam_enc.parameters(),
                "lr": cfg.get("lr_cam_enc"),
                "name": "cam_enc"
            })
        
        if cfg.get("cam_dec_freeze", False):
            for param in model.cam_dec.parameters():
                param.requires_grad = False
            logger.info("Camera decoder parameters are frozen.")
        else:
            param_groups.append({
                "params": model.cam_dec.parameters(),
                "lr": cfg.get("lr_cam_dec"),
                "name": "cam_dec"
            })
            
        if model.gs_head:
            if cfg.get("gs_head_freeze", False):
                for param in model.gs_head.parameters():
                    param.requires_grad = False
                logger.info("GS head parameters are frozen.")
            else:
                param_groups.append({
                    "params": model.gs_head.parameters(),
                    "lr": cfg.get("lr_gs_head"),
                    "name": "gs_head"
                })
                
        exclude_keys = ["backbone", "head", "cam_enc", "cam_dec", "gs_head"]
        
        param_groups.append({
            "params": [
                p for n, p in model.named_parameters()
                if not any(k in n for k in exclude_keys)
            ],
            "lr": cfg.get("lr"),
            "name": "other"
        })
    
    optimizer_type = cfg.get("optimizer_type", "adamw").lower()
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(cfg.get("adam_beta1", 0.9), cfg.get("adam_beta2", 0.95)),
            eps=cfg.get("adam_epsilon", 1e-8),
            weight_decay=cfg.get("adam_weight_decay", 0.01)
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    
    logger.info(f"Optimizer created: {optimizer_type}")
    for i, pg in enumerate(param_groups):
        num_params = sum(p.numel() for p in pg['params'])
        logger.info(f"  Group {i} ({pg['name']}): lr={pg['lr']}, params={num_params:,}")
    
    return optimizer


def build_loss_criterion(cfg: Any) -> MultitaskLoss:
    """
    Build multi-task loss criterion.
    
    Args:
        cfg: Configuration object
        
    Returns:
        MultitaskLoss instance
    """
    if not cfg.get("cam_dec_freeze", True):
        camera={
            "weight": cfg.get("camera_loss_weight", 5.0),
            "loss_type": cfg.get("camera_loss_type", "l1")
        },
    else:
        camera=None
        
    if not cfg.get("head_freeze", True):
        depth={
            "weight": cfg.get("depth_loss_weight", 1.0),
            "gradient_loss_fn": cfg.get("depth_gradient_loss_fn", "grad"),
            "valid_range": cfg.get("depth_valid_range", 0.98)
        },
    else:
        depth=None
        
    if cfg.get("use_ray_pose", True):
        ray={
            "weight": cfg.get("ray_loss_weight", 1.0),
            "loss_type": cfg.get("ray_loss_type", "l1")
        },
    else:
        ray=None
        
    if not cfg.get("seg_head_freeze", True):
        seg_mask={
            "weight": cfg.get("seg_mask_loss_weight", 1.0),
            "delta_pull": cfg.get("seg_mask_delta_pull", 0.25),
            "delta_push": cfg.get("seg_mask_delta_push", 1.0),
            "min_mask_pixels": cfg.get("seg_mask_min_mask_pixels", 50)
        },
    else:
        seg_mask=None
        
    if not cfg.get("gs_head_freeze", True):
        gaussian={
            "weight": cfg.get("gaussian_loss_weight", 1.0),
            "use_conf": cfg.get("gaussian_use_conf", False),
            "use_mask": cfg.get("gaussian_use_mask", True),
            "use_alpha": cfg.get("gaussian_use_alpha", False),
            "use_lpips": cfg.get("gaussian_use_lpips", False),
            "lpips_weight": cfg.get("gaussian_lpips_weight", 1.0),
        }
    else:
        gaussian=None
        
    criterion = MultitaskLoss(
        camera=camera,
        depth=depth,
        ray=ray,
        seg_mask=seg_mask,
        gaussian=gaussian
    )
    
    logger.info("Loss criterion initialized:")
    logger.info(f"  Camera loss weight: {cfg.get('camera_loss_weight', 5.0)}")
    logger.info(f"  Depth loss weight: {cfg.get('depth_loss_weight', 1.0)}")
    logger.info(f"  Ray loss weight: {cfg.get('ray_loss_weight', 1.0)}")
    logger.info(f"  Gaussian loss weight: {cfg.get('gaussian_loss_weight', 1.0)}")
    
    return criterion


def select_camera_gt(S, prob= 0.1, rng=None):
    """
    Select camera ground truth indices.
    
    Args:
        S: Sequence length
        rng: Random number generator
    """
    rng = rng or np.random.default_rng()

    # 33% 的概率直接返回空集
    if rng.random() < prob:
        return []
    # 从 1..S 之间随机选取要保留的个数
    k = rng.integers(0, S + 1)
    if k == 0:
        return []

    # 按顺序从 0 开始选取 k 个
    idx = list(range(k))

    return idx

def select_depth_gt(S, prob= 0.1, rng=None):
    rng = rng or np.random.default_rng()

    # 33% 的概率直接返回空集
    if rng.random() < prob:
        return []

    k = rng.integers(0, S + 1)
    if k == 0:
        return []

    idx = rng.choice(S, size=k, replace=False)

    return sorted(idx.tolist())