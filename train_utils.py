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
    
    # Set requires_grad
    model.requires_grad_(cfg.get("model_requires_grad", True))
    
    # print learnable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / Total parameters: {total_params:,}")
    print(f"[Model Params] Trainable: {trainable_params:,} | Total: {total_params:,}")
    
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
    param_groups = []
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
        
    # if cfg.get("cam_enc_freeze", False):
    #     for param in model.cam_enc.parameters():
    #         param.requires_grad = False
    #     logger.info("Camera encoder parameters are frozen.")
    # else:
    #     param_groups.append({
    #         "params": model.cam_enc.parameters(),
    #         "lr": cfg.get("lr_cam_enc"),
    #         "name": "cam_enc"
    #     })
    
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
    
    exclude_keys = ["backbone", "head", "cam_enc", "cam_dec"]
    
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
        logger.info(f"  Group {i} ({pg['name']}): lr={pg['lr']}")
    
    return optimizer


def build_loss_criterion(cfg: Any) -> MultitaskLoss:
    """
    Build multi-task loss criterion.
    
    Args:
        cfg: Configuration object
        
    Returns:
        MultitaskLoss instance
    """
    criterion = MultitaskLoss(
        camera={
            "weight": cfg.get("camera_loss_weight", 5.0),
            "loss_type": cfg.get("camera_loss_type", "l1")
        },
        depth={
            "weight": cfg.get("depth_loss_weight", 1.0),
            "gradient_loss_fn": cfg.get("depth_gradient_loss_fn", "grad"),
            "valid_range": cfg.get("depth_valid_range", 0.98)
        },
    )
    
    logger.info("Loss criterion initialized:")
    logger.info(f"  Camera loss weight: {cfg.get('camera_loss_weight', 5.0)}")
    logger.info(f"  Depth loss weight: {cfg.get('depth_loss_weight', 1.0)}")
    
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