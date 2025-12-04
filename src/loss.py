# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from lpips import LPIPS
from einops import rearrange
from src.utils.pose_enc import extri_intri_to_pose_encoding
from src.train_utils.general import check_and_fix_inf_nan
from math import ceil, floor


def convert_to_buffer(module: nn.Module, persistent: bool = True):
    """
    Convert all parameters and buffers in a module to buffers.
    This makes them automatically follow the module when moved to different devices.
    
    Args:
        module: The module to convert
        persistent: Whether buffers should be persistent (saved in state_dict)
    """
    # Recurse over child modules
    for name, child in list(module.named_children()):
        convert_to_buffer(child, persistent)
    
    # Convert parameters and buffers to buffers
    for name, parameter_or_buffer in (
        *module.named_parameters(recurse=False),
        *module.named_buffers(recurse=False),
    ):
        value = parameter_or_buffer.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)


@dataclass(eq=False)
class MultitaskLoss(torch.nn.Module):
    """
    Multi-task loss module that combines different loss types for VGGT.
    
    Supports:
    - Camera loss
    - Depth loss 
    - Point loss
    - Tracking loss (not cleaned yet, dirty code is at the bottom of this file)
    """
    def __init__(self, camera=None, depth=None, point=None, ray=None, seg_mask=None, gaussian=None, **kwargs):
        super().__init__()
        # Loss configuration dictionaries for each task
        self.camera = camera
        self.depth = depth
        self.point = point
        self.ray = ray
        self.seg_mask = seg_mask
        self.gaussian = gaussian
        
        # Initialize LPIPS model if gaussian loss uses LPIPS
        self.lpips_model = None
        if gaussian is not None and gaussian.get("use_lpips", False):
            self.lpips_model = LPIPS(net="vgg")
            # Set to eval mode and disable gradients
            self.lpips_model.eval()
            # Convert all parameters to buffers so they automatically move with the module
            convert_to_buffer(self.lpips_model, persistent=False)

    def forward(self, predictions, batch, step=0) -> torch.Tensor:
        """
        Compute the total multi-task loss.
        
        Args:
            predictions: Dict containing model predictions for different tasks
            batch: Dict containing ground truth data and masks
            
        Returns:
            Dict containing individual losses and total objective
        """
        total_loss = 0
        loss_dict = {}
        
        # Camera pose loss - if pose encodings are predicted
        if "extrinsics" in predictions and "intrinsics" in predictions and self.camera is not None:
            camera_loss_dict = compute_camera_loss(predictions, batch, **self.camera)   
            camera_loss = camera_loss_dict["loss_camera"] * self.camera["weight"]   
            total_loss = total_loss + camera_loss
            loss_dict.update(camera_loss_dict)
        
        # Depth estimation loss - if depth maps are predicted
        if "depth" in predictions and self.depth is not None:
            depth_loss_dict = compute_depth_loss(predictions, batch, **self.depth)
            depth_loss = depth_loss_dict["loss_conf_depth"] + depth_loss_dict["loss_reg_depth"] + depth_loss_dict["loss_grad_depth"]
            depth_loss = depth_loss * self.depth["weight"]
            total_loss = total_loss + depth_loss
            loss_dict.update(depth_loss_dict)
            
        if "feat" in predictions and self.seg_mask is not None:
            seg_mask_loss_dict = embedmask_contrastive_loss(predictions, batch, **self.seg_mask)
            seg_mask_loss = seg_mask_loss_dict["loss_seg_mask"] * self.seg_mask["weight"]
            total_loss = total_loss + seg_mask_loss
            loss_dict.update(seg_mask_loss_dict)

        if "ray" in predictions and self.ray is not None:
            ray_loss_dict = compute_ray_loss(predictions, batch, **self.ray)
            ray_loss = ray_loss_dict["loss_ray"] * self.ray["weight"]
            total_loss = total_loss + ray_loss
            loss_dict.update(ray_loss_dict) 
            
        if "gs_rendered" in predictions and self.gaussian is not None:
            gaussian_loss_dict = compute_gaussian_loss(predictions, batch, lpips_model=self.lpips_model, **self.gaussian)
            gaussian_loss = gaussian_loss_dict["loss_gaussian_l1"] * self.gaussian["weight"] + \
                            gaussian_loss_dict["loss_gaussian_lpips"] * self.gaussian["lpips_weight"]
            total_loss = total_loss + gaussian_loss
            loss_dict.update(gaussian_loss_dict)
            
            #TODO
            # gaussian_cosist = compute_consist_loss(predictions, batch, **self.gaussian)
        
        loss_dict["objective"] = total_loss

        return loss_dict

def embedmask_contrastive_loss(predictions, batch, delta_pull=0.25, 
                               delta_push=1.0, min_mask_pixels=50, **kwargs):
    """
    Compute contrastive loss (pull & push) for instance segmentation from feature embeddings.
    
    Pull loss: encourages pixels within the same instance to have similar embeddings (cluster tightly)
    Push loss: encourages embeddings from different instances to be far apart (inter-cluster separation)

    Args:
        predictions: Dict containing 'feat' - feature embeddings of shape (B, N, H, W, C)
                     where B=batch, N=views, H=height, W=width, C=channels
        batch: Dict containing:
               - 'gt_mask': tensor of shape (B*N, H, W, num_instances) with bool masks
                            value=1 indicates pixel belongs to that instance
        delta_pull: margin for pull loss (pixels within instance should be closer than this)
        delta_push: margin for push loss (instance centers should be farther than this)
        min_mask_pixels: minimum number of valid pixels for a mask to be considered

    Returns:
        Dict containing:
            - 'loss_pull': pull loss (intra-instance compactness)
            - 'loss_push': push loss (inter-instance separation)
            - 'loss_seg_mask': total contrastive loss (pull + push)
    """
    features = predictions['feat']  # (B, N, H, W, C)
    gt_masks = batch['gt_mask']  # (B*N, H, W, num_instances)
    
    features = check_and_fix_inf_nan(features, "features")
    
    B, N, H, W, C = features.shape
    device = features.device
    
    # Reshape features: (B, N, H, W, C) -> (B*N, H, W, C) -> (B*N, H*W, C)
    features = features.reshape(B * N, H, W, C)
    features = features.reshape(B * N, H * W, C)
    
    # gt_masks shape: (B*N, H, W, num_instances)
    B, N, H_gt, W_gt, num_instances = gt_masks.shape
    
    # Resize gt_masks if needed
    if H_gt != H or W_gt != W:
        # (B*N, H_gt, W_gt, num_instances) -> (B*N, num_instances, H_gt, W_gt)
        gt_masks = gt_masks.permute(0, 3, 1, 2)
        # Resize to (B*N, num_instances, H, W)
        gt_masks = F.interpolate(
            gt_masks.float(), 
            size=(H, W), 
            mode='nearest'
        ).bool()
        # (B*N, num_instances, H, W) -> (B*N, H, W, num_instances)
        gt_masks = gt_masks.permute(0, 2, 3, 1)
    
    # Reshape gt_masks: (B*N, H, W, num_instances) -> (B*N, H*W, num_instances)
    gt_masks_flat = gt_masks.reshape(B * N, H * W, num_instances)
    
    total_pull_loss = 0.0
    total_push_loss = 0.0
    num_valid_samples = 0

    # Process each sample in B*N
    for idx in range(B * N):
        feat = features[idx]  # (H*W, C)
        masks = gt_masks_flat[idx]  # (H*W, num_instances)
        
        instance_means = []
        pull_loss = 0.0
        valid_instances = 0

        # Pull loss: cluster pixels within each instance
        for i in range(num_instances):
            mask_flat = masks[:, i]  # (H*W,)
            num_pixels = mask_flat.sum()
            
            if num_pixels < min_mask_pixels:
                continue  # Skip tiny or invalid instances
            
            pixel_feats = feat[mask_flat.bool()]  # (num_pixels, C)
            mean_feat = pixel_feats.mean(dim=0, keepdim=True)  # (1, C)
            
            # L2 distance from pixels to their instance center
            distances = torch.norm(pixel_feats - mean_feat, p=2, dim=1)  # (num_pixels,)
            
            # Hinge loss: penalize distances beyond delta_pull
            pull = F.relu(distances - delta_pull).mean()
            pull_loss += pull
            
            instance_means.append(mean_feat)
            valid_instances += 1
        
        if valid_instances == 0:
            continue
        
        pull_loss = pull_loss / valid_instances
        total_pull_loss += pull_loss

        # Push loss: separate different instance centers
        if len(instance_means) > 1:
            means = torch.cat(instance_means, dim=0)  # (num_valid_instances, C)

            # Compute pairwise distances between instance centers
            # Convert to float32 for cdist computation if using bf16 (cdist_cuda doesn't support bf16)
            means_dtype = means.dtype
            if means_dtype == torch.bfloat16:
                dist_mat = torch.cdist(means.float(), means.float(), p=2).to(means_dtype)
            else:
                dist_mat = torch.cdist(means, means, p=2)  # (num_valid_instances, num_valid_instances)
            
            # Mask out diagonal (self-distances)
            eye = torch.eye(len(means), device=device).bool()
            
            # Hinge loss: penalize pairs closer than delta_push
            push_loss = F.relu(delta_push - dist_mat[~eye]).mean()
            total_push_loss += push_loss
        
        num_valid_samples += 1

    # Average over all valid samples
    if num_valid_samples > 0:
        avg_pull = total_pull_loss / num_valid_samples
        avg_push = total_push_loss / num_valid_samples
    else:
        avg_pull = torch.tensor(0.0, device=device)
        avg_push = torch.tensor(0.0, device=device)
    
    total_loss = avg_pull + avg_push

    return {
        'loss_pull': avg_pull,
        'loss_push': avg_push,
        'loss_seg_mask': total_loss
    }
    

def compute_ray_loss(predictions, batch, weight_origins=1.0, weight_directions=1.0, **kwargs):
    """
    Compute L1 loss for ray origins and directions.
    
    Args:
        predictions: Dict containing 'ray' predictions
        batch: Dict containing ground truth 'ray_origins' and 'ray_directions'
        weight_origins: Weight for ray origins loss
        weight_directions: Weight for ray directions loss
    
    Returns:
        Dict containing individual ray losses and total ray loss
    """
    pred_ray = predictions['ray']
    
    # Assuming pred_ray contains both origins and directions
    # pred_ray shape: (B, ..., 6) where first 3 dims are origins, last 3 are directions
    pred_origins = pred_ray[..., :3]
    pred_directions = pred_ray[..., 3:]

    gt_origins = batch['ray_origins']
    gt_directions = batch['ray_directions']
    
    # Check for invalid values
    gt_origins = check_and_fix_inf_nan(gt_origins, "gt_ray_origins", hard_max=None)
    gt_directions = check_and_fix_inf_nan(gt_directions, "gt_ray_directions", hard_max=None)
    
    # Compute L1 loss for origins
    loss_origins = (pred_origins - gt_origins).abs().mean()
    loss_origins = check_and_fix_inf_nan(loss_origins, "loss_ray_origins")
    
    # Compute L1 loss for directions
    loss_directions = (pred_directions - gt_directions).abs().mean()
    loss_directions = check_and_fix_inf_nan(loss_directions, "loss_ray_directions")
    
    # Compute weighted total loss
    total_ray_loss = loss_origins * weight_origins + loss_directions * weight_directions
    
    return {
        "loss_ray": total_ray_loss,
        "loss_ray_origins": loss_origins,
        "loss_ray_directions": loss_directions
    }


def compute_gaussian_loss(predictions, batch, use_conf=False, use_mask=False,
                            use_alpha=False, depth_dict=None, use_lpips=False, 
                            lpips_model=None, **kwargs):
    """
    Compute L1 loss and optionally LPIPS loss for Gaussian splatting rendered images.
    
    Args:
        predictions: Dict containing 'gaussian' with 'gs_output' that has 'alpha' and 'color'
        batch: Dict containing ground truth 'images' and 'valid_mask'
        use_conf: Whether to use confidence mask from depth_dict
        use_mask: Whether to use valid_mask from batch
        use_alpha: Whether to use alpha from gaussian output
        depth_dict: Optional dict containing 'conf_valid_mask' for confidence-based masking
        step: Current training step
        use_lpips: Whether to compute LPIPS loss
        lpips_weight: Weight for LPIPS loss
        lpips_apply_after_step: Step after which to apply LPIPS loss
        lpips_model: LPIPS model instance
    
    Returns:
        Dict containing gaussian L1 loss and optionally LPIPS loss
    """
    # Extract predicted and ground truth data
    gs_output = predictions['gs_rendered']
    alpha = gs_output['alpha'] if isinstance(gs_output, dict) else gs_output.alpha
    color = gs_output['color'] if isinstance(gs_output, dict) else gs_output.color
    valid_mask = batch['valid_mask']

    
    # Get ground truth images shape (B, N, C, H, W)
    gt_images = batch["images"]
    B, N, C, H, W = gt_images.shape
    
    # Reshape color from (B*N, C, H, W) to (B, N, C, H, W)
    color = color.reshape(B, N, C, H, W)
    
    # Reshape alpha from (B*N, H, W) to (B, N, H, W) if needed
    if alpha.dim() == 3:
        alpha = alpha.reshape(B, N, H, W)
    
    color = check_and_fix_inf_nan(color, "color")
    gt_images = check_and_fix_inf_nan(gt_images, "gt_images")
    valid_mask = check_and_fix_inf_nan(valid_mask, "valid_masks")
    
    # Determine which mask to use based on configuration
    if use_mask:
        # valid_mask should already be (B, N, H, W), ensure it's boolean
        mask = valid_mask.bool()
    elif use_alpha:
        mask = alpha
    elif use_conf and depth_dict is not None:
        mask = depth_dict['conf_valid_mask']
    else:
        mask = torch.ones_like(alpha, device=alpha.device).bool()
    
    # Denormalize ground truth images to match predicted images
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=gt_images.device).view(1, 1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=gt_images.device).view(1, 1, 3, 1, 1)

    # Denormalize gt_images: original = normalized * std + mean
    gt_images_denorm = gt_images * imagenet_std + imagenet_mean

    # Permute tensors from (B, N, C, H, W) to (B, N, H, W, C) and apply mask
    pred_img = color.permute(0, 1, 3, 4, 2)[mask]
    gt_img = gt_images_denorm.permute(0, 1, 3, 4, 2)[mask]

    # Check if mask has any valid pixels
    if pred_img.numel() == 0:
        print(f"Warning: No valid pixels in mask for gaussian loss. Mask sum: {mask.sum().item()}")
        # Return zero loss if no valid pixels
        loss_dict = {"loss_gaussian": torch.tensor(0.0, device=color.device, requires_grad=True)}
        if use_lpips:
            loss_dict["loss_gaussian_lpips"] = torch.tensor(0.0, device=color.device, requires_grad=True)
        return loss_dict

    # Compute L1 loss
    delta = pred_img - gt_img
    l1_loss = delta.abs().mean()
    
    # Compute LPIPS loss if model is provided
    if use_lpips and lpips_model is not None:
        # LPIPS model is converted to buffer, so it automatically follows device
        if use_mask or use_alpha or use_conf:
            # Apply mask to images for LPIPS
            # Expand mask to match color channels (B, N, C, H, W)
            expanded_mask = mask.unsqueeze(2).expand(-1, -1, C, -1, -1)
            masked_pred = color * expanded_mask
            masked_gt = gt_images_denorm * expanded_mask
            
            # Reshape to (B*N, C, H, W) for LPIPS
            lpips_loss = lpips_model.forward(
                rearrange(masked_pred, "b v c h w -> (b v) c h w"),
                rearrange(masked_gt, "b v c h w -> (b v) c h w"),
                normalize=True,
            )
        else:
            # No masking, use full images
            lpips_loss = lpips_model.forward(
                rearrange(color, "b v c h w -> (b v) c h w"),
                rearrange(gt_images_denorm, "b v c h w -> (b v) c h w"),
                normalize=True,
            )
            
        # Handle NaN/Inf values in LPIPS loss
        lpips_loss = torch.nan_to_num(lpips_loss.mean(), nan=0.0, posinf=0.0, neginf=0.0)
        
        loss_dict = {
            "loss_gaussian_l1": l1_loss,
            "loss_gaussian_lpips": lpips_loss,
        }

    return loss_dict


def compute_camera_loss(
    pred_dict,              # predictions dict, contains pose encodings
    batch_data,             # ground truth and mask batch dict
    loss_type="l1",         # "l1" or "l2" loss
    pose_encoding_type="absT_quaR_FoV",
    weight_trans=1.0,       # weight for translation loss
    weight_rot=1.0,         # weight for rotation loss
    weight_focal=0.5,       # weight for focal length loss
    **kwargs
):
    
    # Binary mask for valid points per frame (B, N, H, W)
    point_masks = batch_data['valid_mask']
    # Only consider frames with enough valid points (>100)
    valid_frame_mask = point_masks[:, 0].sum(dim=[-1, -2]) > 100

    # Get ground truth camera extrinsics and intrinsics
    gt_extrinsics = batch_data['extrinsic']
    gt_intrinsics = batch_data['intrinsic']
    image_hw = batch_data['images'].shape[-2:]
    
    # Get predicted pose encodings from multiple stages
    pred_extrinsics = pred_dict['extrinsics']
    pred_intrinsics = pred_dict['intrinsics']
    

    # Encode ground truth pose to match predicted encoding format
    gt_pose_encoding = extri_intri_to_pose_encoding(
        gt_extrinsics, gt_intrinsics, image_hw, pose_encoding_type=pose_encoding_type
    )
    # Encode prediction
    pred_pose_encoding = extri_intri_to_pose_encoding(
        pred_extrinsics, pred_intrinsics, image_hw, pose_encoding_type=pose_encoding_type
    )

    # Initialize loss accumulators for translation, rotation, focal length
    total_loss_T = total_loss_R = total_loss_FL = 0


    if valid_frame_mask.sum() == 0:
        # If no valid frames, set losses to zero to avoid gradient issues
        total_loss_T = (pred_pose_encoding * 0).mean()
        total_loss_R = (pred_pose_encoding * 0).mean()
        total_loss_FL = (pred_pose_encoding * 0).mean()
    else:
        # Only consider valid frames for loss computation
        total_loss_T, total_loss_R, total_loss_FL = camera_loss_single(
            pred_pose_encoding[valid_frame_mask].clone(),
            gt_pose_encoding[valid_frame_mask].clone(),
            loss_type=loss_type
        )

    # Compute total weighted camera loss
    total_camera_loss = (
        total_loss_T * weight_trans +
        total_loss_R * weight_rot +
        total_loss_FL * weight_focal
    )

    # Return loss dictionary with individual components
    return {
        "loss_camera": total_camera_loss,
        "loss_T": total_loss_T ,
        "loss_R": total_loss_R,
        "loss_FL": total_loss_FL
    }

def camera_loss_single(pred_pose_enc, gt_pose_enc, loss_type="l1"):
    """
    Computes translation, rotation, and focal loss for a batch of pose encodings.
    
    Args:
        pred_pose_enc: (N, D) predicted pose encoding
        gt_pose_enc: (N, D) ground truth pose encoding
        loss_type: "l1" (abs error) or "l2" (euclidean error)
    Returns:
        loss_T: translation loss (mean)
        loss_R: rotation loss (mean)
        loss_FL: focal length/intrinsics loss (mean)
    
    NOTE: The paper uses smooth l1 loss, but we found l1 loss is more stable than smooth l1 and l2 loss.
        So here we use l1 loss.
    """
    if loss_type == "l1":
        # Translation: first 3 dims; Rotation: next 4 (quaternion); Focal/Intrinsics: last dims
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).abs()
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).abs()
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).abs()
    elif loss_type == "l2":
        # L2 norm for each component
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).norm(dim=-1, keepdim=True)
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).norm(dim=-1)
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).norm(dim=-1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Check/fix numerical issues (nan/inf) for each loss component
    loss_T = check_and_fix_inf_nan(loss_T, "loss_T")
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R")
    loss_FL = check_and_fix_inf_nan(loss_FL, "loss_FL")

    # Clamp outlier translation loss to prevent instability, then average
    loss_T = loss_T.clamp(max=100).mean()
    loss_R = loss_R.mean()
    loss_FL = loss_FL.mean()

    return loss_T, loss_R, loss_FL


def compute_point_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn = None, valid_range=-1, **kwargs):
    """
    Compute point loss.
    
    Args:
        predictions: Dict containing 'world_points' and 'world_points_conf'
        batch: Dict containing ground truth 'world_points' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    pred_points = predictions['world_points']
    pred_points_conf = predictions['world_points_conf']
    gt_points = batch['world_points']
    gt_points_mask = batch['valid_mask']
    
    gt_points = check_and_fix_inf_nan(gt_points, "gt_points", hard_max=None)
    
    if gt_points_mask.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_points).mean()
        loss_dict = {f"loss_conf_point": dummy_loss,
                    f"loss_reg_point": dummy_loss,
                    f"loss_grad_point": dummy_loss,}
        return loss_dict
    
    # Compute confidence-weighted regression loss with optional gradient loss
    loss_conf, loss_grad, loss_reg = regression_loss(pred_points, gt_points, gt_points_mask, conf=pred_points_conf,
                                             gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range)
    
    loss_dict = {
        f"loss_conf_point": loss_conf,
        f"loss_reg_point": loss_reg,
        f"loss_grad_point": loss_grad,
    }
    
    return loss_dict


def compute_depth_loss(predictions, batch, gamma=1.0, alpha=0.2, gradient_loss_fn = None, valid_range=-1, **kwargs):
    """
    Compute depth loss.
    
    Args:
        predictions: Dict containing 'depth' and 'depth_conf'
        batch: Dict containing ground truth 'depths' and 'point_masks'
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        gradient_loss_fn: Type of gradient loss to apply
        valid_range: Quantile range for outlier filtering
    """
    pred_depth = predictions['depth'][...,None]
    pred_depth_conf = predictions['depth_conf']

    gt_depth = batch['depth']
    gt_depth = check_and_fix_inf_nan(gt_depth, "gt_depth")
    gt_depth = gt_depth[..., None]              # (B, H, W, 1)
    gt_depth_mask = batch['valid_mask'].clone()   # 3D points derived from depth map, so we use the same mask

    if gt_depth_mask.sum() < 100:
        # If there are less than 100 valid points, skip this batch
        dummy_loss = (0.0 * pred_depth).mean()
        loss_dict = {f"loss_conf_depth": dummy_loss,
                    f"loss_reg_depth": dummy_loss,
                    f"loss_grad_depth": dummy_loss,}
        return loss_dict

    # NOTE: we put conf inside regression_loss so that we can also apply conf loss to the gradient loss in a multi-scale manner
    # this is hacky, but very easier to implement
    loss_conf, loss_grad, loss_reg = regression_loss(pred_depth, gt_depth, gt_depth_mask, conf=pred_depth_conf,
                                             gradient_loss_fn=gradient_loss_fn, gamma=gamma, alpha=alpha, valid_range=valid_range)

    loss_dict = {
        f"loss_conf_depth": loss_conf,
        f"loss_reg_depth": loss_reg,    
        f"loss_grad_depth": loss_grad,
    }

    return loss_dict


def regression_loss(pred, gt, mask, conf=None, gradient_loss_fn=None, gamma=1.0, alpha=0.2, valid_range=-1):
    """
    Core regression loss function with confidence weighting and optional gradient loss.
    
    Computes:
    1. gamma * ||pred - gt||^2 * conf - alpha * log(conf)
    2. Optional gradient loss
    
    Args:
        pred: (B, S, H, W, C) predicted values
        gt: (B, S, H, W, C) ground truth values
        mask: (B, S, H, W) valid pixel mask
        conf: (B, S, H, W) confidence weights (optional)
        gradient_loss_fn: Type of gradient loss ("normal", "grad", etc.)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
        valid_range: Quantile range for outlier filtering
    
    Returns:
        loss_conf: Confidence-weighted loss
        loss_grad: Gradient loss (0 if not specified)
        loss_reg: Regular L2 loss
    """
    bb, ss, hh, ww, nc = pred.shape

    # Compute L2 distance between predicted and ground truth points
    loss_reg = torch.norm(gt[mask] - pred[mask], dim=-1)
    loss_reg = check_and_fix_inf_nan(loss_reg, "loss_reg")

    # Confidence-weighted loss: gamma * loss * conf - alpha * log(conf)
    # This encourages the model to be confident on easy examples and less confident on hard ones
    loss_conf = gamma * loss_reg * conf[mask] - alpha * torch.log(conf[mask])
    loss_conf = check_and_fix_inf_nan(loss_conf, "loss_conf")
        
    # Initialize gradient loss
    loss_grad = 0

    # Prepare confidence for gradient loss if needed
    if "conf" in gradient_loss_fn:
        to_feed_conf = conf.reshape(bb*ss, hh, ww)
    else:
        to_feed_conf = None

    # Compute gradient loss if specified for spatial smoothness
    if "normal" in gradient_loss_fn:
        # Surface normal-based gradient loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=normal_loss,
            scales=3,
            conf=to_feed_conf,
        )
    elif "grad" in gradient_loss_fn:
        # Standard gradient-based loss
        loss_grad = gradient_loss_multi_scale_wrapper(
            pred.reshape(bb*ss, hh, ww, nc),
            gt.reshape(bb*ss, hh, ww, nc),
            mask.reshape(bb*ss, hh, ww),
            gradient_loss_fn=gradient_loss,
            conf=to_feed_conf,
        )

    # Process confidence-weighted loss
    if loss_conf.numel() > 0:
        # Filter out outliers using quantile-based thresholding
        if valid_range>0:
            loss_conf = filter_by_quantile(loss_conf, valid_range)

        loss_conf = check_and_fix_inf_nan(loss_conf, f"loss_conf_depth")
        loss_conf = loss_conf.mean()
    else:
        loss_conf = (0.0 * pred).mean()

    # Process regular regression loss
    if loss_reg.numel() > 0:
        # Filter out outliers using quantile-based thresholding
        if valid_range>0:
            loss_reg = filter_by_quantile(loss_reg, valid_range)

        loss_reg = check_and_fix_inf_nan(loss_reg, f"loss_reg_depth")
        loss_reg = loss_reg.mean()
    else:
        loss_reg = (0.0 * pred).mean()

    return loss_conf, loss_grad, loss_reg


def gradient_loss_multi_scale_wrapper(prediction, target, mask, scales=4, gradient_loss_fn = None, conf=None):
    """
    Multi-scale gradient loss wrapper. Applies gradient loss at multiple scales by subsampling the input.
    This helps capture both fine and coarse spatial structures.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values  
        mask: (B, H, W) valid pixel mask
        scales: Number of scales to use
        gradient_loss_fn: Gradient loss function to apply
        conf: (B, H, W) confidence weights (optional)
    """
    total = 0
    for scale in range(scales):
        step = pow(2, scale)  # Subsample by 2^scale

        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None
        )

    total = total / scales
    return total


def normal_loss(prediction, target, mask, cos_eps=1e-8, conf=None, gamma=1.0, alpha=0.2):
    """
    Surface normal-based loss for geometric consistency.
    
    Computes surface normals from 3D point maps using cross products of neighboring points,
    then measures the angle between predicted and ground truth normals.
    
    Args:
        prediction: (B, H, W, 3) predicted 3D coordinates/points
        target: (B, H, W, 3) ground-truth 3D coordinates/points
        mask: (B, H, W) valid pixel mask
        cos_eps: Epsilon for numerical stability in cosine computation
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Convert point maps to surface normals using cross products
    pred_normals, pred_valids = point_map_to_normal(prediction, mask, eps=cos_eps)
    gt_normals,   gt_valids   = point_map_to_normal(target,     mask, eps=cos_eps)

    # Only consider regions where both predicted and GT normals are valid
    all_valid = pred_valids & gt_valids  # shape: (4, B, H, W)

    # Early return if not enough valid points
    divisor = torch.sum(all_valid)
    if divisor < 10:
        return 0

    # Extract valid normals
    pred_normals = pred_normals[all_valid].clone()
    gt_normals = gt_normals[all_valid].clone()

    # Compute cosine similarity between corresponding normals
    dot = torch.sum(pred_normals * gt_normals, dim=-1)

    # Clamp dot product to [-1, 1] for numerical stability
    dot = torch.clamp(dot, -1 + cos_eps, 1 - cos_eps)

    # Compute loss as 1 - cos(theta), instead of arccos(dot) for numerical stability
    loss = 1 - dot

    # Return mean loss if we have enough valid points
    if loss.numel() < 10:
        return 0
    else:
        loss = check_and_fix_inf_nan(loss, "normal_loss")

        if conf is not None:
            # Apply confidence weighting
            conf = conf[None, ...].expand(4, -1, -1, -1)
            conf = conf[all_valid].clone()

            loss = gamma * loss * conf - alpha * torch.log(conf)
            return loss.mean()
        else:
            return loss.mean()


def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    """
    Gradient-based loss. Computes the L1 difference between adjacent pixels in x and y directions.
    
    Args:
        prediction: (B, H, W, C) predicted values
        target: (B, H, W, C) ground truth values
        mask: (B, H, W) valid pixel mask
        conf: (B, H, W) confidence weights (optional)
        gamma: Weight for confidence loss
        alpha: Weight for confidence regularization
    """
    # Expand mask to match prediction channels
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    M = torch.sum(mask, (1, 2, 3))

    # Compute difference between prediction and target
    diff = prediction - target
    diff = torch.mul(mask, diff)

    # Compute gradients in x direction (horizontal)
    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    # Compute gradients in y direction (vertical)
    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    # Clamp gradients to prevent outliers
    grad_x = grad_x.clamp(max=100)
    grad_y = grad_y.clamp(max=100)

    # Apply confidence weighting if provided
    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        conf_x = conf[:, :, 1:]
        conf_y = conf[:, 1:, :]

        grad_x = gamma * grad_x * conf_x - alpha * torch.log(conf_x)
        grad_y = gamma * grad_y * conf_y - alpha * torch.log(conf_y)

    # Sum gradients and normal
    # ize by number of valid pixels
    grad_loss = torch.sum(grad_x, (1, 2, 3)) + torch.sum(grad_y, (1, 2, 3))
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        grad_loss = torch.sum(grad_loss) / divisor

    return grad_loss


def point_map_to_normal(point_map, mask, eps=1e-6):
    """
    Convert 3D point map to surface normal vectors using cross products.
    
    Computes normals by taking cross products of neighboring point differences.
    Uses 4 different cross-product directions for robustness.
    
    Args:
        point_map: (B, H, W, 3) 3D points laid out in a 2D grid
        mask: (B, H, W) valid pixels (bool)
        eps: Epsilon for numerical stability in normalization
    
    Returns:
        normals: (4, B, H, W, 3) normal vectors for each of the 4 cross-product directions
        valids: (4, B, H, W) corresponding valid masks
    """
    with torch.amp.autocast('cuda', enabled=False):
        # Pad inputs to avoid boundary issues
        padded_mask = F.pad(mask, (1, 1, 1, 1), mode='constant', value=0)
        pts = F.pad(point_map.permute(0, 3, 1, 2), (1,1,1,1), mode='constant', value=0).permute(0, 2, 3, 1)

        # Get neighboring points for each pixel
        center = pts[:, 1:-1, 1:-1, :]   # B,H,W,3
        up     = pts[:, :-2,  1:-1, :]
        left   = pts[:, 1:-1, :-2 , :]
        down   = pts[:, 2:,   1:-1, :]
        right  = pts[:, 1:-1, 2:,   :]

        # Compute direction vectors from center to neighbors
        up_dir    = up    - center
        left_dir  = left  - center
        down_dir  = down  - center
        right_dir = right - center

        # Compute four cross products for different normal directions
        n1 = torch.cross(up_dir,   left_dir,  dim=-1)  # up x left
        n2 = torch.cross(left_dir, down_dir,  dim=-1)  # left x down
        n3 = torch.cross(down_dir, right_dir, dim=-1)  # down x right
        n4 = torch.cross(right_dir,up_dir,    dim=-1)  # right x up

        # Validity masks - require both direction pixels to be valid
        v1 = padded_mask[:, :-2,  1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, :-2]
        v2 = padded_mask[:, 1:-1, :-2 ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 2:,   1:-1]
        v3 = padded_mask[:, 2:,   1:-1] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, 1:-1, 2:]
        v4 = padded_mask[:, 1:-1, 2:  ] & padded_mask[:, 1:-1, 1:-1] & padded_mask[:, :-2,  1:-1]

        # Stack normals and validity masks
        normals = torch.stack([n1, n2, n3, n4], dim=0)  # shape [4, B, H, W, 3]
        valids  = torch.stack([v1, v2, v3, v4], dim=0)  # shape [4, B, H, W]

        # Normalize normal vectors
        normals = F.normalize(normals, p=2, dim=-1, eps=eps)

    return normals, valids


def filter_by_quantile(loss_tensor, valid_range, min_elements=1000, hard_max=100):
    """
    Filter loss tensor by keeping only values below a certain quantile threshold.
    
    This helps remove outliers that could destabilize training.
    
    Args:
        loss_tensor: Tensor containing loss values
        valid_range: Float between 0 and 1 indicating the quantile threshold
        min_elements: Minimum number of elements required to apply filtering
        hard_max: Maximum allowed value for any individual loss
    
    Returns:
        Filtered and clamped loss tensor
    """
    if loss_tensor.numel() <= min_elements:
        # Too few elements, just return as-is
        return loss_tensor

    # Randomly sample if tensor is too large to avoid memory issues
    if loss_tensor.numel() > 100000000:
        # Flatten and randomly select 1M elements
        indices = torch.randperm(loss_tensor.numel(), device=loss_tensor.device)[:1_000_000]
        loss_tensor = loss_tensor.view(-1)[indices]

    # First clamp individual values to prevent extreme outliers
    loss_tensor = loss_tensor.clamp(max=hard_max)

    # Compute quantile threshold
    quantile_thresh = torch_quantile(loss_tensor.detach(), valid_range)
    quantile_thresh = min(quantile_thresh, hard_max)

    # Apply quantile filtering if enough elements remain
    quantile_mask = loss_tensor < quantile_thresh
    if quantile_mask.sum() > min_elements:
        return loss_tensor[quantile_mask]
    return loss_tensor


def torch_quantile(
    input,
    q,
    dim = None,
    keepdim: bool = False,
    *,
    interpolation: str = "nearest",
    out: torch.Tensor = None,
) -> torch.Tensor:
    """Better torch.quantile for one SCALAR quantile.

    Using torch.kthvalue. Better than torch.quantile because:
        - No 2**24 input size limit (pytorch/issues/67592),
        - Much faster, at least on big input sizes.

    Arguments:
        input (torch.Tensor): See torch.quantile.
        q (float): See torch.quantile. Supports only scalar input
            currently.
        dim (int | None): See torch.quantile.
        keepdim (bool): See torch.quantile. Supports only False
            currently.
        interpolation: {"nearest", "lower", "higher"}
            See torch.quantile.
        out (torch.Tensor | None): See torch.quantile. Supports only
            None currently.
    """
    # https://github.com/pytorch/pytorch/issues/64947
    # Sanitization: q
    try:
        q = float(q)
        assert 0 <= q <= 1
    except Exception:
        raise ValueError(f"Only scalar input 0<=q<=1 is currently supported (got {q})!")

    # Handle dim=None case
    if dim_was_none := dim is None:
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))

    # Set interpolation method
    if interpolation == "nearest":
        inter = round
    elif interpolation == "lower":
        inter = floor
    elif interpolation == "higher":
        inter = ceil
    else:
        raise ValueError(
            "Supported interpolations currently are {'nearest', 'lower', 'higher'} "
            f"(got '{interpolation}')!"
        )

    # Validate out parameter
    if out is not None:
        raise ValueError(f"Only None value is currently supported for out (got {out})!")

    # Compute k-th value
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True, out=out)[0]

    # Handle keepdim and dim=None cases
    if keepdim:
        return out
    if dim_was_none:
        return out.squeeze()
    else:
        return out.squeeze(dim)

    return out


