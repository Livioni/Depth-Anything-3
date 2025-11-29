# flake8: noqa: F722
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from types import SimpleNamespace
from typing import Optional
import numpy as np
import torch
import torch.nn.functional as F
from einops import einsum

from utils.geometry import closed_form_inverse_se3


def as_homogeneous(ext):
    """
    Accept (..., 3,4) or (..., 4,4) extrinsics, return (...,4,4) homogeneous matrix.
    Supports torch.Tensor or np.ndarray.
    """
    if isinstance(ext, torch.Tensor):
        # If already in homogeneous form
        if ext.shape[-2:] == (4, 4):
            return ext
        elif ext.shape[-2:] == (3, 4):
            # Create a new homogeneous matrix
            ones = torch.zeros_like(ext[..., :1, :4])
            ones[..., 0, 3] = 1.0
            return torch.cat([ext, ones], dim=-2)
        else:
            raise ValueError(f"Invalid shape for torch.Tensor: {ext.shape}")

    elif isinstance(ext, np.ndarray):
        if ext.shape[-2:] == (4, 4):
            return ext
        elif ext.shape[-2:] == (3, 4):
            ones = np.zeros_like(ext[..., :1, :4])
            ones[..., 0, 3] = 1.0
            return np.concatenate([ext, ones], axis=-2)
        else:
            raise ValueError(f"Invalid shape for np.ndarray: {ext.shape}")

    else:
        raise TypeError("Input must be a torch.Tensor or np.ndarray.")


@torch.jit.script
def affine_inverse(A: torch.Tensor):
    R = A[..., :3, :3]  # ..., 3, 3
    T = A[..., :3, 3:]  # ..., 3, 1
    P = A[..., 3:, :]  # ..., 1, 4
    return torch.cat([torch.cat([R.mT, -R.mT @ T], dim=-1), P], dim=-2)


def transpose_last_two_axes(arr):
    """
    for np < 2
    """
    if arr.ndim < 2:
        return arr
    axes = list(range(arr.ndim))
    # swap the last two
    axes[-2], axes[-1] = axes[-1], axes[-2]
    return arr.transpose(axes)


def affine_inverse_np(A: np.ndarray):
    R = A[..., :3, :3]
    T = A[..., :3, 3:]
    P = A[..., 3:, :]
    return np.concatenate(
        [
            np.concatenate([transpose_last_two_axes(R), -transpose_last_two_axes(R) @ T], axis=-1),
            P,
        ],
        axis=-2,
    )


def quat_to_mat(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Quaternion Order: XYZW or say ijkr, scalar-last

    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    i, j, k, r = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def mat_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part last, as tensor of shape (..., 4).
        Quaternion Order: XYZW or say ijkr, scalar-last
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(
        batch_dim + (4,)
    )

    # Convert from rijk to ijkr
    out = out[..., [1, 2, 3, 0]]

    out = standardize_quaternion(out)

    return out


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part last,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 3:4] < 0, -quaternions, quaternions)


def sample_image_grid(
    shape: tuple[int, ...],
    device: torch.device = torch.device("cpu"),
) -> tuple[
    torch.Tensor,  # float coordinates (xy indexing), "*shape dim"
    torch.Tensor,  # integer indices (ij indexing), "*shape dim"
]:
    """Get normalized (range 0 to 1) coordinates and integer indices for an image."""

    # Each entry is a pixel-wise integer coordinate. In the 2D case, each entry is a
    # (row, col) coordinate.
    indices = [torch.arange(length, device=device) for length in shape]
    stacked_indices = torch.stack(torch.meshgrid(*indices, indexing="ij"), dim=-1)

    # Each entry is a floating-point coordinate in the range (0, 1). In the 2D case,
    # each entry is an (x, y) coordinate.
    coordinates = [(idx + 0.5) / length for idx, length in zip(indices, shape)]
    coordinates = reversed(coordinates)
    coordinates = torch.stack(torch.meshgrid(*coordinates, indexing="xy"), dim=-1)

    return coordinates, stacked_indices


def homogenize_points(points: torch.Tensor) -> torch.Tensor:  # "*batch dim"  # "*batch dim+1"
    """Convert batched points (xyz) to (xyz1)."""
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)


def homogenize_vectors(vectors: torch.Tensor) -> torch.Tensor:  #  "*batch dim"  # "*batch dim+1"
    """Convert batched vectors (xyz) to (xyz0)."""
    return torch.cat([vectors, torch.zeros_like(vectors[..., :1])], dim=-1)


def transform_rigid(
    homogeneous_coordinates: torch.Tensor,  # "*#batch dim"
    transformation: torch.Tensor,  # "*#batch dim dim"
) -> torch.Tensor:  # "*batch dim"
    """Apply a rigid-body transformation to points or vectors."""
    return einsum(
        transformation,
        homogeneous_coordinates.to(transformation.dtype),
        "... i j, ... j -> ... i",
    )


def transform_cam2world(
    homogeneous_coordinates: torch.Tensor,  # "*#batch dim"
    extrinsics: torch.Tensor,  # "*#batch dim dim"
) -> torch.Tensor:  # "*batch dim"
    """Transform points from 3D camera coordinates to 3D world coordinates."""
    return transform_rigid(homogeneous_coordinates, extrinsics)


def unproject(
    coordinates: torch.Tensor,  # "*#batch dim"
    z: torch.Tensor,  # "*#batch"
    intrinsics: torch.Tensor,  # "*#batch dim+1 dim+1"
) -> torch.Tensor:  # "*batch dim+1"
    """Unproject 2D camera coordinates with the given Z values."""

    # Apply the inverse intrinsics to the coordinates.
    coordinates = homogenize_points(coordinates)
    ray_directions = einsum(
        intrinsics.float().inverse().to(intrinsics),
        coordinates.to(intrinsics.dtype),
        "... i j, ... j -> ... i",
    )

    # Apply the supplied depth values.
    return ray_directions * z[..., None]


def get_world_rays(
    coordinates: torch.Tensor,  # "*#batch dim"
    extrinsics: torch.Tensor,  # "*#batch dim+2 dim+2"
    intrinsics: torch.Tensor,  # "*#batch dim+1 dim+1"
) -> tuple[
    torch.Tensor,  # origins, "*batch dim+1"
    torch.Tensor,  # directions, "*batch dim+1"
]:
    # Get camera-space ray directions.
    directions = unproject(
        coordinates,
        torch.ones_like(coordinates[..., 0]),
        intrinsics,
    )
    directions = directions / directions.norm(dim=-1, keepdim=True)

    # Transform ray directions to world coordinates.
    directions = homogenize_vectors(directions)
    directions = transform_cam2world(directions, extrinsics)[..., :-1]

    # Tile the ray origins to have the same shape as the ray directions.
    origins = extrinsics[..., :-1, -1].broadcast_to(directions.shape)

    return origins, directions


def get_fov(intrinsics: torch.Tensor) -> torch.Tensor:  # "batch 3 3" -> "batch 2"
    intrinsics_inv = intrinsics.float().inverse().to(intrinsics)

    def process_vector(vector):
        vector = torch.tensor(vector, dtype=intrinsics.dtype, device=intrinsics.device)
        vector = einsum(intrinsics_inv, vector, "b i j, j -> b i")
        return vector / vector.norm(dim=-1, keepdim=True)

    left = process_vector([0, 0.5, 1])
    right = process_vector([1, 0.5, 1])
    top = process_vector([0.5, 0, 1])
    bottom = process_vector([0.5, 1, 1])
    fov_x = (left * right).sum(dim=-1).acos()
    fov_y = (top * bottom).sum(dim=-1).acos()
    return torch.stack((fov_x, fov_y), dim=-1)


def map_pdf_to_opacity(
    pdf: torch.Tensor,  # " *batch"
    global_step: int = 0,
    opacity_mapping: Optional[dict] = None,
) -> torch.Tensor:  # " *batch"
    # https://www.desmos.com/calculator/opvwti3ba9

    # Figure out the exponent.
    if opacity_mapping is not None:
        cfg = SimpleNamespace(**opacity_mapping)
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
    else:
        x = 0.0
    exponent = 2**x

    # Map the probability density to an opacity.
    return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

def normalize_homogenous_points(points):
    """Normalize the point vectors"""
    return points / points[..., -1:]

def inverse_intrinsic_matrix(ixts):
    """ """
    return torch.inverse(ixts)

def pixel_space_to_camera_space(pixel_space_points, depth, intrinsics):
    """
    Convert pixel space points to camera space points.

    Args:
        pixel_space_points (torch.Tensor): Pixel space points with shape (h, w, 2)
        depth (torch.Tensor): Depth map with shape (b, v, h, w, 1)
        intrinsics (torch.Tensor): Camera intrinsics with shape (b, v, 3, 3)

    Returns:
        torch.Tensor: Camera space points with shape (b, v, h, w, 3).
    """
    pixel_space_points = homogenize_points(pixel_space_points)
    # camera_space_points = torch.einsum(
    #     "b v i j , h w j -> b v h w i", intrinsics.inverse(), pixel_space_points
    # )
    camera_space_points = torch.einsum(
        "b v i j , h w j -> b v h w i", inverse_intrinsic_matrix(intrinsics), pixel_space_points
    )
    camera_space_points = camera_space_points * depth
    return camera_space_points


def camera_space_to_world_space(camera_space_points, c2w):
    """
    Convert camera space points to world space points.

    Args:
        camera_space_points (torch.Tensor): Camera space points with shape (b, v, h, w, 3)
        c2w (torch.Tensor): Camera to world extrinsics matrix with shape (b, v, 4, 4)

    Returns:
        torch.Tensor: World space points with shape (b, v, h, w, 3).
    """
    camera_space_points = homogenize_points(camera_space_points)
    world_space_points = torch.einsum("b v i j , b v h w j -> b v h w i", c2w, camera_space_points)
    return world_space_points[..., :3]


def camera_space_to_pixel_space(camera_space_points, intrinsics):
    """
    Convert camera space points to pixel space points.

    Args:
        camera_space_points (torch.Tensor): Camera space points with shape (b, v1, v2, h, w, 3)
        c2w (torch.Tensor): Camera to world extrinsics matrix with shape (b, v2, 3, 3)

    Returns:
        torch.Tensor: World space points with shape (b, v1, v2, h, w, 2).
    """
    camera_space_points = normalize_homogenous_points(camera_space_points)
    pixel_space_points = torch.einsum(
        "b u i j , b v u h w j -> b v u h w i", intrinsics, camera_space_points
    )
    return pixel_space_points[..., :2]


def world_space_to_camera_space(world_space_points, c2w):
    """
    Convert world space points to pixel space points.

    Args:
        world_space_points (torch.Tensor): World space points with shape (b, v1, h, w, 3)
        c2w (torch.Tensor): Camera to world extrinsics matrix with shape (b, v2, 4, 4)

    Returns:
        torch.Tensor: Camera space points with shape (b, v1, v2, h, w, 3).
    """
    world_space_points = homogenize_points(world_space_points)
    camera_space_points = torch.einsum(
        "b u i j , b v h w j -> b v u h w i", c2w.inverse(), world_space_points
    )
    return camera_space_points[..., :3]


def unproject_depth(
    depth, intrinsics, c2w=None, ixt_normalized=False, num_patches_x=None, num_patches_y=None
):
    """
    Turn the depth map into a 3D point cloud in world space

    Args:
        depth: (b, v, h, w, 1)
        intrinsics: (b, v, 3, 3)
        c2w: (b, v, 4, 4)

    Returns:
        torch.Tensor: World space points with shape (b, v, h, w, 3).
    """
    if c2w is None:
        c2w = torch.eye(4, device=depth.device, dtype=depth.dtype)
        c2w = c2w[None, None].repeat(depth.shape[0], depth.shape[1], 1, 1)

    if not ixt_normalized:
        # Compute indices of pixels
        h, w = depth.shape[-3], depth.shape[-2]
        x_grid, y_grid = torch.meshgrid(
            torch.arange(w, device=depth.device, dtype=depth.dtype),
            torch.arange(h, device=depth.device, dtype=depth.dtype),
            indexing="xy",
        )  # (h, w), (h, w)
    else:
        # ixt_normalized: h=w=2.0. cx, cy, fx, fy are normalized according to h=w=2.0
        assert num_patches_x is not None and num_patches_y is not None
        dx = 1 / num_patches_x
        dy = 1 / num_patches_y
        max_y = 1 - dy
        min_y = -max_y
        max_x = 1 - dx
        min_x = -max_x

        grid_shift = 1.0
        y_grid, x_grid = torch.meshgrid(
            torch.linspace(
                min_y + grid_shift,
                max_y + grid_shift,
                num_patches_y,
                dtype=torch.float32,
                device=depth.device,
            ),
            torch.linspace(
                min_x + grid_shift,
                max_x + grid_shift,
                num_patches_x,
                dtype=torch.float32,
                device=depth.device,
            ),
            indexing="ij",
        )

    # Compute coordinates of pixels in camera space
    pixel_space_points = torch.stack((x_grid, y_grid), dim=-1)  # (..., h, w, 2)
    camera_points = pixel_space_to_camera_space(
        pixel_space_points, depth, intrinsics
    )  # (..., h, w, 3)

    # Convert points to world space
    world_points = camera_space_to_world_space(camera_points, c2w)  # (..., h, w, 3)

    return world_points

def normalize_extrinsics(ex_t: torch.Tensor | None) -> torch.Tensor | None:
    """
    Normalize extrinsics to canonical coordinate system.
    
    This function:
    1. Aligns all cameras to the first camera's coordinate system
    2. Normalizes the scale based on median camera distance
    
    Args:
        ex_t: Extrinsics tensor of shape (B, N, 3, 4) or (B, N, 4, 4) or (N, 3, 4) or (N, 4, 4)
              where B is batch size, N is number of cameras
              These are world-to-camera transforms
    
    Returns:
        Normalized extrinsics with the same shape as input
    """
    if ex_t is None:
        return None
    
    # Handle both batched and unbatched inputs
    original_shape = ex_t.shape
    is_batched = ex_t.ndim == 4  # (B, N, 3/4, 4)
    
    if not is_batched:
        # Add batch dimension: (N, 3/4, 4) -> (1, N, 3/4, 4)
        ex_t = ex_t.unsqueeze(0)
    
    B, N = ex_t.shape[:2]
    
    # Ensure homogeneous form (B, N, 4, 4)
    ex_t_homog = as_homogeneous(ex_t)
    
    # Get the first camera's extrinsics for each batch: (B, 1, 4, 4)
    first_camera = ex_t_homog[:, :1, :, :]
    
    # Compute inverse to use as alignment transform
    # Reshape to (B, 4, 4) for affine_inverse
    transform = affine_inverse(first_camera.squeeze(1))  # (B, 4, 4)
    
    # Expand transform for broadcasting: (B, 1, 4, 4)
    transform = transform.unsqueeze(1)
    
    # Align all cameras to the first camera: ex_t_norm = ex_t @ transform
    # (B, N, 4, 4) @ (B, 1, 4, 4) -> (B, N, 4, 4)
    ex_t_norm = ex_t_homog @ transform
    
    # Convert to camera-to-world for distance computation
    # Reshape to (B*N, 4, 4)
    ex_t_norm_flat = ex_t_norm.reshape(B * N, 4, 4)
    c2ws_flat = affine_inverse(ex_t_norm_flat)
    c2ws = c2ws_flat.reshape(B, N, 4, 4)
    
    # Extract camera positions (translations) in world space
    translations = c2ws[..., :3, 3]  # (B, N, 3)
    
    # Compute distances from origin for each camera
    dists = translations.norm(dim=-1)  # (B, N)
    
    # Compute median distance for each batch
    median_dist = torch.median(dists, dim=-1)[0]  # (B,)
    median_dist = torch.clamp(median_dist, min=1e-1)
    
    # Normalize translation by median distance
    # Reshape for broadcasting: (B, 1, 1)
    scale_factor = median_dist.view(B, 1, 1)
    
    # Extract and normalize translation component
    # ex_t_norm is (B, N, 4, 4), we want to modify the translation part (last column, first 3 rows)
    translation = ex_t_norm[:, :, :3, 3]  # (B, N, 3)
    translation_normalized = translation / scale_factor  # (B, N, 3) / (B, 1, 1) -> (B, N, 3)
    ex_t_norm[:, :, :3, 3] = translation_normalized
    
    # Restore original shape if input was not batched
    if not is_batched:
        ex_t_norm = ex_t_norm.squeeze(0)  # (1, N, 4, 4) -> (N, 4, 4)
    
    # Restore original last two dimensions if input was (3, 4)
    if original_shape[-2] == 3:
        ex_t_norm = ex_t_norm[..., :3, :]
    
    return ex_t_norm

def generate_rays_from_batch(
    images_shape: tuple[int, ...],
    extrinsics: torch.Tensor,  # (B, N, 3, 4) or (B, N, 4, 4)
    intrinsics: torch.Tensor,  # (B, N, 3, 3)
) -> tuple[torch.Tensor, torch.Tensor]:  # ray_origins, ray_directions (B, N, H, W, 3)
    """
    Generate world-space rays from batch data.

    Args:
        images_shape: Shape of images tensor (B, N, C, H, W)
        extrinsics: Camera extrinsics (B, N, 3, 4) or (B, N, 4, 4)
        intrinsics: Camera intrinsics (B, N, 3, 3)

    Returns:
        ray_origins: Ray origins in world space (B, N, H, W, 3)
        ray_directions: Ray directions in world space (B, N, H, W, 3)
        
    def get_rays_np(H, W, K, c2w):
        i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
        dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
        # Rotate ray directions from camera frame to the world frame
        rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
        return rays_o, rays_d
    """
    # Extract dimensions
    B, N, C, H_orig, W_orig = images_shape
    
    # Compute scaled dimensions
    scale_factor = 1.75
    H = int(H_orig // scale_factor)
    W = int(W_orig // scale_factor)
    
    # Scale intrinsics to match the new resolution
    intrinsics_scaled = intrinsics.clone()
    intrinsics_scaled[..., 0, 0] = intrinsics[..., 0, 0] / scale_factor  # fx
    intrinsics_scaled[..., 1, 1] = intrinsics[..., 1, 1] / scale_factor  # fy
    intrinsics_scaled[..., 0, 2] = intrinsics[..., 0, 2] / scale_factor  # cx
    intrinsics_scaled[..., 1, 2] = intrinsics[..., 1, 2] / scale_factor  # cy

    # Generate normalized pixel coordinates for the image grid
    # coordinates shape: (H, W, 2) with values in [0, 1]
    coordinates, _ = sample_image_grid((H, W), device=extrinsics.device)
    
    # Convert normalized coordinates [0, 1] to pixel coordinates
    # sample_image_grid returns (x, y) in range [0, 1], convert to pixel coords
    coordinates = coordinates * torch.tensor([W, H], device=coordinates.device, dtype=coordinates.dtype)

    # Expand coordinates to match batch dimensions (B, N, H, W, 2)
    coordinates = coordinates.unsqueeze(0).unsqueeze(0).expand(B, N, H, W, 2)

    # Convert extrinsics (w2c) to c2w and ensure homogeneous form
    # Reshape to process all cameras at once
    extrinsics_reshaped = extrinsics.reshape(B * N, *extrinsics.shape[-2:])
    extrinsics_c2w = closed_form_inverse_se3(extrinsics_reshaped)
    extrinsics_c2w = extrinsics_c2w.reshape(B, N, *extrinsics_c2w.shape[-2:])
    extrinsics_homog = as_homogeneous(extrinsics_c2w)

    # Reshape for batch processing
    coordinates_flat = coordinates.reshape(B * N, H, W, 2)
    extrinsics_flat = extrinsics_homog.reshape(B * N, 4, 4).unsqueeze(-3).unsqueeze(-3).expand(-1, H, W, -1, -1)
    intrinsics_flat = intrinsics_scaled.reshape(B * N, 3, 3).unsqueeze(-3).unsqueeze(-3).expand(-1, H, W, -1, -1)

    # Generate world space rays (origins and directions)
    ray_origins, ray_directions = get_world_rays(
        coordinates_flat,
        extrinsics_flat,
        intrinsics_flat
    )

    # Reshape back to (B, N, H, W, 3)
    ray_origins = ray_origins.reshape(B, N, H, W, 3)
    ray_directions = ray_directions.reshape(B, N, H, W, 3)

    return ray_origins, ray_directions


def generate_rays_at_resolution(
    target_height: int,
    target_width: int,
    extrinsics: torch.Tensor,  # (B, N, 3, 4) or (B, N, 4, 4)
    intrinsics: torch.Tensor,  # (B, N, 3, 3)
    original_height: int = None,
    original_width: int = None,
) -> tuple[torch.Tensor, torch.Tensor]:  # ray_origins, ray_directions (B, N, H_target, W_target, 3)
    """
    Generate rays at a specific target resolution.
    
    If the target resolution differs from the original resolution, this function
    automatically adjusts the intrinsics to match the new resolution.
    
    Args:
        target_height: Desired output height
        target_width: Desired output width
        extrinsics: Camera extrinsics (B, N, 3, 4) or (B, N, 4, 4), world-to-camera transform
        intrinsics: Camera intrinsics (B, N, 3, 3) at original resolution
        original_height: Original image height (if None, assumes intrinsics match target resolution)
        original_width: Original image width (if None, assumes intrinsics match target resolution)
    
    Returns:
        ray_origins: Ray origins in world space (B, N, H_target, W_target, 3)
        ray_directions: Ray directions in world space (B, N, H_target, W_target, 3)
    
    Example:
        # Generate rays at original resolution
        rays_o, rays_d = generate_rays_at_resolution(480, 640, extrinsics, intrinsics)
        
        # Generate rays at lower resolution (correctly adjusting intrinsics)
        rays_o_small, rays_d_small = generate_rays_at_resolution(
            240, 320, extrinsics, intrinsics, 
            original_height=480, original_width=640
        )
    """
    B, N = extrinsics.shape[:2]
    device = extrinsics.device
    dtype = extrinsics.dtype
    
    # Adjust intrinsics if target resolution differs from original
    if original_height is not None and original_width is not None:
        scale_h = target_height / original_height
        scale_w = target_width / original_width
        
        intrinsics_adjusted = intrinsics.clone()
        intrinsics_adjusted[..., 0, 0] = intrinsics[..., 0, 0] * scale_w  # fx
        intrinsics_adjusted[..., 1, 1] = intrinsics[..., 1, 1] * scale_h  # fy
        intrinsics_adjusted[..., 0, 2] = intrinsics[..., 0, 2] * scale_w  # cx
        intrinsics_adjusted[..., 1, 2] = intrinsics[..., 1, 2] * scale_h  # cy
    else:
        intrinsics_adjusted = intrinsics
    
    # Create pixel coordinate grids at target resolution
    # Use the same dtype as extrinsics to avoid type mismatch
    # With 'xy' indexing, meshgrid(x, y) returns grids of shape (len(y), len(x))
    # i.e., meshgrid(arange(W), arange(H)) returns (H, W) shaped tensors
    i, j = torch.meshgrid(
        torch.arange(target_width, dtype=dtype, device=device),   # x: [0, W-1]
        torch.arange(target_height, dtype=dtype, device=device),  # y: [0, H-1]
        indexing='xy'
    )
    # i and j are already (H, W) shaped, where i contains x-coords and j contains y-coords
    
    # Extract adjusted intrinsics parameters
    fx = intrinsics_adjusted[..., 0, 0].view(B, N, 1, 1)
    fy = intrinsics_adjusted[..., 1, 1].view(B, N, 1, 1)
    cx = intrinsics_adjusted[..., 0, 2].view(B, N, 1, 1)
    cy = intrinsics_adjusted[..., 1, 2].view(B, N, 1, 1)
    
    # Compute camera-space ray directions
    dirs_x = (i - cx) / fx
    dirs_y = -(j - cy) / fy
    dirs_z = -torch.ones_like(dirs_x)
    dirs = torch.stack([dirs_x, dirs_y, dirs_z], dim=-1)  # (B, N, H_target, W_target, 3)
    
    # Convert extrinsics to c2w
    extrinsics_reshaped = extrinsics.reshape(B * N, *extrinsics.shape[-2:])
    c2w = closed_form_inverse_se3(extrinsics_reshaped)
    c2w = c2w.reshape(B, N, *c2w.shape[-2:])
    
    # Extract rotation and translation
    R = c2w[..., :3, :3]
    T = c2w[..., :3, 3]
    if T.dim() == 4:
        T = T.squeeze(-1)
    
    # Transform ray directions to world space
    R_expanded = R.unsqueeze(2).unsqueeze(2)
    dirs_expanded = dirs.unsqueeze(-1)
    rays_d = torch.matmul(R_expanded, dirs_expanded).squeeze(-1)
    
    # Broadcast ray origins
    rays_o = T.unsqueeze(2).unsqueeze(2).expand(B, N, target_height, target_width, 3)
    
    return rays_o.to(dtype), rays_d.to(dtype)