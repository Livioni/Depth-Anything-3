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

import torch
import torch.nn as nn
from typing import Dict

from depth_anything_3.specs import Gaussians
from depth_anything_3.utils.geometry import as_homogeneous
from depth_anything_3.model.utils.gs_renderer import render_3dgs


class GaussianSplatRenderer(nn.Module):
    """
    Renderer for 3D Gaussian Splats that maintains gradient flow during forward pass.

    This module encapsulates the rendering logic for 3D Gaussian Splats, ensuring
    that all operations remain differentiable and gradients can flow through the
    rendering process for training and optimization.
    """

    def __init__(self):
        """
        Initialize the Gaussian Splat Renderer.
        """
        super().__init__()

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        x: torch.Tensor,
        H: int,
        W: int
    ) -> Dict[str, torch.Tensor]:
        """
        Render 3D Gaussian Splats if gaussians are available in the output.

        This method ensures gradient flow is maintained throughout the rendering
        process by avoiding any operations that could detach tensors from the
        computational graph.

        Args:
            output: Model output dictionary containing gaussians and camera parameters
            x: Input tensor with shape (B, N, ...)
            H: Image height
            W: Image width

        Returns:
            Modified output dictionary with rendered auxiliary features
        """
        # Render 3DGS if gaussians are available
        if "gaussians" in output and output.gaussians is not None:
            B, N = x.shape[:2]

            # Get camera parameters
            extrinsics = output.get("extrinsics", None)
            intrinsics = output.get("intrinsics", None)

            # Ensure extrinsics are homogeneous (B, N, 4, 4)
            extrinsics = as_homogeneous(extrinsics)

            # Lists to collect rendered results from each batch
            rendered_colors = []
            rendered_depths = []

            # Process each batch separately to maintain gradient flow
            for b in range(B):
                # Extract parameters for current batch
                extr_b = extrinsics[b]  # (N, 4, 4)
                intr_b = intrinsics[b]  # (N, 3, 3)

                # Normalize intrinsics for rendering (gs_renderer expects normalized intrinsics)
                # Use in-place division to maintain gradient flow
                intr_normed = intr_b / torch.tensor([[[W, W, W], [H, H, H], [1, 1, 1]]], 
                                                     device=intr_b.device, dtype=intr_b.dtype)

                # Extract gaussian for current batch
                # Create a new Gaussians object with data from batch b (keep batch dim=1)
                gaussian_b = Gaussians(
                    means=output.gaussians.means[b:b+1],
                    scales=output.gaussians.scales[b:b+1],
                    rotations=output.gaussians.rotations[b:b+1],
                    harmonics=output.gaussians.harmonics[b:b+1],
                    opacities=output.gaussians.opacities[b:b+1],
                )

                # Render gaussians with SH colors for this batch
                rendered_color_b, rendered_depth_b = render_3dgs(
                    extrinsics=extr_b,
                    intrinsics=intr_normed,
                    image_shape=(H, W),
                    gaussian=gaussian_b,
                    num_view=N,
                    use_sh=True,  # Important: use spherical harmonics for colors
                    color_mode="RGB+D",  # RGB + depth mode
                )

                # Collect results - ensure tensors maintain gradients
                rendered_colors.append(rendered_color_b)  # (N, 3, H, W)
                rendered_depths.append(rendered_depth_b)  # (N, H, W)

            # Concatenate results from all batches
            rendered_color = torch.cat(rendered_colors, dim=0)  # (B*N, 3, H, W)
            rendered_depth = torch.cat(rendered_depths, dim=0)  # (B*N, H, W)

            # Store rendered results in output
            output["gs_rendered"] = {
                "color": rendered_color,  # (B*N, 3, H, W)
                "depth": rendered_depth,  # (B*N, H, W)
                "alpha": (rendered_depth > 0).float(),  # Simple alpha based on depth
            }

        return output
