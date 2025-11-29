# flake8: noqa E501
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

from typing import Dict as TyDict
from typing import List, Sequence, Tuple
import torch
import torch.nn as nn
from addict import Dict

from depth_anything_3.model.dpt import _make_fusion_block, _make_scratch
from depth_anything_3.model.utils.head_utils import (
    create_uv_grid,
    custom_interpolate,
    position_grid_to_embed,
)


class SegDPT(nn.Module):
    """
    Segmentation DPT for semantic segmentation (no confidence output).

    Architectural notes:
      - Based on DualDPT but simplified to output only semantic predictions
      - No confidence channel in the output
      - Single fusion path (no auxiliary head)
      - `intermediate_layer_idx` is configurable via `layer_indices` parameter (default: (0, 1, 2, 3))
      - Returns a **dict** with key from `head_name`, e.g.:
          { "semantic": tensor }
    """

    def __init__(
        self,
        dim_in: int,
        *,
        patch_size: int = 14,
        output_dim: int = 8,
        activation: str = "linear",
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        pos_embed: bool = True,
        down_ratio: int = 1,
        head_name: str = "semantic",
        layer_indices: Tuple[int, int, int, int] = (0, 1, 2, 3),
    ) -> None:
        super().__init__()

        # -------------------- configuration --------------------
        self.patch_size = patch_size
        self.activation = activation
        self.pos_embed = pos_embed
        self.down_ratio = down_ratio
        self.head_name = head_name
        self.output_dim = output_dim

        # Configurable layer indices to select from feats list
        # Default (0, 1, 2, 3) means use feats[0], feats[1], feats[2], feats[3]
        self.intermediate_layer_idx: Tuple[int, int, int, int] = layer_indices

        # -------------------- token pre-norm + per-stage projection --------------------
        self.norm = nn.LayerNorm(dim_in)
        self.projects = nn.ModuleList(
            [nn.Conv2d(dim_in, oc, kernel_size=1, stride=1, padding=0) for oc in out_channels]
        )

        # -------------------- spatial re-sizers (align to common scale before fusion) --------------------
        # design: stage strides (x4, x2, x1, /2) relative to patch grid to align to a common pivot scale
        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    out_channels[0], out_channels[0], kernel_size=4, stride=4, padding=0
                ),
                nn.ConvTranspose2d(
                    out_channels[1], out_channels[1], kernel_size=2, stride=2, padding=0
                ),
                nn.Identity(),
                nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),
            ]
        )

        # -------------------- scratch: stage adapters + fusion --------------------
        self.scratch = _make_scratch(list(out_channels), features, expand=False)

        # Fusion chain
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        # Output head (no confidence, pure semantic prediction)
        head_features_1 = features
        head_features_2 = 32
        self.scratch.output_conv1 = nn.Conv2d(
            head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
        )
        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
        )

    # -------------------------------------------------------------------------
    # Public forward (supports frame chunking for memory)
    # -------------------------------------------------------------------------

    def forward(
        self,
        feats: List[torch.Tensor],
        H: int,
        W: int,
        patch_start_idx: int,
        chunk_size: int = 8,
    ) -> Dict:
        """
        Args:
            feats: List of 4 entries, each entry is a tensor like [B, S, T, C]
            H, W:  Original image dimensions
            patch_start_idx: Starting index of patch tokens in sequence
            chunk_size:      Chunk size along time dimension S

        Returns:
            Dict[str, Tensor] with key self.head_name
            Shape: {head_name: [B, S, output_dim, H/down_ratio, W/down_ratio]}
        """
        B, S, N, C = feats[0][0].shape
        feats = [feat[0].reshape(B * S, N, C) for feat in feats]

        if chunk_size is None or chunk_size >= S:
            out_dict = self._forward_impl(feats, H, W, patch_start_idx)
            out_dict = {k: v.reshape(B, S, *v.shape[1:]) for k, v in out_dict.items()}
            return Dict(out_dict)

        out_dicts = []
        for s0 in range(0, S, chunk_size):
            s1 = min(s0 + chunk_size, S)
            out_dict = self._forward_impl(
                [feat[s0:s1] for feat in feats],
                H,
                W,
                patch_start_idx,
            )
            out_dicts.append(out_dict)
        out_dict = {
            k: torch.cat([out_dict[k] for out_dict in out_dicts], dim=0)
            for k in out_dicts[0].keys()
        }
        out_dict = {k: v.view(B, S, *v.shape[1:]) for k, v in out_dict.items()}
        return Dict(out_dict)

    # -------------------------------------------------------------------------
    # Internal forward (single chunk)
    # -------------------------------------------------------------------------

    def _forward_impl(
        self,
        feats: List[torch.Tensor],
        H: int,
        W: int,
        patch_start_idx: int,
    ) -> TyDict[str, torch.Tensor]:
        B, _, C = feats[0].shape
        ph, pw = H // self.patch_size, W // self.patch_size
        resized_feats = []
        for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
            x = feats[take_idx][:, patch_start_idx:]
            x = self.norm(x)
            x = x.permute(0, 2, 1).contiguous().reshape(B, C, ph, pw)  # [B*S, C, ph, pw]

            x = self.projects[stage_idx](x)
            if self.pos_embed:
                x = self._add_pos_embed(x, W, H)
            x = self.resize_layers[stage_idx](x)  # align scales
            resized_feats.append(x)

        # 2) Fuse pyramid
        fused = self._fuse(resized_feats)

        # 3) Upsample to target resolution and (optional) add pos-embed again
        h_out = int(ph * self.patch_size / self.down_ratio)
        w_out = int(pw * self.patch_size / self.down_ratio)

        fused = self.scratch.output_conv1(fused)
        fused = custom_interpolate(fused, (h_out, w_out), mode="bilinear", align_corners=True)
        if self.pos_embed:
            fused = self._add_pos_embed(fused, W, H)

        # 4) Semantic head: direct output (no confidence separation)
        logits = self.scratch.output_conv2(fused)  # [B*S, output_dim, H, W]
        semantic_pred = self._apply_activation(logits, self.activation)

        return {
            self.head_name: semantic_pred,
        }

    # -------------------------------------------------------------------------
    # Subroutines
    # -------------------------------------------------------------------------

    def _fuse(self, feats: List[torch.Tensor]) -> torch.Tensor:
        """
        4-layer top-down fusion, returns finest scale features (after fusion, before neck1).
        """
        l1, l2, l3, l4 = feats

        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)

        # 4 -> 3 -> 2 -> 1
        out = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        out = self.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[2:])
        out = self.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[2:])
        out = self.scratch.refinenet1(out, l1_rn)
        return out

    def _add_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """Simple UV positional embedding added to feature maps."""
        pw, ph = x.shape[-1], x.shape[-2]
        pe = create_uv_grid(pw, ph, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        pe = position_grid_to_embed(pe, x.shape[1]) * ratio
        pe = pe.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pe

    def _apply_activation(
        self, x: torch.Tensor, activation: str = "linear"
    ) -> torch.Tensor:
        """
        Apply activation to semantic output.
        Supports: exp / relu / sigmoid / softplus / tanh / linear
        """
        act = activation.lower() if isinstance(activation, str) else activation
        if act == "exp":
            return torch.exp(x)
        if act == "relu":
            return torch.relu(x)
        if act == "sigmoid":
            return torch.sigmoid(x)
        if act == "softplus":
            return torch.nn.functional.softplus(x)
        if act == "tanh":
            return torch.tanh(x)
        # Default linear
        return x
