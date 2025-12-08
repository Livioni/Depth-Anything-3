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

from typing import List, Sequence, Tuple
import torch
import torch.nn as nn
from addict import Dict

from depth_anything_3.model.dpt import _make_fusion_block
from depth_anything_3.model.dualdpt import DualDPT
from depth_anything_3.model.utils.head_utils import Permute


class TriDPT(DualDPT):
    """
    Triple-head DPT for dense prediction with two always-on auxiliary heads.
    Inherits from DualDPT to enable weight loading from DualDPT checkpoints.

    Architectural notes:
      - Inherits main head and aux head from DualDPT (with original naming).
      - Adds a third head (feat head) with its **own** fusion blocks.
      - The feat head is internally multi-level; **only the final level** is returned.
      - Returns a **dict** with keys from `head_names`, e.g.:
          { main_name, f"{main_name}_conf", aux_name, f"{aux_name}_conf", feat_name, f"{feat_name}_conf" }
      - DualDPT weights can be loaded directly into TriDPT; only feat head weights are new.
    """

    def __init__(
        self,
        dim_in: int,
        *,
        patch_size: int = 14,
        output_dim: int = 2,
        activation: str = "exp",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        pos_embed: bool = True,
        down_ratio: int = 1,
        aux_pyramid_levels: int = 4,
        aux_out1_conv_num: int = 5,
        feat_pyramid_levels: int = 4,
        feat_out1_conv_num: int = 5,
        head_names: Tuple[str, str, str] = ("depth", "ray", "seg"),
        feat_output_dim: int = 8,
        layer_indices: Tuple[int, int, int, int] = (0, 1, 2, 3),
    ) -> None:
        # Initialize DualDPT with first two heads
        super().__init__(
            dim_in=dim_in,
            patch_size=patch_size,
            output_dim=output_dim,
            activation=activation,
            conf_activation=conf_activation,
            features=features,
            out_channels=out_channels,
            pos_embed=pos_embed,
            down_ratio=down_ratio,
            aux_pyramid_levels=aux_pyramid_levels,
            aux_out1_conv_num=aux_out1_conv_num,
            head_names=(head_names[0], head_names[1]),  # Pass only first two heads to parent
            layer_indices=layer_indices,
        )

        # -------------------- additional configuration for feat head --------------------
        self.feat_levels = feat_pyramid_levels
        self.feat_out1_conv_num = feat_out1_conv_num

        # Override head_names to include all three heads
        self.head_main, self.head_aux, self.head_feat = head_names

        # -------------------- feat head: fusion chain + output layers --------------------
        # Feature fusion chain (completely separate; no sharing)
        self.scratch.refinenet1_feat = _make_fusion_block(features)
        self.scratch.refinenet2_feat = _make_fusion_block(features)
        self.scratch.refinenet3_feat = _make_fusion_block(features)
        self.scratch.refinenet4_feat = _make_fusion_block(features, has_residual=False)

        # Feat pre-head per level (we will only *return final level*)
        head_features_1 = features
        head_features_2 = 32
        self.scratch.output_conv1_feat = nn.ModuleList(
            [self._make_feat_out1_block(head_features_1) for _ in range(self.feat_levels)]
        )

        # Feat final projection per level (e.g., for segmentation)
        use_ln = True
        ln_seq = (
            [Permute((0, 2, 3, 1)), nn.LayerNorm(head_features_2), Permute((0, 3, 1, 2))]
            if use_ln
            else []
        )
        self.scratch.output_conv2_feat = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1
                    ),
                    *ln_seq,
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_features_2, feat_output_dim, kernel_size=1, stride=1, padding=0),  # e.g., 150 classes for segmentation
                )
                for _ in range(self.feat_levels)
            ]
        )

    # -------------------------------------------------------------------------
    # Override forward to handle three heads
    # -------------------------------------------------------------------------

    def forward(
        self,
        feats: List[torch.Tensor],
        H: int,
        W: int,
        patch_start_idx: int,
        chunk_size: int = 8,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            feats:             List of 4 tensors with [B, S, T, C] from transformer.
            H:                 Original image height.
            W:                 Original image width.
            patch_start_idx:   Patch-token start in the token sequence (to drop non-patch tokens).
            chunk_size:        Optional chunking along S for memory.

        Returns:
            Dict[str, Tensor] with keys based on `head_names`, e.g.:
                self.head_main, f"{self.head_main}_conf",
                self.head_aux,  f"{self.head_aux}_conf",
                self.head_feat, f"{self.head_feat}_conf"
            Shapes:
              main:    [B, S, out_dim, H/down_ratio, W/down_ratio]
              main_cf: [B, S, 1,       H/down_ratio, W/down_ratio]
              aux:     [B, S, 7,       H/down_ratio, W/down_ratio]
              aux_cf:  [B, S, 1,       H/down_ratio, W/down_ratio]
              feat:    [B, S, 150,     H/down_ratio, W/down_ratio]
              feat_cf: [B, S, 1,       H/down_ratio, W/down_ratio]
        """
        B, S, N, C = feats[0][0].shape
        feats = [feat[0].reshape(B * S, N, C) for feat in feats]
        if chunk_size is None or chunk_size >= S:
            out_dict = self._forward_impl(feats, H, W, patch_start_idx)
            out_dict = {k: v.reshape(B, S, *v.shape[1:]) for k, v in out_dict.items()}
            return Dict(out_dict)
        out_dicts = []
        for s0 in range(0, B * S, chunk_size):
            s1 = min(s0 + chunk_size, B * S)
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
    # Override _forward_impl to handle three heads
    # -------------------------------------------------------------------------

    def _forward_impl(
        self,
        feats: List[torch.Tensor],
        H: int,
        W: int,
        patch_start_idx: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Internal forward implementation for a single chunk.
        Extends parent's _forward_impl to add the third head.
        """
        B, _, C = feats[0].shape
        ph, pw = H // self.patch_size, W // self.patch_size
        resized_feats = []
        for stage_idx, take_idx in enumerate(self.intermediate_layer_idx):
            x = feats[take_idx][:, patch_start_idx:]
            x = self.norm(x)
            x = x.permute(0, 2, 1).reshape(B, C, ph, pw)  # [B*S, C, ph, pw]

            x = self.projects[stage_idx](x)
            if self.pos_embed:
                x = self._add_pos_embed(x, W, H)
            x = self.resize_layers[stage_idx](x)  # align scales
            resized_feats.append(x)

        # 2) Fuse pyramid (main & aux & feat are completely independent)
        fused_main, fused_aux_pyr, fused_feat_pyr = self._fuse(resized_feats)

        # 3) Upsample to target resolution and (optional) add pos-embed again
        h_out = int(ph * self.patch_size / self.down_ratio)
        w_out = int(pw * self.patch_size / self.down_ratio)

        from depth_anything_3.model.utils.head_utils import custom_interpolate
        
        fused_main = custom_interpolate(
            fused_main, (h_out, w_out), mode="bilinear", align_corners=True
        )
        if self.pos_embed:
            fused_main = self._add_pos_embed(fused_main, W, H)

        # Primary head: conv1 -> conv2 -> activate
        main_logits = self.scratch.output_conv2(fused_main)
        fmap = main_logits.permute(0, 2, 3, 1)
        main_pred = self._apply_activation_single(fmap[..., :-1], self.activation)
        main_conf = self._apply_activation_single(fmap[..., -1], self.conf_activation)

        # Auxiliary head (multi-level inside) -> only last level returned (after activation)
        last_aux = fused_aux_pyr[-1]
        if self.pos_embed:
            last_aux = self._add_pos_embed(last_aux, W, H)
        last_aux_logits = self.scratch.output_conv2_aux[-1](last_aux)
        fmap_last = last_aux_logits.permute(0, 2, 3, 1)
        aux_pred = self._apply_activation_single(fmap_last[..., :-1], "linear")
        aux_conf = self._apply_activation_single(fmap_last[..., -1], self.conf_activation)

        # Feature head (multi-level inside) -> only last level returned (after activation)
        last_feat = fused_feat_pyr[-1]
        last_feat = custom_interpolate(
            last_feat, (h_out, w_out), mode="bilinear", align_corners=True
        )
        if self.pos_embed:
            last_feat = self._add_pos_embed(last_feat, W, H)
        last_feat_logits = self.scratch.output_conv2_feat[-1](last_feat)
        fmap_feat = last_feat_logits.permute(0, 2, 3, 1)
        feat_pred = self._apply_activation_single(fmap_feat, "linear")

        return {
            self.head_main: main_pred.squeeze(-1),
            f"{self.head_main}_conf": main_conf,
            self.head_aux: aux_pred,
            f"{self.head_aux}_conf": aux_conf,
            self.head_feat: feat_pred,
        }

    # -------------------------------------------------------------------------
    # Override _fuse to handle three heads
    # -------------------------------------------------------------------------

    def _fuse(self, feats: List[torch.Tensor]) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Feature pyramid fusion for three heads.
        Returns:
            fused_main: Tensor at finest scale (after refinenet1)
            aux_pyr:    List of aux tensors at each level (pre out_conv1_aux)
            feat_pyr:   List of feat tensors at each level (pre out_conv1_feat)
        """
        l1, l2, l3, l4 = feats

        l1_rn = self.scratch.layer1_rn(l1)
        l2_rn = self.scratch.layer2_rn(l2)
        l3_rn = self.scratch.layer3_rn(l3)
        l4_rn = self.scratch.layer4_rn(l4)

        # level 4 -> 3
        out = self.scratch.refinenet4(l4_rn, size=l3_rn.shape[2:])
        aux_out = self.scratch.refinenet4_aux(l4_rn, size=l3_rn.shape[2:])
        feat_out = self.scratch.refinenet4_feat(l4_rn, size=l3_rn.shape[2:])
        aux_list: List[torch.Tensor] = []
        feat_list: List[torch.Tensor] = []
        if self.aux_levels >= 4:
            aux_list.append(aux_out)
        if self.feat_levels >= 4:
            feat_list.append(feat_out)

        # level 3 -> 2
        out = self.scratch.refinenet3(out, l3_rn, size=l2_rn.shape[2:])
        aux_out = self.scratch.refinenet3_aux(aux_out, l3_rn, size=l2_rn.shape[2:])
        feat_out = self.scratch.refinenet3_feat(feat_out, l3_rn, size=l2_rn.shape[2:])
        if self.aux_levels >= 3:
            aux_list.append(aux_out)
        if self.feat_levels >= 3:
            feat_list.append(feat_out)

        # level 2 -> 1
        out = self.scratch.refinenet2(out, l2_rn, size=l1_rn.shape[2:])
        aux_out = self.scratch.refinenet2_aux(aux_out, l2_rn, size=l1_rn.shape[2:])
        feat_out = self.scratch.refinenet2_feat(feat_out, l2_rn, size=l1_rn.shape[2:])
        if self.aux_levels >= 2:
            aux_list.append(aux_out)
        if self.feat_levels >= 2:
            feat_list.append(feat_out)

        # level 1 (final)
        out = self.scratch.refinenet1(out, l1_rn)
        aux_out = self.scratch.refinenet1_aux(aux_out, l1_rn)
        feat_out = self.scratch.refinenet1_feat(feat_out, l1_rn)
        aux_list.append(aux_out)
        feat_list.append(feat_out)

        out = self.scratch.output_conv1(out)
        aux_list = [self.scratch.output_conv1_aux[i](aux) for i, aux in enumerate(aux_list)]
        feat_list = [self.scratch.output_conv1_feat[i](feat) for i, feat in enumerate(feat_list)]

        return out, aux_list, feat_list

    # -------------------------------------------------------------------------
    # Helper method for feat head
    # -------------------------------------------------------------------------

    def _make_feat_out1_block(self, in_ch: int) -> nn.Sequential:
        """Factory for the feat pre-head stack before the final 1x1 projection."""
        if self.feat_out1_conv_num == 5:
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1),
                nn.Conv2d(in_ch // 2, in_ch, 3, 1, 1),
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1),
                nn.Conv2d(in_ch // 2, in_ch, 3, 1, 1),
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1),
            )
        if self.feat_out1_conv_num == 3:
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1),
                nn.Conv2d(in_ch // 2, in_ch, 3, 1, 1),
                nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1),
            )
        if self.feat_out1_conv_num == 1:
            return nn.Sequential(nn.Conv2d(in_ch, in_ch // 2, 3, 1, 1))
        raise ValueError(f"feat_out1_conv_num {self.feat_out1_conv_num} not supported")
