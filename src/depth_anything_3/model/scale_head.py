from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_act(name: str) -> nn.Module:
    name = name.lower()
    if name in {"relu"}:
        return nn.ReLU(inplace=True)
    if name in {"gelu"}:
        return nn.GELU()
    if name in {"silu", "swish"}:
        return nn.SiLU(inplace=True)
    if name in {"tanh"}:
        return nn.Tanh()
    if name in {"identity", "linear", "none"}:
        return nn.Identity()
    raise ValueError(f"Unknown activation: {name}")


class ScaleHead(nn.Module):
    """
    一个简单的 n 层 MLP，用于从 token/特征向量回归尺度相关的输出（默认 1 维）。

    约定输入通常是 backbone 最后一层的 cls/cam token：
    - (B, N, C) -> (B, N, out_dim)；可选对 N 做 reduce 得到 (B, out_dim)
    - (B, C) -> (B, out_dim)
    """

    def __init__(
        self,
        dim_in: int,
        hidden_dim: int = 1024,
        num_layers: int = 3,
        out_dim: int = 1,
        activation: str = "relu",
        dropout: float = 0.0,
        reduce: Literal["none", "mean"] = "none",
        final_activation: str = "linear",
    ) -> None:
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        assert dim_in > 0 and hidden_dim > 0 and out_dim > 0
        assert reduce in {"none", "mean"}

        self.dim_in = dim_in
        self.out_dim = out_dim
        self.reduce = reduce

        act = _make_act(activation)
        layers: list[nn.Module] = []

        if num_layers == 1:
            layers.append(nn.Linear(dim_in, out_dim))
        else:
            layers.append(nn.Linear(dim_in, hidden_dim))
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            for _ in range(num_layers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(act)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dim, out_dim))

        self.mlp = nn.Sequential(*layers)

        final_activation = final_activation.lower()
        if final_activation in {"linear", "identity", "none"}:
            self.final_act: nn.Module | None = None
        elif final_activation in {"softplus"}:
            self.final_act = nn.Softplus()
        elif final_activation in {"softplus_1", "softplus+1"}:
            # 保证输出严格为正且 >= 1（常见用于 scale）
            self.final_act = nn.Identity()
        elif final_activation in {"exp"}:
            self.final_act = nn.Identity()
        elif final_activation in {"sigmoid"}:
            self.final_act = nn.Identity()
        else:
            raise ValueError(f"Unknown final_activation: {final_activation}")
        self._final_activation_name = final_activation

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.shape[-1] != self.dim_in:
            raise ValueError(
                f"ScaleHead expected last dim={self.dim_in}, got {feat.shape[-1]}"
            )

        orig_shape = feat.shape[:-1]
        x = feat.reshape(-1, feat.shape[-1])
        y = self.mlp(x.float()).reshape(*orig_shape, self.out_dim)

        if self._final_activation_name in {"softplus_1", "softplus+1"}:
            y = F.softplus(y) + 1.0
        elif self._final_activation_name == "exp":
            y = torch.exp(y)
        elif self._final_activation_name == "sigmoid":
            y = torch.sigmoid(y)
        elif self.final_act is not None:
            y = self.final_act(y)

        # 常见输入是 (B, N, C)，这里支持对 N 做聚合
        if self.reduce == "mean" and y.dim() == 3:
            y = y.mean(dim=1)
        return y