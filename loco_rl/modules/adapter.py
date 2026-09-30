# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage-2 residual adapter.

The MLP maps concatenated observations, frozen-policy actions and world-model
features to a residual action. The final action is the base action plus this
residual. FiLM conditioning is reserved but not implemented."""

from __future__ import annotations

import torch
import torch.nn as nn

from loco_rl.models import MLP


class Adapter(nn.Module):
    """Map (observations, base actions, world-model features) to residual actions."""

    def __init__(self, obs_dim: int, a0_dim: int, wm_feat_dim: int, num_actions: int, cfg: dict):
        super().__init__()
        self.adapter_type = cfg.get("adapter_type", "mlp")
        hidden_dims = list(cfg.get("hidden_dims", [256, 128]))
        activation = cfg.get("activation", "elu")

        # Separate base inputs from world-model conditioning features.
        self.main_dim = obs_dim + a0_dim
        self.cond_dim = wm_feat_dim

        if self.adapter_type == "mlp":
            self.net = MLP(self.main_dim + self.cond_dim, hidden_dims, num_actions, activation=activation)
            self._zero_init_last_layer(self.net)
        elif self.adapter_type == "film":
            raise NotImplementedError("FiLM adapters are not implemented; use adapter_type=mlp.")
        else:
            raise ValueError(f"Unknown adapter_type: {self.adapter_type} (expected 'mlp' or 'film')")

        print(f"Adapter ({self.adapter_type}, main={self.main_dim} + cond={self.cond_dim} -> a_r={num_actions})")

    @staticmethod
    def _zero_init_last_layer(mlp: nn.Module):
        """Zero the output layer so the initial residual is exactly zero."""
        last_linear = None
        for m in mlp.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is not None:
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

    def forward(self, obs: torch.Tensor, a0: torch.Tensor, wm_feat: torch.Tensor) -> torch.Tensor:
        if self.adapter_type == "mlp":
            return self.net(torch.cat([obs, a0, wm_feat], dim=-1))

        raise NotImplementedError(f"adapter_type={self.adapter_type} forward is not implemented")
