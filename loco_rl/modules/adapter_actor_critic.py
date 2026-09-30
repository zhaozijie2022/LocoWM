# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Stage-2 actor-critic with a frozen base policy and world model.

The action mean is the base action plus the learned residual. Gradients update
only the adapter, new transport critic and action-noise parameters.

A Stage-1 policy checkpoint is required. Omitting the world-model checkpoint
uses zero-valued features for the No-WM ablation; a supplied missing path is
an error."""

from __future__ import annotations

import os
import warnings

import torch
import torch.nn as nn
from torch.distributions import Normal

from loco_rl.modules.actor_critic import ActorCritic
from loco_rl.modules.adapter import Adapter
from loco_rl.modules.world_model import WorldModel
from loco_rl.utils import resolve_nn_activation


class AdapterActorCritic(ActorCritic):
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        # Stage-2 settings.
        policy_checkpoint: str = "",
        world_model_checkpoint: str = "",
        adapter: dict | None = None,
        world_model: dict | None = None,
        **kwargs,
    ):
        # Build the new critic and retain the base noise parameterization.
        super().__init__(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
            **kwargs,
        )
        # Actions come from the frozen policy plus the adapter.
        del self.actor

        # Load the actor.* subset from the Stage-1 policy.
        self.frozen_policy = self._build_actor_mlp(num_actor_obs, actor_hidden_dims, num_actions, activation)
        self._load_frozen_policy(policy_checkpoint)
        self._freeze(self.frozen_policy)

        # Construct the frozen world model for inference only.
        wm_cfg = dict(world_model or {})
        self.frozen_wm = WorldModel(wm_cfg)
        self.frozen_wm.build(num_actor_obs, num_actions, for_training=False, device="cpu")
        self._load_frozen_wm(world_model_checkpoint)
        self._freeze(self.frozen_wm)
        self.wm_feat_dim = self.frozen_wm.chunk_len * self.frozen_wm.num_targets

        # Trainable residual adapter.
        self.adapter = Adapter(
            obs_dim=num_actor_obs,
            a0_dim=num_actions,
            wm_feat_dim=self.wm_feat_dim,
            num_actions=num_actions,
            cfg=dict(adapter or {}),
        )

    # region build/load helpers
    @staticmethod
    def _build_actor_mlp(input_dim, hidden_dims, num_actions, activation_name):
        """Match ActorCritic.actor and its layer keys for Stage-1 checkpoint loading."""
        activation = resolve_nn_activation(activation_name)
        layers = [nn.Linear(input_dim, hidden_dims[0]), activation]
        for i in range(len(hidden_dims)):
            if i == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[i], num_actions))
            else:
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                layers.append(activation)
        return nn.Sequential(*layers)

    def _load_frozen_policy(self, path: str):
        """Load actor.* weights from a required Stage-1 model_state_dict."""
        if not path:
            raise ValueError("[AdapterActorCritic] Stage 2 requires a Stage-1 policy_checkpoint.")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[AdapterActorCritic] policy_checkpoint not found: '{path}'.")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        model_sd = ckpt["model_state_dict"]
        actor_sd = {k[len("actor.") :]: v for k, v in model_sd.items() if k.startswith("actor.")}
        self.frozen_policy.load_state_dict(actor_sd)
        print(f"[AdapterActorCritic] Loaded frozen policy: {path}")

    def _load_frozen_wm(self, path: str):
        """Load Stage-1 world-model weights, or zero the output layer for No-WM.

        An empty path enables the ablation. A supplied missing path is an error."""
        if not path:
            self._zero_out_last_linear(self.frozen_wm.net)
            warnings.warn("[AdapterActorCritic] No world_model_checkpoint supplied; using zero world-model features.")
            return
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[AdapterActorCritic] world_model_checkpoint not found: '{path}'.")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.frozen_wm.load_state_dict(ckpt["world_model_state_dict"])
        print(f"[AdapterActorCritic] Loaded frozen world model: {path}")

    @staticmethod
    def _zero_out_last_linear(net: nn.Module):
        """Zero the last linear layer so all inputs produce zero outputs."""
        last_linear = None
        for m in net.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is not None:
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

    @staticmethod
    def _freeze(module: nn.Module):
        for p in module.parameters():
            p.requires_grad_(False)
        module.eval()

    def _base_action_and_feat(self, observations):
        """Compute the frozen base action and flattened predictions without gradients."""
        with torch.no_grad():
            a0 = self.frozen_policy(observations)
            wm_feat = self.frozen_wm.forward(observations, a0)
            wm_feat = wm_feat.reshape(wm_feat.shape[0], -1)  # [B, chunk_len*num_targets]
        return a0, wm_feat

    def _std_from(self, mean):
        """Apply the base-policy noise bounds for scalar or log standard deviations."""
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
            std = torch.clamp(std, min=1e-6, max=5.0)
        elif self.noise_std_type == "log":
            log_std = torch.clamp(self.log_std, min=-10.0, max=2.0)
            std = torch.exp(log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        return std

    def update_distribution(self, observations):
        a0, wm_feat = self._base_action_and_feat(observations)
        a_r = self.adapter(observations, a0, wm_feat)
        mean = a0 + a_r
        self.distribution = Normal(mean, self._std_from(mean))

    def act_inference(self, observations):
        a0, wm_feat = self._base_action_and_feat(observations)
        a_r = self.adapter(observations, a0, wm_feat)
        return a0 + a_r
