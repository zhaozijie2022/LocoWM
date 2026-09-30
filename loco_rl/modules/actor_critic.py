# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL actor-critic with LocoWM's noise bounds and observation helpers."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.modules.actor_critic import ActorCritic as RslRlActorCritic


class ActorCritic(RslRlActorCritic):
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        super().__init__(
            num_actor_obs, num_critic_obs, num_actions,
            actor_hidden_dims, critic_hidden_dims, activation,
            init_noise_std, noise_std_type, **kwargs,
        )
        self.init_noise_std = init_noise_std

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Keep nn.Module's return value and assign support; upstream returns bool.
        return nn.Module.load_state_dict(self, state_dict, strict=strict, assign=assign)

    def update_distribution(self, observations):
        # compute mean
        mean = self.actor(observations)
        # compute standard deviation
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
            std = torch.clamp(std, min=1e-6, max=5.0)
        elif self.noise_std_type == "log":
            log_std = torch.clamp(self.log_std, min=-10.0, max=2.0)
            std = torch.exp(log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        # create distribution
        self.distribution = Normal(mean, std)

    def reset_init_std(self):
        if self.noise_std_type == "scalar":
            self.std.data = self.init_noise_std * torch.ones_like(self.std.data)
        elif self.noise_std_type == "log":
            self.log_std.data = torch.log(self.init_noise_std * torch.ones_like(self.log_std.data))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

    def get_actor_critic_obs_from_obs_dict(self, obs_dict):
        actor_obs = obs_dict["policy"]
        critic_obs = obs_dict.get("critic", actor_obs)
        return actor_obs, critic_obs
