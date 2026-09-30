# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Adapt RSL-RL storage to LocoWM's transition and mini-batch interface."""

from __future__ import annotations

import torch
from rsl_rl.storage import RolloutStorage as RslRlRolloutStorage


class RolloutStorage(RslRlRolloutStorage):
    class Transition(RslRlRolloutStorage.Transition):
        @property
        def critic_observations(self):
            return self.privileged_observations

        @critic_observations.setter
        def critic_observations(self, value):
            self.privileged_observations = value

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        rnd_state_shape=None,
        device="cpu",
    ):
        super().__init__(
            "rl", num_envs, num_transitions_per_env, obs_shape,
            privileged_obs_shape, actions_shape, rnd_state_shape, device,
        )

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        for batch in super().mini_batch_generator(num_mini_batches, num_epochs):
            # Feed-forward PPO uses no recurrent hidden states or masks.
            yield (*batch[:9], batch[-1])

    def get_statistics(self):
        done = self.dones
        done[-1] = 1
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat(
            (flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero(as_tuple=False)[:, 0])
        )
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()
