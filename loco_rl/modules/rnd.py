# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL RND with the original LocoWM target-network mode."""

from rsl_rl.modules.rnd import RandomNetworkDistillation as RslRlRandomNetworkDistillation


class RandomNetworkDistillation(RslRlRandomNetworkDistillation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Preserve the previous target mode (the current MLP has no dropout/BN).
        self.target.train()
