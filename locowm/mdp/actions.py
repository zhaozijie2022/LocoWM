from __future__ import annotations
import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.envs.mdp.actions import JointPositionAction
from isaaclab.envs.mdp.actions import JointVelocityActionCfg
from isaaclab.envs.mdp.actions import JointVelocityAction


# region Low Pass Actions


def _compute_lowpass_alpha(control_frequency: float, cut_off_frequency: float) -> float:
    """Compute the EMA coefficient: alpha = 1 - exp(-2*pi*f_cut/f_control).

    Higher alpha means less smoothing. Second-order filtering cascades two
    stages with the same coefficient."""
    return 1.0 - math.exp(-2.0 * math.pi * cut_off_frequency / control_frequency)


class JointPositionLowPassAction(JointPositionAction):
    """Filter policy actions with one or two EMA stages before scaling and offset.

    Each stage stores its previous filtered output, not the raw action."""

    def __init__(self, cfg: JointPositionLowPassActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._order = cfg.order
        self._alpha = _compute_lowpass_alpha(cfg.control_frequency, cfg.cut_off_frequency)
        # Previous outputs of the cascaded IIR stages.
        self._filtered_1 = torch.zeros_like(self._raw_actions)
        self._filtered_2 = torch.zeros_like(self._raw_actions) if cfg.order >= 2 else None

    def process_actions(self, actions: torch.Tensor):
        a = self._alpha
        # First-order EMA: y[t] = a*x[t] + (1-a)*y[t-1].
        y = a * actions + (1.0 - a) * self._filtered_1
        self._filtered_1[:] = y
        if self._order >= 2:
            # The second stage filters the first-stage output.
            y = a * y + (1.0 - a) * self._filtered_2
            self._filtered_2[:] = y
        super().process_actions(y)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._filtered_1[env_ids] = 0.0
        if self._filtered_2 is not None:
            self._filtered_2[env_ids] = 0.0
        super().reset(env_ids)


@configclass
class JointPositionLowPassActionCfg(JointPositionActionCfg):
    class_type: type[ActionTerm] = JointPositionLowPassAction
    control_frequency: float = 50.0  # Hz
    cut_off_frequency: float = 5.0  # Hz
    order: int = 1  # One or two cascaded stages.

    def __post_init__(self) -> None:
        assert self.order >= 1 and self.order <= 2, "order must be 1 or 2"


# region Velocity Low Pass


class JointVelocityLowPassAction(JointVelocityAction):
    """Apply the same cascaded EMA filtering to wheel-velocity actions."""

    def __init__(self, cfg: JointVelocityLowPassActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._order = cfg.order
        self._alpha = _compute_lowpass_alpha(cfg.control_frequency, cfg.cut_off_frequency)
        self._filtered_1 = torch.zeros_like(self._raw_actions)
        self._filtered_2 = torch.zeros_like(self._raw_actions) if cfg.order >= 2 else None

    def process_actions(self, actions: torch.Tensor):
        a = self._alpha
        y = a * actions + (1.0 - a) * self._filtered_1
        self._filtered_1[:] = y
        if self._order >= 2:
            y = a * y + (1.0 - a) * self._filtered_2
            self._filtered_2[:] = y
        super().process_actions(y)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._filtered_1[env_ids] = 0.0
        if self._filtered_2 is not None:
            self._filtered_2[env_ids] = 0.0
        super().reset(env_ids)


@configclass
class JointVelocityLowPassActionCfg(JointVelocityActionCfg):
    class_type: type[ActionTerm] = JointVelocityLowPassAction
    control_frequency: float = 50.0  # Hz
    cut_off_frequency: float = 5.0  # Hz
    order: int = 1  # One or two cascaded stages.

    def __post_init__(self) -> None:
        assert self.order >= 1 and self.order <= 2, "order must be 1 or 2"


# endregion
