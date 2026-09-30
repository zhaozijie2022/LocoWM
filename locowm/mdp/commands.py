from __future__ import annotations
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.utils import configclass
from isaaclab.envs.mdp.commands import UniformVelocityCommandCfg


class UniformVelocityCommandMultiSampling(UniformVelocityCommand):
    """UniformVelocityCommand + multi-sampling ranges + initial zero command steps + binary maximal command."""

    def __init__(self, cfg: UniformVelocityCommandMultiSamplingCfg, env: "ManagerBasedEnv | ManagerBasedRLEnv"):
        super().__init__(cfg, env)

        # Initialize three-bin sampling.
        p = float(self.cfg.new_command_probs)
        mid = 1.0 - 2.0 * p
        if mid < 0:
            raise ValueError(f"new_command_probs too large: {p}, must satisfy 1-2p >= 0.")
        self.sampling_probs = torch.tensor([p, mid, p], device=self.device, dtype=torch.float32)

        # command buffer + init-zero logic
        self.vel_command_b_buffer = torch.zeros_like(
            self.vel_command_b
        )  # Commands before the zero-command mask, shape [num_envs, 3].
        self.initial_zero_command_steps = int(self.cfg.initial_zero_command_steps)

        # bang_bang control combos
        combos = []
        for i in [self.cfg.ranges.lin_vel_x[0], 0, self.cfg.ranges.lin_vel_x[1]]:
            for j in [self.cfg.ranges.lin_vel_y[0], 0, self.cfg.ranges.lin_vel_y[1]]:
                for k in [self.cfg.ranges.ang_vel_z[0], 0, self.cfg.ranges.ang_vel_z[1]]:
                    if i > 1e-2 or j > 1e-2 or k > 1e-2:
                        combos.append([i, j, k])  # Avoid sampling a zero command.
        self.bang_bang_commands = torch.tensor(combos, device=self.device, dtype=torch.float32)

        self.metrics["tracking_x_ema"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["tracking_y_ema"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["tracking_z_ema"] = torch.zeros(self.num_envs, device=self.device)

    def _update_metrics(self):

        super()._update_metrics()  # Update linear-velocity and yaw tracking metrics.

        if hasattr(self, "_command_x_level"):
            self.metrics["tracking_x_ema"][:] = float(self._tracking_x_ema.mean().item())

        if hasattr(self, "_command_y_level"):
            self.metrics["tracking_y_ema"][:] = float(self._tracking_y_ema.mean().item())

        if hasattr(self, "_command_z_level"):
            self.metrics["tracking_z_ema"][:] = float(self._tracking_z_ema.mean().item())

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return

        super()._resample_command(env_ids)

        # Mix bang-bang and binned sampling.
        p_bang_bang = float(self.cfg.bang_bang_envs)
        use_bang_bang = torch.rand(env_ids.numel(), device=self.device) < p_bang_bang
        bang_ids = env_ids[use_bang_bang]
        bucket_ids = env_ids[~use_bang_bang]
        if bang_ids.numel() > 0:
            idx = torch.randint(0, self.bang_bang_commands.shape[0], (bang_ids.numel(),), device=self.device)
            self.vel_command_b[bang_ids] = self.bang_bang_commands[idx]
        if bucket_ids.numel() > 0:
            if hasattr(self, "_command_x_level"):
                self._sample_dim_with_bins(bucket_ids, dim=0, ranges=self.cfg.ranges.lin_vel_x)
            if hasattr(self, "_command_y_level"):
                self._sample_dim_with_bins(bucket_ids, dim=1, ranges=self.cfg.ranges.lin_vel_y)
            if hasattr(self, "_command_z_level"):
                self._sample_dim_with_bins(bucket_ids, dim=2, ranges=self.cfg.ranges.ang_vel_z)

        self.vel_command_b_buffer[env_ids] = self.vel_command_b[env_ids].clone()

        self._set_zero_command_for_beginning_steps()

    def _sample_dim_with_bins(self, env_ids: torch.Tensor, dim: int, ranges: tuple[float, float]):
        """Sample ranges derived from the current and previous command curriculum levels.

        The curriculum supplies per-axis levels; ranges gives the unscaled interval."""
        # Non-increasing levels sample the full scaled interval.
        # Increasing levels use three bins around the previous interval.
        # Choose either expansion tail with probability p, or the previous interval with 1-2p.
        min_v, max_v = ranges

        axis_name = "xyz"[dim]
        level = getattr(self, f"_command_{axis_name}_level")[env_ids]
        prev_level = getattr(self, f"_previous__command_{axis_name}_level")[env_ids]

        # Non-increasing level: sample a single interval.
        mask_same = level <= prev_level
        if mask_same.any():
            low = min_v * level[mask_same]
            high = max_v * level[mask_same]
            n = mask_same.sum().item()
            u = torch.rand(n, device=self.device, dtype=torch.float32)
            self.vel_command_b[env_ids[mask_same], dim] = low + u * (high - low)

        # Increasing level: choose among three bins.
        mask_expand = level > prev_level
        if mask_expand.any():
            n = mask_expand.sum().item()
            u = torch.rand(n, device=self.device, dtype=torch.float32)

            bin_idx = torch.multinomial(self.sampling_probs, n, replacement=True)
            lvl = level[mask_expand]
            prev_lvl = prev_level[mask_expand]

            # bin 0: [min_v*lvl, min_v*prev_lvl], p
            mask = bin_idx == 0
            if mask.any():
                low = min_v * lvl[mask]
                high = min_v * prev_lvl[mask]
                uu = torch.rand(mask.sum().item(), device=self.device, dtype=torch.float32)
                self.vel_command_b[env_ids[mask_expand][mask], dim] = low + uu * (high - low)

            # bin 1: [min_v*pl, max_v*pl]，1-2p
            mask = bin_idx == 1
            if mask.any():
                low = min_v * prev_lvl[mask]
                high = max_v * prev_lvl[mask]
                uu = torch.rand(mask.sum().item(), device=self.device, dtype=torch.float32)
                self.vel_command_b[env_ids[mask_expand][mask], dim] = low + uu * (high - low)

            # bin 2: [max_v*pl, max_v*l]，p
            mask = bin_idx == 2
            if mask.any():
                low = max_v * prev_lvl[mask]
                high = max_v * lvl[mask]
                uu = torch.rand(mask.sum().item(), device=self.device, dtype=torch.float32)
                self.vel_command_b[env_ids[mask_expand][mask], dim] = low + uu * (high - low)

    def _update_command(self):

        self._set_zero_command_for_beginning_steps()
        self._recover_command_for_beginning_steps()
        super()._update_command()

    def _set_zero_command_for_beginning_steps(self):
        """Hold commands at zero for initial_zero_command_steps after each reset."""
        if self.initial_zero_command_steps <= 0:
            return
        mask = self._env.episode_length_buf < self.initial_zero_command_steps
        if mask.any():
            ids = mask.nonzero(as_tuple=True)[0]
            self.vel_command_b[ids] = 0.0

    def _recover_command_for_beginning_steps(self):
        """Restore buffered commands after the initial zero-command interval."""
        if self.initial_zero_command_steps <= 0:
            return
        mask = self._env.episode_length_buf == self.initial_zero_command_steps
        if mask.any():
            ids = mask.nonzero(as_tuple=True)[0]
            self.vel_command_b[ids] = self.vel_command_b_buffer[ids].clone()


@configclass
class UniformVelocityCommandMultiSamplingCfg(UniformVelocityCommandCfg):
    class_type: type = UniformVelocityCommandMultiSampling

    new_command_probs: float = 0.15
    initial_zero_command_steps: int = 50  # Number of initial episode steps with zero commands.
    bang_bang_envs: float = 0.05  # Fraction of environments using bang-bang commands.


# =============================================================================
# Evaluation commands ramp to a fixed forward speed; lateral/yaw commands stay zero.
# =============================================================================


class EvalRampVelocityCommand(UniformVelocityCommand):
    """Ramp forward speed linearly with episode time, then hold the target speed.

    Lateral and yaw commands remain zero. With zero spawn yaw, this gives a
    flat-ground acceleration phase followed by forward terrain traversal."""

    cfg: "EvalRampVelocityCommandCfg"

    def __init__(self, cfg: "EvalRampVelocityCommandCfg", env):
        super().__init__(cfg, env)
        self.target_speed = float(cfg.target_speed)
        self.accel_steps = max(1, int(cfg.accel_steps))

    def _resample_command(self, env_ids: Sequence[int]):
        # Commands are determined by episode time rather than random sampling.
        return

    def _update_command(self):
        t = self._env.episode_length_buf.to(self.vel_command_b.dtype)
        frac = torch.clamp(t / self.accel_steps, max=1.0)
        self.vel_command_b[:, 0] = frac * self.target_speed
        self.vel_command_b[:, 1] = 0.0
        self.vel_command_b[:, 2] = 0.0


@configclass
class EvalRampVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = EvalRampVelocityCommand

    target_speed: float = 1.0
    """Forward velocity after the acceleration ramp (m/s)."""
    accel_steps: int = 125
    """Steps to ramp from zero to target_speed; 125 steps at 50 Hz is 2.5 seconds."""
