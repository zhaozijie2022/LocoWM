from __future__ import annotations
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Curriculum based on the distance the robot walked when commanded to move at a desired velocity.

    This term is used to increase the difficulty of the terrain when the robot walks far enough and decrease the
    difficulty when the robot walks less than half of the distance required by the commanded velocity.

    .. note::
        It is only possible to use this term with the terrain type ``generator``. For further information
        on different terrain types, check the :class:`isaaclab.terrains.TerrainImporter` class.

    Returns:
        The mean terrain level for the given environment ids.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain level
    return torch.mean(terrain.terrain_levels.float())


def command_axis_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,  # track_lin_vel_x_exp
    range_multiplier: Sequence[float] = (0.1, 1.0),  # Initial and final curriculum levels.
    delta: float = 0.1,
    upper_threshold: float = 0.8,
    lower_threshold: float = 0.5,
    ema_alpha: float = 0.5,
) -> torch.Tensor:
    """Update the command-axis range using per-environment tracking-reward EMAs."""
    base_velocity = env.command_manager.get_term("base_velocity")
    axis_name = reward_term_name.split("_")[-2]
    ema_name = f"_tracking_{axis_name}_ema"
    cmd_level_name = f"_command_{axis_name}_level"

    if env.common_step_counter == 0:
        # Per-environment EMA of tracking performance.
        setattr(base_velocity, ema_name, torch.zeros(env.num_envs, device=env.device, dtype=torch.float32))
        # Initialize each environment at the minimum command level.
        setattr(
            base_velocity,
            cmd_level_name,
            torch.ones(env.num_envs, device=env.device, dtype=torch.float32) * range_multiplier[0],
        )
        # Keep previous levels for command sampling.
        setattr(
            base_velocity,
            f"_previous_{cmd_level_name}",
            torch.ones(env.num_envs, device=env.device, dtype=torch.float32) * range_multiplier[0],
        )

    ema = getattr(base_velocity, ema_name)
    cmd_level = getattr(base_velocity, cmd_level_name)
    previous_cmd_level = getattr(base_velocity, f"_previous_{cmd_level_name}")

    cmd_level_mean = cmd_level.mean().item()

    if len(env_ids) > 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)

        # Normalize by the maximum episode duration so early falls reduce performance.
        per_env_reward = (episode_sums[env_ids] / env.cfg.episode_length_s).float()

        # Update the per-environment EMA.
        ema[env_ids] = (1.0 - ema_alpha) * ema[env_ids] + ema_alpha * per_env_reward
        previous_cmd_level[env_ids] = cmd_level[env_ids]

        # cmd_level move up or down
        cmd_level[env_ids] = torch.where(
            ema[env_ids] > (upper_threshold * reward_term_cfg.weight),
            torch.clamp(cmd_level[env_ids] + delta, min=range_multiplier[0], max=range_multiplier[1]),
            cmd_level[env_ids],
        )
        cmd_level[env_ids] = torch.where(
            ema[env_ids] < (lower_threshold * reward_term_cfg.weight),
            torch.clamp(cmd_level[env_ids] - delta, min=range_multiplier[0], max=range_multiplier[1]),
            cmd_level[env_ids],
        )

    return torch.tensor(cmd_level_mean, device=env.device)
