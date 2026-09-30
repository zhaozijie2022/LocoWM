from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    """Terminate when the actor move too close to the edge of the terrain.

    If the actor moves too close to the edge of the terrain, the termination is activated. The distance
    to the edge of the terrain is calculated based on the size of the terrain and the distance buffer.
    """
    if env.scene.cfg.terrain.terrain_type == "plane":
        # we have infinite terrain because it is a plane
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size
        n_rows, n_cols = terrain_gen_cfg.num_rows, terrain_gen_cfg.num_cols
        border_width = terrain_gen_cfg.border_width
        # compute the size of the map
        map_width = n_rows * grid_width + 2 * border_width
        map_height = n_cols * grid_length + 2 * border_width

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]

        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
        y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")


# =============================================================================
# Success-evaluation termination terms.
# A payload center below the base center counts as a drop.
# Drops during the command ramp reset the episode without counting a failure.
# =============================================================================


def _accel_steps(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> int:
    """Read the acceleration duration from the evaluation command term."""
    try:
        return int(getattr(env.command_manager.get_term(command_name), "accel_steps", 0))
    except Exception:
        return 0


def object_dropped(
    env: ManagerBasedRLEnv,
    during_accel: bool,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Detect a payload root below the base root during or after acceleration.

    Acceleration-phase drops trigger retries; later drops count as failures.
    Root positions approximate centers of mass to avoid repeated physics COM
    queries across all environments."""
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    dropped = obj.data.root_pos_w[:, 2] < robot.data.root_pos_w[:, 2]
    in_accel = env.episode_length_buf < _accel_steps(env, command_name)
    return dropped & (in_accel if during_accel else ~in_accel)


def forward_traversed(
    env: ManagerBasedRLEnv,
    distance: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Succeed after forward traversal with the payload still above the base.

    With zero spawn yaw, displacement is root_pos_w.x minus env_origin.x.
    Root positions approximate centers of mass, as in object_dropped()."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    disp_x = robot.data.root_pos_w[:, 0] - env.scene.env_origins[:, 0]
    up = obj.data.root_pos_w[:, 2] >= robot.data.root_pos_w[:, 2]
    return (disp_x > distance) & up
