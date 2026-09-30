from __future__ import annotations
import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse
from typing import TYPE_CHECKING, Tuple
import isaaclab.envs.mdp.rewards as isaaclab_rewards

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from .observations import ideal_projected_gravity
from .sanitize import sanitize, DEFAULT_CLIP


def stand_still_without_cmd(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # Set asset_cfg.joint_names to the leg joints.
    use_gravity_gating: bool = False,
    gating_max: float = 0.7,
    clip: float = DEFAULT_CLIP,
) -> torch.Tensor:

    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.abs(diff_angle), dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold
    if use_gravity_gating:
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, gating_max) / gating_max
    return sanitize(reward, clip)


def hip_deviation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # params["asset_cfg"].joint_names = hip_joint_names
    clip: float = DEFAULT_CLIP,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    joint_ids = asset_cfg.joint_ids
    q = asset.data.joint_pos[:, joint_ids]
    q0 = asset.data.default_joint_pos[:, joint_ids]

    return sanitize(torch.sum(torch.square(q - q0), dim=1), clip)


def joint_deviation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # params["asset_cfg"].joint_names = leg_joint_names
    clip: float = DEFAULT_CLIP,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    joint_ids = asset_cfg.joint_ids
    q = asset.data.joint_pos[:, joint_ids]
    q0 = asset.data.default_joint_pos[:, joint_ids]

    return sanitize(torch.sum(torch.square(q - q0), dim=1), clip)


def hip_action_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),  # params["asset_cfg"].joint_names = hip_joint_names
) -> torch.Tensor:
    """Penalize hip joint actions (L2 squared)."""
    action = env.action_manager.action
    joint_ids = asset_cfg.joint_ids

    reward = torch.sum(torch.square(action[:, joint_ids]), dim=1)
    return reward


def custom_track_lin_vel_x_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    gravity_z_power: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    clip: float = DEFAULT_CLIP,
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 0] - asset.data.root_lin_vel_b[:, 0])
    reward = torch.exp(-lin_vel_error / std**2)
    if gravity_z_power is not None:
        reward *= -((env.scene["robot"].data.projected_gravity_b[:, 2]) ** gravity_z_power)
    else:
        reward *= -env.scene["robot"].data.projected_gravity_b[:, 2]
    return sanitize(reward, clip)


def custom_track_lin_vel_y_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    gravity_z_power: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    clip: float = DEFAULT_CLIP,
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 1] - asset.data.root_lin_vel_b[:, 1])
    reward = torch.exp(-lin_vel_error / std**2)
    if gravity_z_power is not None:
        reward *= -((env.scene["robot"].data.projected_gravity_b[:, 2]) ** gravity_z_power)
    else:
        reward *= -env.scene["robot"].data.projected_gravity_b[:, 2]
    return sanitize(reward, clip)


def custom_track_ang_vel_z_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    gravity_z_power: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    clip: float = DEFAULT_CLIP,
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    if gravity_z_power is not None:
        reward *= -((env.scene["robot"].data.projected_gravity_b[:, 2]) ** gravity_z_power)
    else:
        reward *= -env.scene["robot"].data.projected_gravity_b[:, 2]
    return sanitize(reward, clip)


def safe_action_rate(
    env: ManagerBasedRLEnv,
    threshold: float = 7.0,
    clip: float = DEFAULT_CLIP,
) -> torch.Tensor:
    delta_action = env.action_manager.action - env.action_manager.prev_action
    if torch.max(torch.abs(delta_action)) > threshold:
        print(f"[WARN] safe_action_rate: delta_action exceeds threshold {threshold}!")
        delta_action = torch.clamp(delta_action, min=-threshold, max=threshold)
    pen = torch.sum(torch.square(delta_action), dim=1)
    return sanitize(pen, clip)


def safe_base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    terrain_height_threshold: Tuple[float, float] = (-0.2, 0.2),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    clip: float = DEFAULT_CLIP,
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        base_ray_hits_w = sensor.data.ray_hits_w[..., 2]
        # Clamp base_ray_hits_w to avoid NaN and Inf (including -Inf/Inf) before usage
        base_ray_hits_w = torch.nan_to_num(
            base_ray_hits_w, nan=0.0, posinf=terrain_height_threshold[1], neginf=terrain_height_threshold[0]
        )
        base_ray_hits_w = torch.clamp(base_ray_hits_w, min=terrain_height_threshold[0], max=terrain_height_threshold[1])
        adjusted_target_height = target_height + torch.mean(base_ray_hits_w, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return sanitize(torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height), clip)


def custom_base_acc_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: Tuple[float, float] = (1.5, 5.0),
    xyz: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    clip: float = DEFAULT_CLIP,
) -> torch.Tensor:
    """Penalize squared base acceleration with per-axis weights."""
    assert sum(xyz) > 0 and min(xyz) >= 0
    thr0, thr1 = threshold
    assert thr1 > thr0 >= 0

    asset: Articulation = env.scene[asset_cfg.name]
    acc_w = asset.data.body_com_lin_acc_w[:, asset_cfg.body_ids].squeeze()
    quat = asset.data.body_quat_w[:, asset_cfg.body_ids].squeeze()
    acc_b = quat_apply_inverse(quat, acc_w)

    pen = torch.clamp(torch.square(acc_b) - thr0, min=0.0, max=(thr1 - thr0) ** 2)
    return sanitize(torch.sum(pen * torch.tensor(xyz, device=pen.device), dim=1) / sum(xyz), clip)


def custom_gravity_track_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    std: float = 0.5,
    clip: float = DEFAULT_CLIP,
) -> torch.Tensor:
    """Track the horizontal components of the ideal projected-gravity direction."""
    asset: Articulation = env.scene[asset_cfg.name]
    g_b = asset.data.projected_gravity_b
    g_ideal_b = ideal_projected_gravity(env, asset_cfg)
    # Compare only the horizontal components associated with roll and pitch.
    err_xy = torch.sum(torch.square(g_b[:, :2] - g_ideal_b[:, :2]), dim=-1)
    return sanitize(torch.exp(-err_xy / (std**2)), clip)


def custom_gravity_pen_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    clip: float = DEFAULT_CLIP,
) -> torch.Tensor:
    """Penalize squared distance from the ideal gravity direction."""
    asset: Articulation = env.scene[asset_cfg.name]
    g_b = asset.data.projected_gravity_b
    g_ideal_b = ideal_projected_gravity(env, asset_cfg)
    return sanitize(torch.sum(torch.square(g_b - g_ideal_b), dim=-1), clip)


# =============================================================================
# Sanitized wrappers around Isaac Lab rewards.
# Keep SceneEntityCfg arguments at the top level for joint/body resolution.
# Each reward can override the permissive clipping bound.
# =============================================================================
def safe_lin_vel_z_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    clip: float = DEFAULT_CLIP,
) -> torch.Tensor:
    return sanitize(isaaclab_rewards.lin_vel_z_l2(env, asset_cfg), clip)


def safe_ang_vel_xy_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    clip: float = DEFAULT_CLIP,
) -> torch.Tensor:
    return sanitize(isaaclab_rewards.ang_vel_xy_l2(env, asset_cfg), clip)


def safe_flat_orientation_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    clip: float = DEFAULT_CLIP,
) -> torch.Tensor:
    return sanitize(isaaclab_rewards.flat_orientation_l2(env, asset_cfg), clip)


def safe_joint_torques_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    clip: float = DEFAULT_CLIP,
) -> torch.Tensor:
    return sanitize(isaaclab_rewards.joint_torques_l2(env, asset_cfg), clip)


def safe_joint_acc_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    clip: float = DEFAULT_CLIP,
) -> torch.Tensor:
    return sanitize(isaaclab_rewards.joint_acc_l2(env, asset_cfg), clip)


def safe_undesired_contacts(
    env: ManagerBasedRLEnv,
    threshold: float,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    clip: float = DEFAULT_CLIP,
) -> torch.Tensor:
    return sanitize(isaaclab_rewards.undesired_contacts(env, threshold=threshold, sensor_cfg=sensor_cfg), clip)
