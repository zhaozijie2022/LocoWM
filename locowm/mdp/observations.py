from __future__ import annotations
import torch
from typing import TYPE_CHECKING, Tuple
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


from isaaclab.assets import Articulation
from .databuffer import StepwiseLPF


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


import isaaclab.utils.math as math_utils


def base_lin_acc(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The linear acceleration of the base link of the asset."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids].squeeze()
    base_lin_acc_w = asset.data.body_com_lin_acc_w[:, asset_cfg.body_ids].squeeze()
    return math_utils.quat_apply_inverse(body_quat, base_lin_acc_w)


def base_lin_acc_w(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return unfiltered world-frame linear acceleration with shape [N, 3]."""
    asset: Articulation = env.scene[asset_cfg.name]
    acc_w = asset.data.body_com_lin_acc_w[:, asset_cfg.body_ids].squeeze(1)
    return acc_w


def ideal_projected_gravity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Compute the desired gravity direction from world-frame base acceleration."""
    acc_gain = env.cfg.acc_track.acc_gain
    zero_threshold = env.cfg.acc_track.zero_threshold
    asset: Articulation = env.scene[asset_cfg.name]
    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids].squeeze()
    acc_w = custom_base_lin_acc(env, asset_cfg, frame="world")
    g_mag = 9.81
    ax, ay = acc_w[:, 0], acc_w[:, 1]
    # Apply the deadband before amplifying the target acceleration.
    ax = torch.where(torch.abs(ax) < zero_threshold, torch.zeros_like(ax), ax) * acc_gain
    ay = torch.where(torch.abs(ay) < zero_threshold, torch.zeros_like(ay), ay) * acc_gain
    norm = torch.sqrt(ax**2 + ay**2 + g_mag**2).clamp(min=1e-9)
    g_ideal_w = torch.stack([-ax / norm, -ay / norm, -g_mag / norm], dim=-1)
    return math_utils.quat_apply_inverse(body_quat, g_ideal_w)


def custom_height_scan(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.5,
    terrain_height_threshold: Tuple[float, float] = (-0.2, 0.2),
) -> torch.Tensor:
    """The height scan of the sensor."""
    sensor = env.scene.sensors[sensor_cfg.name]
    base_ray_hits_w = sensor.data.ray_hits_w[..., 2]
    # Clamp base_ray_hits_w to avoid NaN and Inf (including -Inf/Inf) before usage
    base_ray_hits_w = torch.nan_to_num(
        base_ray_hits_w, nan=0.0, posinf=terrain_height_threshold[1], neginf=terrain_height_threshold[0]
    )
    base_ray_hits_w = torch.clamp(base_ray_hits_w, min=terrain_height_threshold[0], max=terrain_height_threshold[1])
    return sensor.data.pos_w[:, 2].unsqueeze(1) - base_ray_hits_w - offset


def custom_base_lin_acc(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    frame: str = "world",  # or "base"
) -> torch.Tensor:
    """Return base acceleration in the world or base frame, optionally low-pass filtered."""
    assert frame in ("world", "base"), "Only 'world' and 'base' frames are supported"

    use_lpf = env.cfg.acc_track.use_lpf
    cut_off_frequency = env.cfg.acc_track.cut_off_frequency
    control_frequency = env.cfg.acc_track.control_frequency

    asset: Articulation = env.scene[asset_cfg.name]
    body_quat = asset.data.body_quat_w[:, asset_cfg.body_ids].squeeze(1)
    acc_w = asset.data.body_com_lin_acc_w[:, asset_cfg.body_ids].squeeze(1)

    if use_lpf:
        if not hasattr(env, "_base_lin_acc_lpf"):
            # Derive the filter frequency from the environment control timestep.
            env._base_lin_acc_lpf = StepwiseLPF(cut_off_frequency, control_frequency)
        reset_mask = (env.episode_length_buf == 0).unsqueeze(-1)
        acc_w = env._base_lin_acc_lpf(acc_w, step=env.common_step_counter, reset_mask=reset_mask)

    if frame == "base":
        return math_utils.quat_apply_inverse(body_quat, acc_w)
    return acc_w
