"""Deterministic terrain and payload evaluation environments.

Sixteen columns represent four terrain types at four difficulty tiers.
EVAL_CONDITIONS maps each column to its condition; explicit geometry fields
replace the terrain generator difficulty. Environments start at zero yaw,
ramp their forward speed and carry identical cylindrical payloads. Domain
randomization is disabled.

Transport and base variants retain their training-time critic observation
dimensions so checkpoints can be loaded without modification."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg, TerminationTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

import locowm.mdp as mdp
import locowm.terrains as custom_terrain_gen
from locowm.config.go2w.locomotion_env_cfg import LocomotionEnvCfg, BASE_LINK_NAME
from locowm.config.go2w.transport_env_cfg import TransportEnvCfg


# =============================================================================
# Shared terrain conditions for metric aggregation.
# Condition c maps to terrain c // 4 and tier EVAL_TIERS[c % 4].

# =============================================================================
EVAL_TERRAIN_TYPES = ["slope", "bump", "perlin", "bridge"]
EVAL_TIERS = [25, 50, 75, 100]
EVAL_CONDITIONS: list[tuple[str, int]] = [(t, tier) for t in EVAL_TERRAIN_TYPES for tier in EVAL_TIERS]

# Tier-100 geometry; lower tiers scale these values linearly.
_EVAL_MAX = {
    "slope_tan": 0.30,  # 16.7deg
    "bump_height": 0.12,
    "amplitude": 0.10,
    "bridge_height": 0.25,
}

# Nominal payload cylinder dimensions.
_OBJ_RADIUS = 0.032
_OBJ_HEIGHT = 0.16  # hr_ratio=5
_OBJ_MASS = 0.5


def _build_eval_terrain_generator() -> TerrainGeneratorCfg:
    sub: dict = {}
    for ttype in EVAL_TERRAIN_TYPES:
        for tier in EVAL_TIERS:
            s = tier / 100.0
            key = f"{ttype}_{tier}"
            if ttype == "slope":
                sub[key] = custom_terrain_gen.HfEvalSlopeHillTerrainCfg(
                    proportion=1.0, slope_tan=_EVAL_MAX["slope_tan"] * s
                )
            elif ttype == "bump":
                sub[key] = custom_terrain_gen.HfEvalVerticalBumpTerrainCfg(
                    proportion=1.0, bump_height=_EVAL_MAX["bump_height"] * s
                )
            elif ttype == "perlin":
                sub[key] = custom_terrain_gen.HfEvalPerlinTerrainCfg(
                    proportion=1.0, amplitude=_EVAL_MAX["amplitude"] * s
                )
            elif ttype == "bridge":
                sub[key] = custom_terrain_gen.HfEvalUnilateralBridgeTerrainCfg(
                    proportion=1.0, bridge_height=_EVAL_MAX["bridge_height"] * s
                )
    return TerrainGeneratorCfg(
        size=(8.0, 4.0),  # Forward extent and lane width in meters.
        border_width=20.0,
        num_rows=4,  # Four repeated rows per terrain condition.
        num_cols=16,  # One column per terrain/tier combination.
        horizontal_scale=0.1,  # Match the height scanner resolution to limit collision mesh size.
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains=sub,
        curriculum=True,  # Deterministic columns with independently specified geometry.
        seed=1,
    )


def _apply_eval_overrides(cfg) -> None:
    """Apply shared deterministic evaluation settings to the environment config."""

    # All environments start in row zero; disable terrain progression.
    cfg.scene.terrain.terrain_generator = _build_eval_terrain_generator()
    cfg.scene.terrain.terrain_generator.curriculum = True
    cfg.scene.terrain.max_init_terrain_level = 0
    if getattr(cfg, "curriculum", None) is not None:
        for attr in ("terrain_levels", "command_x_levels", "command_y_levels", "command_z_levels"):
            if getattr(cfg.curriculum, attr, None) is not None:
                setattr(cfg.curriculum, attr, None)

    # Ramp the forward speed; lateral and yaw commands remain zero.
    cfg.commands.base_velocity = mdp.EvalRampVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.EvalRampVelocityCommandCfg.Ranges(lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)),
        target_speed=1.0,
        accel_steps=125,  # 2.5s @ 50Hz
    )

    # Disable domain randomization and start at zero yaw and velocity.
    for attr in (
        "randomize_base_mass",
        "randomize_foot_physics_material",
        "randomize_rigid_body_inertia",
        "randomize_com_positions",
        "randomize_actuator_gains",
        "randomize_apply_external_force_torque",
        "push_robot",
    ):
        if getattr(cfg.events, attr, None) is not None:
            setattr(cfg.events, attr, None)

    cfg.events.reset_base.params = {
        "pose_range": {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.05, 0.05),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
        "velocity_range": {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        },
    }
    if getattr(cfg.events, "randomize_reset_joints", None) is not None:
        cfg.events.randomize_reset_joints.params["position_range"] = (1.0, 1.0)
        cfg.events.randomize_reset_joints.params["velocity_range"] = (0.0, 0.0)

    # Keep timeout and boundary safeguards; retain falls in the measurement window.
    for attr in ("base_contact", "hip_contact"):
        if getattr(cfg.terminations, attr, None) is not None:
            setattr(cfg.terminations, attr, None)

    # Identical payloads allow physics replication across environments.

    cfg.scene.replicate_physics = True
    cfg.scene.env_spacing = 2.5
    cfg.sim.physx.gpu_max_rigid_patch_count = 2**21

    cfg.scene.object = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.CylinderCfg(
            radius=_OBJ_RADIUS,
            height=_OBJ_HEIGHT,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=_OBJ_MASS),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.005, rest_offset=0.0
            ),
            # Skip visual materials to avoid expensive Kit undo operations in headless scenes.
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # Reset the payload above the base with matching orientation and 1 cm clearance.
    cfg.events.reset_object = EventTermCfg(
        func=mdp.set_rigid_object_relative_to_robot,
        mode="reset",
        params={
            "base_asset_cfg": SceneEntityCfg("robot", body_names=[BASE_LINK_NAME]),
            "target_asset_cfg": SceneEntityCfg("object"),
            "relative_pose": {
                "x": 0.0,
                "y": 0.0,
                "z": _OBJ_HEIGHT / 2 + 0.01,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            },
        },
    )

    # Keep episode duration longer than the measurement window.
    cfg.episode_length_s = 60.0


@configclass
class EvalEnvCfg(TransportEnvCfg):
    """Evaluation with transport critic observations for E2E and adapter policies."""

    def __post_init__(self):
        self.scene.num_envs = 256
        super().__post_init__()
        _apply_eval_overrides(self)


@configclass
class EvalBaseEnvCfg(LocomotionEnvCfg):
    """Use locomotion critic dimensions while sharing the transport evaluation setup."""

    def __post_init__(self):
        self.scene.num_envs = 256
        super().__post_init__()
        _apply_eval_overrides(self)


# =============================================================================
# Success-rate evaluation.
#
# Reuse the payload, ramp command and deterministic initial state.
# One bounded terrain feature per trial, with flat entry and exit sections.
# Reset after crossing the feature; do not enter another terrain cell.
# A payload below the base fails the trial; acceleration drops only trigger retries.
# Forward displacement beyond SUCC_TRAVERSE_DISTANCE with the payload retained succeeds.
# See locowm/scripts/succ_eval.py for aggregation.
# =============================================================================

# Traversal threshold must extend beyond the feature into the exit section.
SUCC_TRAVERSE_DISTANCE = 5.0


def _build_success_terrain_generator() -> TerrainGeneratorCfg:
    sub: dict = {}
    for ttype in EVAL_TERRAIN_TYPES:
        for tier in EVAL_TIERS:
            s = tier / 100.0
            key = f"{ttype}_{tier}"
            if ttype == "slope":
                sub[key] = custom_terrain_gen.HfEvalSlopeHillTerrainCfg(
                    proportion=1.0,
                    slope_tan=_EVAL_MAX["slope_tan"] * s,
                    pre_flat=2.0,
                    ramp_run=1.0,
                    top_len=0.5,
                )
            elif ttype == "bump":
                sub[key] = custom_terrain_gen.HfSuccBumpTerrainCfg(
                    proportion=1.0, bump_height=_EVAL_MAX["bump_height"] * s
                )
            elif ttype == "perlin":
                sub[key] = custom_terrain_gen.HfSuccPerlinTerrainCfg(
                    proportion=1.0, amplitude=_EVAL_MAX["amplitude"] * s
                )
            elif ttype == "bridge":
                sub[key] = custom_terrain_gen.HfSuccBridgeTerrainCfg(
                    proportion=1.0, bridge_height=_EVAL_MAX["bridge_height"] * s
                )
    return TerrainGeneratorCfg(
        size=(12.0, 4.0),  # The positive-x half contains the flat entry, feature and flat exit.
        border_width=20.0,
        num_rows=1,  # A single row prevents entering a second terrain cell.
        num_cols=16,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        use_cache=False,
        sub_terrains=sub,
        curriculum=True,
        seed=1,
    )


def _apply_success_overrides(cfg, traverse_distance: float = SUCC_TRAVERSE_DISTANCE) -> None:
    """Configure a single-feature traversal trial on top of evaluation settings."""
    # One feature band in a single terrain row.
    cfg.scene.terrain.terrain_generator = _build_success_terrain_generator()
    cfg.scene.terrain.terrain_generator.curriculum = True
    cfg.scene.terrain.max_init_terrain_level = 0

    # Bound each trial; timing out before traversal counts as failure.
    cfg.episode_length_s = 15.0

    # Separate success, payload failure and acceleration retries; success precedes the boundary.
    cfg.terminations.terrain_out_of_bounds = None
    cfg.terminations.traversed_success = TerminationTermCfg(
        func=mdp.forward_traversed,
        params={
            "distance": float(traverse_distance),
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("object"),
        },
    )
    cfg.terminations.payload_dropped_fail = TerminationTermCfg(
        func=mdp.object_dropped,
        params={
            "during_accel": False,
            "object_cfg": SceneEntityCfg("object"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.terminations.payload_dropped_accel = TerminationTermCfg(
        func=mdp.object_dropped,
        params={
            "during_accel": True,
            "object_cfg": SceneEntityCfg("object"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class SuccessEnvCfg(EvalEnvCfg):
    """Payload success evaluation for E2E and adapter policies."""

    def __post_init__(self):
        super().__post_init__()
        _apply_success_overrides(self)


@configclass
class SuccessBaseEnvCfg(EvalBaseEnvCfg):
    """Payload success evaluation with locomotion critic observations."""

    def __post_init__(self):
        super().__post_init__()
        _apply_success_overrides(self)
