from __future__ import annotations
import math
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import (
    SceneEntityCfg,
    EventTermCfg,
    RewardTermCfg,
    ObservationTermCfg,
    ObservationGroupCfg,
    CurriculumTermCfg,
    TerminationTermCfg,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import locowm.mdp as mdp
import locowm.mdp.rewards as custom_rewards
from locowm.assets.go2w import Go2W_CFG as Robot_CFG
import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
import locowm.terrains as custom_terrain_gen


BASE_LINK_NAME = "base"
FOOT_LINK_NAME = ".*_foot"
LEG_JOINT_NAMES = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]
WHEEL_JOINT_NAMES = [
    "FR_foot_joint",
    "FL_foot_joint",
    "RR_foot_joint",
    "RL_foot_joint",
]
JOINT_NAMES = LEG_JOINT_NAMES + WHEEL_JOINT_NAMES
HIP_JOINT_NAMES = ["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"]


# Physical bounds for custom observations.
# Unlisted terms use sanitize.DEFAULT_CLIP and NaN/Inf protection.
OBS_CLIP_BOUNDS = {
    "projected_gravity": 1.0,  # Unit-vector components.
    "ideal_projected_gravity": 1.0,  # Unit-vector components.
    "base_ang_vel": 100.0,  # rad/s; well below the PhysX velocity limit.
    "base_lin_vel": 100.0,  # m/s
    "base_lin_acc_w": 1000.0,  # m/s^2, world frame.
    "base_lin_acc": 1000.0,  # m/s^2, base frame.
    "joint_vel": 1000.0,  # rad/s; 2x actuator velocity_limit(30)
    "joint_pos": 10.0,  # Joint displacement from the default position, in radians.
    "last_action": 100.0,  # Match the action clip bounds.
    "velocity_commands": 10.0,  # Permissive bounds for manager-generated commands.
    "height_scan": 5.0,  # Height scans are already clamped to +/-1 m.
}


# =============================================================================
# region -- Scene --
# =============================================================================
@configclass
class SceneCfg(InteractiveSceneCfg):
    replicate_physics = False
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
                "perlin_rough": custom_terrain_gen.HfPerlinNoiseTerrainCfg(
                    proportion=0.2,
                    noise_range=(0.00, 0.10),
                    noise_step=0.005,
                    frequency=0.7,
                    octaves=2,
                    lacunarity=2.0,
                    persistence=0.5,
                    border_width=0.25,
                ),
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.2,
                    noise_range=(0.00, 0.05),
                    noise_step=0.005,
                    border_width=0.25,
                ),
            },
            seed=1,
        ),
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = Robot_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot_contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/(?!sensor.*).*",
        history_length=3,
        track_air_time=True,
    )
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    height_scanner_base = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.1, 0.1)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=1000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.6, 0.6, 0.6), intensity=1000.0),
    )


# endregion -- Scene --

# =============================================================================
# region -- Commands --
# =============================================================================


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandMultiSamplingCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        initial_zero_command_steps=50,
        rel_heading_envs=0.0,
        heading_command=False,
        ranges=mdp.UniformVelocityCommandMultiSamplingCfg.Ranges(
            lin_vel_x=(-1.5, 1.5),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-math.pi / 4, math.pi / 4),
        ),
    )


# endregion -- Commands --

# =============================================================================
# region -- Observations --
# =============================================================================


@configclass
class ObservationsCfg:
    @configclass
    class NoisyProprioceptionCfg(ObservationGroupCfg):
        velocity_commands = ObservationTermCfg(
            func=mdp.generated_commands,
            scale=1.0,
            params={"command_name": "base_velocity"},
            history_length=6,
        )
        base_ang_vel = ObservationTermCfg(
            func=mdp.base_ang_vel,
            scale=0.25,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=6,
        )
        projected_gravity = ObservationTermCfg(
            func=mdp.projected_gravity,
            scale=1.0,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=6,
        )
        joint_pos = ObservationTermCfg(
            func=mdp.joint_pos_rel_without_wheel,
            scale=1.0,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=6,
            params={"wheel_asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES)},
        )
        joint_vel = ObservationTermCfg(
            func=mdp.joint_vel_rel,
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            history_length=6,
        )
        last_action = ObservationTermCfg(
            func=mdp.last_action,
            scale=1.0,
            history_length=1,
        )
        base_lin_vel = None
        height_scan = None

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = None  # Apply to all observation terms in this group.

    @configclass
    class DenoisedProprioceptionCfg(NoisyProprioceptionCfg):
        base_lin_vel = ObservationTermCfg(
            func=mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            scale=2.0,
        )
        height_scan = ObservationTermCfg(
            func=mdp.custom_height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                "offset": 0.4,
            },
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.history_length = None

    policy: NoisyProprioceptionCfg = NoisyProprioceptionCfg()
    critic: DenoisedProprioceptionCfg = DenoisedProprioceptionCfg()

    @configclass
    class WMTargetCfg(ObservationGroupCfg):
        """Post-step task-state targets for the world model (state t+1).

        No low-pass filtering is applied. The nine target values are angular
        velocity, world-frame linear acceleration and projected gravity; scaling
        follows the corresponding observation conventions."""

        base_ang_vel = ObservationTermCfg(
            func=mdp.base_ang_vel,
            scale=0.25,  # Match policy angular-velocity scaling.
        )
        base_lin_acc_w = ObservationTermCfg(
            func=mdp.base_lin_acc_w,
            scale=0.25,  # Scale raw world-frame acceleration without filtering.
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[BASE_LINK_NAME])},
        )
        projected_gravity = ObservationTermCfg(
            func=mdp.projected_gravity,
            scale=1.0,  # Match policy projected-gravity scaling.
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = None

    wm_target: WMTargetCfg = WMTargetCfg()


# endregion -- Observations --

# =============================================================================
# region -- Actions --
# =============================================================================


@configclass
class ActionsCfg:
    leg_joint_pos = mdp.JointPositionLowPassActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES,
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
        control_frequency=50.0,
        cut_off_frequency=5.0,
        order=1,
    )
    wheel_joint_vel = mdp.JointVelocityLowPassActionCfg(
        asset_name="robot",
        joint_names=WHEEL_JOINT_NAMES,
        scale=10.0,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        control_frequency=50.0,
        cut_off_frequency=15.0,
        order=1,
    )


# endregion -- Actions --


# =============================================================================
# region -- Rewards --
# =============================================================================


@configclass
class RewardsCfg:
    # Velocity tracking.
    track_lin_vel_x_exp = RewardTermCfg(
        func=custom_rewards.custom_track_lin_vel_x_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_lin_vel_y_exp = RewardTermCfg(
        func=custom_rewards.custom_track_lin_vel_y_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewardTermCfg(
        func=custom_rewards.custom_track_ang_vel_z_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    # Posture and angular velocity with sanitized reward outputs.
    # Clipping applies to rewards, not raw sensor values.
    lin_vel_z_l2 = RewardTermCfg(
        func=custom_rewards.safe_lin_vel_z_l2,
        weight=-2.0,
        params={"clip": 4.0},  # CLIP = v_z², peak = 2 m/s
    )
    ang_vel_xy_l2 = RewardTermCfg(
        func=custom_rewards.safe_ang_vel_xy_l2,
        weight=-0.05,
        params={"clip": 100.0},  # CLIP = ωx²+ωy², peak = 10 rad/s
    )
    flat_orientation_l2 = RewardTermCfg(
        func=custom_rewards.safe_flat_orientation_l2,
        weight=-0.5,
        params={"clip": 0.2},  # gx²+gy²
    )
    # Joint torques and accelerations.
    joint_torques_l2 = RewardTermCfg(
        func=custom_rewards.safe_joint_torques_l2,
        weight=-2.0e-4,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES + WHEEL_JOINT_NAMES),
            "clip": 0.5e4,  # Στ²
        },
    )
    leg_joint_acc_l2 = RewardTermCfg(
        func=custom_rewards.safe_joint_acc_l2,
        weight=-2.5e-7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
            "clip": 0.4e7,  # Σa²
        },
    )
    wheel_joint_acc_l2 = RewardTermCfg(
        func=custom_rewards.safe_joint_acc_l2,
        weight=-2.5e-9,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=WHEEL_JOINT_NAMES),
            "clip": 0.04e9,  # Σa²
        },
    )
    # Base height and contacts.
    base_height_l2 = RewardTermCfg(
        func=custom_rewards.safe_base_height_l2,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[BASE_LINK_NAME]),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            "target_height": 0.40,
            "terrain_height_threshold": (-0.5, 0.5),
            "clip": 0.09,  # Squared height error relative to the 0.3 m target.
        },
    )
    undesired_contacts = RewardTermCfg(
        func=custom_rewards.safe_undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[f"^(?!.*{FOOT_LINK_NAME}).*"]),
            "threshold": 0.1,
        },
    )
    action_rate_l2 = RewardTermCfg(
        func=custom_rewards.safe_action_rate,
        weight=-0.01,
        params={
            "threshold": 7.0,
        },
    )
    # locowm/mdp/rewards.py
    stand_still_without_cmd = RewardTermCfg(
        func=custom_rewards.stand_still_without_cmd,
        weight=-0.25,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
            "clip": 4.0,
        },
    )
    hip_deviation_l2 = RewardTermCfg(
        func=custom_rewards.hip_deviation_l2,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=HIP_JOINT_NAMES),
            "clip": 1e6,
        },
    )
    joint_deviation_l2 = RewardTermCfg(
        func=custom_rewards.joint_deviation_l2,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES),
            "clip": 1e6,
        },
    )


# =============================================================================
# region -- Events --
# =============================================================================


@configclass
class EventCfg:
    randomize_base_mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[BASE_LINK_NAME]),
            "mass_distribution_params": (-1, 2),
            "operation": "add",
        },
    )
    randomize_foot_physics_material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot"),
            "static_friction_range": (0.4, 2.0),
            "dynamic_friction_range": (0.4, 2.0),
            "restitution_range": (0.0, 0.5),
            "make_consistent": True,
            "num_buckets": 4000,
        },
    )
    randomize_rigid_body_inertia = EventTermCfg(
        func=mdp.randomize_rigid_body_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "inertia_distribution_params": (0.5, 1.5),
            "operation": "scale",
        },
    )
    randomize_com_positions = EventTermCfg(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[BASE_LINK_NAME]),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    reset_base = EventTermCfg(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (0.05, 0.05),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.15, 0.15),
                "z": (-0.2, 0.2),
                "roll": (-0.35, 0.35),
                "pitch": (-0.35, 0.35),
                "yaw": (-0.35, 0.35),
            },
        },
    )
    randomize_reset_joints = EventTermCfg(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.95, 1.05), "velocity_range": (0.0, 0.0)},
    )
    randomize_apply_external_force_torque = EventTermCfg(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[BASE_LINK_NAME]),
            "force_range": (-10.0, 10.0),
            "torque_range": (-10.0, 10.0),
        },
    )
    randomize_actuator_gains = EventTermCfg(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    push_robot = EventTermCfg(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(6.0, 10.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.3, 0.3),
                "z": (-0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )


# endregion -- Events --


# =============================================================================
# region -- Terminations --
# =============================================================================


@configclass
class TerminationsCfg:
    time_out = TerminationTermCfg(func=mdp.time_out, time_out=True)
    base_contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("robot_contact_sensor", body_names=[BASE_LINK_NAME]), "threshold": 1.0},
    )
    hip_contact = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("robot_contact_sensor", body_names=".*hip"), "threshold": 1.0},
    )
    terrain_out_of_bounds = TerminationTermCfg(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )


# =============================================================================
# region -- Curriculum --
# =============================================================================


@configclass
class CurriculumCfg:
    velocity_commands = None

    command_x_levels = CurriculumTermCfg(
        func=mdp.command_axis_levels_vel,
        params={
            "reward_term_name": "track_lin_vel_x_exp",
            "range_multiplier": (0.1, 1.0),
            "upper_threshold": 0.8,
            "lower_threshold": 0.5,
            "ema_alpha": 0.5,
        },
    )
    command_y_levels = CurriculumTermCfg(
        func=mdp.command_axis_levels_vel,
        params={
            "reward_term_name": "track_lin_vel_y_exp",
            "range_multiplier": (0.1, 1.0),
            "upper_threshold": 0.8,
            "lower_threshold": 0.5,
            "ema_alpha": 0.5,
        },
    )
    command_z_levels = CurriculumTermCfg(
        func=mdp.command_axis_levels_vel,
        params={
            "reward_term_name": "track_ang_vel_z_exp",
            "range_multiplier": (0.1, 1.0),
            "upper_threshold": 0.6,
            "lower_threshold": 0.3,
            "ema_alpha": 0.5,
        },
    )
    terrain_levels = CurriculumTermCfg(func=mdp.terrain_levels_vel)


# =============================================================================
# LocomotionEnvCfg
# =============================================================================


def _smaller_scene_for_playing(env_cfg: "LocomotionEnvCfg") -> None:
    env_cfg.scene.num_envs = 50
    env_cfg.scene.env_spacing = 2.5
    env_cfg.sim.physx.gpu_max_rigid_patch_count = 5 * 2**15


@configclass
class LocomotionEnvCfg(ManagerBasedRLEnvCfg):
    base_link_name = BASE_LINK_NAME
    foot_link_name = FOOT_LINK_NAME
    leg_joint_names = LEG_JOINT_NAMES
    wheel_joint_names = WHEEL_JOINT_NAMES
    joint_names = JOINT_NAMES

    scene: SceneCfg = SceneCfg(num_envs=4096, env_spacing=2.5)
    viewer = ViewerCfg(
        eye=(5.0, 5.0, 4.0),
        resolution=(1920, 1080),
        lookat=(-2.0, -2.0, 0.0),
        origin_type="world",
        env_index=0,
        asset_name="robot",
    )
    commands: CommandsCfg = CommandsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True

        # Bind the simulation material to the terrain configuration.
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"

        # Sensor periods depend on the configured simulation timestep.
        if getattr(self.scene, "robot_contact_sensor", None) is not None:
            self.scene.robot_contact_sensor.update_period = self.sim.dt
        if getattr(self.scene, "contact_forces", None) is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # ---------- Curriculum ----------
        # Disable the terrain curriculum when no generator is configured.
        if self.scene.terrain.terrain_type != "generator":
            self.curriculum.terrain_levels = None

        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if getattr(self.scene.terrain, "terrain_generator", None) is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if getattr(self.scene.terrain, "terrain_generator", None) is not None:
                self.scene.terrain.terrain_generator.curriculum = False

        self._disable_zero_weight_rewards()

        # Sanitize observations before noise and scaling.
        # OBS_CLIP_BOUNDS overrides the permissive defaults per term.
        mdp.add_obs_sanitizers(self.observations, overrides=OBS_CLIP_BOUNDS)

    def _disable_zero_weight_rewards(self):
        for attr in dir(self.rewards):
            if not attr.startswith("__"):
                reward_attr = getattr(self.rewards, attr)
                if not callable(reward_attr) and getattr(reward_attr, "weight", None) == 0:
                    setattr(self.rewards, attr, None)


@configclass
class LocomotionPlayEnvCfg(LocomotionEnvCfg):
    def __post_init__(self) -> None:
        self.scene.num_envs = 20
        super().__post_init__()
        self.scene.robot = Robot_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        _smaller_scene_for_playing(self)

        self.commands.base_velocity.bang_bang_envs = 0.5

        if getattr(self, "curriculum", None) is not None:
            if getattr(self.curriculum, "command_x_levels", None) is not None:
                self.curriculum.command_x_levels.params["range_multiplier"] = (1.0, 1.0)
            if getattr(self.curriculum, "command_y_levels", None) is not None:
                self.curriculum.command_y_levels.params["range_multiplier"] = (1.0, 1.0)
            if getattr(self.curriculum, "command_z_levels", None) is not None:
                self.curriculum.command_z_levels.params["range_multiplier"] = (1.0, 1.0)
