import math
import numpy as np
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import (
    SceneEntityCfg,
    EventTermCfg,
    RewardTermCfg,
    ObservationTermCfg,
)
from isaaclab.utils import configclass

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

import locowm.mdp as mdp
import locowm.mdp.rewards as custom_rewards
import locowm.terrains as custom_terrain_gen
from locowm.assets.go2w import Go2W_CFG as Robot_CFG
from locowm.config.go2w.locomotion_env_cfg import LocomotionEnvCfg, OBS_CLIP_BOUNDS


@configclass
class AccTrackCfg:
    """Shared acceleration-filter and ideal-gravity settings for observations and rewards."""

    use_lpf: bool = True  # Smooth base acceleration before computing ideal projected gravity.
    cut_off_frequency: float = 1.0
    control_frequency: float = 50.0
    acc_gain: float = 1.5  # Amplify horizontal acceleration to increase the target lean angle.
    zero_threshold: float = 0.1  # Suppress low-amplitude acceleration noise.


@configclass
class TransportEnvCfg(LocomotionEnvCfg):
    """Train the payload platform to follow the ideal gravity direction."""

    # Shared acceleration-filter and tilt-tracking parameters for observations and rewards.
    acc_track: AccTrackCfg = AccTrackCfg()

    def __post_init__(self):
        super().__post_init__()

        # Use the robot model with a payload platform.
        self.scene.replicate_physics = False
        self.scene.robot = Robot_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Replace aggressive terrain features with transport-oriented terrain.
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=10,
            num_cols=20,
            horizontal_scale=0.05,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
                "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                    proportion=0.2, noise_range=(0.00, 0.10), noise_step=0.005, border_width=0.25
                ),
                "perlin_rough": custom_terrain_gen.HfPerlinNoiseTerrainCfg(
                    proportion=0.2,
                    noise_range=(0.00, 0.10),
                    noise_step=0.005,
                    frequency=0.75,
                    octaves=2,
                    lacunarity=2.0,
                    persistence=0.5,
                    border_width=0.25,
                ),
                "speed_bump": custom_terrain_gen.HfSpeedBumpTerrainCfg(
                    proportion=0.5,
                    num_bumps=8,
                    bump_height_range=(0.00, 0.20),
                    random_flat_ratio=(0.0, 0.40),
                    random_bump_width=(0.20, 0.40),
                    num_gaps=2,
                    random_gap_length=(0.5, 1.0),
                    gap_margin=0.5,
                    platform_width=2.0,
                    border_width=0.25,
                ),
                "x_wave": custom_terrain_gen.HfXWaveTerrainCfg(
                    proportion=0.0, amplitude_range=(0.00, 0.10), wave_length=(1.55, 1.65), border_width=0.25
                ),
            },
            seed=1,
        )
        self.scene.terrain.terrain_generator.curriculum = True

        # Only the critic receives acceleration and ideal-gravity observations.
        self.observations.critic.base_lin_acc = ObservationTermCfg(
            func=mdp.base_lin_acc,
            scale=0.25,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[self.base_link_name])},
            history_length=6,
        )
        self.observations.critic.ideal_projected_gravity = ObservationTermCfg(
            func=mdp.ideal_projected_gravity,
            scale=1.0,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[self.base_link_name])},
            history_length=6,
        )

        # region events
        self.events.reset_base.params = {
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
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        }

        self.events.randomize_reset_joints.params["position_range"] = (1.0, 1.0)
        # endregion

        # region Terminations
        self.terminations.base_contact = None
        self.terminations.hip_contact = None

        # region Command
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-math.pi / 4, math.pi / 4)
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.initial_zero_command_steps = 50
        self.commands.base_velocity.resampling_time_range = (6.0, 8.0)
        self.commands.base_velocity.bang_bang_envs = 0.05

        # region Curriculum
        self.curriculum.command_y_levels = None

        # region Reward
        # Increase forward-velocity tracking weight.
        self.rewards.track_lin_vel_x_exp.weight = 1.0

        self.rewards.base_height_l2 = None
        # Penalize vertical platform motion more strongly.
        self.rewards.lin_vel_z_l2.weight = -10.0

        # Track the ideal gravity direction instead of enforcing a level platform.
        self.rewards.flat_orientation_l2 = None
        self.rewards.track_gravity_exp = RewardTermCfg(
            func=custom_rewards.custom_gravity_track_exp,
            weight=0.5,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.base_link_name]),
                "std": math.sqrt(0.25),
            },
        )
        # Penalize excessive horizontal acceleration.
        self.rewards.base_acc_l2 = RewardTermCfg(
            func=custom_rewards.custom_base_acc_l2,
            weight=-0.2,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.base_link_name]),
                "threshold": (1.0, 10.0),
                "xyz": (1.0, 1.0, 0.0),
            },
        )

        # Sanitize the transport-specific critic observations.
        # Already configured terms are skipped; bounds come from OBS_CLIP_BOUNDS.
        mdp.add_obs_sanitizers(self.observations, overrides=OBS_CLIP_BOUNDS)


@configclass
class TransportPlayEnvCfg(TransportEnvCfg):
    """Playback environment with a cylindrical payload on the robot platform."""

    def __post_init__(self):
        self.scene.num_envs = 20
        super().__post_init__()

        # Flat terrain for playback.
        self.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=20.0,
            num_rows=1,
            num_cols=1,
            horizontal_scale=0.05,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
            },
            seed=1,
        )
        self.scene.terrain.terrain_generator.curriculum = True

        # Disable external forces and pushes during playback.
        self.events.reset_base.params = {
            "pose_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (0.01, 0.01),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-math.pi, math.pi),
            },
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        }
        self.events.randomize_apply_external_force_torque = None
        self.events.push_robot = None

        # Increase rigid-patch capacity for independently instantiated payloads.
        self.scene.env_spacing = 2.5
        self.sim.physx.gpu_max_rigid_patch_count = 5 * 2**15

        # Payload cylinders.
        env_num = self.scene.num_envs
        radius_range = (0.032, 0.032)
        hr_ratio_range = (5.0, 5.0)
        radii = np.random.uniform(radius_range[0], radius_range[1], size=(env_num, 1))
        hr_ratios = np.random.uniform(hr_ratio_range[0], hr_ratio_range[1], size=(env_num, 1))
        heights = radii * hr_ratios
        size_samples = np.concatenate([radii, heights], axis=1)
        color_samples = np.random.uniform(0.0, 1.0, (env_num, 3)).astype(np.float32)

        self.scene.object = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Object",
            spawn=sim_utils.MultiAssetSpawnerCfg(  # Create one independent payload per environment.
                assets_cfg=[
                    sim_utils.CylinderCfg(
                        radius=float(size_samples[i, 0]),
                        height=float(size_samples[i, 1]),
                        axis="Z",
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(map(float, color_samples[i]))),
                    )  # type: ignore
                    for i in range(env_num)
                ],
                random_choice=False,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                activate_contact_sensors=True,
                mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=True,
                    contact_offset=0.005,
                    rest_offset=0.0,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        )

        self.events.reset_object_position = EventTermCfg(
            func=mdp.ResetObjectStateUniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.00, 0.00),
                    "y": (-0.00, 0.00),
                    "z": (0.01, 0.01),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (-0.0, 0.0),
                },
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object", body_names="Object"),
                "reference_asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # Playback commands.
        self.commands.base_velocity.bang_bang_envs = 0.00
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.initial_zero_command_steps = 100
        self.commands.base_velocity.resampling_time_range = (1.5, 2.0)

        if getattr(self, "curriculum", None) is not None:
            if getattr(self.curriculum, "command_x_levels", None) is not None:
                self.curriculum.command_x_levels.params["range_multiplier"] = (1.0, 1.0)
            if getattr(self.curriculum, "command_z_levels", None) is not None:
                self.curriculum.command_z_levels.params["range_multiplier"] = (1.0, 1.0)
