from __future__ import annotations
import torch
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ResetObjectStateUniform(ManagerTermBase):
    """Place a payload above the reference root and align its orientation."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("object"))
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        self.reference_asset_cfg: SceneEntityCfg = cfg.params.get("reference_asset_cfg", SceneEntityCfg("robot"))
        self.reference_asset: RigidObject | Articulation = env.scene[self.reference_asset_cfg.name]
        pose_range = cfg.params.get("pose_range", {})
        pose_range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.pose_ranges = torch.tensor(pose_range_list, device=self.asset.device)
        velocity_range = cfg.params.get("velocity_range", {})
        velocity_range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        self.velocity_ranges = torch.tensor(velocity_range_list, device=self.asset.device)

        object_height = []
        for asset_cfg in self.asset.cfg.spawn.assets_cfg:
            if isinstance(asset_cfg, sim_utils.CuboidCfg):
                object_height.append(asset_cfg.size[2] / 2)
            elif isinstance(asset_cfg, sim_utils.SphereCfg):
                object_height.append(asset_cfg.radius)
            elif isinstance(asset_cfg, sim_utils.CylinderCfg) or isinstance(asset_cfg, sim_utils.CapsuleCfg):
                object_height.append(asset_cfg.height / 2)
        self.object_height = torch.tensor(object_height, device=self.asset.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        pose_range: dict[str, tuple[float, float]],
        velocity_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        reference_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ):
        pose_rand_samples = math_utils.sample_uniform(
            self.pose_ranges[:, 0], self.pose_ranges[:, 1], (len(env_ids), 6), device=self.asset.device
        )
        reference_frame_states = self.reference_asset.data.root_state_w[env_ids].clone()

        # poses
        positions = reference_frame_states[:, 0:3] + pose_rand_samples[:, 0:3]
        positions[:, 2] += self.object_height[env_ids]
        orientations_delta = math_utils.quat_from_euler_xyz(
            pose_rand_samples[:, 3], pose_rand_samples[:, 4], pose_rand_samples[:, 5]
        )
        orientations = math_utils.quat_mul(reference_frame_states[:, 3:7], orientations_delta)

        # velocities
        velocity_rand_samples = math_utils.sample_uniform(
            self.velocity_ranges[:, 0], self.velocity_ranges[:, 1], (len(env_ids), 6), device=self.asset.device
        )
        velocities = reference_frame_states[:, 7:13] + velocity_rand_samples

        # set into the physics simulation
        self.asset.write_root_link_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=env_ids)
        self.asset.write_root_com_velocity_to_sim(velocities, env_ids=env_ids)


class randomize_friction_restitution(ManagerTermBase):
    # based on the original randomize_rigid_body_material but:
    # 1. sample friction and restitution values each time the environment is reset
    # 2. apply the sampled values to all the rigid bodies in the asset_cfg
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # extract the used quantities (to enable type-hinting)
        self.asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        self.asset: RigidObject | Articulation = env.scene[self.asset_cfg.name]

        if not isinstance(self.asset, (RigidObject, Articulation)):
            raise ValueError(
                f"Randomization term 'randomize_rigid_body_material' not supported for asset: '{self.asset_cfg.name}'"
                f" with type: '{type(self.asset)}'."
            )

        # obtain number of shapes per body (needed for indexing the material properties correctly)
        # note: this is a workaround since the Articulation does not provide a direct way to obtain the number of shapes
        #  per body. We use the physics simulation view to obtain the number of shapes per body.
        if isinstance(self.asset, Articulation) and self.asset_cfg.body_ids != slice(None):
            self.num_shapes_per_body = []
            for link_path in self.asset.root_physx_view.link_paths[0]:
                link_physx_view = self.asset._physics_sim_view.create_rigid_body_view(link_path)  # type: ignore
                self.num_shapes_per_body.append(link_physx_view.max_shapes)
            # ensure the parsing is correct
            num_shapes = sum(self.num_shapes_per_body)
            expected_shapes = self.asset.root_physx_view.max_shapes
            if num_shapes != expected_shapes:
                raise ValueError(
                    "Randomization term 'randomize_rigid_body_material' failed to parse the number of shapes per body."
                    f" Expected total shapes: {expected_shapes}, but got: {num_shapes}."
                )
        else:
            # in this case, we don't need to do special indexing
            self.num_shapes_per_body = None

        # obtain parameters for sampling friction and restitution values
        static_friction_range = cfg.params.get("static_friction_range", (1.0, 1.0))
        dynamic_friction_range = cfg.params.get("dynamic_friction_range", (1.0, 1.0))
        restitution_range = cfg.params.get("restitution_range", (0.0, 0.0))

        # sample material properties from the given ranges
        # note: we only sample the materials once during initialization
        #   afterwards these are randomly assigned to the geometries of the asset
        range_list = [static_friction_range, dynamic_friction_range, restitution_range]
        self.ranges = torch.tensor(range_list, device="cpu")

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        static_friction_range: tuple[float, float],
        dynamic_friction_range: tuple[float, float],
        restitution_range: tuple[float, float],
        asset_cfg: SceneEntityCfg,
        make_consistent: bool = False,
    ):
        # resolve environment ids
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()

        # retrieve material buffer from the physics simulation
        materials = self.asset.root_physx_view.get_material_properties()
        sampled_materials = (
            torch.rand((len(env_ids), 3), device="cpu") * (self.ranges[:, 1] - self.ranges[:, 0]) + self.ranges[:, 0]
        )
        if make_consistent:
            sampled_materials[:, 1] = torch.min(sampled_materials[:, 0], sampled_materials[:, 1])

        # update material buffer with new samples
        if self.num_shapes_per_body is not None:
            # sample material properties from the given ranges
            for body_id in self.asset_cfg.body_ids:
                # obtain indices of shapes for the body
                start_idx = sum(self.num_shapes_per_body[:body_id])
                end_idx = start_idx + self.num_shapes_per_body[body_id]
                # assign the new materials
                materials[env_ids, start_idx:end_idx] = sampled_materials.unsqueeze(1)
        else:
            # assign all the materials
            materials[env_ids] = sampled_materials[:]

        # apply to simulation
        self.asset.root_physx_view.set_material_properties(materials, env_ids)


def randomize_rigid_body_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    inertia_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the inertia tensors of the bodies by adding, scaling, or setting random values.

    This function allows randomizing only the diagonal inertia tensor components (xx, yy, zz) of the bodies.
    The function samples random values from the given distribution parameters and adds, scales, or sets the values
    into the physics simulation based on the operation.

    .. tip::
        This function uses CPU tensors to assign the body inertias. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current inertia tensors of the bodies (num_assets, num_bodies, 9 for articulations or 9 for rigid objects)
    inertias = asset.root_physx_view.get_inertias()

    # apply randomization on default values
    inertias[env_ids[:, None], body_ids, :] = asset.data.default_inertia[env_ids[:, None], body_ids, :].clone()

    # randomize each diagonal element (xx, yy, zz -> indices 0, 4, 8)
    for idx in [0, 4, 8]:
        # Extract and randomize the specific diagonal element
        randomized_inertias = _randomize_prop_by_op(
            inertias[:, :, idx],
            inertia_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        # Assign the randomized values back to the inertia tensor
        inertias[env_ids[:, None], body_ids, idx] = randomized_inertias

    # set the inertia tensors into the physics simulation
    asset.root_physx_view.set_inertias(inertias, env_ids)


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    # resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data


# =============================================================================
# Place a rigid payload relative to a robot body.
# Adapted from SteadyTray's set_rigid_object_relative_to_robot.
# =============================================================================

# stores the last relative placement (in base local frame) per env; exposed as the
# event term's return value.
_all_rel_pos = torch.tensor([])


def _sample_or_fill(spec, num, device):
    """Return a (num,) tensor: uniform in range if `spec` is a (lo, hi) tuple, else constant."""
    if isinstance(spec, tuple):
        return torch.rand(num, device=device) * (spec[1] - spec[0]) + spec[0]
    return torch.full((num,), float(spec), device=device)


def set_rigid_object_relative_to_robot(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    base_asset_cfg: SceneEntityCfg,
    target_asset_cfg: SceneEntityCfg,
    relative_pose: dict[str, float | tuple[float, float]],
    relative_velocity: dict[str, float] = {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
    },
) -> torch.Tensor:
    """On reset, place ``target`` (e.g. the tray) at ``relative_pose`` w.r.t. a ``base`` body's
    current world pose (base may be an Articulation body or a RigidObject). Each of x/y/z/roll/
    pitch/yaw may be a scalar or a (lo, hi) range to randomize. Returns per-env relative position.
    """
    global _all_rel_pos
    if _all_rel_pos.numel() == 0 or _all_rel_pos.shape[0] < env.num_envs:
        _all_rel_pos = torch.zeros((env.num_envs, 3), device=env.device)
    if _all_rel_pos.device != env.device:
        _all_rel_pos = _all_rel_pos.to(env.device)

    base_asset_cfg.resolve(env.scene)
    target_asset_cfg.resolve(env.scene)

    base_asset: Articulation | RigidObject = env.scene[base_asset_cfg.name]
    target_object: RigidObject = env.scene[target_asset_cfg.name]

    # base world pose (body link for articulation, root for rigid object)
    if isinstance(base_asset, Articulation):
        if isinstance(base_asset_cfg.body_ids, slice):
            raise ValueError("base_asset_cfg.body_ids must be a list/int, not a slice.")
        body_idx = base_asset_cfg.body_ids[0]
        base_pos = base_asset.data.body_link_pos_w[env_ids, body_idx]
        base_quat = base_asset.data.body_link_quat_w[env_ids, body_idx]
    else:
        base_pos = base_asset.data.root_pos_w[env_ids]
        base_quat = base_asset.data.root_quat_w[env_ids]

    n = len(env_ids)
    rel_pos_local = torch.stack(
        [
            _sample_or_fill(relative_pose["x"], n, env.device),
            _sample_or_fill(relative_pose["y"], n, env.device),
            _sample_or_fill(relative_pose["z"], n, env.device),
        ],
        dim=1,
    )
    object_pos = base_pos + math_utils.quat_apply(base_quat, rel_pos_local)

    rel_quat = math_utils.quat_from_euler_xyz(
        _sample_or_fill(relative_pose.get("roll", 0.0), n, env.device),
        _sample_or_fill(relative_pose.get("pitch", 0.0), n, env.device),
        _sample_or_fill(relative_pose.get("yaw", 0.0), n, env.device),
    )
    object_quat = math_utils.quat_mul(base_quat, rel_quat)

    rel_lin_vel = torch.stack(
        [torch.full((n,), float(relative_velocity[k]), device=env.device) for k in ("x", "y", "z")], dim=1
    )
    rel_ang_vel = torch.stack(
        [torch.full((n,), float(relative_velocity[k]), device=env.device) for k in ("roll", "pitch", "yaw")], dim=1
    )
    object_lin_vel = math_utils.quat_apply(base_quat, rel_lin_vel)
    object_ang_vel = math_utils.quat_apply(base_quat, rel_ang_vel)

    root_state = target_object.data.default_root_state[env_ids].clone()
    root_state[:, :3] = object_pos
    root_state[:, 3:7] = object_quat
    root_state[:, 7:10] = object_lin_vel
    root_state[:, 10:13] = object_ang_vel
    target_object.write_root_state_to_sim(root_state, env_ids)

    _all_rel_pos[env_ids] = rel_pos_local
    return _all_rel_pos
