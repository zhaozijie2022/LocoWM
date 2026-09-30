from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class WorldModelCfg:
    """World-model settings for joint Stage-1 training with PPO.

    Predict future angular velocity, linear acceleration and projected gravity
    from policy observations and actions. Targets come from wm_target, and the
    separate optimizer does not backpropagate into the policy."""

    enabled: bool = True
    mlp_hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu"
    use_resnet: bool = False
    chunk_len: int = 5  # Prediction horizon in steps.
    num_targets: int = 9  # base_ang_vel(3)+base_lin_acc_w(3)+projected_gravity_b(3)
    # Reconstruction predicts the current state with a one-step output.
    reconstruct_current: bool = False
    learning_rate: float = 1.0e-3
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    max_grad_norm: float = 1.0


@configclass
class LocomotionPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 80000
    save_interval = 500
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
    )
    world_model: WorldModelCfg = WorldModelCfg()
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    experiment_name = "locomotion"
    logger = "wandb"
    wandb_project = "LocoWM"


@configclass
class TransportPPORunnerCfg(LocomotionPPORunnerCfg):
    """Transport PPO settings using the locomotion network architecture."""

    def __post_init__(self):
        self.experiment_name = "transport"
        self.wandb_project = "LocoWM"


# =============================================================================
# Stage 2: freeze the base policy and world model; train the residual adapter.
# =============================================================================
@configclass
class AdapterCfg:
    """Stage-2 residual-adapter settings."""

    adapter_type: str = "mlp"  # FiLM is not implemented.
    hidden_dims: list[int] = [256, 128]
    activation: str = "elu"


@configclass
class TransportAdapterPolicyCfg(RslRlPpoActorCriticCfg):
    """Frozen Stage-1 policy and world model with a learned adapter and new critic."""

    class_name: str = "AdapterActorCritic"
    noise_std_type: str = "log"
    # Stage-1 checkpoints must match the frozen network architecture.
    policy_checkpoint: str = ""
    world_model_checkpoint: str = ""
    adapter: AdapterCfg = AdapterCfg()
    # Frozen world-model architecture must match the Stage-1 checkpoint.
    world_model: WorldModelCfg = WorldModelCfg()


@configclass
class TransportAdapterPPORunnerCfg(TransportPPORunnerCfg):
    """Train the transport adapter while keeping the world model frozen."""

    def __post_init__(self):
        super().__post_init__()  # transport experiment_name / wandb_project
        self.experiment_name = "transport_adapter"
        # Actor dimensions must match the Stage-1 policy.
        # Supply frozen-component checkpoints through the command line.

        self.policy = TransportAdapterPolicyCfg(
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
            init_noise_std=1.0,
        )
        # The world model is frozen within the adapter policy.
        self.world_model.enabled = False


# =============================================================================
# Each ablation has a separate experiment name and logging group.


#   group3 e2e :  Isaac-TransportGo2W-v1 (transport)


# =============================================================================
@configclass
class LocomotionReconWMPPORunnerCfg(LocomotionPPORunnerCfg):
    """Stage-1 ablation that reconstructs current task state instead of predicting it."""

    def __post_init__(self):
        self.experiment_name = "locomotion_reconwm"
        self.world_model.reconstruct_current = True
        self.world_model.chunk_len = 1


@configclass
class TransportAdapterNoWMPPORunnerCfg(TransportAdapterPPORunnerCfg):
    """Stage-2 ablation with zero world-model features and the same base actions."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "transport_adapter_nowm"
        # Supply the base policy checkpoint and omit the world-model checkpoint.
        # The adapter then receives zero world-model features.
        self.policy.world_model_checkpoint = ""


@configclass
class TransportAdapterReconWMPPORunnerCfg(TransportAdapterPPORunnerCfg):
    """Stage-2 adapter conditioned on the reconstruction world model."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "transport_adapter_reconwm"
        # Supply checkpoints from the reconstruction variant of Stage 1.
        # Match the frozen model to current-state reconstruction.
        self.policy.world_model.reconstruct_current = True
        self.policy.world_model.chunk_len = 1
