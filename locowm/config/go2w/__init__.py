import gymnasium as gym
from .agents import rsl_rl_ppo_cfg
from . import locomotion_env_cfg
from . import transport_env_cfg
from . import eval_env_cfg

# ----------------------------------- Locomotion Go2W  -----------------------------------
gym.register(
    id="Isaac-LocomotionGo2W-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": locomotion_env_cfg.LocomotionEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.LocomotionPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-LocomotionGo2W-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": locomotion_env_cfg.LocomotionPlayEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.LocomotionPPORunnerCfg,
    },
)


# ----------------------------------- Transport Go2W-----------------------------------
gym.register(
    id="Isaac-TransportGo2W-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": transport_env_cfg.TransportEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.TransportPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-TransportGo2W-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": transport_env_cfg.TransportPlayEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.TransportPPORunnerCfg,
    },
)


# Stage-2 transport with a residual adapter.
gym.register(
    id="Isaac-TransportGo2W-Adapter-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": transport_env_cfg.TransportEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.TransportAdapterPPORunnerCfg,
    },
)

gym.register(
    id="Isaac-TransportGo2W-Adapter-Play-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": transport_env_cfg.TransportPlayEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.TransportAdapterPPORunnerCfg,
    },
)


# Go2W ablations.
# No-WM adapter: observations, base actions and zero prediction features.
gym.register(
    id="Isaac-TransportGo2W-Adapter-NoWM-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": transport_env_cfg.TransportEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.TransportAdapterNoWMPPORunnerCfg,
    },
)

# Stage-1 current-state reconstruction.
gym.register(
    id="Isaac-LocomotionGo2W-ReconWM-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": locomotion_env_cfg.LocomotionEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.LocomotionReconWMPPORunnerCfg,
    },
)

# Stage-2 adapter with the reconstruction world model.
gym.register(
    id="Isaac-TransportGo2W-Adapter-ReconWM-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": transport_env_cfg.TransportEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.TransportAdapterReconWMPPORunnerCfg,
    },
)


# =============================================================================
# Multi-terrain tracking and payload evaluation.
# Four terrain types and four tiers; accelerate on flat ground before traversal.
# Policy configurations preserve the critic dimensions used during training.
#     Base(group2 locomotion) / E2E(group3 transport) / Adapter(group1) / NoWM(group4)
# See locowm/scripts/eval.py and scripts/run_eval.sh.
# =============================================================================

# group2 base locomotion policy (locomotion critic obs)
gym.register(
    id="Isaac-EvalGo2W-Base-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": eval_env_cfg.EvalBaseEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.LocomotionPPORunnerCfg,
    },
)

# End-to-end transport policy.
gym.register(
    id="Isaac-EvalGo2W-E2E-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": eval_env_cfg.EvalEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.TransportPPORunnerCfg,
    },
)

# Full adapter; requires Stage-1 policy and world-model checkpoints.
gym.register(
    id="Isaac-EvalGo2W-Adapter-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": eval_env_cfg.EvalEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.TransportAdapterPPORunnerCfg,
    },
)

# No-WM adapter; requires a Stage-1 policy checkpoint.
gym.register(
    id="Isaac-EvalGo2W-NoWM-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": eval_env_cfg.EvalEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.TransportAdapterNoWMPPORunnerCfg,
    },
)


# =============================================================================
# Payload success-rate evaluation.
# Each trial traverses one feature; payload center below base center means failure.
# Drops during acceleration reset the trial without counting a failure.
# See locowm/scripts/succ_eval.py.
# =============================================================================
gym.register(
    id="Isaac-SuccGo2W-Base-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": eval_env_cfg.SuccessBaseEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.LocomotionPPORunnerCfg,
    },
)
gym.register(
    id="Isaac-SuccGo2W-E2E-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": eval_env_cfg.SuccessEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.TransportPPORunnerCfg,
    },
)
gym.register(
    id="Isaac-SuccGo2W-Adapter-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": eval_env_cfg.SuccessEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.TransportAdapterPPORunnerCfg,
    },
)
gym.register(
    id="Isaac-SuccGo2W-NoWM-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": eval_env_cfg.SuccessEnvCfg,
        "rsl_rl_cfg_entry_point": rsl_rl_ppo_cfg.TransportAdapterNoWMPPORunnerCfg,
    },
)
