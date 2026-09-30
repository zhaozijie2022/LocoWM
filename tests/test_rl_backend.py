"""Behavioral contracts at the RSL-RL / LocoWM boundary (no simulator required)."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.modules import ActorCritic as RslRlActorCritic

from loco_rl.modules import ActorCritic, AdapterActorCritic
from loco_rl.runners import OnPolicyRunner
from loco_rl.storage import RolloutStorage


@pytest.mark.parametrize("noise_type", ["scalar", "log"])
def test_actor_checkpoint_noise_bounds_and_reset(noise_type):
    upstream = RslRlActorCritic(3, 4, 2, [8], [8], noise_std_type=noise_type)
    policy = ActorCritic(3, 4, 2, [8], [8], "elu", 0.7, noise_type)
    result = policy.load_state_dict(upstream.state_dict(), assign=True)
    assert result.missing_keys == result.unexpected_keys == []
    obs = torch.randn(4, 3)
    torch.testing.assert_close(policy.act_inference(obs), upstream.act_inference(obs))
    with torch.no_grad():
        parameter = policy.std if noise_type == "scalar" else policy.log_std
        parameter.copy_(torch.tensor([-30.0, 30.0]))
    policy.update_distribution(obs)
    expected = torch.tensor([1e-6, 5.0]) if noise_type == "scalar" else torch.tensor([-10.0, 2.0]).exp()
    torch.testing.assert_close(policy.action_std, expected.expand(4, 2))
    policy.reset_init_std()
    policy.update_distribution(obs)
    torch.testing.assert_close(policy.action_std, torch.full((4, 2), 0.7))
    actor, critic = policy.get_actor_critic_obs_from_obs_dict({"policy": obs})
    assert actor is critic is obs


@pytest.mark.parametrize("privileged", [False, True])
@pytest.mark.parametrize("rnd", [False, True])
def test_storage_gae_episode_boundaries_and_batch_contract(privileged, rnd):
    storage = RolloutStorage(1, 3, [1], [1] if privileged else None, [1], [1] if rnd else None)
    transition = storage.Transition()
    for step in range(3):
        transition.observations = torch.tensor([[float(step)]])
        transition.critic_observations = transition.observations + 10
        transition.actions = torch.zeros(1, 1)
        transition.rewards = torch.tensor([step + 1.0])
        transition.dones = torch.tensor([step == 1])
        transition.values = torch.tensor([[0.5]])
        transition.actions_log_prob = torch.zeros(1)
        transition.action_mean = torch.zeros(1, 1)
        transition.action_sigma = torch.ones(1, 1)
        transition.rnd_state = transition.observations + 20
        storage.add_transitions(transition)
        transition.clear()
        assert transition.critic_observations is None
    storage.compute_returns(torch.tensor([[4.0]]), gamma=0.9, lam=1.0, normalize_advantage=False)
    # Step 1 ends an episode; step 2 bootstraps the final critic value.
    torch.testing.assert_close(storage.returns[:, 0, 0], torch.tensor([2.8, 2.0, 6.6]))
    torch.testing.assert_close(storage.advantages, storage.returns - 0.5)
    batches = list(storage.mini_batch_generator(3, 2))
    assert len(batches) == 6
    for batch in batches:
        assert len(batch) == 10
        obs, critic_obs = batch[:2]
        torch.testing.assert_close(critic_obs, obs + 10 if privileged else obs)
        if rnd:
            torch.testing.assert_close(batch[-1], obs + 20)
        else:
            assert batch[-1] is None
    assert sorted(batch[0].item() for batch in batches[:3]) == [0.0, 1.0, 2.0]
    storage.clear()
    assert storage.step == 0


def make_runner(rnd):
    env = SimpleNamespace(
        num_envs=2, num_actions=2,
        get_observations=lambda: (torch.zeros(2, 3), {"observations": {"critic": torch.zeros(2, 4),
                                                                           "rnd_state": torch.zeros(2, 3)}}),
        unwrapped=SimpleNamespace(step_dt=0.02),
    )
    cfg = {
        "policy": {"class_name": "ActorCritic", "actor_hidden_dims": [8], "critic_hidden_dims": [8]},
        "algorithm": {"class_name": "PPO", "num_learning_epochs": 1, "num_mini_batches": 2},
        "world_model": {"enabled": True, "chunk_len": 2, "num_targets": 2, "mlp_hidden_dims": [8],
                        "num_learning_epochs": 1, "num_mini_batches": 2},
        "num_steps_per_env": 4, "save_interval": 1, "empirical_normalization": True,
    }
    if rnd:
        cfg["algorithm"]["rnd_cfg"] = {"num_outputs": 2, "predictor_hidden_dims": [8],
                                      "target_hidden_dims": [8], "weight": 0.1,
                                      "state_normalization": True, "reward_normalization": True}
    runner = OnPolicyRunner(env, cfg)
    runner.logger_type = "tensorboard"
    return runner


@pytest.mark.parametrize("rnd", [False, True])
def test_ppo_world_model_resume_and_stage2_loading(tmp_path, rnd):
    runner = make_runner(rnd)
    before = deepcopy(runner.alg.actor_critic.state_dict())
    wm_before = deepcopy(runner.alg.world_model.state_dict())
    for step in range(4):
        with torch.no_grad():
            runner.alg.act(runner.obs_normalizer(torch.randn(2, 3)),
                           runner.critic_obs_normalizer(torch.randn(2, 4)))
            runner.alg.process_env_step(
                torch.randn(2), torch.tensor([step == 1, False]),
                {"observations": {"wm_target": torch.randn(2, 2), "rnd_state": torch.randn(2, 3)},
                 "time_outs": torch.tensor([step == 1, False])},
            )
    runner.alg.compute_returns(torch.randn(2, 4))
    metrics = runner.alg.update()
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics if value is not None)
    assert runner.alg.storage.step == runner.alg.world_model.step == 0
    assert any(not torch.equal(before[k], v) for k, v in runner.alg.actor_critic.state_dict().items())
    assert any(not torch.equal(wm_before[k], v) for k, v in runner.alg.world_model.state_dict().items())
    runner.current_learning_iteration = 7
    checkpoint = tmp_path / "model_7.pt"
    runner.save(str(checkpoint), infos={"test": True})
    restored = make_runner(rnd)
    assert restored.load(str(checkpoint)) == {"test": True}
    assert restored.current_learning_iteration == 7
    for old, new in ((runner.alg.actor_critic, restored.alg.actor_critic),
                     (runner.alg.world_model, restored.alg.world_model),
                     (runner.obs_normalizer, restored.obs_normalizer),
                     (runner.critic_obs_normalizer, restored.critic_obs_normalizer),
                     (runner.alg.optimizer, restored.alg.optimizer),
                     (runner.alg.world_model.optimizer, restored.alg.world_model.optimizer)):
        torch.testing.assert_close(old.state_dict(), new.state_dict(), rtol=0, atol=0)
    if rnd:
        torch.testing.assert_close(runner.alg.rnd.state_dict(), restored.alg.rnd.state_dict(), rtol=0, atol=0)
        torch.testing.assert_close(runner.alg.rnd_optimizer.state_dict(), restored.alg.rnd_optimizer.state_dict())
    obs = torch.randn(2, 3)
    torch.testing.assert_close(runner.get_inference_policy()(obs), restored.get_inference_policy()(obs))
    adapter = AdapterActorCritic(
        3, 4, 2, [8], [8], policy_checkpoint=str(checkpoint),
        world_model_checkpoint=str(tmp_path / "world_model_7.pt"),
        adapter={"hidden_dims": [8]}, world_model={"chunk_len": 2, "num_targets": 2, "mlp_hidden_dims": [8]},
    )
    # The zero residual reproduces the stored base policy exactly.
    torch.testing.assert_close(adapter.act_inference(obs), runner.alg.actor_critic.act_inference(obs))
    assert all(not p.requires_grad for p in adapter.frozen_policy.parameters())
    assert all(not p.requires_grad for p in adapter.frozen_wm.parameters())
    # Also accept the previous combined policy + WM checkpoint format.
    combined = torch.load(checkpoint, weights_only=False)
    combined.update(torch.load(tmp_path / "world_model_7.pt", weights_only=False))
    torch.save(combined, tmp_path / "legacy.pt")
    restored.load(str(tmp_path / "legacy.pt"))
    torch.testing.assert_close(runner.alg.world_model.state_dict(), restored.alg.world_model.state_dict())
