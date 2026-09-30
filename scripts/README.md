# Experiment script templates

The scripts in this directory are portable templates. Activate the Isaac Sim
environment before running them, or set `ISAACLAB_ENV_SCRIPT` to a shell script
that performs the activation. All scripts resolve the repository root from
their own location.

Common variables:

- `PYTHON_BIN`: Python executable, default `python`.
- `NUM_ENVS`: number of parallel environments.
- `LOGGER`: `tensorboard` by default; use `wandb` or `neptune` when configured.

Stage 2 scripts additionally require `POLICY_CHECKPOINT`; the full adapter
script also requires `WORLD_MODEL_CHECKPOINT`. The evaluation script requires
`BASE_RUN`, `BASE_CHECKPOINT`, `E2E_RUN`, `E2E_CHECKPOINT`, `ADAPTER_RUN`,
`ADAPTER_CHECKPOINT`, `NOWM_RUN`, and `NOWM_CHECKPOINT`.
