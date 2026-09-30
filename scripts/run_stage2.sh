#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

NUM_ENVS="${NUM_ENVS:-4096}"
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:?Set POLICY_CHECKPOINT to a Stage-1 policy checkpoint}"
WORLD_MODEL_CHECKPOINT="${WORLD_MODEL_CHECKPOINT:?Set WORLD_MODEL_CHECKPOINT to a Stage-1 world-model checkpoint}"

"${PYTHON_BIN}" -m locowm.scripts.train \
  --task Isaac-TransportGo2W-Adapter-v1 \
  --num_envs "${NUM_ENVS}" \
  --headless \
  --logger "${LOGGER}" \
  --policy_checkpoint "${POLICY_CHECKPOINT}" \
  --world_model_checkpoint "${WORLD_MODEL_CHECKPOINT}"
