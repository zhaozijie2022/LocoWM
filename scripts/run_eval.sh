#!/usr/bin/env bash
# Evaluate the four Go2W policies. Set the run/checkpoint variables before use.
source "$(dirname "$0")/common.sh"

NUM_ENVS="${NUM_ENVS:-256}"
BASE_RUN="${BASE_RUN:?Set BASE_RUN}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:?Set BASE_CHECKPOINT}"
E2E_RUN="${E2E_RUN:?Set E2E_RUN}"
E2E_CHECKPOINT="${E2E_CHECKPOINT:?Set E2E_CHECKPOINT}"
ADAPTER_RUN="${ADAPTER_RUN:?Set ADAPTER_RUN}"
ADAPTER_CHECKPOINT="${ADAPTER_CHECKPOINT:?Set ADAPTER_CHECKPOINT}"
NOWM_RUN="${NOWM_RUN:?Set NOWM_RUN}"
NOWM_CHECKPOINT="${NOWM_CHECKPOINT:?Set NOWM_CHECKPOINT}"
POLICY_CHECKPOINT="${POLICY_CHECKPOINT:?Set POLICY_CHECKPOINT to the Stage-1 policy checkpoint}"
WORLD_MODEL_CHECKPOINT="${WORLD_MODEL_CHECKPOINT:?Set WORLD_MODEL_CHECKPOINT to the Stage-1 world-model checkpoint}"

COMMON=(--num_envs "${NUM_ENVS}" --headless --logger "${LOGGER}")

"${PYTHON_BIN}" -m locowm.scripts.eval --task Isaac-EvalGo2W-Base-v1 "${COMMON[@]}" \
  --load_run "${BASE_RUN}" --checkpoint "${BASE_CHECKPOINT}"

"${PYTHON_BIN}" -m locowm.scripts.eval --task Isaac-EvalGo2W-E2E-v1 "${COMMON[@]}" \
  --load_run "${E2E_RUN}" --checkpoint "${E2E_CHECKPOINT}"

"${PYTHON_BIN}" -m locowm.scripts.eval --task Isaac-EvalGo2W-Adapter-v1 "${COMMON[@]}" \
  --load_run "${ADAPTER_RUN}" --checkpoint "${ADAPTER_CHECKPOINT}" \
  --policy_checkpoint "${POLICY_CHECKPOINT}" \
  --world_model_checkpoint "${WORLD_MODEL_CHECKPOINT}"

"${PYTHON_BIN}" -m locowm.scripts.eval --task Isaac-EvalGo2W-NoWM-v1 "${COMMON[@]}" \
  --load_run "${NOWM_RUN}" --checkpoint "${NOWM_CHECKPOINT}" \
  --policy_checkpoint "${POLICY_CHECKPOINT}"
