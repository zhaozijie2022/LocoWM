#!/usr/bin/env bash
source "$(dirname "$0")/common.sh"

NUM_ENVS="${NUM_ENVS:-4096}"
"${PYTHON_BIN}" -m locowm.scripts.train \
  --task Isaac-TransportGo2W-v1 \
  --num_envs "${NUM_ENVS}" \
  --headless \
  --logger "${LOGGER}"
