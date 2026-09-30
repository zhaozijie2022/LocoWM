#!/usr/bin/env bash

# Shared shell setup for the public experiment templates. An Isaac Lab
# activation script can be supplied when the caller does not activate the
# environment beforehand: ISAACLAB_ENV_SCRIPT=/path/to/env.sh ./scripts/...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOGGER="${LOGGER:-tensorboard}"

if [[ -n "${ISAACLAB_ENV_SCRIPT:-}" ]]; then
  # shellcheck disable=SC1090
  source "${ISAACLAB_ENV_SCRIPT}"
fi

cd "${ROOT_DIR}"
