#!/usr/bin/env bash
set -euo pipefail

REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export REPO_PATH
export EMBODIED_PATH="$REPO_PATH/examples/embodiment"
export RLINF_NODE_RANK="${RLINF_NODE_RANK:-0}"
# Reuse an independently started Ray head.
export RAY_ADDRESS="${RAY_ADDRESS:-auto}"
export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export VK_DRIVER_FILES="${VK_DRIVER_FILES:-$VK_ICD_FILENAMES}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
config_name="${1:-realagainst_peg_slot_adversary_sac}"
if [ "$#" -gt 0 ]; then shift; fi
exec python "$EMBODIED_PATH/train_peg_slot_sac.py" --config-name "$config_name" "$@"
