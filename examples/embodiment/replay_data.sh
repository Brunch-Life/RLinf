#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/replay_real_data.py"

export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

export HYDRA_FULL_ERROR=1


if [ -z "$1" ]; then
    CONFIG_NAME="realworld_replay_joint_dual_franka"
else
    CONFIG_NAME=$1
    shift
fi

# Remaining args are forwarded as Hydra overrides, e.g.
#   bash replay_data.sh realworld_replay_joint_dual_franka \
#       replay.episode_index=3 replay.fps=10
EXTRA_ARGS=("$@")

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/replay_real_data.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR} ${EXTRA_ARGS[*]}"
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
