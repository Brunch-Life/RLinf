#! /bin/bash
# Real-world evaluation launcher.
#
# Usage:
#   bash eval_realworld.sh [CONFIG_NAME]
#
# Examples:
#   bash eval_realworld.sh realworld_eval_zed_robotiq
#   bash eval_realworld.sh realworld_eval_zed_robotiq  # default if no arg

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/eval_realworld.py"

export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
export HYDRA_FULL_ERROR=1

if [ -z "$1" ]; then
    CONFIG_NAME="realworld_eval_zed_robotiq"
else
    CONFIG_NAME=$1
fi

echo "Using Python at $(which python)"
echo "Config: ${CONFIG_NAME}"

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-eval-${CONFIG_NAME}"
MEGA_LOG_FILE="${LOG_DIR}/eval_realworld.log"
mkdir -p "${LOG_DIR}"

CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
