#! /bin/bash

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/collect_real_data.py"

export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

export HYDRA_FULL_ERROR=1

CONFIG_NAME="${1:-dosw1_collect_data}"

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/dosw1-collect/$(date +'%Y%m%d-%H%M%S')"
MEGA_LOG_FILE="${LOG_DIR}/collect_dosw1_data.log"
mkdir -p "${LOG_DIR}"
CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR}"
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
