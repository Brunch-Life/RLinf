#! /bin/bash

set -euo pipefail

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/eval_embodied_agent.py"

export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config_name> [hydra_overrides...]" >&2
    exit 1
fi

CONFIG_NAME=$1
shift

EXTRA_ARGS=("$@")

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
MEGA_LOG_FILE="${LOG_DIR}/run_realworld_eval.log"
mkdir -p "${LOG_DIR}"

# Shared diag dir picked up by rlinf.utils.diag_logger. The driver reads
# RLINF_DIAG_DIR; Ray actors (which don't inherit shell env) follow the
# logs/diag/current symlink. Disable with RLINF_DIAG_DISABLE=1.
export RLINF_DIAG_DIR="${LOG_DIR}/diag"
mkdir -p "${RLINF_DIAG_DIR}" "${REPO_PATH}/logs/diag"
ln -sfn "${RLINF_DIAG_DIR}" "${REPO_PATH}/logs/diag/current"

CMD="python ${SRC_FILE} --config-path ${EMBODIED_PATH}/config/ --config-name ${CONFIG_NAME} runner.logger.log_path=${LOG_DIR} ${EXTRA_ARGS[*]}"
echo ${CMD} > ${MEGA_LOG_FILE}
${CMD} 2>&1 | tee -a ${MEGA_LOG_FILE}
