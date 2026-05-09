#!/bin/bash
#
# Move the Franka arm to the GELLO's current joint positions.
#
# Run this BEFORE collect_data.sh so the robot and GELLO start
# at the same pose — avoids the violent reset motion at startup.
#
# Required env vars (override on the command line if your setup differs):
#
#   FRANKA_ROBOT_IP        Franka FCI IP                  (default: 172.16.0.2)
#   GELLO_PORT             GELLO Dynamixel by-id path      (default: FTAJEDPC by-id)
#
# The Robotiq gripper port is auto-discovered from the local FTDI USB-RS485
# adapter (each dual-Franka node has exactly one), so no override is needed.
#
# Usage:
#
#   bash examples/embodiment/gello_reset_to_gello.sh

set -euo pipefail

EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
REPO_PATH="$(dirname "$(dirname "$EMBODIED_PATH")")"
SRC_FILE="${REPO_PATH}/toolkits/realworld_check/gello.py"

export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export FRANKA_ROBOT_IP="${FRANKA_ROBOT_IP:-172.16.0.2}"
export GELLO_PORT="${GELLO_PORT:-/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAJEDPC-if00-port0}"
unset RAY_ADDRESS

echo "Using Python at $(which python)"
echo "  FRANKA_ROBOT_IP = ${FRANKA_ROBOT_IP}"
echo "  GELLO_PORT      = ${GELLO_PORT}"
echo

if [ ! -f "${SRC_FILE}" ]; then
    echo "ERROR: ${SRC_FILE} not found" >&2
    exit 1
fi

exec python "${SRC_FILE}" reset-to-gello
