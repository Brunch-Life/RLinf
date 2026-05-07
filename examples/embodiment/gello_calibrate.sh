#!/bin/bash
#
# Two-pose GELLO calibration: solve joint_signs AND joint_offsets at the
# same time by sampling raw Dynamixel motor positions at two known
# Franka joint configurations.
#
# Run this whenever:
#   - You first install a new GELLO unit
#   - The alignment script (gello_align.sh) reports "aligned" at deltas
#     near zero but the GELLO leader's physical pose visibly does not
#     match the robot
#   - You suspect a sign is flipped on some joint
#
# The script will safely move the robot between two preset poses via
# reset_joint (effective dynamics ~4.5%, max_joint_delta=1.5 rad guard).
# At each pose you'll be asked to physically pose the GELLO to match.
#
# Required env vars (override on the command line if your setup differs):
#
#   FRANKA_ROBOT_IP        Franka FCI IP                  (default: 172.16.0.2)
#   GELLO_PORT             GELLO Dynamixel by-id path      (default: FTAJEDPC by-id)
#   GELLO_BAUDRATE         Dynamixel baud (Franka GELLO    (default: 1000000)
#                          uses 1 Mbps)
#
# The Robotiq gripper port is auto-discovered from the local FTDI USB-RS485
# adapter (each dual-Franka node has exactly one), so no override is needed.
#
# Usage:
#
#   bash examples/embodiment/gello_calibrate.sh
#
# After it prints the new DynamixelRobotConfig snippet, paste it into
# /home/i-yinuo/cynws/third_party/gello_software/gello/agents/gello_agent.py
# (replacing the existing FTAJEDPC entry), then re-run
# bash examples/embodiment/gello_align.sh to verify.

set -euo pipefail

EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
REPO_PATH="$(dirname "$(dirname "$EMBODIED_PATH")")"
SRC_FILE="${REPO_PATH}/toolkits/realworld_check/gello_calibrate.py"

export PYTHONPATH="${REPO_PATH}:${PYTHONPATH:-}"
export FRANKA_ROBOT_IP="${FRANKA_ROBOT_IP:-172.16.0.2}"
export GELLO_PORT="${GELLO_PORT:-/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAJEDPC-if00-port0}"
export GELLO_BAUDRATE="${GELLO_BAUDRATE:-1000000}"
unset RAY_ADDRESS

echo "Using Python at $(which python)"
echo "  FRANKA_ROBOT_IP = ${FRANKA_ROBOT_IP}"
echo "  GELLO_PORT      = ${GELLO_PORT}"
echo "  GELLO_BAUDRATE  = ${GELLO_BAUDRATE}"
echo

if [ ! -f "${SRC_FILE}" ]; then
    echo "ERROR: ${SRC_FILE} not found" >&2
    exit 1
fi

exec python "${SRC_FILE}"
