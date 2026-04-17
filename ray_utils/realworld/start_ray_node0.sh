#!/bin/bash
# Launch Ray head (node 0) for dual-Franka realworld collection.
#
# Usage (run on the head machine, e.g. ubuntu-franka-slave / 192.168.120.43):
#   bash ray_utils/realworld/start_ray_node0.sh
#
# Tears down any stale Ray instance, activates the project venv, exports
# the env vars RLinf captures at ray-start time, and starts the head.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_PATH"

# --- venv & PYTHONPATH ---------------------------------------------------
source "$REPO_PATH/.venv/bin/activate"
export PYTHONPATH="$REPO_PATH:${PYTHONPATH:-}"

# --- RLinf multi-node env (captured by Ray at start time) ----------------
export RLINF_NODE_RANK=0
# Uncomment and set to the interface that connects the two nodes if the
# default NCCL autodetect picks the wrong one (e.g. docker0, tailscale):
# export RLINF_COMM_NET_DEVICES="rlinf"

# --- Ray head ------------------------------------------------------------
HEAD_IP="192.168.120.43"
RAY_PORT=6379

ray stop --force || true
ray start --head --port="$RAY_PORT" --node-ip-address="$HEAD_IP"
