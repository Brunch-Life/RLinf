#!/bin/bash
# Launch Ray head (node 0) for dual-Franka realworld collection.
#
# Usage (run on the head machine, e.g. ubuntu-franka-slave / 192.168.120.43):
#   bash ray_utils/realworld/start_ray_node0.sh
#
# Tears down any stale Ray instance, activates the project venv, exports
# the env vars RLinf captures at ray-start time, and starts the head.

set -eo pipefail  # no -u: venv activate touches unset LD_LIBRARY_PATH

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_PATH"

# --- venv & PYTHONPATH ---------------------------------------------------
source "$REPO_PATH/.venv/bin/activate"
export PYTHONPATH="$REPO_PATH:${PYTHONPATH:-}"

# Tear down any stale Ray instance before exporting new env vars / starting.
ray stop --force || true

# --- RLinf multi-node env (captured by Ray at start time) ----------------
export RLINF_NODE_RANK=0
# Uncomment and set to the interface that connects the two nodes if the
# default NCCL autodetect picks the wrong one (e.g. docker0, tailscale):
# export RLINF_COMM_NET_DEVICES="rlinf"

# --- Ray head ------------------------------------------------------------
# Direct point-to-point USB-Ethernet link between the two nodes:
#   node 0  enx207bd232e224  10.10.10.1  <--->  10.10.10.2  enx00e04c364742  node 1
# sub-ms RTT on this link vs ~25 ms mdev over the shared WiFi — Ray cross-
# node RPC (franky streamer, gripper events, controller state polls) is the
# only thing we want going through the direct cable, so bind Ray to the
# direct-link IP here and on node 1.  Old WiFi IP: 192.168.120.43.
HEAD_IP="10.10.10.1"
RAY_PORT=6379

ray start --head --port="$RAY_PORT" --node-ip-address="$HEAD_IP"
