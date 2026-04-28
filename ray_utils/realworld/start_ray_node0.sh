#!/bin/bash
# Launch Ray head (node 0) for dual-Franka realworld collection.
#
# Usage (run on the head machine, slave / 192.168.120.143):
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
# Shared lab LAN: node 0 (this host, slave) = 192.168.120.143,
#                 node 1 (master)          = 192.168.120.140.
# Earlier rigs used a 10.10.10.0/24 USB-Ethernet point-to-point link or
# the older 192.168.120.43/42 LAN IPs — both retired in favour of these.
HEAD_IP="192.168.120.143"
RAY_PORT=6379

ray start --head --port="$RAY_PORT" --node-ip-address="$HEAD_IP"
