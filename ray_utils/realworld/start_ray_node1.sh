#!/bin/bash
# Join Ray cluster as worker (node 1) for dual-Franka realworld collection.
#
# Usage (run on the worker machine, master / 192.168.120.140):
#   bash ray_utils/realworld/start_ray_node1.sh
#
# Tears down any stale Ray instance, activates the project venv, exports
# the env vars RLinf captures at ray-start time, and joins the head.
#
# NOTE: Edit WORKER_IP below to match this machine's IP on the same subnet
# as the head (git-synced from node 0 — the head side keeps its own script).

set -eo pipefail  # no -u: venv activate touches unset LD_LIBRARY_PATH

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_PATH"

# --- venv & PYTHONPATH ---------------------------------------------------
source "$REPO_PATH/.venv/bin/activate"
export PYTHONPATH="$REPO_PATH:${PYTHONPATH:-}"

# Tear down any stale Ray instance before exporting new env vars / joining.
ray stop --force || true

# --- RLinf multi-node env (captured by Ray at start time) ----------------
export RLINF_NODE_RANK=1
# Keep the same interface as node 0, or override if this host's routing
# to the head goes through a different NIC:
# export RLINF_COMM_NET_DEVICES="rlinf"

# --- Ray worker ----------------------------------------------------------
# Shared lab LAN: head (node 0, slave) = 192.168.120.143,
#                 this host (node 1, master) = 192.168.120.140.
# Earlier rigs used the 10.10.10.0/24 USB-Ethernet direct link or the
# older 192.168.120.43/42 LAN IPs — both retired in favour of these.
HEAD_IP="192.168.120.143"
RAY_PORT=6379
WORKER_IP="192.168.120.140"  # <-- this node's IP on the LAN

ray start --address="$HEAD_IP:$RAY_PORT" --node-ip-address="$WORKER_IP"
