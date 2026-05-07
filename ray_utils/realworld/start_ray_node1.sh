#!/bin/bash
# Join Ray cluster as worker for multi-node realworld collection.
# Usage: bash ray_utils/realworld/start_ray_node1.sh

set -eo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_PATH"

source "$REPO_PATH/.venv/bin/activate"
export PYTHONPATH="$REPO_PATH:${PYTHONPATH:-}"

ray stop --force || true

export RLINF_NODE_RANK=1

HEAD_IP="HEAD_IP"      # Replace: head node's LAN IP
WORKER_IP="WORKER_IP"  # Replace: this host's LAN IP
RAY_PORT=6379

ray start --address="$HEAD_IP:$RAY_PORT" --node-ip-address="$WORKER_IP"
