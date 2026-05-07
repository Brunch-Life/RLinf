#!/bin/bash
# Launch Ray head for multi-node realworld collection.
# Usage: bash ray_utils/realworld/start_ray_node0.sh

set -eo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_PATH"

source "$REPO_PATH/.venv/bin/activate"
export PYTHONPATH="$REPO_PATH:${PYTHONPATH:-}"

ray stop --force || true

export RLINF_NODE_RANK=0

HEAD_IP="HEAD_IP"  # Replace: this host's LAN IP (shared with worker nodes)
RAY_PORT=6379

ray start --head --port="$RAY_PORT" --node-ip-address="$HEAD_IP"
