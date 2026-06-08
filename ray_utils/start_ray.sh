#!/usr/bin/env bash

set -euo pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PATH="$(dirname "$SCRIPT_PATH")"
RAY_HEAD_IP_FILE="${RAY_HEAD_IP_FILE:-$REPO_PATH/ray_utils/ray_head_ip.txt}"

if [[ "$REPO_PATH" == "/workspace/RLinf" || "$REPO_PATH" == "/home/arm/RLinf_feature_turtle2_deploy" || "$(hostname)" == "arm-turtle" ]]; then
    export RLINF_NODE_RANK=1
    export RLINF_COMM_NET_DEVICES=enp1s0
    export RAY_HEAD_IP=192.168.120.237
else
    export RLINF_NODE_RANK=0
    export RLINF_COMM_NET_DEVICES=enp132s0
fi

RANK_VALUE="$RLINF_NODE_RANK"
export RLINF_NODE_RANK="$RANK_VALUE"

RAY_PORT="${RAY_PORT:-6379}"
RAY_STOP_BEFORE_START="${RAY_STOP_BEFORE_START:-1}"
RAY_BLOCK="${RAY_BLOCK:-0}"
RAY_TEMP_DIR="${RAY_TEMP_DIR:-/tmp/ray-rlinf}"

unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

get_ip_from_interface() {
    local iface="$1"
    ip -4 -o addr show dev "$iface" scope global 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -n1
}

get_default_ip() {
    hostname -I | awk '{print $1}'
}

NODE_IP="${RAY_NODE_IP_ADDRESS:-}"
if [[ -z "$NODE_IP" && -n "${RLINF_COMM_NET_DEVICES:-}" ]]; then
    NODE_IP="$(get_ip_from_interface "$RLINF_COMM_NET_DEVICES")"
fi
if [[ -z "$NODE_IP" ]]; then
    NODE_IP="$(get_default_ip)"
fi
if [[ -z "$NODE_IP" ]]; then
    echo "Error: could not determine node IP. Set RAY_NODE_IP_ADDRESS explicitly." >&2
    exit 1
fi

export PYTHONPATH="$REPO_PATH:${PYTHONPATH:-}"

if [[ "$RAY_STOP_BEFORE_START" == "1" ]]; then
    ray stop --force || true
    pkill -9 gcs_server 2>/dev/null || true
    pkill -9 raylet 2>/dev/null || true
fi

ray_args=(--node-ip-address="$NODE_IP" --disable-usage-stats)

if [[ -n "${RAY_TEMP_DIR:-}" ]]; then
    ray_args+=(--temp-dir="$RAY_TEMP_DIR")
fi
if [[ -n "${RAY_MEMORY:-}" ]]; then
    ray_args+=(--memory="$RAY_MEMORY")
fi
if [[ -n "${RAY_OBJECT_STORE_MEMORY:-}" ]]; then
    ray_args+=(--object-store-memory="$RAY_OBJECT_STORE_MEMORY")
fi
if [[ "$RAY_BLOCK" == "1" ]]; then
    ray_args+=(--block)
fi

if [[ "$RANK_VALUE" -eq 0 ]]; then
    RAY_NUM_GPUS="${RAY_NUM_GPUS:-1}"
    if [[ -n "${RAY_NUM_GPUS:-}" ]]; then
        ray_args+=(--num-gpus="$RAY_NUM_GPUS")
    fi

    echo "Starting Ray head: rank=$RANK_VALUE ip=$NODE_IP port=$RAY_PORT"
    ray start --head --port="$RAY_PORT" "${ray_args[@]}"

    echo "$NODE_IP" > "$RAY_HEAD_IP_FILE"
    echo "Head node IP written to $RAY_HEAD_IP_FILE"
else
    HEAD_ADDRESS="${RAY_ADDRESS:-}"
    if [[ -z "$HEAD_ADDRESS" ]]; then
        HEAD_IP="${RAY_HEAD_IP:-}"
        if [[ -z "$HEAD_IP" && -f "$RAY_HEAD_IP_FILE" ]]; then
            HEAD_IP="$(cat "$RAY_HEAD_IP_FILE")"
        fi
        if [[ -z "$HEAD_IP" ]]; then
            echo "Error: set RAY_HEAD_IP or RAY_ADDRESS on worker nodes." >&2
            exit 1
        fi
        HEAD_ADDRESS="$HEAD_IP:$RAY_PORT"
    fi

    echo "Starting Ray worker: rank=$RANK_VALUE ip=$NODE_IP head=$HEAD_ADDRESS"
    ray start --address="$HEAD_ADDRESS" "${ray_args[@]}"
fi
