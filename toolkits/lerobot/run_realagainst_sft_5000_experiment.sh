#!/usr/bin/env bash

set -euo pipefail

ROOT="/mnt/public/chenyinuo/RealAgainst_new/RLinf"
DATA="$ROOT/data/realagainst_peg_slot_expert_5000"
TRAIN_LOG="$ROOT/logs/realagainst_peg_slot_sft_5000_100k"
FINAL_CKPT="$TRAIN_LOG/realagainst_peg_slot_pi05_sft_5000_100k/checkpoints/global_step_100000"
SFT_CONFIG="realagainst_peg_slot_sft_5000_openpi_pi05"
EVAL_CONFIG="realagainst_peg_slot_sft_5000_openpi_rlinf_eval"

export PATH="$ROOT/.venv/bin:$PATH"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export REPO_PATH="$ROOT"
export EMBODIED_PATH="$ROOT/examples/sft"
export VK_ICD_FILENAMES="/etc/vulkan/icd.d/nvidia_icd.json"
export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export CUDA_VISIBLE_DEVICES="0,1"
export RLINF_NODE_RANK="0"
export TOKENIZERS_PARALLELISM="false"
export PYTHONDONTWRITEBYTECODE="1"
export HF_DATASETS_DISABLE_PROGRESS_BARS="1"

cd "$ROOT"

if [[ "${1:-}" != "--skip-collection" ]]; then
    CUDA_VISIBLE_DEVICES=0 python -u toolkits/lerobot/collect_realagainst_peg_slot_vectorized.py \
        --num-episodes 5000 \
        --max-attempts 6000 \
        --num-envs 16 \
        --seed 0 \
        --overwrite
fi

python - "$DATA/meta/info.json" <<'PY'
import json
import sys

info = json.load(open(sys.argv[1]))
assert info["total_episodes"] == 5000, info
assert info["total_frames"] > 0, info
print(
    f"validated dataset: episodes={info['total_episodes']} "
    f"frames={info['total_frames']}"
)
PY

python toolkits/lerobot/calculate_norm_stats.py \
    --config-name pi05_realagainst_peg_slot \
    --repo-id "$DATA"

test -f "$DATA/norm_stats.json"

if ! ray status >/dev/null 2>&1; then
    ray start --head --disable-usage-stats
fi

mkdir -p "$TRAIN_LOG"
python examples/sft/train_vla_sft.py \
    --config-path "$ROOT/examples/sft/config/" \
    --config-name "$SFT_CONFIG" \
    2>&1 | tee "$TRAIN_LOG/run_sft.log"

test -d "$FINAL_CKPT"

export EMBODIED_PATH="$ROOT/examples/embodiment"
bash evaluations/run_eval.sh maniskill "$EVAL_CONFIG"
