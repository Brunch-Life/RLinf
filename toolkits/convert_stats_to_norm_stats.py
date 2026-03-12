"""Convert a LeRobot dataset stats.json to OpenPI norm_stats.json.

Reads the per-column statistics produced by the data collection pipeline
(``dataset/<repo_id>/meta/stats.json``) and writes the ``norm_stats.json``
file that OpenPI expects under the checkpoint assets directory.

Transformations applied:
  - ``state``: select dimensions specified by ``--select-state-dims``
    (default matches ``pi0_realworld_pnp``: tcp_pose(6) + gripper(1) from 19D),
    then zero-pad to ``--action-dim`` (default 32).
  - ``actions``: zero-pad to ``--action-dim``.

Usage:
    python toolkits/convert_stats_to_norm_stats.py \
        --stats-json dataset/<user>/<dataset>/meta/stats.json \
        --output-dir checkpoints/torch/pi0_base/<user>/<dataset>

    # Or with custom dims:
    python toolkits/convert_stats_to_norm_stats.py \
        --stats-json dataset/<user>/<dataset>/meta/stats.json \
        --output-dir checkpoints/torch/pi0_base/<user>/<dataset> \
        --select-state-dims 4 5 6 7 8 9 0 \
        --action-dim 32
"""

import argparse
import json
import pathlib


def _pad(arr: list[float], target_len: int, pad_value: float = 0.0) -> list[float]:
    return arr + [pad_value] * (target_len - len(arr))


def main():
    parser = argparse.ArgumentParser(description="Convert LeRobot stats.json → OpenPI norm_stats.json")
    parser.add_argument("--stats-json", type=str, required=True, help="Path to the dataset stats.json")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to write norm_stats.json into")
    parser.add_argument(
        "--select-state-dims",
        type=int,
        nargs="+",
        default=[4, 5, 6, 7, 8, 9, 0],
        help="Indices to select from the raw state vector (default: pi0_realworld_pnp mapping)",
    )
    parser.add_argument("--action-dim", type=int, default=32, help="Pi0 max action/state dim for zero-padding")
    args = parser.parse_args()

    with open(args.stats_json) as f:
        stats = json.load(f)

    dim = args.action_dim
    indices = args.select_state_dims

    raw_state = stats["state"]
    selected_state = {
        "mean": [raw_state["mean"][i] for i in indices],
        "std": [raw_state["std"][i] for i in indices],
    }

    raw_actions = stats["actions"]

    norm_stats = {
        "norm_stats": {
            "state": {
                "mean": _pad(selected_state["mean"], dim),
                "std": _pad(selected_state["std"], dim),
                "q01": None,
                "q99": None,
            },
            "actions": {
                "mean": _pad(raw_actions["mean"], dim),
                "std": _pad(raw_actions["std"], dim),
                "q01": None,
                "q99": None,
            },
        }
    }

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "norm_stats.json"
    with open(out_path, "w") as f:
        json.dump(norm_stats, f, indent=2)

    print(f"Written norm_stats.json to: {out_path}")
    print(f"  state:   {len(indices)}D selected from {len(raw_state['mean'])}D → padded to {dim}D")
    print(f"  actions: {len(raw_actions['mean'])}D → padded to {dim}D")


if __name__ == "__main__":
    main()
