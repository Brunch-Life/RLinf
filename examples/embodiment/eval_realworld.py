# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real-world evaluation with a remote policy server.

This script runs evaluation episodes on a real Franka robot, querying a remote
policy server (e.g. ``serve_policy.py``) for actions via WebSocket.  The policy
server can run on a different machine / environment from the robot control loop,
enabling full decoupling of the inference environment from the robot environment.

The protocol is fully compatible with OpenPI's ``WebsocketClientPolicy``, so you
can also point this evaluator at a vanilla OpenPI server.

Usage
-----
1. Start the policy server (on a GPU machine)::

    python serve_policy.py --config pi0_custom --checkpoint-dir /path/to/ckpt

2. Run evaluation (on the robot-control machine)::

    bash eval_realworld.sh realworld_eval_zed_robotiq
"""

import functools
import logging
import os
import time
from collections import OrderedDict

import hydra
import msgpack
import numpy as np
import torch

from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.envs.wrappers.collect_episode import CollectEpisode
from rlinf.scheduler import Cluster, ComponentPlacement, Worker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# msgpack-numpy helpers (same wire format as OpenPI / serve_policy.py)
# ---------------------------------------------------------------------------

def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported numpy dtype: {obj.dtype}")
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"],
            dtype=np.dtype(obj[b"dtype"]),
            shape=obj[b"shape"],
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


_Packer = functools.partial(msgpack.Packer, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


# ---------------------------------------------------------------------------
# WebSocket client (OpenPI-compatible)
# ---------------------------------------------------------------------------

class PolicyClient:
    """Connects to a WebSocket policy server and queries actions.

    Wire-compatible with OpenPI's ``WebsocketPolicyServer`` and with
    ``openpi_client.WebsocketClientPolicy``.
    """

    def __init__(self, host: str = "localhost", port: int = 8000):
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = _Packer()
        self._ws, self._server_metadata = self._wait_for_server()

    def _wait_for_server(self, timeout: float = 300):
        import websockets.sync.client

        logger.info("Connecting to policy server at %s ...", self._uri)
        start = time.time()
        while True:
            try:
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None,
                )
                metadata = _unpackb(conn.recv())
                logger.info("Connected to policy server. Metadata: %s", metadata)
                return conn, metadata
            except (ConnectionRefusedError, OSError) as exc:
                elapsed = time.time() - start
                if timeout and elapsed > timeout:
                    raise TimeoutError(
                        f"Could not connect to policy server at {self._uri} "
                        f"after {elapsed:.0f}s. Is the server running?"
                    ) from exc
                logger.info(
                    "Server not ready (%s), retrying in 5s ... (%.0fs / %.0fs)",
                    exc.__class__.__name__, elapsed, timeout,
                )
                time.sleep(5)

    def infer(self, obs: dict) -> dict:
        """Send an observation dict, receive an action dict."""
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error from policy server:\n{response}")
        return _unpackb(response)

    def close(self):
        if self._ws is not None:
            self._ws.close()


# ---------------------------------------------------------------------------
# Observation formatting
# ---------------------------------------------------------------------------

def build_state_index_map(obs_space):
    """Build a mapping from state key to slice in the concatenated state vector.

    ``RealWorldEnv._wrap_obs`` concatenates all state sub-spaces in
    **alphabetical** key order.  This helper recovers each key's slice so
    that downstream code can extract ``tcp_pose`` and ``gripper_position``
    reliably regardless of how many keys exist or their dimensions.
    """
    state_space = obs_space["state"]
    offset = 0
    index_map = {}
    for key in sorted(state_space.spaces.keys()):
        dim = state_space[key].shape[-1]
        index_map[key] = slice(offset, offset + dim)
        offset += dim
    return index_map


def format_obs_for_policy(
    obs: dict, task_description: str, state_index_map: dict
) -> dict:
    """Convert RealWorldEnv observation to the OpenPI input format.

    The policy server expects ``observation/state`` as
    ``[x, y, z, rx, ry, rz, gripper]`` (7-D).

    Extra camera images from ``extra_view_images`` are sent as
    ``observation/extra_image_0``, ``observation/extra_image_1``, etc.
    so that ``FrankaEEInputs`` on the server can assign them to the
    correct Pi0 image slots via ``pi0_slot_keys``.
    """
    states = obs["states"]
    if isinstance(states, torch.Tensor):
        states = states.cpu().numpy()
    if states.ndim == 2:
        states = states[0]

    main_image = obs["main_images"]
    if isinstance(main_image, torch.Tensor):
        main_image = main_image.cpu().numpy()
    if main_image.ndim == 4:
        main_image = main_image[0]

    tcp_pose = states[state_index_map["tcp_pose"]]       # [x,y,z,rx,ry,rz]
    gripper = states[state_index_map["gripper_position"]]  # [g]
    state_7d = np.concatenate([tcp_pose, gripper]).astype(np.float32)

    result = {
        "observation/image": np.ascontiguousarray(main_image),
        "observation/state": state_7d,
        "prompt": task_description,
    }

    extra_views = obs.get("extra_view_images")
    if extra_views is not None:
        if isinstance(extra_views, torch.Tensor):
            extra_views = extra_views.cpu().numpy()
        if extra_views.ndim == 5:  # [num_envs, num_cameras, H, W, C]
            extra_views = extra_views[0]
        for i in range(extra_views.shape[0]):
            result[f"observation/extra_image_{i}"] = np.ascontiguousarray(
                extra_views[i]
            )

    return result


# ---------------------------------------------------------------------------
# EvalCollectEpisode — records eval episodes into LeRobot datasets
# ---------------------------------------------------------------------------

class EvalCollectEpisode(CollectEpisode):
    """CollectEpisode variant for eval that infers success from terminal reward.

    Works with ``keyboard_reward_wrapper`` (single_stage) where the operator
    presses ``c`` (success, reward=+1) or ``a`` (failure, reward=-1) to end
    each episode.  Falls back to reward-based success detection when no
    explicit success flag is found in info.
    """

    def _get_episode_success(self, buf: dict, env_idx: int) -> bool:
        if self._has_explicit_success_flag(buf):
            return super()._get_episode_success(buf, env_idx)
        if buf["rewards"]:
            last_reward = buf["rewards"][-1]
            r = (
                last_reward.item()
                if isinstance(last_reward, torch.Tensor)
                else float(last_reward)
            )
            return r > 0
        return False

    @staticmethod
    def _has_explicit_success_flag(buf: dict) -> bool:
        for info in reversed(buf["infos"]):
            if not isinstance(info, dict):
                continue
            for src in (info.get("final_info"), info.get("episode"), info):
                if not isinstance(src, dict):
                    continue
                for key in ("success_once", "success_at_end", "success"):
                    if src.get(key) is not None:
                        return True
        return False

    def flush_if_pending(self, is_success: bool | None = None) -> None:
        """Flush any in-progress episode that was not terminated/truncated.

        Called by the eval loop when an episode ends due to step limit rather
        than an environment signal.
        """
        for env_idx in range(self.num_envs):
            buf = self._buffers[env_idx]
            if not buf["actions"]:
                continue
            success = (
                is_success
                if is_success is not None
                else self._get_episode_success(buf, env_idx)
            )
            if not self.only_success or success:
                self._flush_episode(env_idx, success)
            self._reset_env_buffer(env_idx)


# ---------------------------------------------------------------------------
# RealWorldEvaluator (Ray Worker)
# ---------------------------------------------------------------------------

class RealWorldEvaluator(Worker):
    """Runs eval episodes on a real robot, querying a remote policy server."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_eval_episodes = cfg.runner.get("num_eval_episodes", 20)
        self.max_episode_steps = cfg.runner.get("max_episode_steps", 1000)

        server_cfg = cfg.get("policy_server", {})
        self.server_host = server_cfg.get("host", "localhost")
        self.server_port = server_cfg.get("port", 8000)

        self.dry_run = cfg.runner.get("dry_run", False)
        self.dry_run_steps = cfg.runner.get("dry_run_steps", 5)

        # "absolute" = policy outputs target EE pose, convert to delta before env.step()
        # "delta"    = policy outputs delta directly, pass through to env.step()
        self.action_type = cfg.runner.get("action_type", "absolute")

        self._collect_enabled = False
        self._debug_dir = None

    def _build_env(self):
        base_env = RealWorldEnv(
            self.cfg.env.eval,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=self.worker_info,
        )
        self._state_index_map = build_state_index_map(base_env.observation_space)
        self.log_info(f"State index map: {self._state_index_map}")

        collect_cfg = self.cfg.get("collect", {})
        self._collect_enabled = collect_cfg.get("enabled", False)
        if self._collect_enabled:
            log_path = self.cfg.runner.logger.get("log_path", "logs")
            save_dir = os.path.join(log_path, "lerobot_dataset")
            base_env = EvalCollectEpisode(
                env=base_env,
                save_dir=save_dir,
                num_envs=1,
                export_format=collect_cfg.get("export_format", "lerobot"),
                robot_type=collect_cfg.get("robot_type", "panda"),
                fps=collect_cfg.get("fps", 10),
                only_success=collect_cfg.get("only_success", False),
                show_goal_site=False,
            )
            self.log_info(f"Data collection enabled → {save_dir}")

        return base_env

    def _init_debug_dir(self):
        """Create debug output directory. Uses cwd (the repo root)."""
        debug_dir = os.path.join(os.getcwd(), "tmp", "eval_debug")
        os.makedirs(debug_dir, exist_ok=True)
        self._debug_dir = debug_dir
        self.log_info(f"Debug output dir: {debug_dir}")

    def _convert_action(self, raw_action: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        """Convert policy output to delta action for env.step().

        When action_type == "absolute", the policy returns a target EE pose
        [x,y,z,rx,ry,rz,gripper].  We subtract the current state to get
        delta [dx,dy,dz,drx,dry,drz,gripper_target].
        Gripper (index 6) is kept as-is since it's a binary command, not a pose.
        """
        if self.action_type == "delta":
            return raw_action

        delta = raw_action.copy()
        delta[:6] = raw_action[:6] - current_state[:6]
        return delta

    def _get_current_state(self, obs):
        """Extract current EE state [x,y,z,rx,ry,rz,gripper] from observation."""
        states = obs["states"]
        if isinstance(states, torch.Tensor):
            states = states.cpu().numpy()
        if states.ndim == 2:
            states = states[0]
        tcp_pose = states[self._state_index_map["tcp_pose"]]
        gripper = states[self._state_index_map["gripper_position"]]
        return np.concatenate([tcp_pose, gripper])

    def _save_debug(
        self, ep_idx, step, policy_input, raw_action, delta_action, current_state
    ):
        """Save debug info: image + action log. Also log to stdout."""
        prompt = policy_input.get("prompt", "")
        state_str = np.array2string(current_state[:7], precision=4, separator=", ")
        raw_str = np.array2string(raw_action, precision=4, separator=", ")
        delta_str = np.array2string(delta_action, precision=4, separator=", ")

        self.log_info(
            f"[debug] ep={ep_idx} step={step} | "
            f"state={state_str} | raw_action={raw_str} | delta_to_env={delta_str}"
        )

        if self._debug_dir is None:
            return
        try:
            prefix = f"ep{ep_idx:03d}_step{step:04d}"
            image = policy_input.get("observation/image")
            if image is not None:
                from PIL import Image

                img = np.asarray(image)
                if img.dtype != np.uint8:
                    img = (img * 255).clip(0, 255).astype(np.uint8)
                img_path = os.path.join(self._debug_dir, f"{prefix}_image.jpg")
                Image.fromarray(img).save(img_path)
                self.log_info(f"[debug] saved image {img.shape} → {img_path}")
            else:
                self.log_warning("[debug] No image in policy_input to save.")
            with open(os.path.join(self._debug_dir, f"{prefix}_info.txt"), "w") as f:
                f.write(f"prompt: {prompt}\n")
                f.write(f"current_state: {state_str}\n")
                f.write(f"raw_action ({self.action_type}): {raw_str}\n")
                f.write(f"delta_to_env: {delta_str}\n")
        except Exception as exc:
            self.log_warning(f"Failed to save debug files: {exc}")

    def run(self):
        self.log_info("Building environment ...")
        env = self._build_env()
        self.log_info("Environment ready.")

        self._init_debug_dir()

        self.log_info(
            f"Connecting to policy server at "
            f"{self.server_host}:{self.server_port} ..."
        )
        client = PolicyClient(host=self.server_host, port=self.server_port)
        self.log_info("Policy server connected.")

        task_desc = env.task_descriptions[0] if env.task_descriptions else ""
        self.log_info(f"Task: {task_desc}")
        self.log_info(
            f"Action type: {self.action_type} "
            f"({'policy output = target pose, will subtract state' if self.action_type == 'absolute' else 'policy output = delta, pass through'})"
        )

        if self.dry_run:
            self.log_info(
                f"*** DRY RUN MODE *** — will query policy for "
                f"{self.dry_run_steps} steps but NOT send actions to robot."
            )
            results = [self._run_dry_episode(env, client, task_desc, 0)]
            env.close()
            client.close()
            self._report_summary(results)
            return results

        self.log_info(
            f"Evaluating {self.num_eval_episodes} episodes "
            f"(max {self.max_episode_steps} steps each)"
        )

        results = []
        for ep_idx in range(self.num_eval_episodes):
            self.log_info(f"Starting episode {ep_idx + 1}/{self.num_eval_episodes} ...")
            ep_result = self._run_episode(env, client, task_desc, ep_idx)
            results.append(ep_result)
            self.log_info(
                f"Episode {ep_idx + 1}/{self.num_eval_episodes}: "
                f"success={ep_result['success']}, "
                f"return={ep_result['return']:.2f}, "
                f"steps={ep_result['steps']}"
            )

        env.close()
        client.close()

        self._report_summary(results)
        return results

    def _run_dry_episode(self, env, client, task_desc, ep_idx):
        """Query the policy without executing actions — for debugging."""
        self.log_info("Resetting environment (dry run) ...")
        obs, _ = env.reset()
        self.log_info("Environment reset done (dry run).")

        current_state = self._get_current_state(obs)
        self.log_info(
            f"[dry_run] Initial absolute state: "
            f"{np.array2string(current_state, precision=4, separator=', ')}"
        )

        for step in range(self.dry_run_steps):
            policy_input = format_obs_for_policy(obs, task_desc, self._state_index_map)
            current_state = self._get_current_state(obs)
            result = client.infer(policy_input)

            actions = result.get("actions", None)
            if actions is None:
                self.log_error("Policy server returned no 'actions' key.")
                break

            actions = np.asarray(actions, dtype=np.float32)
            if actions.ndim == 2:
                raw_action = actions[0]
                self.log_info(
                    f"[dry_run] step {step}: action chunk shape={actions.shape}, "
                    f"showing first action"
                )
            else:
                raw_action = actions if actions.ndim == 1 else actions[0]

            delta_action = self._convert_action(raw_action, current_state)
            self._save_debug(
                ep_idx, step, policy_input, raw_action, delta_action, current_state,
            )
            self.log_info(f"[dry_run] step {step}: action NOT sent to robot.")

        return {"episode": ep_idx, "success": False, "return": 0.0, "steps": 0}

    def _run_episode(self, env, client, task_desc, ep_idx):
        self.log_info(f"Resetting environment for episode {ep_idx} ...")
        obs, _ = env.reset()
        self.log_info("Environment reset done.")

        episode_return = 0.0
        success = False
        steps = 0

        for step in range(self.max_episode_steps):
            policy_input = format_obs_for_policy(obs, task_desc, self._state_index_map)
            current_state = self._get_current_state(obs)
            result = client.infer(policy_input)

            actions = result.get("actions", None)
            if actions is None:
                self.log_error("Policy server returned no 'actions' key.")
                break

            actions = np.asarray(actions, dtype=np.float32)

            if actions.ndim == 2:
                for chunk_idx in range(actions.shape[0]):
                    raw_action = actions[chunk_idx]
                    delta = self._convert_action(raw_action, current_state)

                    if chunk_idx == 0:
                        self._save_debug(
                            ep_idx, steps, policy_input,
                            raw_action, delta, current_state,
                        )

                    action_for_env = delta[np.newaxis, :]  # [1, 7]
                    obs, reward, terminated, truncated, info = env.step(action_for_env)
                    current_state = self._get_current_state(obs)
                    steps += 1

                    if isinstance(reward, torch.Tensor):
                        reward = reward.item()
                    episode_return += reward

                    term = terminated.any() if hasattr(terminated, "any") else bool(terminated)
                    trunc = truncated.any() if hasattr(truncated, "any") else bool(truncated)
                    if term:
                        success = reward > 0
                    if term or trunc or steps >= self.max_episode_steps:
                        break
            else:
                raw_action = actions if actions.ndim == 1 else actions[0]
                delta = self._convert_action(raw_action, current_state)
                self._save_debug(
                    ep_idx, steps, policy_input,
                    raw_action, delta, current_state,
                )

                action_for_env = delta[np.newaxis, :]  # [1, 7]
                obs, reward, terminated, truncated, info = env.step(action_for_env)
                current_state = self._get_current_state(obs)
                steps += 1

                if isinstance(reward, torch.Tensor):
                    reward = reward.item()
                episode_return += reward

                term = terminated.any() if hasattr(terminated, "any") else bool(terminated)
                trunc = truncated.any() if hasattr(truncated, "any") else bool(truncated)
                if term:
                    success = reward > 0

            if (terminated.any() if hasattr(terminated, "any") else bool(terminated)) or \
               (truncated.any() if hasattr(truncated, "any") else bool(truncated)) or \
               steps >= self.max_episode_steps:
                break

        if self._collect_enabled and hasattr(env, "flush_if_pending"):
            env.flush_if_pending(is_success=success)

        return {
            "episode": ep_idx,
            "success": success,
            "return": episode_return,
            "steps": steps,
        }

    def _report_summary(self, results):
        num_episodes = len(results)
        num_success = sum(r["success"] for r in results)
        avg_return = np.mean([r["return"] for r in results])
        avg_steps = np.mean([r["steps"] for r in results])

        self.log_info("=" * 60)
        self.log_info("Evaluation Summary")
        self.log_info(f"  Episodes:     {num_episodes}")
        self.log_info(f"  Success rate: {num_success}/{num_episodes} = {num_success / max(num_episodes, 1):.2%}")
        self.log_info(f"  Avg return:   {avg_return:.3f}")
        self.log_info(f"  Avg steps:    {avg_steps:.1f}")
        if self._collect_enabled:
            log_path = self.cfg.runner.logger.get("log_path", "logs")
            self.log_info(f"  Dataset:      {os.path.join(log_path, 'lerobot_dataset')}")
        self.log_info("=" * 60)


# ---------------------------------------------------------------------------
# Hydra entry-point
# ---------------------------------------------------------------------------

@hydra.main(
    version_base="1.1", config_path="config", config_name="realworld_eval_zed_robotiq"
)
def main(cfg):
    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = ComponentPlacement(cfg, cluster)
    env_placement = component_placement.get_strategy("env")

    print("[main] Launching RealWorldEvaluator worker ...", flush=True)
    evaluator = RealWorldEvaluator.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )
    print("[main] Worker launched, calling run() ...", flush=True)
    try:
        results = evaluator.run().wait()
        print(f"[main] Evaluation complete. {len(results)} episodes finished.", flush=True)
    except Exception as e:
        print(f"[main] Evaluation failed with error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
