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

"""Dual-GELLO intervention wrapper for dual-arm Franka environments.

When ``use_joint_control=True`` (the default), GELLO joint positions are
sent directly to the robot via ZMQ → Polymetis joint impedance at kHz,
completely bypassing the cartesian impedance controller.  This requires
running ``launch_nodes.py --robot fr3`` on each controller node inside the
``polymetis-local`` conda environment.

When ``use_joint_control=False``, the original cartesian-delta mode is
used (FK → delta → env.step → ROS impedance).
"""

from __future__ import annotations

import threading
import time

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.common.gello.gello_expert import GelloExpert


class DualGelloIntervention(gym.ActionWrapper):
    """Override the policy action with two GELLO teleoperation devices.

    Args:
        env: The wrapped dual-arm environment.
        left_port: Serial port for the left GELLO device.
        right_port: Serial port for the right GELLO device.
        gripper_enabled: Whether the gripper channel is present.
        use_joint_control: If ``True``, bypass the env step and send
            GELLO joints directly to the robot via ZMQ/Polymetis.
        left_zmq_host: ZMQ host for the left arm Polymetis bridge.
        right_zmq_host: ZMQ host for the right arm Polymetis bridge.
        left_zmq_port: ZMQ port for the left arm.
        right_zmq_port: ZMQ port for the right arm.
        joint_control_hz: Control loop frequency for joint mode.
    """

    def __init__(
        self,
        env: gym.Env,
        left_port: str,
        right_port: str,
        gripper_enabled: bool = True,
        use_joint_control: bool = True,
        left_zmq_host: str = "127.0.0.1",
        right_zmq_host: str = "127.0.0.1",
        left_zmq_port: int = 5555,
        right_zmq_port: int = 5556,
        joint_control_hz: float = 100.0,
    ):
        super().__init__(env)
        self.gripper_enabled = gripper_enabled
        self.use_joint_control = use_joint_control

        from gello.agents.gello_agent import GelloAgent

        self.left_gello = GelloAgent(port=left_port)
        self.right_gello = GelloAgent(port=right_port)

        self.last_intervene = 0
        self._intervening = False

        if use_joint_control:
            from gello.zmq_core.robot_node import ZMQClientRobot

            self._left_robot = ZMQClientRobot(
                port=left_zmq_port, host=left_zmq_host
            )
            self._right_robot = ZMQClientRobot(
                port=right_zmq_port, host=right_zmq_host
            )
            self._joint_control_hz = joint_control_hz
            self._stop_event = threading.Event()
            self._joint_thread = threading.Thread(
                target=self._joint_control_loop, daemon=True
            )
            self._joint_thread.start()
        else:
            self.left_expert = GelloExpert(port=left_port)
            self.right_expert = GelloExpert(port=right_port)

    def _joint_control_loop(self):
        """Background thread: read GELLO joints -> send to robot at fixed rate."""
        dt = 1.0 / self._joint_control_hz
        max_delta = 0.05

        while not self._stop_event.is_set():
            t0 = time.perf_counter()

            left_joints = self.left_gello.act({})
            right_joints = self.right_gello.act({})

            left_ok = right_ok = False

            try:
                left_current = self._left_robot.get_observations()["joint_positions"]
                delta_l = np.clip(
                    left_joints - left_current,
                    -max_delta, max_delta,
                )
                self._left_robot.command_joint_state(left_current + delta_l)
                left_ok = True
            except Exception:
                pass

            try:
                right_current = self._right_robot.get_observations()["joint_positions"]
                delta_r = np.clip(
                    right_joints - right_current,
                    -max_delta, max_delta,
                )
                self._right_robot.command_joint_state(right_current + delta_r)
                right_ok = True
            except Exception:
                pass

            any_active = (
                left_ok and np.abs(left_joints[:-1] - left_current[:-1]).max() > 0.01
            ) or (
                right_ok and np.abs(right_joints[:-1] - right_current[:-1]).max() > 0.01
            )
            if any_active:
                self.last_intervene = time.time()
            self._intervening = time.time() - self.last_intervene < 0.5

            elapsed = time.perf_counter() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        if self.use_joint_control:
            return action, self._intervening

        if not self.left_expert.ready or not self.right_expert.ready:
            return action, False

        tcp_pose = self.get_wrapper_attr("get_tcp_pose")()
        action_scale = self.get_wrapper_attr("get_action_scale")()

        left_tcp = tcp_pose[:7]
        right_tcp = tcp_pose[7:]

        left_a = self._compute_delta(self.left_expert, left_tcp, action_scale)
        right_a = self._compute_delta(self.right_expert, right_tcp, action_scale)

        any_active = np.linalg.norm(left_a[:6]) > 0.001 or np.linalg.norm(right_a[:6]) > 0.001
        if self.gripper_enabled:
            any_active = any_active or np.abs(left_a[6]) > 0.5 or np.abs(right_a[6]) > 0.5

        if any_active:
            self.last_intervene = time.time()

        expert_a = np.concatenate([left_a, right_a])

        if time.time() - self.last_intervene < 0.5:
            return expert_a, True
        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)
        if self.use_joint_control and replaced:
            # In joint control mode, the background thread already moves the
            # robot.  The env step should still run for observations/cameras
            # but we send a zero-delta action so move_arm is a no-op
            # (current pose → current pose).
            zero_action = np.zeros_like(new_action)
            obs, rew, done, truncated, info = self.env.step(zero_action)
            info["intervene_action"] = new_action
            info["intervene_flag"] = np.ones(1)
            return obs, rew, done, truncated, info

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
            info["intervene_flag"] = np.ones(1)
        return obs, rew, done, truncated, info

    def _compute_delta(
        self, expert: GelloExpert, tcp_pose: np.ndarray, action_scale: np.ndarray,
    ) -> np.ndarray:
        target_pos, target_quat, target_gripper = expert.get_action()
        r_target = R.from_quat(target_quat.copy())
        tcp_pos = tcp_pose[:3]
        r_tcp = R.from_quat(tcp_pose[3:].copy())

        delta_pos = (target_pos - tcp_pos) / action_scale[0]
        delta_euler = (r_target * r_tcp.inv()).as_euler("xyz") / action_scale[1]

        expert_a = np.clip(np.concatenate([delta_pos, delta_euler]), -1.0, 1.0)

        if self.gripper_enabled:
            grip = target_gripper / action_scale[2]
            grip = np.clip(-(2 * grip - 1.0), -1.0, 1.0)
            expert_a = np.concatenate([expert_a, grip])

        return expert_a

    def close(self):
        if self.use_joint_control:
            self._stop_event.set()
            self._joint_thread.join(timeout=2)
            self._left_robot.close()
            self._right_robot.close()
        super().close()
