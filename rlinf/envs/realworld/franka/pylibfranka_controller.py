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

"""Franka robot controller using pylibfranka with a 1 kHz real-time control loop.

This module provides a :class:`Worker`-based controller that communicates
directly with the Franka robot via the official pylibfranka Python bindings.
Unlike :class:`FrankyController` (which uses blocking waypoint motions),
this controller runs a 1 kHz ``readOnce()``/``writeOnce()`` control loop
in a background thread, enabling:

* Continuous 1 kHz state updates (no ``_moving`` flag needed).
* Non-blocking ``move_joints()`` — the caller sets a target and the RT
  loop smoothly tracks it via joint impedance control.
* Natural extension to torque control or variable-impedance in the future.

Architecture::

    PylibfrankaController (Worker, runs as Ray actor)
    │
    ├── _rt_thread  (1 kHz joint position control)
    │   ├── readOnce()  → _update_state()
    │   └── writeOnce() ← _target (set by move_joints / move_arm)
    │
    ├── _idle_thread (100 Hz, active when RT loop is not running)
    │   └── robot.read_once() → _update_state()
    │
    └── Gripper (independent network channel, safe during RT control)
"""

import copy
import threading
import time
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.common.gripper import create_gripper
from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker
from rlinf.utils.logging import get_logger

from .franka_robot_state import FrankaRobotState

# Franka Emika Panda joint position limits (radians)
JOINT_LIMITS_LOWER = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
)
JOINT_LIMITS_UPPER = np.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
)

# Default collision behaviour — conservative thresholds (Nm / N)
_DEFAULT_TORQUE_THRESHOLD = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
_DEFAULT_FORCE_THRESHOLD = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]

# Default joint impedance stiffness (Nm/rad)
_DEFAULT_JOINT_IMPEDANCE = [600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0]


class PylibfrankaController(Worker):
    """Franka robot arm controller using pylibfranka's real-time interface.

    Runs a 1 kHz ``readOnce()``/``writeOnce()`` joint position control loop
    in a background thread.  ``move_joints()`` is non-blocking — it simply
    updates the target that the RT loop tracks.

    Args:
        robot_ip: IP address of the Franka robot.
        gripper_type: ``"pylibfranka"`` (native Franka gripper) or
            ``"robotiq"`` (Modbus RTU).
        gripper_connection: Serial port for Robotiq (e.g. ``"/dev/ttyUSB0"``).
            Ignored when *gripper_type* is ``"pylibfranka"``.
    """

    @staticmethod
    def launch_controller(
        robot_ip: str,
        env_idx: int = 0,
        node_rank: int = 0,
        worker_rank: int = 0,
        gripper_type: str = "pylibfranka",
        gripper_connection: Optional[str] = None,
    ):
        """Launch a PylibfrankaController on the specified node.

        Uses the standard RLinf Worker factory pattern
        (``create_group().launch()``).
        """
        cluster = Cluster()
        placement = NodePlacementStrategy(node_ranks=[node_rank])
        return PylibfrankaController.create_group(
            robot_ip, gripper_type, gripper_connection
        ).launch(
            cluster=cluster,
            placement_strategy=placement,
            name=f"PylibfrankaController-{worker_rank}-{env_idx}",
        )

    def __init__(
        self,
        robot_ip: str,
        gripper_type: str = "pylibfranka",
        gripper_connection: Optional[str] = None,
    ):
        super().__init__()
        self._logger = get_logger()
        self._robot_ip = robot_ip
        self._gripper_type = gripper_type

        # ── pylibfranka imports (lazy, only on controller machine) ──
        from pylibfranka import (
            ControllerMode,
            JointPositions,
            RealtimeConfig,
            Robot,
        )

        self._pylibfranka_Robot = Robot
        self._pylibfranka_JointPositions = JointPositions
        self._pylibfranka_ControllerMode = ControllerMode
        self._pylibfranka_RealtimeConfig = RealtimeConfig

        # ── Robot connection ────────────────────────────────────────
        self._robot = Robot(robot_ip, RealtimeConfig.kIgnore)
        self._model = self._robot.load_model()

        # Configure collision thresholds
        self._robot.set_collision_behavior(
            _DEFAULT_TORQUE_THRESHOLD,
            _DEFAULT_TORQUE_THRESHOLD,
            _DEFAULT_FORCE_THRESHOLD,
            _DEFAULT_FORCE_THRESHOLD,
        )
        self._robot.set_joint_impedance(_DEFAULT_JOINT_IMPEDANCE)

        # ── State management ────────────────────────────────────────
        self._state = FrankaRobotState()
        self._state_lock = threading.Lock()
        self._has_valid_state = False  # True after first successful state read

        # ── RT control target ───────────────────────────────────────
        self._target: Optional[list[float]] = None
        self._target_lock = threading.Lock()

        # ── Lifecycle flags ─────────────────────────────────────────
        self._rt_active = False  # Is the RT control loop running?
        self._running = True  # Master shutdown flag

        # ── Gripper (independent network channel) ───────────────────
        if gripper_type == "pylibfranka":
            self._gripper = create_gripper(
                gripper_type="pylibfranka",
                robot_ip=robot_ip,
            )
        elif gripper_type == "robotiq":
            self._gripper = create_gripper(
                gripper_type="robotiq",
                port=gripper_connection,
            )
        else:
            raise ValueError(
                f"Unsupported gripper_type={gripper_type!r} for "
                f"PylibfrankaController. Supported: 'pylibfranka', 'robotiq'."
            )

        # Cached gripper state (read outside state_lock to avoid
        # blocking the RT loop with gripper I/O).
        self._cached_gripper_position: float = 0.0
        self._cached_gripper_open: bool = True

        # ── Start background threads ────────────────────────────────
        self._idle_thread = threading.Thread(
            target=self._idle_state_loop, daemon=True
        )
        self._idle_thread.start()

        # Start the 1 kHz RT control loop
        self._start_rt_loop()

        self._logger.info(
            f"PylibfrankaController connected to robot at {robot_ip}"
        )

    # ═══════════════════════════════════════════════════════════════
    #  RT Control Loop
    # ═══════════════════════════════════════════════════════════════

    def _start_rt_loop(self):
        """Start the 1 kHz real-time joint position control loop."""
        if self._rt_active:
            return
        self._rt_active = True
        self._rt_thread = threading.Thread(
            target=self._rt_control_loop, daemon=True
        )
        self._rt_thread.start()

    def _stop_rt_loop(self):
        """Stop the RT control loop and wait for it to exit."""
        if not self._rt_active:
            return
        self._rt_active = False
        if hasattr(self, "_rt_thread") and self._rt_thread.is_alive():
            self._rt_thread.join(timeout=3.0)

    def _rt_control_loop(self):
        """Background thread running the 1 kHz readOnce/writeOnce cycle.

        The loop enters joint impedance control mode and continuously
        streams the current ``_target`` to the robot.  When ``_rt_active``
        is set to ``False``, the loop sends a final ``motion_finished``
        command and exits cleanly.
        """
        JointPositions = self._pylibfranka_JointPositions
        ControllerMode = self._pylibfranka_ControllerMode

        try:
            active = self._robot.start_joint_position_control(
                ControllerMode.JointImpedance
            )

            # Read initial state and use current desired position as target
            initial_state, _ = active.readOnce()
            self._update_state(initial_state)

            with self._target_lock:
                if self._target is None:
                    self._target = list(initial_state.q_d)

            while self._rt_active:
                state, _duration = active.readOnce()
                self._update_state(state)

                with self._target_lock:
                    target = list(self._target)

                cmd = JointPositions(target)
                active.writeOnce(cmd)

            # ── Graceful exit: hold current desired position and finish ──
            state, _ = active.readOnce()
            self._update_state(state)
            cmd = JointPositions(list(state.q_d))
            cmd.motion_finished = True
            active.writeOnce(cmd)

        except Exception as e:
            self._logger.warning(f"RT control loop error: {e}")
            self._rt_active = False

    # ═══════════════════════════════════════════════════════════════
    #  Idle State Reading (when RT loop is not active)
    # ═══════════════════════════════════════════════════════════════

    def _idle_state_loop(self):
        """Read state at ~100 Hz when the RT loop is not running.

        ``robot.read_once()`` cannot be called while an active control
        session exists, so this loop only runs when ``_rt_active`` is
        ``False``.
        """
        while self._running:
            if not self._rt_active:
                try:
                    robot_state = self._robot.read_once()
                    self._update_state(robot_state)
                except Exception:
                    pass  # Robot may not be ready yet or RT loop just started
            time.sleep(0.01)  # 100 Hz

    # ═══════════════════════════════════════════════════════════════
    #  State Update
    # ═══════════════════════════════════════════════════════════════

    def _update_state(self, robot_state):
        """Parse a pylibfranka RobotState into :class:`FrankaRobotState`.

        Called from either the RT loop (1 kHz) or the idle loop (100 Hz).
        """
        # TCP pose from O_T_EE (16-element column-major 4×4)
        tmatrix = np.array(robot_state.O_T_EE).reshape(4, 4, order="F")
        r = R.from_matrix(tmatrix[:3, :3].copy())
        tcp_pose = np.concatenate([tmatrix[:3, 3], r.as_quat()])

        # Joint state
        joint_pos = np.array(robot_state.q)
        joint_vel = np.array(robot_state.dq)

        # External forces / torques
        K_F_ext = np.array(robot_state.K_F_ext_hat_K)
        tcp_force = K_F_ext[:3]
        tcp_torque = K_F_ext[3:]

        # Jacobian (from the dynamics model)
        try:
            jacobian = np.array(
                self._model.zero_jacobian(robot_state)
            ).reshape(6, 7, order="F")
        except Exception:
            jacobian = np.zeros((6, 7))

        # TCP velocity via Jacobian
        try:
            tcp_vel = jacobian @ joint_vel
        except Exception:
            tcp_vel = np.zeros(6)

        with self._state_lock:
            self._state.tcp_pose = tcp_pose
            self._state.arm_joint_position = joint_pos
            self._state.arm_joint_velocity = joint_vel
            self._state.tcp_force = tcp_force
            self._state.tcp_torque = tcp_torque
            self._state.arm_jacobian = jacobian
            self._state.tcp_vel = tcp_vel
            self._has_valid_state = True

    # ═══════════════════════════════════════════════════════════════
    #  Public API
    # ═══════════════════════════════════════════════════════════════

    def is_robot_up(self) -> bool:
        """Check if the robot state is valid and the gripper is ready."""
        try:
            with self._state_lock:
                has_state = self._has_valid_state
            if not has_state:
                return False
            return self._gripper.is_ready()
        except Exception:
            return False

    def get_state(self) -> FrankaRobotState:
        """Get the current robot state.

        The arm state is updated at 1 kHz by the RT loop.  Gripper state
        is read here (outside the state lock) to avoid blocking the RT
        loop with potentially slow gripper I/O.
        """
        try:
            self._cached_gripper_position = self._gripper.position
            self._cached_gripper_open = self._gripper.is_open
        except Exception as e:
            self._logger.warning(f"Gripper state read failed: {e}")

        with self._state_lock:
            self._state.gripper_position = self._cached_gripper_position
            self._state.gripper_open = self._cached_gripper_open
            return copy.deepcopy(self._state)

    def reconfigure_compliance_params(self, params: dict[str, float]):
        """Reconfigure joint impedance stiffness.

        Accepts a dict with an optional ``"Kq"`` key mapping to a list
        of 7 stiffness values (Nm/rad).  Other keys are ignored (Cartesian
        impedance params are not applicable to joint position control).
        """
        Kq = params.get("Kq", None)
        if Kq is not None:
            try:
                self._robot.set_joint_impedance(list(Kq))
            except Exception as e:
                self._logger.warning(f"Failed to set joint impedance: {e}")

    def clear_errors(self):
        """Attempt automatic error recovery."""
        try:
            self._robot.automatic_error_recovery()
        except Exception as e:
            self._logger.warning(f"Error recovery failed: {e}")

    # ═══════════════════════════════════════════════════════════════
    #  Joint Control (non-blocking — sets RT loop target)
    # ═══════════════════════════════════════════════════════════════

    def move_joints(self, joint_positions: np.ndarray):
        """Set target joint positions (non-blocking).

        The 1 kHz RT loop will continuously stream these targets to the
        robot's joint impedance controller.

        Args:
            joint_positions: Target joint positions in radians, shape ``(7,)``.
        """
        assert len(joint_positions) == 7, (
            f"Expected 7 joint positions, got {len(joint_positions)}"
        )
        positions = np.clip(
            joint_positions, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER
        )
        with self._target_lock:
            self._target = positions.tolist()

    def reset_joint(self, reset_pos: list[float]):
        """Reset to target joint position (blocking).

        Stops the RT loop, runs a temporary 1 kHz control loop that
        smoothly interpolates to the target, then restarts the RT loop.

        Args:
            reset_pos: Desired joint positions (7-DOF).
        """
        assert len(reset_pos) == 7, (
            f"Invalid reset position, expected 7 dims but got {len(reset_pos)}"
        )

        was_active = self._rt_active
        self._stop_rt_loop()
        self.clear_errors()

        JointPositions = self._pylibfranka_JointPositions
        ControllerMode = self._pylibfranka_ControllerMode
        target = np.array(reset_pos, dtype=np.float64)

        try:
            active = self._robot.start_joint_position_control(
                ControllerMode.JointImpedance
            )
            max_steps = 10000  # 10 seconds at 1 kHz
            for _ in range(max_steps):
                state, _ = active.readOnce()
                self._update_state(state)
                current = np.array(state.q)

                # Smooth interpolation: limit to ~1 rad/s at 1 kHz
                delta = target - current
                max_delta = 0.001  # rad per 1 ms tick
                delta = np.clip(delta, -max_delta, max_delta)
                cmd_pos = (current + delta).tolist()

                cmd = JointPositions(cmd_pos)
                if np.allclose(current, target, atol=1e-2):
                    cmd.motion_finished = True
                    active.writeOnce(cmd)
                    break
                active.writeOnce(cmd)
            else:
                # Timed out — finish the motion anyway
                state, _ = active.readOnce()
                cmd = JointPositions(list(state.q_d))
                cmd.motion_finished = True
                active.writeOnce(cmd)
                self._logger.warning(
                    "reset_joint timed out before reaching target"
                )
        except Exception as e:
            self._logger.warning(f"Joint reset failed: {e}")
            self.clear_errors()

        self._wait_for_joint(reset_pos)
        with self._state_lock:
            final_pos = self._state.arm_joint_position.copy()
        self._logger.debug(f"Joint reset complete: {final_pos}")

        # Reset the target to the new position
        with self._target_lock:
            self._target = list(reset_pos)

        if was_active:
            self._start_rt_loop()

    # ═══════════════════════════════════════════════════════════════
    #  Cartesian Motion (for reset compatibility)
    # ═══════════════════════════════════════════════════════════════

    def move_arm(self, position: np.ndarray):
        """Move toward a Cartesian pose via linearised IK (non-blocking).

        Converts the Cartesian target to joint positions using a single-step
        damped least-squares IK from the current state, then updates the
        RT loop target.  This is used by ``FrankaEnv._interpolate_move()``
        during reset, where successive small-delta calls at 10 Hz produce
        smooth motion.

        Args:
            position: 7-D array ``[x, y, z, qx, qy, qz, qw]``.
        """
        assert len(position) == 7, (
            f"Invalid position, expected 7 dims but got {len(position)}"
        )

        with self._state_lock:
            current_pose = self._state.tcp_pose.copy()
            current_joints = self._state.arm_joint_position.copy()
            jacobian = self._state.arm_jacobian.copy()

        # Position error
        pos_error = position[:3] - current_pose[:3]

        # Orientation error via rotation vector
        R_target = R.from_quat(position[3:])
        R_current = R.from_quat(current_pose[3:])
        orient_error = (R_target * R_current.inv()).as_rotvec()

        # Full 6-D Cartesian error
        x_error = np.concatenate([pos_error, orient_error])

        # Damped pseudo-inverse: Δq = J^T (J J^T + λ²I)^{-1} Δx
        damping = 1e-3
        JJT = jacobian @ jacobian.T + damping**2 * np.eye(6)
        delta_q = jacobian.T @ np.linalg.solve(JJT, x_error)

        target_joints = np.clip(
            current_joints + delta_q, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER
        )

        with self._target_lock:
            self._target = target_joints.tolist()

    # ═══════════════════════════════════════════════════════════════
    #  Gripper (delegates to self._gripper)
    # ═══════════════════════════════════════════════════════════════

    def open_gripper(self):
        self._gripper.open()
        self.log_debug("Open gripper")

    def close_gripper(self):
        self._gripper.close()
        self.log_debug("Close gripper")

    def move_gripper(self, position: int, speed: float = 0.3):
        assert 0 <= position <= 255, (
            f"Invalid gripper position {position}, must be between 0 and 255"
        )
        self._gripper.move(position, speed)
        self.log_debug(f"Move gripper to position: {position}")

    # ═══════════════════════════════════════════════════════════════
    #  Internals
    # ═══════════════════════════════════════════════════════════════

    def _wait_for_joint(self, target_pos: list[float], timeout: int = 30):
        """Poll until joints reach *target_pos* or timeout."""
        wait_time = 0.01
        waited_time = 0.0
        target_pos = np.array(target_pos)

        while waited_time < timeout:
            with self._state_lock:
                current = self._state.arm_joint_position.copy()
            if np.allclose(target_pos, current, atol=1e-2, rtol=1e-2):
                break
            time.sleep(wait_time)
            waited_time += wait_time

        if waited_time >= timeout:
            self._logger.warning("Joint position wait timeout exceeded")
        else:
            with self._state_lock:
                current = self._state.arm_joint_position.copy()
            self._logger.debug(f"Joint position reached {current}")

    def cleanup(self):
        """Stop all threads and release resources."""
        self._running = False
        self._stop_rt_loop()
        if hasattr(self, "_idle_thread") and self._idle_thread.is_alive():
            self._idle_thread.join(timeout=2.0)
        self._gripper.cleanup()
