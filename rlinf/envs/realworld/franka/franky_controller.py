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

"""Franka robot controller using Franky (libfranka) for joint-space control.

Thin wrapper around the ``franky`` library.  All real-time work — 1 kHz
``robot.control()``, Ruckig online trajectory generation, motion
preemption — runs inside franky's C++ std::thread, and every franky
Python binding releases the GIL via
``py::call_guard<py::gil_scoped_release>()``.  That means this
controller does no RT work in Python at all:

* ``move_joints`` / ``move_arm`` just hand a ``JointMotion`` or
  ``CartesianMotion`` to ``robot.move(..., asynchronous=True)`` and
  return immediately.  The next call preempts the previous one and
  Ruckig re-plans from the current state — so streaming GELLO joint
  targets at 1 kHz works out of the box.
* ``reset_joint`` is the one place we want a blocking wait (env
  ``reset()`` semantics), so we submit the motion synchronously.
* ``get_state`` calls ``robot.read_once()`` on demand; there is no
  background state thread fighting for the GIL with Ray dispatch.

See ``franky_install.md`` for the PREEMPT_RT kernel / CPU governor /
rtprio limits that must be set before this controller will behave
deterministically.
"""

import ctypes
import ctypes.util
import os
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

# Collision thresholds (Nm / N) — conservative defaults that let
# deliberate streaming moves through without tripping reflexes.
_DEFAULT_TORQUE_THRESHOLD = [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0]
_DEFAULT_FORCE_THRESHOLD = [20.0, 20.0, 20.0, 25.0, 25.0, 25.0]

# Joint impedance (Nm/rad).  Lower than libfranka's defaults on joints
# 5-7 so small command noise from a 10 Hz upstream doesn't translate
# into audible buzz.  Tune via reconfigure_compliance_params at runtime.
_DEFAULT_JOINT_IMPEDANCE = [400.0, 400.0, 400.0, 400.0, 200.0, 100.0, 40.0]

# Ruckig dynamics scale.  0.15 = 15% of Franka's hard v/a/j limits.
_DEFAULT_DYNAMICS_FACTOR = 0.15

# Real-time priority for the Python main/dispatch thread.  80 is a
# polite high — leaves headroom for kernel threads (>=90).
_RT_PRIORITY = 80

# mlockall flags (MCL_CURRENT=1, MCL_FUTURE=2).  2 alone would miss
# existing pages, 3 locks everything now and pins future allocations.
_MCL_CURRENT = 1
_MCL_FUTURE = 2


class FrankyController(Worker):
    """Franka robot arm controller backed by ``franky``.

    Args:
        robot_ip: IP address of the Franka robot.
        gripper_type: ``"robotiq"`` (Modbus RTU parallel-jaw).
        gripper_connection: Serial port for Robotiq (e.g.
            ``"/dev/ttyUSB0"``).
    """

    @staticmethod
    def launch_controller(
        robot_ip: str,
        env_idx: int = 0,
        node_rank: int = 0,
        worker_rank: int = 0,
        gripper_type: str = "robotiq",
        gripper_connection: Optional[str] = None,
    ):
        """Launch a ``FrankyController`` as a Ray actor on *node_rank*."""
        cluster = Cluster()
        placement = NodePlacementStrategy(node_ranks=[node_rank])
        return FrankyController.create_group(
            robot_ip, gripper_type, gripper_connection
        ).launch(
            cluster=cluster,
            placement_strategy=placement,
            name=f"FrankyController-{worker_rank}-{env_idx}",
        )

    def __init__(
        self,
        robot_ip: str,
        gripper_type: str = "robotiq",
        gripper_connection: Optional[str] = None,
    ):
        super().__init__()
        self._logger = get_logger()
        self._robot_ip = robot_ip
        self._gripper_type = gripper_type

        # ── Lock memory + raise priority BEFORE touching the robot ──
        # Done first so any allocation during the franky import is
        # already under the mlockall umbrella.
        self._apply_rt_hardening()

        # ── Robot connection ────────────────────────────────────────
        import franky

        self._franky = franky
        self._robot = franky.Robot(robot_ip)
        self._robot.recover_from_errors()
        self._robot.relative_dynamics_factor = _DEFAULT_DYNAMICS_FACTOR

        # Collision + impedance configuration.  franky exposes a
        # two-argument ``set_collision_behavior(torque, force)`` that
        # applies the same threshold to lower/upper.
        try:
            self._robot.set_collision_behavior(
                _DEFAULT_TORQUE_THRESHOLD,
                _DEFAULT_FORCE_THRESHOLD,
            )
        except Exception as e:
            self._logger.warning(f"set_collision_behavior failed: {e}")

        try:
            self._robot.set_joint_impedance(_DEFAULT_JOINT_IMPEDANCE)
        except Exception as e:
            self._logger.warning(f"set_joint_impedance failed: {e}")

        # ── Gripper (Robotiq Modbus RTU) ──────────────────────────────
        if gripper_type != "robotiq":
            raise ValueError(
                f"Unsupported gripper_type={gripper_type!r} for "
                f"FrankyController. Currently only 'robotiq' is supported."
            )
        if gripper_connection is None:
            raise ValueError(
                "gripper_connection (serial port) must be specified "
                "for Robotiq grippers (e.g. '/dev/ttyUSB0'). "
                "Set it in the hardware config: "
                "cluster.node_groups[].hardware.configs[].gripper_connection"
            )
        self._gripper = create_gripper(
            gripper_type="robotiq",
            port=gripper_connection,
        )

        # Cached gripper state (read on get_state, not on every RT tick).
        self._cached_gripper_position: float = 0.0
        self._cached_gripper_open: bool = True

        self._logger.info(
            f"FrankyController connected to robot at {robot_ip} "
            f"(impedance={_DEFAULT_JOINT_IMPEDANCE}, "
            f"dynamics_factor={_DEFAULT_DYNAMICS_FACTOR})"
        )

    # ═══════════════════════════════════════════════════════════════
    #  Real-time hardening (Linux PREEMPT_RT scheduling + mlockall)
    # ═══════════════════════════════════════════════════════════════

    def _apply_rt_hardening(self) -> None:
        """Lock memory, raise priority, pin affinity.

        Every step is best-effort: we log and continue on failure.  The
        sudo commands in ``franky_install.md`` grant the permissions
        these calls need; without them the calls are no-ops (the
        controller still works, just with normal OS scheduling).
        """
        # --- mlockall ---------------------------------------------------
        libc_name = ctypes.util.find_library("c") or "libc.so.6"
        try:
            libc = ctypes.CDLL(libc_name, use_errno=True)
            rc = libc.mlockall(_MCL_CURRENT | _MCL_FUTURE)
            if rc != 0:
                err = os.strerror(ctypes.get_errno())
                self._logger.warning(
                    f"mlockall failed ({err}); page faults may cause RT jitter"
                )
            else:
                self._logger.info("mlockall: memory pages pinned")
        except Exception as e:
            self._logger.warning(f"mlockall unavailable: {e}")

        # --- SCHED_FIFO -------------------------------------------------
        try:
            param = os.sched_param(_RT_PRIORITY)
            os.sched_setscheduler(0, os.SCHED_FIFO, param)
            sched = os.sched_getscheduler(0)
            self._logger.info(
                f"SCHED_FIFO priority {_RT_PRIORITY} applied (policy={sched})"
            )
        except PermissionError:
            self._logger.warning(
                "SCHED_FIFO not granted — run the one-time cap/limits "
                "setup in franky_install.md so rtprio>=80 is allowed."
            )
        except Exception as e:
            self._logger.warning(f"SCHED_FIFO setup failed: {e}")

        # --- CPU affinity ----------------------------------------------
        # Leave CPUs 2-3 free for franky's C++ control thread and its
        # network interrupts; pin everything else Python touches to
        # the remaining cores.  On machines with <4 cores this is a
        # no-op.
        try:
            ncpu = os.cpu_count() or 1
            if ncpu >= 6:
                allowed = {0, 1} | set(range(4, ncpu))
                os.sched_setaffinity(0, allowed)
                self._logger.info(
                    f"Python thread affinity set to {sorted(allowed)}; "
                    f"CPUs 2-3 reserved for franky RT thread"
                )
        except Exception as e:
            self._logger.warning(f"sched_setaffinity failed: {e}")

    # ═══════════════════════════════════════════════════════════════
    #  State
    # ═══════════════════════════════════════════════════════════════

    def _compute_full_state(self, robot_state) -> FrankaRobotState:
        """Parse a ``franky.RobotState`` into our ``FrankaRobotState``.

        Called off the RT path (from ``get_state``), so it can afford
        FK + Jacobian computation.

        ``franky.RobotState.O_T_EE`` is a ``franky.Affine`` (not a raw
        4x4) — we use its ``.translation`` and ``.quaternion`` accessors
        directly.  ``q``, ``dq``, ``K_F_ext_hat_K`` come through as
        numpy arrays via pybind11/eigen.
        """
        O_T_EE = robot_state.O_T_EE
        translation = np.asarray(O_T_EE.translation, dtype=np.float64)
        # franky.Affine.quaternion returns Eigen coeffs [x, y, z, w] —
        # same layout scipy uses, so no remap needed.
        quat = np.asarray(O_T_EE.quaternion, dtype=np.float64)
        tcp_pose = np.concatenate([translation, quat])

        joint_pos = np.asarray(robot_state.q, dtype=np.float64)
        joint_vel = np.asarray(robot_state.dq, dtype=np.float64)

        K_F_ext = np.asarray(robot_state.K_F_ext_hat_K, dtype=np.float64)
        tcp_force = K_F_ext[:3]
        tcp_torque = K_F_ext[3:]

        try:
            jacobian = np.asarray(
                self._robot.model.zero_jacobian(
                    self._franky.Frame.EndEffector, robot_state
                ),
                dtype=np.float64,
            )
            if jacobian.shape != (6, 7):
                jacobian = jacobian.reshape(6, 7)
        except Exception as e:
            self._logger.warning(f"zero_jacobian failed: {e}")
            jacobian = np.zeros((6, 7))

        try:
            tcp_vel = jacobian @ joint_vel
        except Exception:
            tcp_vel = np.zeros(6)

        s = FrankaRobotState()
        s.tcp_pose = tcp_pose
        s.arm_joint_position = joint_pos
        s.arm_joint_velocity = joint_vel
        s.tcp_force = tcp_force
        s.tcp_torque = tcp_torque
        s.arm_jacobian = jacobian
        s.tcp_vel = tcp_vel
        return s

    # ═══════════════════════════════════════════════════════════════
    #  Public API
    # ═══════════════════════════════════════════════════════════════

    def is_robot_up(self) -> bool:
        """Check robot link + gripper handshake.

        ``robot.state`` is the franky way to poll the latest cached
        state — it falls back to an internal ``readOnce`` when no
        motion is active, and returns the cached control-thread state
        otherwise.  Either way the GIL is released during the call.
        """
        try:
            _ = self._robot.state
        except Exception:
            return False
        try:
            return self._gripper.is_ready()
        except Exception:
            return False

    def get_state(self) -> FrankaRobotState:
        """Read the current robot state (off the RT hot path).

        Uses the ``robot.state`` property (GIL-released; see
        ``franky/python/bind_robot.cpp:237``).  Safe to call at any
        time: when a motion is active it returns the cached state from
        franky's control thread; when idle it does a one-shot libfranka
        readOnce internally.
        """
        try:
            self._cached_gripper_position = self._gripper.position
            self._cached_gripper_open = self._gripper.is_open
        except Exception as e:
            self._logger.warning(f"Gripper state read failed: {e}")

        try:
            raw = self._robot.state
        except Exception as e:
            self._logger.warning(f"robot.state failed: {e}")
            s = FrankaRobotState()
            s.gripper_position = self._cached_gripper_position
            s.gripper_open = self._cached_gripper_open
            return s

        s = self._compute_full_state(raw)
        s.gripper_position = self._cached_gripper_position
        s.gripper_open = self._cached_gripper_open
        return s

    def reconfigure_compliance_params(self, params: dict[str, float]):
        """Apply runtime joint impedance updates.

        Accepts a dict with an optional ``"Kq"`` key mapping to a list
        of 7 stiffness values (Nm/rad).  Cartesian params are ignored
        — joint-space control doesn't use them.
        """
        Kq = params.get("Kq", None)
        if Kq is not None:
            try:
                self._robot.set_joint_impedance(list(Kq))
                self._logger.info(f"Joint impedance updated: {list(Kq)}")
            except Exception as e:
                self._logger.warning(f"Failed to set joint impedance: {e}")

    def clear_errors(self):
        """Trigger libfranka's automatic error recovery."""
        try:
            self._robot.recover_from_errors()
        except Exception as e:
            self._logger.warning(f"Error recovery failed: {e}")

    # ═══════════════════════════════════════════════════════════════
    #  Joint control (non-blocking — franky Ruckig re-plans on preempt)
    # ═══════════════════════════════════════════════════════════════

    def move_joints(self, joint_positions: np.ndarray):
        """Submit a new joint-space target (non-blocking).

        Builds a ``JointMotion(target, ReferenceType.Absolute)`` and
        calls ``robot.move(motion, asynchronous=True)``.  Successive
        calls preempt the in-flight motion; franky's Ruckig re-plans
        from the current state/velocity/acceleration, so streaming
        targets at 1 kHz is fine.
        """
        assert len(joint_positions) == 7, (
            f"Expected 7 joint positions, got {len(joint_positions)}"
        )
        q = np.clip(
            joint_positions, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER
        ).tolist()

        franky = self._franky
        motion = franky.JointMotion(
            q, reference_type=franky.ReferenceType.Absolute
        )
        try:
            self._robot.move(motion, asynchronous=True)
        except Exception as e:
            self._logger.warning(f"Joint motion failed: {e}")
            self.clear_errors()

    def reset_joint(self, reset_pos: list[float]):
        """Blocking joint reset — matches env ``reset()`` semantics.

        Uses franky's synchronous ``robot.move`` on a ``JointMotion``
        so Ruckig picks a smooth duration and we return only when the
        robot has actually arrived.  No manual ``_wait_for_joint``
        polling needed.
        """
        assert len(reset_pos) == 7, (
            f"Invalid reset position, expected 7 dims but got {len(reset_pos)}"
        )
        self.clear_errors()

        franky = self._franky
        motion = franky.JointMotion(
            list(reset_pos),
            reference_type=franky.ReferenceType.Absolute,
        )
        try:
            self._robot.move(motion)  # synchronous
        except Exception as e:
            self._logger.warning(f"Joint reset failed: {e}")
            self.clear_errors()

        try:
            final = list(self._robot.current_joint_positions)
            self._logger.debug(f"Joint reset complete: {final}")
        except Exception:
            pass

    # ═══════════════════════════════════════════════════════════════
    #  Cartesian motion (non-blocking — for env reset interpolation)
    # ═══════════════════════════════════════════════════════════════

    def move_arm(self, position: np.ndarray):
        """Submit a Cartesian target (non-blocking).

        Called at 10 Hz from ``FrankaEnv._interpolate_move`` during a
        Cartesian reset.  Must be async so each 10 Hz tick completes
        immediately; franky's Ruckig interpolates between successive
        targets.
        """
        assert len(position) == 7, (
            f"Invalid position, expected 7 dims but got {len(position)}"
        )
        franky = self._franky

        translation = np.asarray(position[:3], dtype=np.float64)
        quat = np.asarray(position[3:], dtype=np.float64)
        rotation_matrix = R.from_quat(quat).as_matrix()
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        T[:3, 3] = translation

        motion = franky.CartesianMotion(
            franky.Affine(T),
            reference_type=franky.ReferenceType.Absolute,
        )
        try:
            self._robot.move(motion, asynchronous=True)
        except Exception as e:
            self._logger.warning(f"Cartesian motion failed: {e}")
            self.clear_errors()

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
    #  Lifecycle
    # ═══════════════════════════════════════════════════════════════

    def cleanup(self):
        """Join any in-flight motion and release the Robotiq gripper handle."""
        try:
            self._robot.join_motion()
        except Exception:
            pass
        try:
            self._gripper.cleanup()
        except Exception:
            pass
