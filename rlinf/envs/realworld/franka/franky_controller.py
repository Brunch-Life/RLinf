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
torque loop and impedance computation — runs inside franky's C++
std::thread; franky bindings release the GIL via
``py::call_guard<py::gil_scoped_release>()``.

Torque law used by ``JointImpedanceTrackingMotion``
(franky ``src/motion/joint_impedance_motion.cpp``)::

    τ = Kq·(q_ref − q) + Kqd·(dq_ref − dq) + coriolis + τ_ff
    τ ← saturateTorqueRate(τ, τ_prev, max_delta_tau)

Torque law traces polymetis's pure-joint ``JointImpedanceControl``
(``adaptive=False``) branch — franky has no ``HybridJointImpedanceControl``
path (``Kp_eff = Jᵀ Kx J + Kq``).  Default Kq/Kqd below are hand-tuned
for dual-Franka GELLO teleop, not polymetis defaults; collision
thresholds are also raised (see ``_DEFAULT_TORQUE_THRESHOLD`` below).

See ``requirements/embodied/franky_install.md`` for the PREEMPT_RT kernel / CPU governor /
rtprio limits that must be set before this controller will behave
deterministically.
"""

import ctypes
import ctypes.util
import os
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
JOINT_LIMITS_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

# Joint velocity limits (rad/s), Franka hard limits minus 0.1 rad/s
# margin — same figures polymetis uses.  Used to clamp dq feedforward.
JOINT_VEL_LIMITS = np.array([2.075, 2.075, 2.075, 2.075, 2.51, 2.51, 2.51])

# Collision thresholds (joint torques Nm; Cartesian Fxyz N + Mxyz Nm).
# Raised well above polymetis defaults so GELLO teleop tracking errors
# don't trip the reflex; joint values stay under libfranka's physical
# limits [87,87,87,87,12,12,12] Nm.
_DEFAULT_TORQUE_THRESHOLD = [80.0, 80.0, 80.0, 80.0, 11.0, 11.0, 11.0]
_DEFAULT_FORCE_THRESHOLD = [100.0, 100.0, 100.0, 25.0, 25.0, 25.0]

# Joint impedance PD gains (Nm/rad, Nms/rad) — hand-tuned for dual-Franka
# GELLO joint teleop; see toolkits/realworld_check/tune_impedance.py for
# the live tuner and keep it in sync with these values.
_DEFAULT_JOINT_STIFFNESS = [103.75, 265.734, 227.273, 221.445, 13.5, 12.818, 5.134]
_DEFAULT_JOINT_DAMPING = [16.7, 40.263, 25.0, 12.862, 1.5, 2.0, 1.331]

# Cartesian impedance gains used by ``stream_tcp_impedance`` — franky
# defaults translate to N/m for xyz and Nm/rad for rotation.  Kept at
# franky defaults; bump if the TCP-pose policy's tracking is too soft.
_DEFAULT_TRANS_STIFFNESS = 2000.0
_DEFAULT_ROT_STIFFNESS = 200.0
# Per-motion "duration" legacy constant — retained for
# ``stream_tcp_impedance``'s API signature (kept for backwards compat).
# The current implementation uses a long-lived CartesianImpedanceTracker
# and does not submit per-call motions, so this value is unused.
_TCP_STREAM_DURATION_S = 0.3

# TCP velocity feedforward clamps — mirror the joint-space JOINT_VEL_LIMITS
# role. Ballpark Franka TCP ceilings are ~1.7 m/s linear and ~2.5 rad/s
# angular; these are conservative enough that a bad policy step can't
# drive an unsafe step-velocity into the impedance controller.
_TCP_LIN_VEL_LIMIT = 1.0  # m/s
_TCP_ANG_VEL_LIMIT = 2.5  # rad/s

# Null-space stiffness for the Cartesian impedance tracker. Franka has
# 7 DOF while the TCP task constrains 6, so the remaining redundant DOF
# (chiefly J7 / elbow circle) is an undamped oscillator when this is
# zero. We cannot refresh nullspace_target at runtime (franky's Tracker
# API only takes it once at construction), so we anchor on the
# configuration q at tracker start. A large stiffness then pulls q back
# toward that frozen snapshot as the TCP moves — the null-space torque
# is in principle orthogonal to the 6-DOF task, but in practice anything
# above single digits visibly disturbs Cartesian tracking. Keep it small
# so it only supplies damping on the redundant DOF; J7 runaway is still
# bounded by the joint-limit pushback the tracker applies below.
_DEFAULT_NULLSPACE_STIFFNESS = 5.0

# Global dynamics factor scales JointMotion/CartesianMotion planned
# moves (reset path).  0.3 ≈ polymetis min-jerk feel; not applied to
# the torque-mode impedance tracker.
_DEFAULT_DYNAMICS_FACTOR = 0.3

# Minimum dt for dq feedforward.  Guards against divide-by-zero when
# two move_joints calls land in the same millisecond.
_DQ_MIN_DT_S = 1e-3

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

        # Must precede the franky import so mlockall catches its allocations.
        self._apply_rt_hardening()

        import franky

        self._franky = franky
        self._robot = franky.Robot(robot_ip)
        self._robot.recover_from_errors()
        self._robot.relative_dynamics_factor = _DEFAULT_DYNAMICS_FACTOR

        # franky exposes a two-argument ``set_collision_behavior(torque,
        # force)`` that applies the same threshold to lower/upper.
        try:
            self._robot.set_collision_behavior(
                _DEFAULT_TORQUE_THRESHOLD,
                _DEFAULT_FORCE_THRESHOLD,
            )
        except Exception as e:
            self._logger.warning(f"set_collision_behavior failed: {e}")

        # Note: libfranka ``set_joint_impedance`` only affects position-
        # mode moves (our JointMotion reset path).  The torque-mode
        # impedance tracker reads its own stiffness/damping, so we skip
        # that call and let libfranka keep its internal defaults for the
        # reset path.

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

        # Joint-space impedance tracker (lazy-started on first move_joints).
        # Cartesian impedance tracker (lazy-started on first stream_tcp_impedance).
        # They are mutually exclusive in franky — starting one stops the other.
        self._tracker = None
        self._cart_tracker = None

        # Previous commanded TCP target [xyz, quat_xyzw] + timestamp, used
        # to derive the linear/angular velocity feedforward passed to the
        # Cartesian tracker on each stream_tcp_impedance call. Mirrors the
        # joint-space dq feedforward — see the docstring on move_joints
        # for why feedforward is essential at 10 Hz env.step cadence.
        self._prev_tcp_target: Optional[np.ndarray] = None
        self._prev_tcp_target_ts: Optional[float] = None

        # Previous commanded target + timestamp, used to derive the
        # dq feedforward passed to the tracker each move_joints call.
        self._prev_target_q: Optional[np.ndarray] = None
        self._prev_target_ts: Optional[float] = None

        # Cached gripper state (read on get_state, not on every RT tick).
        self._cached_gripper_position: float = 0.0
        self._cached_gripper_open: bool = True

        self._logger.info(
            f"FrankyController connected to robot at {robot_ip} "
            f"(Kq={_DEFAULT_JOINT_STIFFNESS}, Kqd={_DEFAULT_JOINT_DAMPING}, "
            f"dynamics_factor={_DEFAULT_DYNAMICS_FACTOR})"
        )

    def _apply_rt_hardening(self) -> None:
        """Lock memory, raise priority, pin affinity.

        Every step is best-effort: we log and continue on failure.  The
        sudo commands in ``requirements/embodied/franky_install.md`` grant the permissions
        these calls need; without them the calls are no-ops (the
        controller still works, just with normal OS scheduling).
        """
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
                "setup in requirements/embodied/franky_install.md so rtprio>=80 is allowed."
            )
        except Exception as e:
            self._logger.warning(f"SCHED_FIFO setup failed: {e}")

        # Reserve CPUs 2-3 for franky's RT control thread + NIC IRQs.
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

        Accepts a dict with optional ``"Kq"`` (stiffness, Nm/rad) and
        ``"Kqd"`` (damping, Nms/rad) keys, each a 7-element list.
        Any other keys (e.g. the Cartesian compliance dict from the
        legacy franka_controller config) are silently ignored so the
        same reset hook works for both controllers.  When a tracker is
        active, gains are smoothed via its exponential interpolation.
        """
        Kq = params.get("Kq", None)
        Kqd = params.get("Kqd", None)
        if Kq is None and Kqd is None:
            return
        if self._tracker is None:
            return
        try:
            self._tracker.set_gains(
                stiffness=np.array(Kq, dtype=np.float64) if Kq is not None else None,
                damping=np.array(Kqd, dtype=np.float64) if Kqd is not None else None,
            )
            self._logger.info(f"Tracker gains updated: Kq={Kq}, Kqd={Kqd}")
        except Exception as e:
            self._logger.warning(f"Failed to set tracker gains: {e}")

    def clear_errors(self):
        """Trigger libfranka's automatic error recovery."""
        try:
            self._robot.recover_from_errors()
        except Exception as e:
            self._logger.warning(f"Error recovery failed: {e}")

    def _ensure_tracking_motion(self):
        """Start the impedance tracking motion if not already active.

        Creates a ``JointImpedanceTracker`` which internally builds a
        ``JointImpedanceTrackingMotion``, seeds the reference from the
        current joint positions, and starts the async control loop.
        Subsequent calls to ``set_target()`` update the reference that
        the C++ RT thread tracks via impedance torques.
        """
        if self._tracker is not None:
            return

        # Flush any lingering async motion (e.g. a failed Cartesian
        # reset) so franky allows switching to joint impedance mode.
        try:
            self._robot.join_motion()
        except Exception:
            pass
        self.clear_errors()

        self._tracker = self._franky.JointImpedanceTracker(
            self._robot,
            stiffness=np.array(_DEFAULT_JOINT_STIFFNESS, dtype=np.float64),
            damping=np.array(_DEFAULT_JOINT_DAMPING, dtype=np.float64),
            compensate_coriolis=True,
        )
        self._prev_target_q = None
        self._prev_target_ts = None
        self._logger.info("Joint impedance tracking motion started")

    def move_joints(self, joint_positions: np.ndarray):
        """Update the impedance tracking reference (non-blocking).

        Target velocity feedforward is derived from the last commanded
        target and elapsed wall time, clamped to Franka's joint
        velocity limits.  Without it the PD term has to eat all
        tracking error through the position channel, which produces
        visible lag and overshoot at the 10 Hz env.step rate even when
        the direct-stream daemon pushes at 1 kHz.
        """
        assert len(joint_positions) == 7, (
            f"Expected 7 joint positions, got {len(joint_positions)}"
        )
        q = np.clip(
            np.asarray(joint_positions, dtype=np.float64),
            JOINT_LIMITS_LOWER,
            JOINT_LIMITS_UPPER,
        )

        now = time.perf_counter()
        if self._prev_target_q is not None and self._prev_target_ts is not None:
            dt = max(now - self._prev_target_ts, _DQ_MIN_DT_S)
            dq_ff = np.clip(
                (q - self._prev_target_q) / dt, -JOINT_VEL_LIMITS, JOINT_VEL_LIMITS
            )
        else:
            dq_ff = None

        try:
            self._ensure_tracking_motion()
            self._tracker.set_target(q, dq=dq_ff)
        except Exception as e:
            self._logger.warning(f"Joint impedance update failed: {e}")
            self._stop_tracking_motion()
            try:
                self._robot.join_motion()
            except Exception:
                pass
            self.clear_errors()
            return

        self._prev_target_q = q
        self._prev_target_ts = now

    def _stop_tracking_motion(self):
        """Stop any active impedance tracker (joint and/or Cartesian).

        After stopping, also flush any in-flight motion and clear errors
        so libfranka is guaranteed to be in a quiescent state on return.
        Without this, a tracker that was oscillating (e.g. Cartesian
        nullspace chatter) can leave stale torque commands that prevent
        the next motion from switching control mode cleanly.
        """
        had_tracker = self._tracker is not None or self._cart_tracker is not None
        if self._tracker is not None:
            try:
                self._tracker.stop()
            except Exception:
                pass
            self._tracker = None
        if self._cart_tracker is not None:
            try:
                self._cart_tracker.stop()
            except Exception:
                pass
            self._cart_tracker = None
        self._prev_target_q = None
        self._prev_target_ts = None
        self._prev_tcp_target = None
        self._prev_tcp_target_ts = None

        if had_tracker:
            try:
                self._robot.join_motion()
            except Exception:
                pass
            try:
                self._robot.recover_from_errors()
            except Exception:
                pass

    def _ensure_cart_tracking_motion(
        self,
        *,
        translational_stiffness: float = _DEFAULT_TRANS_STIFFNESS,
        rotational_stiffness: float = _DEFAULT_ROT_STIFFNESS,
    ):
        """Lazy-start a long-lived ``CartesianImpedanceTracker`` session.

        The joint tracker cannot coexist with Cartesian torque mode, so
        it is stopped first. Subsequent ``stream_tcp_impedance`` calls
        just update the reference via ``set_target``.
        """
        if self._cart_tracker is not None:
            return
        if self._tracker is not None:
            try:
                self._tracker.stop()
            except Exception:
                pass
            self._tracker = None
            self._prev_target_q = None
            self._prev_target_ts = None
        try:
            self._robot.join_motion()
        except Exception:
            pass
        self.clear_errors()

        # Snapshot the current joints as the null-space anchor. Without
        # this Franka's redundant DOF (J7 / elbow) floats freely and
        # chatters violently once the TCP target stops moving. Also pass
        # the per-joint limits so the tracker pushes back softly before
        # we hit libfranka's hard limit + reflex state.
        try:
            current_q = np.asarray(
                self._robot.current_joint_positions, dtype=np.float64
            )
        except Exception as e:
            self._logger.warning(
                f"Could not read current_joint_positions for nullspace anchor: {e}; "
                f"starting tracker without null-space target"
            )
            current_q = None

        self._cart_tracker = self._franky.CartesianImpedanceTracker(
            self._robot,
            translational_stiffness=translational_stiffness,
            rotational_stiffness=rotational_stiffness,
            nullspace_target=current_q,
            nullspace_stiffness=(
                _DEFAULT_NULLSPACE_STIFFNESS if current_q is not None else 0.0
            ),
            lower_joint_limits=JOINT_LIMITS_LOWER,
            upper_joint_limits=JOINT_LIMITS_UPPER,
        )
        self._logger.info(
            "Cartesian impedance tracking motion started "
            f"(nullspace_stiffness={_DEFAULT_NULLSPACE_STIFFNESS if current_q is not None else 0.0})"
        )

    def reset_joint(self, reset_pos: list[float]):
        """Blocking joint reset — matches env ``reset()`` semantics.

        Stops any active impedance tracking, then uses a synchronous
        ``JointMotion`` so the robot moves smoothly to the target and
        we return only when it has arrived.
        """
        assert len(reset_pos) == 7, (
            f"Invalid reset position, expected 7 dims but got {len(reset_pos)}"
        )
        self._stop_tracking_motion()
        try:
            self._robot.join_motion()
        except Exception:
            pass
        self.clear_errors()

        franky = self._franky
        motion = franky.JointMotion(
            franky.JointState(position=np.asarray(reset_pos, dtype=np.float64)),
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

    def stream_tcp_impedance(
        self,
        position: np.ndarray,
        duration_s: float = _TCP_STREAM_DURATION_S,  # noqa: ARG002 — kept for API compat
        translational_stiffness: float = _DEFAULT_TRANS_STIFFNESS,
        rotational_stiffness: float = _DEFAULT_ROT_STIFFNESS,
    ):
        """Update the Cartesian impedance reference (non-blocking).

        Uses franky's long-lived ``CartesianImpedanceTracker`` session —
        lazy-started on first call — then just refreshes the target via
        ``set_target``. This is the Cartesian analogue of ``move_joints``
        on top of ``JointImpedanceTracker``.

        The earlier implementation submitted a fresh ``CartesianImpedance
        Motion`` with ``asynchronous=True`` on every call; at 10 Hz each
        submission pre-empted the previous motion before its tracker
        finished spinning up, leaving the robot motionless until the
        stream stopped. The tracker session avoids that entirely.

        The joint impedance tracker cannot coexist with Cartesian torque
        mode, so it is stopped inside ``_ensure_cart_tracking_motion`` on
        first use.
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

        # Velocity feedforward from finite-differenced targets — without
        # it the PD eats all tracking error through position alone, which
        # at 10 Hz env.step cadence produces visible chatter/overshoot
        # even when each target itself is reachable (same reason the
        # joint path passes dq_ff to JointImpedanceTracker.set_target).
        now = time.perf_counter()
        twist = None
        if self._prev_tcp_target is not None and self._prev_tcp_target_ts is not None:
            dt = max(now - self._prev_tcp_target_ts, _DQ_MIN_DT_S)
            prev_t = self._prev_tcp_target[:3]
            prev_q = self._prev_tcp_target[3:]
            linear = (translation - prev_t) / dt
            lin_norm = float(np.linalg.norm(linear))
            if lin_norm > _TCP_LIN_VEL_LIMIT:
                linear = linear * (_TCP_LIN_VEL_LIMIT / lin_norm)
            rot_delta = R.from_quat(quat) * R.from_quat(prev_q).inv()
            angular = rot_delta.as_rotvec() / dt
            ang_norm = float(np.linalg.norm(angular))
            if ang_norm > _TCP_ANG_VEL_LIMIT:
                angular = angular * (_TCP_ANG_VEL_LIMIT / ang_norm)
            twist = franky.Twist(linear_velocity=linear, angular_velocity=angular)

        try:
            self._ensure_cart_tracking_motion(
                translational_stiffness=translational_stiffness,
                rotational_stiffness=rotational_stiffness,
            )
            self._cart_tracker.set_target(franky.Affine(T), twist=twist)
        except Exception as e:
            self._logger.warning(f"Cartesian impedance update failed: {e}")
            self._stop_tracking_motion()
            try:
                self._robot.join_motion()
            except Exception:
                pass
            self.clear_errors()
            return

        self._prev_tcp_target = np.concatenate([translation, quat])
        self._prev_tcp_target_ts = now

    def move_waypoints(
        self,
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        relative_dynamics_factor: Optional[float] = None,
    ):
        """Submit a pose-chunk as one Ruckig-planned ``CartesianWaypointMotion``.

        Args:
            positions: ``(N, 7)`` array of ``[xyz, quat_xyzw]`` waypoints.
            velocities: Optional ``(N, 6)`` array of ``[linear(3),
                angular(3)]`` twists attached to each waypoint. Non-zero
                rows tell Ruckig to blend **through** the waypoint at that
                speed rather than decelerate to zero (franky's default for
                pose-only waypoints). Typical usage: finite-difference the
                pose stream and pass all rows except the last, leaving the
                last as zeros for a graceful stop that the next chunk will
                preempt. ``angular`` is rotvec (axis * rate, rad/s) — not
                Euler rates — to match franky's ``Twist`` convention.
            relative_dynamics_factor: Optional Ruckig v/a/j scale on top
                of the robot-level factor. Defaults to the motion's own
                default (1.0) if ``None``.

        Submitted with ``asynchronous=True`` + ``return_when_finished=False``
        so the next chunk can preempt via another ``robot.move`` call
        before Ruckig finishes decelerating on the current chunk's last
        waypoint. Any impedance tracker (joint- or Cartesian-space) is
        stopped first — franky forbids control-mode switches mid-flight.
        """
        positions = np.asarray(positions, dtype=np.float64)
        assert positions.ndim == 2 and positions.shape[1] == 7, (
            f"positions must be (N, 7) [xyz, quat_xyzw]; got {positions.shape}"
        )
        if velocities is not None:
            velocities = np.asarray(velocities, dtype=np.float64)
            assert velocities.shape == (positions.shape[0], 6), (
                f"velocities must be (N, 6) matching positions; got {velocities.shape}"
            )

        n = positions.shape[0]
        if n == 0:
            return

        self._stop_tracking_motion()
        franky = self._franky

        waypoints = []
        for i in range(n):
            pos = positions[i, :3]
            quat = positions[i, 3:]
            T = np.eye(4)
            T[:3, :3] = R.from_quat(quat).as_matrix()
            T[:3, 3] = pos
            affine = franky.Affine(T)

            if velocities is not None and float(np.linalg.norm(velocities[i])) > 1e-8:
                twist = franky.Twist(
                    linear_velocity=velocities[i, :3],
                    angular_velocity=velocities[i, 3:],
                )
                target = franky.CartesianState(affine, twist)
            else:
                # Pose-only waypoint: Ruckig decelerates here. Use on the
                # final waypoint for graceful stop.
                target = affine

            waypoints.append(franky.CartesianWaypoint(target))

        kwargs = {"return_when_finished": False}
        if relative_dynamics_factor is not None:
            kwargs["relative_dynamics_factor"] = relative_dynamics_factor
        motion = franky.CartesianWaypointMotion(waypoints, **kwargs)

        try:
            self._robot.move(motion, asynchronous=True)
        except Exception as e:
            # Preempt-of-other-modes or stored errors from prior motions
            # surface here; clear and let the caller retry on the next
            # chunk instead of killing the whole rollout.
            self._logger.warning(f"move_waypoints dispatch failed: {e}")
            try:
                self._robot.join_motion()
            except Exception:
                pass
            self.clear_errors()

    def move_arm(self, position: np.ndarray):
        """Submit a Cartesian target (non-blocking).

        Called at 10 Hz from ``FrankaEnv._interpolate_move`` during a
        Cartesian reset.  Stops any active impedance tracking first,
        then submits a ``CartesianMotion`` asynchronously.
        """
        assert len(position) == 7, (
            f"Invalid position, expected 7 dims but got {len(position)}"
        )
        self._stop_tracking_motion()
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
            try:
                self._robot.join_motion()
            except Exception:
                pass
            self.clear_errors()

    def open_gripper(self):
        self._gripper.open(speed=1.0)
        self.log_debug("Open gripper")

    def close_gripper(self):
        self._gripper.close(speed=1.0)
        self.log_debug("Close gripper")

    def move_gripper(self, position: int, speed: float = 1.0):
        assert 0 <= position <= 255, (
            f"Invalid gripper position {position}, must be between 0 and 255"
        )
        self._gripper.move(position, speed)
        self.log_debug(f"Move gripper to position: {position}")

    def cleanup(self):
        """Stop tracking motion, join in-flight motion, release gripper."""
        self._stop_tracking_motion()
        try:
            self._robot.join_motion()
        except Exception:
            pass
        try:
            self._gripper.cleanup()
        except Exception:
            pass
