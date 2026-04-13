Real-World Franka Joint Control with Franky + GELLO
=====================================================

This guide explains how to use **joint-space control** for the Franka robot
via the `franky <https://github.com/TimSchneider42/franky>`_ backend, paired
with a **GELLO** teleoperation device for data collection.

.. note::

   This page extends the base :doc:`franka` and :doc:`franka_gello`
   documentation. Please read those first for general Franka setup and
   GELLO installation instructions.


Why Joint-Space Control with Franky?
-------------------------------------

The default Franka control path in RLinf uses **ROS + Cartesian impedance
control**. While effective, it has two limitations for certain use cases:

1. **Cartesian IK overhead**: Converting joint-space GELLO readings to
   Cartesian targets and back loses information and adds latency.
2. **Python GIL contention**: Any Python-based 1 kHz control loop competes
   with Ray actor dispatch, camera I/O, and garbage collection for the GIL,
   causing jitter that triggers libfranka safety reflexes.

The **franky** backend solves both problems:

- ``franky`` runs the 1 kHz ``robot.control()`` loop and impedance
  torque computation entirely inside a C++ ``std::thread``.
- All pybind11 bindings release the GIL via
  ``py::call_guard<py::gil_scoped_release>()``, so Python threads never
  block the RT path.
- Joint targets from GELLO map **directly** to robot joints — no FK/IK
  round-trip needed.

The result is **jitter-free real-time control** even under heavy Python/Ray
load, with a simple joint-space action space ideal for RL training.


Architecture
-------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Component
     - Role
     - Key Class
   * - ``FrankyController``
     - Ray Worker: connects to robot via libfranka, applies RT hardening
       (mlockall, SCHED_FIFO), exposes ``move_joints`` / ``get_state``
     - ``rlinf/envs/realworld/franka/franky_controller.py``
   * - ``FrankaJointEnv``
     - Gymnasium env: 8D action space ``[j1..j7, gripper]``, joint-limit
       and velocity-limit safety clipping
     - ``rlinf/envs/realworld/franka/franka_joint_env.py``
   * - ``GelloJointIntervention``
     - ActionWrapper: overrides policy actions with GELLO joint readings;
       optional 1 kHz direct-stream mode
     - ``rlinf/envs/realworld/common/wrappers/gello_joint_intervention.py``
   * - ``GelloJointExpert``
     - Background thread polling GELLO at ~1 kHz, exposes raw 7-DOF
       joint positions + gripper
     - ``rlinf/envs/realworld/common/gello/gello_joint_expert.py``


Two streaming modes for GELLO:

- **Step-gated** (default): ``GelloJointIntervention.action()`` returns
  GELLO joints at the env's ``step_frequency`` (~10 Hz). Simple but
  discards most GELLO samples.
- **Direct-stream** (``direct_stream=True``): a daemon thread reads GELLO
  at ~1 kHz and pushes joint targets straight to the controller, bypassing
  the env step rate gate. Paired with ``teleop_direct_stream=True`` in the
  env config so ``env.step()`` only does bookkeeping — the wrapper owns the
  1 kHz hardware loop.


Dependency Installation
------------------------

1. Install the franky backend
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # Option A: via install.sh
   bash requirements/install.sh embodied --env franka --franka-backend franky

   # Option B: manual pip install (if you already have a venv)
   pip install "franky-control>=0.15.0"

For the PREEMPT_RT kernel, CPU governor, and rtprio/memlock limits that
franky needs for deterministic behaviour, follow the detailed guide in
``requirements/embodied/franky_install.md``.

2. Install GELLO software
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Follow the GELLO installation steps in :doc:`franka_gello`:

.. code-block:: bash

   # gello driver
   git clone https://github.com/wuphilipp/gello_software.git
   cd gello_software && git submodule init && git submodule update
   pip install -e . && pip install -e third_party/DynamixelSDK/python

   # gello-teleop
   pip install git+https://github.com/RLinf/gello-teleop.git

3. Apply per-boot RT tuning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before running the controller, apply the system tuning:

.. code-block:: bash

   # CPU performance governor
   sudo bash -c 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$g"; done'

   # Allow unlimited RT scheduling
   sudo sysctl -w kernel.sched_rt_runtime_us=-1

Verify the one-time limits:

.. code-block:: bash

   ulimit -r   # expected: 99 or unlimited
   ulimit -l   # expected: unlimited


Quick Start: Data Collection
------------------------------

Use the provided config for GELLO joint-space data collection:

.. code-block:: bash

   source .venv/bin/activate
   export EMBODIED_PATH=$(pwd)/examples/embodiment
   python examples/embodiment/train_embodied_agent.py \
       --config-name realworld_collect_data_gello_joint

**Key YAML settings** (``realworld_collect_data_gello_joint.yaml``):

.. code-block:: yaml

   env:
     eval:
       use_gello_joint: True
       gello_port: "/dev/serial/by-id/your-gello-serial-port"
       override_cfg:
         joint_action_mode: "absolute"
         teleop_direct_stream: true   # enables 1 kHz streaming

   cluster:
     node_groups:
       - label: franka
         hardware:
           configs:
             - robot_ip: ROBOT_IP
               gripper_connection: /dev/ttyUSB0  # Robotiq serial port

.. list-table:: Key configuration fields
   :header-rows: 1
   :widths: 30 15 55

   * - Field
     - Default
     - Description
   * - ``use_gello_joint``
     - ``False``
     - Enable GELLO **joint-space** teleoperation (mutually exclusive with
       ``use_gello`` and ``use_spacemouse``).
   * - ``joint_action_mode``
     - ``"absolute"``
     - ``"absolute"`` — GELLO joints used directly as targets;
       ``"delta"`` — joint increments scaled by ``joint_action_scale``.
   * - ``teleop_direct_stream``
     - ``False``
     - When ``True``, a 1 kHz daemon thread in the wrapper pushes GELLO
       joints directly to the controller, and ``env.step()`` skips its own
       ``move_joints`` call.
   * - ``joint_velocity_limit``
     - ``0.5``
     - Maximum joint velocity (rad/s per joint) for safety clipping.
   * - ``gripper_connection``
     - ``null``
     - Serial port for the Robotiq gripper (e.g. ``/dev/ttyUSB0``).
       **Required** for ``FrankyController``.


Environment Configuration
---------------------------

The env YAML for joint-space control is
``examples/embodiment/config/env/realworld_franka_joint.yaml``:

.. code-block:: yaml

   env_type: realworld
   use_gello_joint: True
   use_relative_frame: False   # not applicable for joint-space

   override_cfg:
     joint_action_mode: "absolute"
     joint_velocity_limit: 0.5
     gripper_type: robotiq
     gripper_connection: /dev/ttyUSB0

The action space is 8D: ``[j1, j2, j3, j4, j5, j6, j7, gripper]``.

- In **absolute** mode, ``j1..j7`` are target joint positions (radians).
- In **delta** mode, ``j1..j7`` are scaled joint increments.


Verification
-------------

Before running data collection, verify the controller independently:

.. code-block:: bash

   python toolkits/realworld_check/test_franky_controller.py \
       --robot-ip <ROBOT_IP> \
       --gripper-type robotiq \
       --gripper-connection /dev/ttyUSB0

This interactive tool supports commands like ``getjoint``, ``home``,
``nudge``, ``open``/``close``, ``impedance``, and ``stream`` (1 kHz
streaming test).


Troubleshooting
----------------

**"SCHED_FIFO not granted" warning**

- The per-boot RT tuning was not applied, or ``/etc/security/limits.d/``
  does not grant your user ``rtprio >= 80``. Follow
  ``requirements/embodied/franky_install.md``.

**"gripper_connection must be specified" error**

- Add ``gripper_connection: /dev/ttyUSB0`` to either
  ``override_cfg`` in the env YAML or ``cluster.node_groups[].hardware.configs[]``.

**Robot buzzes or jerks under load**

``FrankyController`` uses ``JointImpedanceTrackingMotion`` for streaming,
which computes impedance torques to smoothly track the latest reference
without trajectory re-planning per call.  If buzz still occurs:

- Ensure the CPU governor is set to ``performance``.
- Check that ``kernel.sched_rt_runtime_us=-1`` is active.
- Verify with ``cyclictest -p 80 -t 1 -n -i 1000 -l 10000`` that worst-case
  latency is < 100 μs.
- Try lowering joint impedance stiffness via
  ``controller.reconfigure_compliance_params({"Kq": [300,300,300,300,150,80,30]})``.

**GELLO readings stop updating**

- Check that the GELLO device is powered and the serial port is accessible.
- ``GelloJointExpert`` marks itself as not-ready after 50 consecutive
  read errors, which causes the wrapper to fall back to the policy action.

For general Franka troubleshooting, refer to the :doc:`franka` and
`FAQ <https://rlinf.readthedocs.io/en/latest/rst_source/faq.html>`_.
