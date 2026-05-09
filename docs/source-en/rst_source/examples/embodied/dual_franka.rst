Real-World Dual-Franka: GELLO Collection, π₀.₅ SFT, Deployment
================================================================

This guide is the end-to-end recipe for the **dual-arm Franka** rig in
RLinf — bringing up two physical compute nodes, collecting bimanual
GELLO joint-space teleoperation data at 1 kHz, fine-tuning π₀.₅ on the
data in a 20-D rot6d action space, and deploying the trained policy
back to the real robot driven by a foot pedal.

This page assumes you have already read:

* :doc:`franka` — single-arm Franka basics, Ray cluster setup, the
  RealSense + SpaceMouse data-collection path. Read this first if
  none of the words "FrankaController", "FCI", or "RLINF_NODE_RANK"
  ring a bell.
* :doc:`franka_gello` — GELLO hardware install, Dynamixel SDK,
  ``gello-teleop`` package, USB-FTDI permissions.

This page focuses on what changes for the dual-arm rig:

* the **franky** low-level backend (libfranka via ``franky-control``)
  shared by both arms — replacing the legacy ROS / serl path used by
  :doc:`franka`,
* three new dual-arm environments — ``DualFrankaEnv`` (legacy 14-D
  Cartesian delta), ``DualFrankaJointEnv`` (16-D joint, used at
  collection), ``DualFrankaRot6dEnv`` (20-D TCP-rot6d, used at SFT
  and deployment),
* the **rot6d / SE(3) body-frame delta** action representation that
  replaces openpi's component-wise ``DeltaActions``,
* multi-task / resume-aware data collection driven by a 3-key foot
  pedal,
* a 2-physical-node Ray cluster where each node owns one Franka
  controller and the env worker / GPU live on node 0.


Why this rig (and what it is *not*)
-----------------------------------

The dual-Franka rig in RLinf is designed for **bimanual manipulation
SFT** — collecting high-quality teleoperation demonstrations and
fine-tuning a foundation VLA (π₀ / π₀.₅) on them. Compared with the
single-arm SAC / PPO loops in :doc:`franka`, this rig:

* **Targets imitation learning, not online RL.** The collection path
  is GELLO joint teleop; the deployment path autonomously runs the
  SFT policy with a foot pedal that owns episode boundaries. There is
  no reward labelling and no RL update on the collected data.
* **Uses one libfranka backend (``franky-control``) for both arms.**
  All low-level control runs in C++ at 1 kHz inside ``franky``;
  Python only updates references. This avoids the GIL-induced jitter
  that pure-Python control loops on top of ROS suffer from.
* **Encodes orientation as Zhou et al. 2019 6D rotation, not Euler.**
  Euler-based state/action pollutes π₀ / π₀.₅ with ±π wrap
  discontinuities (one frame's roll = +3.14 rad → next frame's roll
  = −3.14 rad ⇒ a "−2π" pseudo-delta the policy memorises). Switching
  to rot6d + SE(3) body-frame deltas removes that class of bug
  entirely.
* **Splits two arms across two compute nodes.** Each node has a
  direct Ethernet link to one Franka (FCI ``172.16.0.2``); both nodes
  reuse the same FCI IP because the two ``172.16.0.0/24`` subnets are
  physically separate (one cable per arm, one NIC per node). The two
  nodes share a LAN used only by Ray and tensor sync.

If you want online RL on a single Franka with SAC/PPO, you want
:doc:`franka`, not this page.


Hardware topology
-----------------

.. list-table::
   :header-rows: 1
   :widths: 18 35 47

   * - Node
     - Role
     - Hardware on this node
   * - **node 0** (head)
     - Ray head; env worker; left ``FrankyController``;
       actor / rollout (during eval); base camera capture
     - 1× GPU (e.g. RTX 4090) — only used at SFT and deployment;
       left Franka FR3 on a directly-cabled NIC at FCI IP
       ``172.16.0.2``;
       left GELLO Dynamixel chain (USB-FTDI);
       left Robotiq 2F-85 (USB-RS485 Modbus);
       left wrist Lumos USB-3 camera;
       base RealSense D435i (third-person view);
       PCsensor 3-key FootSwitch (optional — can live on either node)
   * - **node 1** (worker)
     - Ray worker; right ``FrankyController``; right wrist camera capture
     - Optional GPU (not used for inference);
       right Franka FR3 on its own directly-cabled NIC at FCI IP
       ``172.16.0.2``;
       right GELLO Dynamixel chain (USB-FTDI);
       right Robotiq 2F-85;
       right wrist Lumos USB-3 camera

.. warning::

   Both Franka arms answer at ``172.16.0.2``. This is **not** an IP
   collision — each arm sits in its own physically separate
   ``172.16.0.0/24`` subnet attached to a dedicated NIC on its node.
   ``ping 172.16.0.2`` from node 0 only reaches the left arm; the
   same command on node 1 only reaches the right arm. Do **not** try
   to "fix" this by renumbering the FCI — Franka Desk only exposes
   the FCI on the standard subnet, and the per-node NIC isolation is
   what keeps the two control loops independent.

Camera roles (the wrapper stack uses
``main_image_key: left_wrist_0_rgb`` so π₀.₅'s ``observation/image``
slot is the *left* wrist; ``base_0_rgb`` and ``right_wrist_0_rgb`` go
into ``observation/extra_view_image-{0,1}``):

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - Camera slot
     - Backend
     - Purpose
   * - ``base_0_rgb``
     - RealSense D435i
     - Third-person view shared by both arms (lives on node 0)
   * - ``left_wrist_0_rgb``
     - Lumos USB 3 (XVisio vSLAM)
     - Left arm wrist camera, used as π₀.₅'s primary ``image``
   * - ``right_wrist_0_rgb``
     - Lumos USB 3 (XVisio vSLAM)
     - Right arm wrist camera

Foot pedal: a 3-key PCsensor FootSwitch flashed to send key codes
``a`` / ``b`` / ``c`` (the three pedals). ``KeyboardListener`` reads
``evdev`` from a local path inside the wrapper process, and the env
worker is pinned to node 0 by the shipped placement
(``component_placement.env.placement: 0`` +
``DualFrankaConfig.node_rank: 0``) — so **the pedal must be plugged
into node 0**. If you re-place the env worker to a different node,
move the pedal to that node. Whichever node owns the pedal must
export ``RLINF_KEYBOARD_DEVICE=/dev/input/eventXX`` *before*
``ray start`` so Ray captures the path. No ``DISPLAY``, no ``xev``,
no terminal focus.


Software stack
--------------

The data path during **collection** is::

  GELLO arm (Dynamixel)                     env worker (node 0)
        │                                          │
        ▼                                          ▼
  GelloJointExpert (1 kHz read)            DualFrankaJointEnv.step
        │ ±2π unwrap                              │ 10 Hz
        ▼                                          │
  DualGelloJointIntervention                       │
   (direct_stream daemon, 1 kHz)                   │  (env.step reads
        │                                          │   state + grippers
        └─move_joints─► FrankyController(left)  ◄──┘   only — does NOT
        └─move_joints─► FrankyController(right)        forward motion)
                              │ C++ 1 kHz JointImpedanceTracker
                              ▼
                        Franka FR3

The data path during **deployment** is::

  observation (state[20] + 3 cams)
        │
        ▼
  DualFrankaRot6dInputs ─► RigidBodyDeltaActions ─► π₀ / π₀.₅
                                                       │
                                                       ▼
                            RigidBodyAbsoluteActions ◄┘  (T_abs = T_state @ T_delta)
                                       │
                                       ▼
                            DualFrankaRot6dOutputs (slice 20-D)
                                       │
                                       ▼
                  DualFrankaRot6dEnv.step (per-arm move_tcp_pose)
                                       │ C++ 1 kHz CartesianImpedanceTracker
                                       ▼
                                 Franka FR3

The two trackers (``JointImpedanceTracker`` and
``CartesianImpedanceTracker``) are **mutually exclusive** inside
``FrankyController`` — switching from collection (joint impedance)
to deployment (Cartesian impedance) automatically stops the previous
tracker, so you do not need to restart franky between sessions.


Installation (per node)
-----------------------

Repeat this section on **both** ``node 0`` and ``node 1``. The nodes
have separate checkouts and separate venvs; they only share the LAN.

1. Real-time prerequisites
~~~~~~~~~~~~~~~~~~~~~~~~~~

The franky backend assumes a PREEMPT_RT kernel + unrestricted RT
limits + per-boot tuning. All of that — kernel version pin
(``5.15.133-rt69``), ``/etc/security/limits.d`` entries, the
``performance`` governor / ``sched_rt_runtime_us=-1`` /
``ethtool -C`` per-boot tuning, the ``cyclictest`` acceptance
criterion, and the full "why franky vs a hand-written Python 1 kHz
loop" rationale — is documented in
``requirements/embodied/franky_install.md``.

For the kernel build itself, follow Franka's official guide
`Setting up the real-time kernel
<https://frankarobotics.github.io/docs/libfranka/docs/real_time_kernel.html>`_,
then come back and run through ``franky_install.md``.

2. RLinf + franky
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

   # System deps for franky (rt-tests, ethtool, eigen, pinocchio).
   bash requirements/embodied/franky_install.sh

   # RLinf Python deps + franky-control wheel from PyPI.
   bash requirements/install.sh embodied --env dual-franka --use-mirror

   source .venv/bin/activate

The ``--env dual-franka`` target pins the franky path
(``franky-control >= 0.15.0`` from PyPI) and **skips** the legacy
``serl_franka_controllers`` ROS / catkin build used by
:doc:`franka`. The ``--use-mirror`` flag is for mainland China users
(switches PyPI / GitHub / HuggingFace mirrors).

If the franky-control wheel does not match your Python + libfranka
combo, ``pip`` falls back to a source build that needs libfranka
headers. ``franky_install.sh`` already pulls eigen / poco / fmt /
pinocchio so the source build path is reasonably hands-off, but the
PyPI wheel matches Python 3.11 + libfranka 0.15.x out of the box.

.. warning::

   **Avoid libfranka 0.18.0 specifically.** Franka's official 0.18.0
   release notes flag a regression in the impedance / Cartesian
   control path; under the joint / Cartesian impedance trackers we
   use, the arm presents as severely under-torqued — limp, sagging
   under its own gravity, unable to track even gentle GELLO motion.
   Pick whichever ``libfranka`` version matches your Franka firmware
   per the official `compatibility matrix
   <https://frankarobotics.github.io/docs/compatibility.html>`_,
   just **not 0.18.0** (0.18.x patches once released, 0.17.x, and
   0.15.x have all been used in practice). Check
   ``franky.__libfranka_version__`` after install if you are unsure.

3. GELLO (env-worker node)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Both GELLO USB-FTDI cables plug into the env-worker node (**node 0**
in the shipped placement) and stay there during data collection.
``DualGelloJointIntervention`` opens both serial ports from inside
the env-worker process and reads them at ~1 kHz — routing through
the LAN to a GELLO physically wired to node 1 would blow the
real-time budget, drop samples, and cause tracker reference jumps.

For the actual install commands (``gello`` + ``gello-teleop`` +
USB-FTDI permission, with the rationale for why only the
``DynamixelSDK`` submodule is initialised), see :doc:`franka_gello`.
Run those commands on **node 0 only**, in the same venv as RLinf —
``DualGelloJointIntervention`` imports both packages in-process when
the env wrapper stack is built.

4. Foot pedal
~~~~~~~~~~~~~

The PCsensor FootSwitch is wired so its three pedals send Linux key
codes ``a`` / ``b`` / ``c`` (re-flashable via the vendor-supplied
Windows tool — flash once, the codes are stored in firmware and
persist across reboots). Verify and grant access:

.. code-block:: bash

   ls -l /dev/input/by-id/*-event-kbd
   #  expect: usb-PCsensor_FootSwitch-event-kbd → ../eventXX

   sudo chmod 666 /dev/input/eventXX

   # Export BEFORE `ray start` so Ray captures the path:
   export RLINF_KEYBOARD_DEVICE=/dev/input/eventXX

The ``KeyboardListener`` reads ``evdev`` directly, supports auto-reopen
on ``ENODEV`` (handles a USB hiccup mid-session), and uses an
edge-triggered press queue so a tap shorter than the polling period
is never missed. You can also export the device in
``ray_utils/realworld/setup_before_ray.sh`` to persist across sessions.


Hardware verification
---------------------

Before bringing up Ray, smoke-test each hardware piece per node.

Cameras
~~~~~~~

.. code-block:: bash

   # Health check: enumerate, USB negotiation, single-frame read,
   # plus a 60-second 3-pane live preview.
   python -m toolkits.dual_franka.check_cameras

   # Skip the preview when running over SSH without an X display.
   python -m toolkits.dual_franka.check_cameras --no-stream

The script flags USB-2 fallback for Lumos (XVisio vSLAM cameras
*sometimes* negotiate as 2.0 on a marginal cable — they will then
silently produce empty frames at ``select()`` timeout) and walks the
standard ``/dev/v4l/by-id`` recovery path (double-open + I420 buffer
warmup) used in ``LumosCamera``. It also lists processes holding
``/dev/video*`` via ``fuser`` so you can identify the leaked Ray
worker that needs killing before the next launch.

GELLO
~~~~~

.. code-block:: bash

   python -m gello_teleop.gello_expert \
       --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_<id>-if00-port0

You should see continuously updating joint readings as you move the
GELLO arm. If readings freeze or jump by ±2π, run the calibration
toolkit (covered in the next section).

Per-arm Franka link
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   FRANKA_ROBOT_IP=172.16.0.2 \
   FRANKA_GRIPPER_TYPE=robotiq \
   FRANKA_GRIPPER_PORT=/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_<id>-if00-port0 \
       python toolkits/realworld_check/test_franky_controller.py

Inside the REPL:

* ``getjoint`` — print current joint angles
* ``home`` — synchronous reset to ``HOME_JOINTS``
* ``hold 30`` — hold for 30 s, listen for buzzing
* ``stream 4 0.001 500`` — push 500 J4 += 0.001 rad commands at 1 kHz
  (streaming preemption stress test)
* ``open`` / ``close`` — gripper sanity

Acceptance criteria (no buzz at hold, > 800 Hz on stream, clean
``home`` from any legal pose) are enumerated in
``requirements/embodied/franky_install.md``. Run separately on each
node, against its own arm. **Both arms must pass before you bring up
Ray.**


GELLO calibration
-----------------

GELLO maps Dynamixel motor positions to Franka joint angles via a
``DynamixelRobotConfig`` block (joint signs + offsets). Each GELLO
unit needs its own config. Two scripts cover the workflow:

1. **Calibrate** (once per GELLO unit, or after replacing a motor):

   .. code-block:: bash

      bash examples/embodiment/gello_calibrate.sh

   The script moves the robot to two known poses (``POSE_A`` =
   Franka home, ``POSE_B`` = π/4 multiples), prompts you to physically
   match the GELLO leader to each pose, then solves
   ``joint_signs`` and ``joint_offsets`` from the difference. Output
   is a paste-ready ``DynamixelRobotConfig`` block to drop into
   ``gello_software/gello/agents/gello_agent.py``::

       "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_<id>-if00-port0":
           DynamixelRobotConfig(
               joint_ids=(1, 2, 3, 4, 5, 6, 7),
               joint_offsets=(...),
               joint_signs=(...),
               gripper_config=(8, ..., ...),
               baudrate=1_000_000,
           ),

   The ``gello`` package is editable-installed, so no reinstall is
   needed after pasting — just restart the next process that imports
   ``gello``.

2. **Align** (run when the GELLO leader and the arm visibly disagree
   — e.g. someone hand-moved the arm, the rig has been idle for a
   while, or you just want to confirm before a fresh collection
   session):

   .. code-block:: bash

      bash examples/embodiment/gello_align.sh

   Drives the robot to a fixed alignment HOME pose (J4 = −π/2,
   J6 = +π/2 etc.), then walks you through aligning J1 → J7 one at
   a time with a per-joint progress bar and live deltas. As soon as
   each joint stays within ±0.10 rad for 8 consecutive frames, it
   advances to the next.

Both scripts auto-discover the local Robotiq port by globbing
``/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_*-if00-port0``, so you do
not need to know whether you are on the left or right node.


Hardware YAML
-------------

The dual-Franka hardware contract lives in
``examples/embodiment/config/env/realworld_franka_joint_dual.yaml``
(joint collection) and
``examples/embodiment/config/env/realworld_franka_rot6d_dual.yaml``
(rot6d eval). The cluster-level wiring is identical; the override
overlays differ.

Cluster block (collection):

.. code-block:: yaml

   cluster:
     num_nodes: 2
     component_placement:
       env:
         node_group: franka
         placement: 0
     node_groups:
       - label: franka
         node_ranks: 0-1
         hardware:
           type: DualFranka
           configs:
             # Same FCI IP for both arms — see "Hardware topology" warning.
             - left_robot_ip:  "172.16.0.2"
               right_robot_ip: "172.16.0.2"
               base_camera_serials: ["<librealsense serial>"]
               base_camera_type:  realsense
               left_camera_serials:  ["usb-XVisio_..._<sn-left>-video-index0"]
               left_camera_type:  lumos
               right_camera_serials: ["usb-XVisio_..._<sn-right>-video-index0"]
               right_camera_type: lumos
               left_gripper_type:  robotiq
               right_gripper_type: robotiq
               left_gripper_connection:  "/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_<id-l>-if00-port0"  # on node 0
               right_gripper_connection: "/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_<id-r>-if00-port0"  # on node 1
               left_controller_node_rank:  0
               right_controller_node_rank: 1
               node_rank: 0  # env worker + cameras live on node 0

Field-by-field reference for ``DualFrankaConfig``
(``rlinf/scheduler/hardware/robots/dual_franka.py``):

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Field
     - Meaning
   * - ``left_robot_ip`` / ``right_robot_ip``
     - FCI IPs (validated as IP addresses at config-load time). Same
       value is fine because the two subnets are physically separate.
   * - ``left_camera_serials`` / ``right_camera_serials`` / ``base_camera_serials``
     - Serial numbers as the SDK reports them. RealSense uses the
       ``librealsense`` serial (e.g. ``941322072906``);
       Lumos uses the ``/dev/v4l/by-id`` USB descriptor
       (``usb-XVisio_..._video-index0``);
       ZED uses the ZED SDK device list serial.
   * - ``camera_type``
     - Default backend (``realsense`` / ``zed`` / ``lumos``) when a
       per-slot type is not given.
   * - ``base_camera_type`` / ``left_camera_type`` / ``right_camera_type``
     - Per-slot override. Lets you mix backends on the same rig — the
       reference rig uses ``realsense`` for the base view and
       ``lumos`` for both wrists.
   * - ``left_gripper_type`` / ``right_gripper_type``
     - For this Franky path use ``robotiq`` (RS-485 Modbus on
       USB-FTDI, no ROS dependency). The ``franka`` Franka-Hand
       backend in ``common/gripper/franka_gripper.py`` requires a
       ROS controller and is **not** wired up under
       ``FrankyController`` — it raises at construction time.
   * - ``left_gripper_connection`` / ``right_gripper_connection``
     - Stable ``/dev/serial/by-id`` path. ``by-id`` is **strongly**
       recommended over ``/dev/ttyUSB*`` because the ``ttyUSB*`` index
       is shuffled on reboot and on hot-plug; ``by-id`` is keyed on
       the FTDI chip's burned-in serial.
   * - ``left_controller_node_rank`` / ``right_controller_node_rank``
     - Node rank where each arm's ``FrankyController`` Ray actor is
       placed. Set to ``0`` and ``1`` for the canonical "one arm per
       node" topology. Both can also be set to ``0`` for a single-node
       dev rig with both arms cabled to the same machine.
   * - ``node_rank``
     - Where the env worker (camera capture, wrappers, action
       dispatch) runs. Pin to the node that owns the base camera.

The override overlay sets per-arm safety boxes
(``ee_pose_limit_min/max``), reset poses, and either
``joint_action_mode`` (joint env) or rot6d-specific fields. See the
shipped configs for full examples.


Ray cluster bring-up
--------------------

Ray captures the active Python interpreter and the *exported
environment variables* when ``ray start`` runs, and worker actors
inherit that snapshot. Packages added to the venv after ``ray start``
are picked up at next import (Ray does not freeze ``site-packages``),
but env vars are not — anything you forget to export before
``ray start`` will be missing from the worker's environment forever.
Order:

1. **On every node**: activate the venv, export
   ``RLINF_NODE_RANK`` and (optionally) ``RLINF_COMM_NET_DEVICES``,
   and export ``RLINF_KEYBOARD_DEVICE`` if this node owns the foot
   pedal. Verify ``franky``, ``gello``, ``gello_teleop`` import.
2. **Then** ``ray start`` — head on node 0, worker on node 1.

Templates are provided:

.. code-block:: bash

   # node 0 (Ray head)
   source .venv/bin/activate
   export RLINF_NODE_RANK=0
   export RLINF_KEYBOARD_DEVICE=/dev/input/eventXX  # if pedal lives here
   bash ray_utils/realworld/start_ray_node0.sh   # edit HEAD_IP first

.. code-block:: bash

   # node 1 (Ray worker)
   source .venv/bin/activate
   export RLINF_NODE_RANK=1
   bash ray_utils/realworld/start_ray_node1.sh   # edit HEAD_IP / WORKER_IP first

Verify on node 0:

.. code-block:: bash

   ray status
   # expected: 2 nodes, both ALIVE, with the cluster GPU/CPU resources
   # you expect.

.. warning::

   The two physical nodes have **independent checkouts**. After every
   code change on node 0, sync to node 1 (``rsync -av --delete RLinf/
   <node1>:/path/to/RLinf/``) **and** restart Ray on the affected
   node so the new code is captured by Ray. Forgetting this leads to
   cryptic ``ImportError`` or "feature works on node 0 but not on
   node 1" symptoms.


Data Collection (GELLO joint-space)
------------------------------------

The collection path uses ``DualFrankaJointEnv-v1`` with
``teleop_direct_stream: true``. A daemon thread inside
``DualGelloJointIntervention`` reads the GELLO Dynamixel servos at
~1 kHz and pushes joint targets straight into both
``FrankyController`` actors (which forward them to franky's
``JointImpedanceTracker``). ``env.step`` runs at 10 Hz and only reads
state, fires gripper open/close on edge transitions, and dispatches
camera captures — it does **not** call ``move_joints``.

Why direct-stream and not env-step-gated? At 10 Hz, sampling a
freehand teleop trajectory under-samples high-frequency wrist motion.
The 1 kHz daemon path samples the operator's actual hand motion at
GELLO's native rate, then env.step reads the *resulting* joint state
at 10 Hz — so the dataset captures what the operator did, not a
1-tap-per-100ms aliased view of it. The 10 Hz state read is what
π₀.₅ then sees as input.

Configuration
~~~~~~~~~~~~~

Use the shipped config:
``examples/embodiment/config/realworld_collect_data_gello_joint_dual_franka.yaml``.

Key fields you will edit before each session:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - YAML field
     - Meaning
   * - ``runner.num_data_episodes``
     - Target episode count. Combined with
       ``data_collection.resume``, this is the *total target across
       all sessions*, not just this session.
   * - ``env.eval.left_gello_port`` / ``right_gello_port``
     - Override here if you swap GELLO units this session. Otherwise
       inherits from the env yaml.
   * - ``env.eval.override_cfg.task_description``
     - Prompt recorded into every frame's ``task`` field. In
       multi-task mode each task overrides this with its own prompt.
   * - ``env.eval.override_cfg.joint_action_mode``
     - ``absolute`` (1:1 GELLO mapping for collection); ``delta`` for
       offline RL on the same env class.
   * - ``env.eval.override_cfg.teleop_direct_stream``
     - ``true`` for the 1 kHz GELLO daemon path. Setting this to
       ``false`` falls back to env.step gating and is **not the
       recommended collection path**.
   * - ``data_collection.save_dir``
     - Base directory for the dataset. Each session by default writes
       to ``${runner.logger.log_path}/collected_data``; override on
       the command line to accumulate sessions under one root.
   * - ``data_collection.resume``
     - ``true`` to seed the episode counter from existing ``id_*``
       shards under ``save_dir/rank_0``.
   * - ``data_collection.tasks``
     - Optional. List of ``{name, prompt}`` entries for round-robin
       multi-task collection. Comment out for single-task mode.

Running
~~~~~~~

Three terminals once Ray is up.

**Terminal 1** — launcher (on node 0):

.. code-block:: bash

   bash examples/embodiment/collect_data.sh \
        realworld_collect_data_gello_joint_dual_franka 2>&1 \
        | tee logs/collect.log

   # Optional Hydra overrides go after the config name, e.g.:
   #   bash examples/embodiment/collect_data.sh \
   #        realworld_collect_data_gello_joint_dual_franka \
   #        env.eval.data_collection.save_dir=/data/dual_franka_v1 \
   #        env.eval.data_collection.resume=true \
   #        runner.num_data_episodes=200 \
   #        2>&1 | tee logs/collect.log

**Terminal 2** — live progress monitor (on node 0):

.. code-block:: bash

   python toolkits/realworld_check/collect_monitor.py logs/collect.log

The monitor exists because the collector runs as a Ray worker, whose
stdout is batched (~500 ms) by Ray's log monitor — that batching
breaks ``tqdm``'s ``\r`` in-place refresh. The monitor tails the log
file in its own TTY and renders a clean tqdm bar showing success
count, latest keyboard event, and last reward. By default it replays
the existing log at startup so episodes saved before the monitor
came up are reflected in the bar's initial position; pass
``--no-replay`` to start from EOF instead. ``--source=worker``
(default ``auto``) tails the Ray per-worker stdout file under
``/tmp/ray/session_latest/logs/worker-*-<pid>.out`` to bypass log
monitor batching entirely (~1–2 min faster), falling back to the
tee log when the worker file is on a different node.

**Terminal 3** — optional camera preview (useful when re-positioning
props between episodes):

.. code-block:: bash

   python toolkits/dual_franka/check_cameras.py --stream-secs 9999

Per-episode workflow
~~~~~~~~~~~~~~~~~~~~

Once both arms are tracking GELLO (``gello_align.sh`` reports
"ALL JOINTS ALIGNED"):

1. **(pre)** The arms align to the GELLO operator pose at every
   ``reset`` (``KeyboardStartEndWrapper`` +
   ``DualGelloJointIntervention`` skip the home slew via
   ``options["skip_reset_to_home"]=True``). The arms stay where the
   operator's hands hold them.
2. **press ``a``** to begin recording the current pose as frame 0.
3. **demonstrate the task.** Per-frame data is buffered. The robot
   tracks the GELLO leader at 1 kHz; cameras capture at 10 Hz.
4. **press ``b``** at sub-task boundaries — increments the per-frame
   ``segment_id`` (debounced at 1 s; back-to-back presses are
   ignored). Use this to mark "approach" / "grasp" / "transfer" /
   "place" so a downstream policy can be conditioned on segment id.
5. **press ``c``** to mark success → reward = 1.0,
   ``terminated=True``, ``CollectEpisode`` flushes the buffer to a
   LeRobot shard.
6. **press ``a`` again** during recording to **abort** — drops the
   buffer, returns to ``pre`` phase. The arms stay where they are
   (no home reset) so the operator can immediately re-attempt
   without GELLO discontinuity.

Multi-task collection
~~~~~~~~~~~~~~~~~~~~~

Set ``data_collection.tasks`` (commented out by default in the shipped
config):

.. code-block:: yaml

   data_collection:
     tasks:
       - name: pour_water
         prompt: "Pour water from the cup into the bowl"
       - name: pick_cup
         prompt: "Pick up the cup from the table"

Episodes round-robin across tasks: ep0 → ``tasks[0]``, ep1 →
``tasks[1]``, ep2 → ``tasks[0]``, … Each task gets its own LeRobot
dataset rooted at ``<save_dir>/<name>/``, and the per-frame ``task``
field is recorded with that task's prompt.

Aborts (``a`` mid-recording) **do not** consume a task slot — only
successes (``c``) advance the rotation. This keeps task counts
balanced even when one task is harder than another.

In multi-task mode ``resume`` is **ignored**: each session starts
fresh ``id_*`` shards under each task's directory.

Output format
~~~~~~~~~~~~~

LeRobot v2.1, one shard per session under
``<save_dir>/[<task>/]rank_0/id_{N}/``:

* ``meta/info.json`` — feature schema. ``state`` is fixed-size
  ``[68]``; ``actions`` is ``[16]`` for joint or ``[20]`` for rot6d.
* ``meta/episodes_stats.jsonl`` — per-episode min / max / mean / std
  for ``state`` and ``actions``.
* ``data/episode_NNNNNN.parquet`` — per-step rows.

Per-frame fields:

* ``state`` — ``DualFrankaJointEnv.STATE_LAYOUT`` flat concat
  ``[gripper_position(2), joint_position(14), joint_velocity(14),
  tcp_force(6), tcp_pose(14), tcp_torque(6), tcp_vel(12)]`` = 68.
  The first 2 slots are ``[L_grip, R_grip]`` to match the
  rot6d-policy's ``_rearrange_state`` slicing assumption.
* ``actions`` — what the GELLO daemon dispatched at each step
  (16-D for joint mode: ``[L_jpos(7), L_grip, R_jpos(7), R_grip]``).
* ``image`` — ``left_wrist_0_rgb`` (the ``main_image_key``).
* ``wrist_image-0`` / ``wrist_image-1`` — fanned-out per-arm wrist
  views via ``CollectEpisode._expand_multi_view_images``.
* ``extra_view_image-0`` / ``extra_view_image-1`` — base + right
  wrist views, ordered as ``("base_0_rgb", "right_wrist_0_rgb")``.
  The order is asserted in
  ``DualFrankaRot6dInputs._extract_extra_views`` so a rig rename
  fails loudly instead of silently swapping camera meanings.
* ``task`` — the prompt for this episode's task.
* ``is_success`` — sticky flag; ``True`` for every frame of an
  episode that ended via pedal ``c``.
* ``done`` — only the *last* frame of an episode has ``True``.
* ``intervene_flag`` — always ``True`` for collection (the GELLO
  daemon's command is the action).
* ``segment_id`` — uint8; advances on pedal ``b``.

Resume
~~~~~~

Set ``data_collection.resume: true`` and re-launch with the same
``save_dir`` — ``CollectEpisode._count_existing_lerobot_episodes``
sums ``total_episodes`` across existing ``id_*`` shards (skipping
malformed shards, so an aborted session that left a corrupt shard
does not break resume), and the new session writes to a fresh
``id_{N}`` shard so previously-finalised data is never touched.

The progress bar's initial position is seeded from the existing
count, so ``num_data_episodes: 200`` plus 50 already-saved successes
means the new session targets 150 more.

Combine with task-mode by overriding ``save_dir`` per-session:

.. code-block:: bash

   bash examples/embodiment/collect_data.sh \
        realworld_collect_data_gello_joint_dual_franka \
        env.eval.data_collection.save_dir=/data/dual_franka_v1 \
        env.eval.data_collection.resume=true \
        2>&1 | tee logs/collect.log


Backfill rot6d and norm_stats
-----------------------------

The collected dataset cannot be fed directly to π₀.₅ — it must be
rewritten offline:

* **State is 68-D** — the ``DualFrankaJointEnv.STATE_LAYOUT``
  concat: ``gripper_position(2) + joint_position(14) +
  joint_velocity(14) + tcp_force(6) + tcp_pose(14) + tcp_torque(6) +
  tcp_vel(12)`` = 68.
* **Actions are 16-D joint targets** (GELLO streams joint angles,
  not TCP poses; per arm: ``[j(7) + grip(1)]``).

π₀.₅ on the SFT path expects **20-D rot6d** for both state and
action: ``[xyz(3) + rot6d(6) + grip(1)] × 2 arms``. So the dataset
needs an offline rewrite: actions widen 16 → 20; state stays 68-D
on disk (so the parquet schema is untouched) but the first 20 slots
are overwritten with the rot6d prefix — π₀.₅'s ``_rearrange_state``
only slices ``[:20]``, the remaining 48 slots are ignored at
training time.

.. code-block:: bash

   export PYTHONPATH=$(pwd)
   python toolkits/dual_franka/backfill_rot6d.py \
       --src $HF_LEROBOT_HOME/<repo_id>/joint_v1 \
       --dst $HF_LEROBOT_HOME/<repo_id>/rot6d_v1

What ``backfill_rot6d.py`` does:

* **State prefix rewrite.** ``state[:, 0:20]`` becomes
  ``[L_grip, R_grip, L_xyz(3), L_rot6d(6), R_xyz(3), R_rot6d(6)]``,
  derived from the original ``state[:, 36:43]`` (left tcp_pose) and
  ``state[:, 43:50]`` (right tcp_pose), with quaternions mapped
  through ``quat_xyzw_to_rot6d``. The remaining ``state[:, 20:68]``
  slots (joint velocity tail, tcp_force, full tcp_pose, tcp_torque,
  tcp_vel) are preserved verbatim — π₀.₅'s ``_rearrange_state``
  slices to ``:20`` and ignores the rest.
* **Actions widen 16 → 20.** Each frame's 20-D vector is
  ``[L_xyz, L_rot6d, L_grip, R_xyz, R_rot6d, R_grip]``, where the
  xyz / rot6d halves come from the **next-frame** ``tcp_pose`` (best
  available proxy for "commanded TCP target the GELLO operator was
  steering toward at this step"; running FK on the joint command
  would require a live ``franky.Model`` binding which is not
  available offline). The gripper slot is the original trigger from
  ``action[7]`` / ``action[15]``. The last frame repeats current
  state (no motion).
* **Schema patch.** The HuggingFace metadata embedded in the parquet
  schema's ``actions.length`` field is bumped from 16 to 20;
  per-episode ``state`` / ``actions`` stats are recomputed.
* **Idempotency.** If you point ``--src`` at an already-backfilled
  dataset, the script refuses with a clear error rather than
  silently re-writing.

Once backfilled, compute norm stats:

.. code-block:: bash

   export HF_LEROBOT_HOME=/path/to/lerobot_root
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi05_dualfranka_rot6d \
       --repo-id <repo_id>/rot6d_v1

This iterates the dataset through the SFT data pipeline
(``RepackTransform`` → ``DualFrankaRot6dInputs`` →
``RigidBodyDeltaActions``) and saves ``norm_stats.json`` under
``<openpi_assets_dirs>/<data_config.repo_id>/``. The same
``<repo_id>`` becomes the lookup key the rollout worker uses to
load these stats at deployment — see "Checkpoint / norm_stats
lock-step" below for the full path-resolution rule.

The norm stats must be recomputed **after** backfill, not before —
they need to see the body-frame deltas the policy will actually
predict, not the absolute targets on disk.


SFT (π₀.₅, rot6d_v1)
--------------------

Configuration
~~~~~~~~~~~~~

``examples/sft/config/dual_franka_rot6d_sft_openpi.yaml``. Edit
before launch:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Field
     - Set to
   * - ``data.train_data_paths``
     - LeRobot root containing your backfilled rot6d_v1 dataset.
       This value is exported as ``HF_LEROBOT_HOME`` by
       ``train_vla_sft.py`` before validation, so openpi's
       data loader picks it up automatically.
   * - ``actor.model.model_path``
     - π₀ / π₀.₅ base ckpt (the torch-converted weights, e.g.
       ``checkpoints/torch/pi05_base/``).
   * - ``actor.model.action_dim``
     - ``20`` (must match the rot6d data layout).
   * - ``actor.model.num_action_chunks``
     - ``20`` (matches the model's ``action_horizon`` from the
       ``pi05_dualfranka_rot6d`` TrainConfig).
   * - ``actor.model.openpi.config_name``
     - ``pi05_dualfranka_rot6d``.
   * - ``actor.optim.lr``
     - ``7.91e-6`` is a reasonable default for π₀.₅ on this dataset.
   * - ``actor.fsdp_config.sharding_strategy``
     - ``full_shard`` (``hybrid_shard`` if you have >8 GPUs and
       want the inter-replica all-reduce instead of all-gather).
   * - ``runner.save_interval``
     - ``500`` (steps); checkpoints land in
       ``${runner.logger.log_path}/checkpoints/global_step_<N>/``.

Launch
~~~~~~

.. code-block:: bash

   # Single node, 4 GPU slots — cluster.num_nodes: 1,
   # component_placement.actor,env,rollout uses GPUs 0..3.
   bash examples/sft/run_vla_sft.sh dual_franka_rot6d_sft_openpi

The runner writes checkpoints every ``runner.save_interval`` steps
(default 500) to ``<log_path>/checkpoints/global_step_<N>/`` with
this layout:

.. code-block:: text

   <log_path>/checkpoints/global_step_<N>/
   ├── actor/
   │   └── model_state_dict/
   │       └── full_weights.pt
   └── <asset_id>/                        # e.g. "YinuoTHU/Dual-franka-rot6d"
       └── norm_stats.json                # pinned norm stats for inference

Real-world deployment reads the policy weights from
``<model_path>/actor/model_state_dict/full_weights.pt`` and norm
stats from ``<model_path>/<asset_id>/norm_stats.json``.

Body-frame SE(3) delta math
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The transform stack (set up in ``DualFrankaRot6dDataConfig.create``):

.. code-block:: text

   on disk:                  state[68] (rot6d prefix in [:20]) +
                             actions[H, 20]  (absolute targets)
                                            │
                                            ▼
   DualFrankaRot6dInputs:    slice state to [:20], rearrange to training layout
                             ([L_xyz, L_rot6d, L_grip, R_xyz, R_rot6d, R_grip]);
                             pad state and actions to action_dim;
                             extract base / right_wrist images (asserts order)
                                            │
                                            ▼
   RigidBodyDeltaActions:    for each pose6d slot in DUAL_ARM_ROT6D_LAYOUT,
                             T_state = pose_to_SE3(state.xyz, state.rot6d)
                             T_abs   = pose_to_SE3(actions.xyz, actions.rot6d)
                             T_delta = inv(T_state) @ T_abs
                             write T_delta back into actions.xyz / .rot6d
                             scalar_abs slots (gripper, idx 9 and 19) untouched
                                            │
                                            ▼
                                        π₀ / π₀.₅
                                            │
                                            ▼ (inference)
   RigidBodyAbsoluteActions: T_abs = T_state @ T_delta
                             write T_abs back into actions.xyz / .rot6d
                                            │
                                            ▼
   DualFrankaRot6dOutputs:   slice predictions back to 20-D
                                            │
                                            ▼
                              env.step (move_tcp_pose per arm)

Why SE(3) instead of openpi's component-wise ``DeltaActions``: rot6d
is the first two columns of an ``SO(3)`` matrix flattened as
``[r1; r2]``. Componentwise subtraction
``rot6d_delta = rot6d_abs − rot6d_state`` produces a non-orthogonal
``r1, r2`` — Gram-Schmidt would need an extra projection just to
map back into ``SO(3)``, and the resulting rotation has nothing to do
with the original "body-frame delta" semantics. Operating in
``SE(3)`` keeps the rotation part exactly orthonormal end-to-end and
preserves the geometric interpretation.

The body-frame convention (``T_delta = inv(T_state) @ T_abs``) means
the delta is expressed *in the current end-effector frame*, so
distribution shift from a slightly different reset pose at deployment
does not drag the predicted action off the manifold. The action
layout is configured in
``DUAL_ARM_ROT6D_LAYOUT`` in
``rlinf/models/embodiment/openpi/transforms/rigid_body_delta.py``::

   ({"kind": "pose6d", "xyz": slice(0, 3),  "rot6d": slice(3, 9)}),
   ({"kind": "scalar_abs", "idx": 9}),                        # L gripper
   ({"kind": "pose6d", "xyz": slice(10, 13), "rot6d": slice(13, 19)}),
   ({"kind": "scalar_abs", "idx": 19})                        # R gripper

Action slots ``9`` and ``19`` (the per-arm grippers) are *absolute*
trigger signals in [-1, 1]; only the two pose6d slots take the SE(3)
round-trip.

The transform is round-trip-tested by
``tests/unit_tests/test_rigid_body_delta.py``: 50 random
``(state, abs chunk)`` pairs, ``delta → absolute`` recovers the
original chunk to ``atol=1e-5``; the gripper channels and pad tail
are asserted untouched.


Real-world deployment
---------------------

Same Ray cluster as collection. Different entry script + config.

Configuration
~~~~~~~~~~~~~

``examples/embodiment/config/realworld_eval_dual_franka.yaml``.
Placeholders are flagged with ``# Replace:`` comments. Most-edited
fields:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Field
     - Set to
   * - ``rollout.model.model_path``
     - ``<sft_log>/checkpoints/global_step_<N>/`` — must contain
       ``actor/model_state_dict/full_weights.pt`` and
       ``<data_config.repo_id>/norm_stats.json`` (see
       "Checkpoint / norm_stats lock-step" below for how
       ``data_config.repo_id`` is resolved).
   * - ``actor.model.openpi_data.repo_id``
     - Forwarded as ``data_kwargs`` to ``get_openpi_config``; this
       overrides ``data_config.repo_id``, which is the lookup key
       for ``norm_stats.json`` at deployment. Keep it consistent
       with what ``calculate_norm_stats.py --repo-id`` was given.
   * - ``env.eval.override_cfg.task_description``
     - Prompt the policy was trained against.
   * - ``env.eval.override_cfg.joint_reset_qpos``
     - Recompute from your SFT dataset's first-frame joint means;
       stale values push the initial obs out of training distribution.
   * - ``env.eval.override_cfg.target_ee_pose`` / ``reset_ee_pose``
     - Match the workspace used at collection.
   * - ``cluster.node_groups[*].env_configs[0].python_interpreter_path``
     - Path to the openpi venv's Python on node 0 (the env worker /
       rollout actor read this to ensure imports resolve).

Hardware ``configs`` should be identical to the collection yaml —
same IPs, same camera serials, same gripper connections. The
wrappers attach based on ``env.eval.use_*`` flags, so the only
non-hardware difference between collection and eval yaml is:

* ``use_gello_joint: false`` (collection: ``true``)
* ``keyboard_reward_wrapper: eval_control`` (collection:
  ``start_end``)
* ``use_relative_frame: false`` — required for rot6d eval, since
  ``DualRelativeFrame`` would corrupt the rot6d state.

Launch
~~~~~~

.. code-block:: bash

   bash examples/embodiment/run_realworld_eval.sh realworld_eval_dual_franka

   # Optional Hydra overrides:
   #   bash examples/embodiment/run_realworld_eval.sh realworld_eval_dual_franka \
   #        rollout.model.model_path=/sft/global_step_5000 \
   #        env.eval.override_cfg.task_description="pour water"

Eval workflow (per episode)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

``KeyboardEvalControlWrapper`` swaps the keyboard wrapper for one
tuned to autonomous rollout:

1. After ``env.reset()``, both arms hold the reset pose.
   ``env.step()`` is intercepted in **idle** mode — it does *not*
   forward to the inner env (so the impedance controller keeps the
   target from the last reset; the arms stay physically still while
   you stage the workpiece). The wrapper still returns the most
   recent observation so the policy's chunked-rollout loop keeps
   cycling without committing fresh joint commands.
2. Press ``a`` → wrapper switches to **running**. The next
   ``env.step`` forwards to the policy's chunked rollout.
3. Press ``c`` → success: ``terminated=True``, ``reward=1.0``,
   ``info["eval_result"]="success"``. The wrapper internally calls
   ``env.reset()`` so the arms drive back to home immediately, then
   sits idle again — this is what makes the pedal feel "live"
   even when the eval ``env_worker`` runs with ``auto_reset=False``.
4. Press ``b`` → failure: same as success but ``reward=0.0``,
   ``info["eval_result"]="failure"``.
5. While running, the wrapper forces ``terminated`` / ``truncated``
   to ``False`` unless the pedal fires — the env's own
   ``max_episode_steps`` does not cut off the policy. Set
   ``max_episode_steps`` large enough that the pedal is always the
   boundary owner (the shipped config uses ``10000``).

Checkpoint / norm_stats lock-step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The single most common deployment failure is a ``norm_stats``
mismatch. The rollout worker resolves the norm_stats path inside
``rlinf/models/embodiment/openpi/__init__.py``::

   pinned_path = <model_path>/<data_config.asset_id>/norm_stats.json
   if pinned_path exists:
       use it
   else:
       fall back to data_config.norm_stats with a loud warning

``data_config.asset_id`` is what
``DualFrankaRot6dDataConfig.create()`` resolves at SFT time (it
inherits the ``AssetsConfig.asset_id`` field, which falls back to
``data_config.repo_id`` when not set explicitly). The same key is
used by ``calculate_norm_stats.py``, which writes its output under
``<openpi_assets_dirs>/<data_config.repo_id>/``. So the path under
``<model_path>/...`` must match what the SFT pipeline actually used.

In practice:

* If you keep the shipped ``actor.model.openpi_data.repo_id``
  (``YinuoTHU/Dual-franka-rot6d``), norm_stats live under
  ``<model_path>/YinuoTHU/Dual-franka-rot6d/norm_stats.json``.
* If you override ``actor.model.openpi_data.repo_id`` (passed
  through as ``data_kwargs``) to point at a local backfilled
  dataset, ``data_config.repo_id`` is replaced and the lookup key
  becomes the new value. **Run** ``calculate_norm_stats.py``
  **with the same** ``--repo-id`` **and copy the result into**
  ``<model_path>/<that_repo_id>/norm_stats.json``.

Verify before launch:

.. code-block:: bash

   # Whatever value resolves on this rollout run — grep the SFT log for
   # the exact path the rollout worker is going to look up:
   grep "norm_stats" <sft_log>/run_embodiment.log | tail
   # Or just confirm the file exists at the path the model_path implies:
   find <model_path> -maxdepth 3 -name norm_stats.json
   ls <model_path>/actor/model_state_dict/full_weights.pt

Mismatched stats silently produce out-of-distribution states; the
policy will collapse to a single canned trick (drift to a corner,
stuck-open gripper, etc.) without any error message. The fallback
path *does* log a ``"norm_stats fallback: ... verify they match
training or inference will be wrong"`` warning — grep for it
before you assume the rollout is healthy.


Configuration cheat-sheet
-------------------------

Joint collection
(``realworld_collect_data_gello_joint_dual_franka.yaml``)

* ``env.eval.max_episode_steps: null`` — let the pedal own
  boundaries.
* ``env.eval.override_cfg.teleop_direct_stream: true`` — 1 kHz
  GELLO daemon.
* ``env.eval.override_cfg.joint_action_mode: absolute`` — direct
  GELLO mapping.
* ``data_collection.fps: 10`` — collection step rate.
* ``data_collection.only_success: true`` — drop aborted episodes.
* ``data_collection.finalize_interval: 100`` — flush ``info.json``
  every 100 saves.
* ``data_collection.export_format: lerobot``.
* ``data_collection.robot_type: dual_FR3`` — recorded in dataset
  metadata.

Rot6d eval (``realworld_eval_dual_franka.yaml``)

* ``env.eval.use_relative_frame: false`` — rot6d state has no euler
  frame, ``DualRelativeFrame`` would corrupt it.
* ``env.eval.use_gello: false`` / ``use_gello_joint: false`` /
  ``use_spacemouse: false`` — fully autonomous rollout.
* ``env.eval.keyboard_reward_wrapper: eval_control``.
* ``env.eval.override_cfg.success_hold_steps: 1`` — terminate
  immediately on pedal.
* ``algorithm.eval_rollout_epoch: 100`` — rollout this many epochs
  before the runner exits.
* ``rollout.backend: huggingface`` — π₀ / π₀.₅ inference path.

Per-arm impedance overrides (env vars, read into module-level
constants when ``franky_controller`` is imported by the Ray worker
— so export them on the controller node *before* ``ray start``)

* ``RLINF_CART_K_T`` — translational stiffness, default ``1000``
  N/m.
* ``RLINF_CART_K_R`` — rotational stiffness, default ``50`` Nm/rad.
* ``RLINF_CART_MAX_DTAU`` — max torque step per 1 kHz cycle.
* ``RLINF_CART_ERR_CLIP_M`` / ``RLINF_CART_ERR_CLIP_RAD`` —
  tracking-error clip (impedance saturation).
* ``RLINF_CART_GAINS_TC`` — gains time constant.
* ``RLINF_CART_MAX_STEP_M`` — per-call rate limit on translational
  target jumps (default ``0.03`` m, mostly there to clamp
  single-frame dataset jumps into a slew rather than a step input).
* ``RLINF_CART_MAX_STEP_RAD`` — same for rotation, default
  ``0.15`` rad.

Tune the ``MAX_STEP`` knobs first when chasing tracking jitter —
they cost nothing and turn most policy-induced jumps into smooth
slews.


Troubleshooting
---------------

**GELLO daemon never starts**
   ``DualGelloJointIntervention._start_stream_thread`` only spins up
   the daemon once *both* experts report ``ready``.
   ``GelloJointExpert`` gates ``ready`` on the first successful
   Dynamixel read — and falls back to ``ready=False`` after 50
   consecutive errors. Power-cycle the GELLO arm, re-plug the FTDI,
   and verify with
   ``python -m gello_teleop.gello_expert --port /dev/...``.

**Ray worker silently dies on import**
   Usually one of: (a) ``franky``, ``gello``, or ``gello_teleop`` is
   not installed in the venv that Ray is currently using on that node;
   (b) ``ray start`` was launched from a different venv than the one
   you ``pip install``-ed into; (c) a native dep (libfranka, Dynamixel
   SDK, X11 libs for Lumos) is missing. ``ray status`` shows the
   worker disappear; ``/tmp/ray/session_latest/logs/worker-*.err``
   has the ``ImportError``. Confirm
   ``which python && python -c "import franky, gello, gello_teleop"``
   from the same shell that ran ``ray start``.

**One arm hangs at reset**
   Check ``ping -c 100 172.16.0.2`` on the controller node. If the
   FCI link drops a packet during the first
   ``FrankyController.__init__``, ``recover_from_errors`` no-ops and
   subsequent ``move_*`` calls fail. Power-cycle the affected arm
   and re-launch.

**"Permission denied" on the foot pedal**
   ``RLINF_KEYBOARD_DEVICE`` points at ``/dev/input/eventXX`` but
   ``chmod`` was reverted on reboot. Either re-run ``sudo chmod 666``
   or add a udev rule
   (``KERNEL=="event*", SUBSYSTEM=="input",
   ATTRS{name}=="PCsensor FootSwitch", MODE="0666"``).

**RealSense falls back to USB 2.x**
   ``check_cameras.py`` reports ``WARN`` with the USB descriptor.
   Replace the cable, plug into a root USB-3 port (the ones marked
   blue), and re-run.

**Lumos cold-start fails on first invocation**
   ``LumosCamera`` and ``check_cameras._open_lumos`` do a double-open
   + I420 buffer warmup that handles the cold-start ``STREAMON``
   bandwidth race. If failures persist after a reboot, the driver
   state is wedged — re-plug the USB cable.

**Eval idle forever**
   ``KeyboardEvalControlWrapper`` waits for pedal ``a``. Confirm
   ``RLINF_KEYBOARD_DEVICE`` points at the right
   ``/dev/input/eventXX`` and that ``chmod 666`` is still in effect.
   Note that the wrapper polls every ``IDLE_POLL_S = 0.05`` s and
   intercepts ``env.step`` *before* it forwards to the inner env, so
   ``ray status`` will look healthy even when the wrapper is stuck
   in idle.

**Tracking jitter at deployment**
   Lower ``RLINF_CART_K_R``, raise ``RLINF_CART_GAINS_TC``, or cap
   ``RLINF_CART_MAX_STEP_RAD`` more aggressively. As a last resort
   shorten the policy's chunk horizon — long chunks amplify a stale
   ``T_state`` if the SFT distribution was tighter than the eval
   one.

**Multi-task round-robin doesn't advance**
   Only ``c`` (success) advances the rotation; ``a`` aborts do not
   consume slots. If you set ``data_collection.only_success: true``
   (default) and the episode reaches ``c`` but ``reward != 1.0``,
   the episode is silently dropped and the rotation does not
   advance. Pedal ``c`` always sets ``reward=1.0`` in
   ``KeyboardStartEndWrapper``, so this should not happen unless
   the wrapper was disabled or replaced.

**``norm_stats.json`` not found at deployment**
   The rollout worker reads
   ``<model_path>/<data_config.asset_id>/norm_stats.json``.
   ``data_config.asset_id`` follows ``data_config.repo_id`` unless
   the dataconfig sets it explicitly — and ``data_config.repo_id``
   is whatever ``actor.model.openpi_data.repo_id`` was set to in
   the SFT yaml. If the SFT ckpt dir does not contain it (for
   example you trained on one machine and rsync'd only the
   ``actor/`` subtree), copy ``norm_stats.json`` from where
   ``calculate_norm_stats.py`` wrote it
   (``<openpi_assets_dirs>/<repo_id>/``) into the SFT ckpt dir.
   The fallback path logs a ``"norm_stats fallback"`` warning
   — grep for it before assuming inference is fine.

**collect_monitor frozen**
   The launcher script must redirect stdout to a log file that the
   monitor tails (``2>&1 | tee logs/collect.log``). If you skipped
   the ``tee``, the monitor has nothing to read. Pass
   ``--source=worker`` when the env worker is on a different node
   from the launcher.

**``sched_setaffinity failed`` warning at controller start**
   The host has fewer than 6 CPU cores or the user is missing
   ``CAP_SYS_NICE``. The controller still works — but RT thread
   pinning is best-effort, and jitter will be worse on a heavily
   loaded host. Either run on a 6+ core machine or grant the
   capability via ``sudo setcap cap_sys_nice=eip $(which python)``
   on the venv interpreter.

**Both arms move at reset but only one tracks GELLO afterwards**
   ``DualGelloJointIntervention._start_stream_thread`` returned
   early because one expert was not ready by the time the daemon
   was queried. Run ``gello_align_check.py`` per arm to confirm both
   GELLOs produce continuous joint readings, then re-launch.


References
----------

Configs

* ``examples/embodiment/config/realworld_collect_data_gello_joint_dual_franka.yaml``
* ``examples/embodiment/config/realworld_eval_dual_franka.yaml``
* ``examples/embodiment/config/env/realworld_franka_joint_dual.yaml``
* ``examples/embodiment/config/env/realworld_franka_rot6d_dual.yaml``
* ``examples/embodiment/config/env/realworld_dual_franka.yaml`` —
  legacy 14-D Cartesian dual env (kept for compatibility, not used
  by the rot6d SFT path).
* ``examples/sft/config/dual_franka_rot6d_sft_openpi.yaml``

Toolkits

* ``toolkits/dual_franka/check_cameras.py`` — 3-cam health check +
  live preview.
* ``toolkits/dual_franka/backfill_rot6d.py`` — joint-space → rot6d_v1
  rewrite.
* ``toolkits/realworld_check/collect_monitor.py`` —
  out-of-process tqdm monitor.
* ``toolkits/realworld_check/test_franky_controller.py`` — per-arm
  REPL smoke test.
* ``toolkits/realworld_check/gello_calibrate.py`` /
  ``gello_align_check.py`` / ``gello_align_sequential.py`` —
  GELLO calibration helpers.
* ``toolkits/lerobot/calculate_norm_stats.py`` — π₀ / π₀.₅ norm
  stats.

Code (canonical entry points)

* ``rlinf/scheduler/hardware/robots/dual_franka.py`` —
  ``DualFrankaConfig``.
* ``rlinf/envs/realworld/franka/franky_controller.py`` — libfranka
  backend.
* ``rlinf/envs/realworld/franka/dual_franka_franky_env.py`` — shared
  env scaffold for joint + rot6d.
* ``rlinf/envs/realworld/franka/dual_franka_joint_env.py`` — 16-D
  joint env.
* ``rlinf/envs/realworld/franka/dual_franka_rot6d_env.py`` — 20-D
  rot6d env.
* ``rlinf/envs/realworld/common/wrappers/dual_gello_joint_intervention.py``
  — 1 kHz GELLO daemon.
* ``rlinf/envs/realworld/common/wrappers/keyboard_start_end_wrapper.py``
  — collection pedal.
* ``rlinf/envs/realworld/common/wrappers/keyboard_eval_control_wrapper.py``
  — eval pedal.
* ``rlinf/envs/realworld/common/wrappers/apply.py`` — wrapper
  composition.
* ``rlinf/utils/rot6d.py`` — rot6d math + SE(3) helpers.
* ``rlinf/models/embodiment/openpi/transforms/rigid_body_delta.py``
  — ``RigidBodyDeltaActions`` / ``RigidBodyAbsoluteActions``.
* ``rlinf/models/embodiment/openpi/policies/dual_franka_rot6d_policy.py``
  — data transform inputs / outputs.
* ``rlinf/models/embodiment/openpi/dataconfig/dual_franka_rot6d_dataconfig.py``
  — openpi data config wiring.
* ``rlinf/envs/wrappers/collect_episode.py`` — multi-task LeRobot
  writer + resume.
* ``examples/embodiment/collect_real_data.py`` — collection driver.

Related guides

* :doc:`franka` — single-arm Franka basics.
* :doc:`franka_gello` — GELLO hardware install.
* :doc:`franka_pi0_sft_deploy` — single-arm π₀ SFT deployment
  example.
* :doc:`sft_openpi` — full / LoRA OpenPI SFT pipeline overview.
