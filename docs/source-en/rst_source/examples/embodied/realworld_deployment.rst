Real-World Deployment: Franka + ZED + Robotiq (Pi0)
====================================================

This document describes the end-to-end workflow for deploying RLinf on a
Franka Panda real-world setup with ZED cameras and a Robotiq gripper:
**Data Collection → SFT Training → Real-World Deployment & Evaluation**.

- **SFT Training phase**: Remote A100 server (training only; checkpoint transferred to 4090 after training)
- **Deployment / Online RL phase**: 4090 server + NUC two-node deployment

.. contents:: Table of Contents
   :local:
   :depth: 2

----

1. Hardware Architecture
------------------------

The setup uses the following machines:

.. code-block:: text

   ┌──────────────────────────────────────────────────────────────┐
   │  Remote A100 Server (SFT Training Only)                     │
   │   - Multi-GPU A100                                          │
   │   - RLinf SFT training inside Docker                        │
   │   - Transfer checkpoint to 4090 server after training       │
   └──────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────┐
   │  4090 Server (node 0, Ubuntu 22.04)                         │
   │   - 3× ZED cameras                                         │
   │   - GPU for inference + training + env worker               │
   │   - actor + rollout worker + env worker all run here        │
   │   - Ray Head node: RLINF_NODE_RANK=0                        │
   └───────────────────────────┬──────────────────────────────────┘
                               │ LAN (direct / switch)
   ┌───────────────────────────┴──────────────────────────────────┐
   │  NUC (node 1, Ubuntu 20.04)                                 │
   │   - Franka Panda arm (robot_ip: 172.16.0.2)                 │
   │   - Robotiq gripper (/dev/ttyUSB0, USB-RS485)               │
   │   - GELLO teleoperation handle                              │
   │   - ROS Noetic                                              │
   │   - FrankaController runs here                              │
   │   - Ray Worker node: RLINF_NODE_RANK=1                      │
   └──────────────────────────────────────────────────────────────┘

.. note::

   **Modify the following values to match your own environment:**

   - ZED camera serial numbers (defaults: ``10848563``, ``39651335``, ``34303972``)
   - Franka arm IP (default: ``172.16.0.2``)
   - Robotiq gripper serial port (default: ``/dev/ttyUSB0``)
   - Python virtual-environment path on the NUC (default: ``/home/franka/workspace/RLinf/.venv/bin/python3``)
   - IP addresses of the 4090 server and NUC
   - Remote A100 server SSH address and port

----

2. Installation
---------------

2.1 NUC (Franka Controller Node)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The NUC needs ROS Noetic, the Franka driver, and RLinf's ``franka`` environment
dependencies.

.. code-block:: bash

   # 1. Clone RLinf (--recurse-submodules pulls the GELLO teleop toolkit)
   git clone --recurse-submodules https://github.com/Brunch-Life/RLinf.git
   cd RLinf
   git checkout feature/real_sft
   # If already cloned but missing submodules:
   # git submodule update --init --recursive

   # 2. Install franka environment dependencies (installs ROS, franka_ros, etc.)
   #    Add SKIP_ROS=1 to skip system-package installation if ROS is already present
   bash requirements/install.sh embodied --env franka --install-rlinf

   # 3. Activate the virtual environment
   source .venv/bin/activate

.. note::

   The NUC does **not** need model dependencies — it only runs the
   FrankaController (low-level arm control).

2.2 4090 Server (Env Worker Node)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The 4090 server also installs the ``franka`` environment dependencies plus the
OpenPI model dependencies.

.. code-block:: bash

   # 1. Clone RLinf
   git clone --recurse-submodules https://github.com/Brunch-Life/RLinf.git
   cd RLinf
   git checkout feature/real_sft

   # 2. Install franka env + RLinf
   bash requirements/install.sh embodied --env franka --install-rlinf

   # 3. Activate the virtual environment
   source .venv/bin/activate

2.3 Remote A100 Server (Training Node)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # 1. Clone RLinf
   git clone https://github.com/Brunch-Life/RLinf.git
   cd RLinf
   git checkout feature/real_sft

Using Docker is recommended so that no manual dependency installation is
required.

.. code-block:: bash

   # Pull the Docker image
   docker pull rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0

   # Start the container
   docker run -it --gpus all \
       --shm-size 100g \
       --net=host \
       --name rlinf \
       -e NVIDIA_DRIVER_CAPABILITIES=all \
       -v /path/to/RLinf:/workspace/RLinf \
       rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0 /bin/bash

   cd /workspace/RLinf

.. note::

   If you prefer not to use Docker, you can install via ``install.sh`` (Ubuntu
   22.04 only):

   .. code-block:: bash

      bash requirements/install.sh embodied --model openpi --env maniskill_libero

----

3. Data Collection
------------------

Data collection uses two nodes: the 4090 server (Ray Head + env worker) and the
NUC (FrankaController).

Two collection scripts are available:

- ``collect_data.sh`` — uses the raw ``TrajectoryReplayBuffer`` format (``.pt``),
  suitable for RLPD / online-RL training. Records continuously with **no
  keyboard control**.
- ``collect_data_with_wrapper.sh`` — uses the ``CollectEpisode`` wrapper and
  outputs **LeRobot v2.0** format. Supports **keyboard-interactive recording**
  (``a`` = start recording, ``b`` = mark failure, ``c`` = mark success).

**This document uses** ``collect_data_with_wrapper.sh`` **as the primary
example.**

3.1 Configuration
~~~~~~~~~~~~~~~~~

The corresponding config file is
``examples/embodiment/config/realworld_collect_data_zed_robotiq.yaml``.

.. note::

   To collect ``.pt`` data for RLPD / online-RL training, use
   ``collect_data.sh`` with ``realworld_collect_data.yaml`` instead. The
   wrapper-specific parameters below do not apply to that workflow.

Key configuration items:

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Config Key
     - Value
     - Description
   * - ``cluster.num_nodes``
     - ``2``
     - Dual-node: 4090 + NUC
   * - ``env.eval.use_gello``
     - ``True``
     - Use GELLO teleoperation
   * - ``env.eval.use_spacemouse``
     - ``False``
     - Do not use SpaceMouse
   * - ``runner.num_data_episodes``
     - ``50``
     - Collect 50 episodes
   * - ``runner.export_format``
     - ``"lerobot"``
     - Output in LeRobot v2.0 format (wrapper only)
   * - ``runner.fps``
     - ``10``
     - Collection frame rate 10 Hz (wrapper only)
   * - ``runner.only_success``
     - ``True``
     - Save only successful episodes (wrapper only)
   * - ``camera_type``
     - ``"zed"``
     - ZED camera
   * - ``gripper_type``
     - ``"robotiq"``
     - Robotiq gripper

Modify the config file to match your actual **camera serial numbers**,
**arm IP**, **gripper serial port**, **GELLO serial port**, **NUC Python
path**, and **initial end-effector pose** (``target_ee_pose``).

3.2 Launch Steps
~~~~~~~~~~~~~~~~

**Step 1 — NUC side:**

.. code-block:: bash

   # 1. Source the ROS environment
   source /opt/ros/noetic/setup.bash

   # 2. Activate the RLinf virtual environment
   source ~/RLinf/.venv/bin/activate

   # 3. Join the Ray cluster as a worker
   RLINF_NODE_RANK=1 ray start --address=<4090_SERVER_IP>:6379

.. note::

   You **must** source ROS and the virtual environment **before**
   ``ray start``, because Ray captures environment variables at startup.

**Step 2 — 4090 server side:**

.. code-block:: bash

   # 1. Enter the RLinf directory
   cd /path/to/RLinf

   # 2. Activate the virtual environment
   source .venv/bin/activate

   # 3. Set environment variables
   export EMBODIED_PATH="$(pwd)/examples/embodiment"

   # 4. Start the Ray Head node
   RLINF_NODE_RANK=0 ray start --head --port=6379 --node-ip-address=<4090_SERVER_IP>

   # 5. (Optional) Wait for the NUC to join the cluster
   ray status

   # 6. Start data collection (LeRobot format + keyboard control)
   bash examples/embodiment/collect_data_with_wrapper.sh realworld_collect_data_zed_robotiq

.. note::

   To collect ``.pt`` data for RLPD training, use ``collect_data.sh`` instead:

   .. code-block:: bash

      bash examples/embodiment/collect_data.sh realworld_collect_data_zed_robotiq

3.3 Keyboard Controls (Wrapper Version Only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following keyboard controls are active **only** when using
``collect_data_with_wrapper.sh``. The original ``collect_data.sh`` records
continuously with no keyboard interaction.

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - Key
     - Action
   * - **a**
     - **Start recording.** Before pressing, you can freely move the arm with
       GELLO to reach the desired start pose — this phase is not recorded.
   * - **c**
     - **Mark success** and end the current episode (reward = +1).
   * - **b**
     - **Mark failure** and end the current episode (reward = −1).

Typical workflow:

1. Move the arm to a suitable start pose using GELLO.
2. Press ``a`` to begin recording.
3. Operate the arm with GELLO to complete the task.
4. Press ``c`` on success or ``b`` on failure.
5. The system automatically advances to the next episode. Repeat.

3.4 Data Output
~~~~~~~~~~~~~~~

After collection, data is saved in LeRobot v2.0 format:

.. code-block:: text

   logs/<timestamp>/lerobot_dataset/
   ├── data/
   │   └── chunk-000/
   │       ├── episode_000000.parquet
   │       ├── episode_000001.parquet
   │       └── ...
   └── meta/
       ├── episodes.jsonl
       ├── info.json
       ├── stats.json
       └── tasks.jsonl

----

4. Data Processing
------------------

4.1 Transfer Data to the Training Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sync the collected dataset to the remote training server:

.. code-block:: bash

   # From the local 4090 server to the remote A100
   rsync -avhzP logs/<timestamp>/lerobot_dataset/ \
       <REMOTE_HOST>:<REMOTE_RLINF_PATH>/dataset/<DATASET_REPO_ID>/

.. note::

   **Placeholder descriptions:**

   - ``<REMOTE_HOST>`` — SSH alias or ``user@ip:port``
     (e.g. ``my-server`` or ``user@192.168.1.100``)
   - ``<REMOTE_RLINF_PATH>`` — path to RLinf on the remote server
     (e.g. ``/workspace/RLinf``)
   - ``<DATASET_REPO_ID>`` — dataset identifier
     (e.g. ``<user>/<dataset>``)

4.2 Convert norm_stats
~~~~~~~~~~~~~~~~~~~~~~~

The Pi0 model requires ``norm_stats.json`` for normalization. On the training
server, generate it from ``stats.json``:

.. code-block:: bash

   python toolkits/convert_stats_to_norm_stats.py \
       --stats-json dataset/<DATASET_REPO_ID>/meta/stats.json \
       --output-dir checkpoints/torch/pi0_base/<DATASET_REPO_ID> \
       --select-state-dims 4 5 6 7 8 9 0 \
       --action-dim 32

Arguments:

- ``--stats-json`` — path to the dataset's ``stats.json``
- ``--output-dir`` — output directory for ``norm_stats.json``, should
  correspond to the dataset name under the model checkpoint path
- ``--select-state-dims`` — dimension indices to select from the raw state
  vector (matching the ``pi0_realworld_pnp`` config)
- ``--action-dim`` — maximum Pi0 action/state dimension (zero-padded to this
  size)

.. note::

   If using the OpenPI native training path (Option B), you also need to copy
   ``norm_stats.json`` to OpenPI's assets directory. See
   :ref:`section-5-3-option-b`.

----

5. Training
-----------

Training runs on the remote A100 server. Two options are available:

- **Option A: RLinf SFT** — uses RLinf's distributed training pipeline with
  support for a value head and subsequent RL fine-tuning.
- **Option B: OpenPI Native PyTorch** — uses OpenPI's built-in ``torchrun``
  training script; lighter-weight.

Checkpoints from either option can be used for :ref:`section-6-deployment`.

5.1 Prerequisites (Common)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ensure the following files are ready:

- Dataset: ``<REMOTE_RLINF_PATH>/dataset/<DATASET_REPO_ID>/``
- Pi0 pre-trained weights: ``<REMOTE_RLINF_PATH>/checkpoints/torch/pi0_base``
- norm_stats: ``checkpoints/torch/pi0_base/<DATASET_REPO_ID>/norm_stats.json``

5.2 Option A: RLinf SFT Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Config file: ``examples/sft/config/custom_sft_openpi.yaml``

.. note::

   The ``train_data_paths`` and ``model_path`` values in this config file are
   **placeholders** (``/path/to/custom-data`` and ``/path/to/pi0-model``).
   You **must** replace them with your actual paths before training.

Key training parameters:

.. list-table::
   :header-rows: 1
   :widths: 28 22 18 32

   * - Parameter
     - Config Default
     - Recommended (8-GPU)
     - Description
   * - ``data.train_data_paths``
     - ``"/path/to/custom-data"``
     - Absolute dataset path
     - e.g. ``"/workspace/RLinf/dataset/<DATASET_REPO_ID>"``
   * - ``actor.model.model_path``
     - ``"/path/to/pi0-model"``
     - Pi0 pre-trained weights path
     - e.g. ``"/workspace/RLinf/checkpoints/torch/pi0_base"``
   * - ``actor.model.openpi.config_name``
     - ``"pi0_realworld_pnp"``
     - ``"pi0_realworld_pnp"``
     - OpenPI config name (ZED+Robotiq real-world)
   * - ``actor.micro_batch_size``
     - ``1``
     - ``16``
     - Per-GPU batch size; adjust based on GPU memory
   * - ``actor.global_batch_size``
     - ``16``
     - ``128``
     - Global batch size (8 GPUs × 16 = 128)
   * - ``actor.optim.lr``
     - ``7.91e-6``
     - ``2.5e-5``
     - Learning rate; tune as needed
   * - ``runner.max_steps``
     - ``-1`` (unlimited)
     - ``30000``
     - Total training steps; ``-1`` means controlled by ``max_epochs``
   * - ``runner.save_interval``
     - ``10``
     - ``1000``
     - Save a checkpoint every N steps
   * - ``actor.model.add_value_head``
     - ``True``
     - ``True``
     - Add a value head (set ``False`` for pure SFT)

**Launch training:**

.. code-block:: bash

   # Enter the Docker container
   docker run -it --gpus all --shm-size 100g --net=host --name rlinf \
       -e NVIDIA_DRIVER_CAPABILITIES=all -v /mnt:/mnt \
       rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0 /bin/bash

   # Inside the container
   cd /workspace/RLinf

   # Switch to the openpi environment (inside Docker image)
   source switch_env openpi

   # Start Ray
   ray start --head

   # Set environment variables
   export EMBODIED_PATH="$(pwd)/examples/embodiment"

   # Launch training
   bash examples/sft/run_vla_sft.sh custom_sft_openpi

**RLinf training output:**

.. code-block:: text

   logs/<timestamp>/test_openpi/checkpoints/
   ├── global_step_1000/
   │   └── actor/
   │       └── model_state_dict/
   │           └── full_weights.pt
   ├── global_step_2000/
   │   └── ...
   └── global_step_30000/
       └── ...

For evaluation, use the ``actor/model_state_dict/`` directory (containing
``.pt`` or ``.safetensors`` files).

.. _section-5-3-option-b:

5.3 Option B: OpenPI Native PyTorch Training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you do not need RLinf's distributed scheduling or value head, or want to run
a comparison, you can use OpenPI's native training script directly.

Prerequisites
^^^^^^^^^^^^^

1. Install the OpenPI repository:

.. code-block:: bash

   git clone https://github.com/RLinf/openpi.git
   cd openpi
   pip install -e .

2. **Align the data processing pipeline:** the ``pi0_realworld_pnp`` config on the
   OpenPI side must be consistent with RLinf's data processing. Key files to
   check (ensure your OpenPI version includes these changes):

   - ``src/openpi/policies/realworld_policy.py`` — ``RealworldInputs`` state
     dimension selection (19D → 7D), multi-camera slot mapping
   - ``src/openpi/training/config.py`` — ``pi0_realworld_pnp`` ``state_indices``,
     ``extra_image_keys``, ``pi0_slot_keys``, etc.

3. Copy ``norm_stats`` to the OpenPI assets directory:

.. code-block:: bash

   cp <REMOTE_RLINF_PATH>/checkpoints/torch/pi0_base/<DATASET_REPO_ID>/norm_stats.json \
      <OPENPI_PATH>/assets/pi0_realworld_pnp/<DATASET_REPO_ID>/norm_stats.json

Camera Slot Mapping
^^^^^^^^^^^^^^^^^^^

The Pi0 pre-trained model has fixed semantic camera slots. Map your cameras
accordingly:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Pi0 Slot
     - Pre-trained Semantics
     - Actual Camera
   * - ``base_0_rgb``
     - Global / third-person view
     - ``extra_image_0`` (left standing camera)
   * - ``left_wrist_0_rgb``
     - Wrist / close-up view
     - ``image`` (wrist camera)
   * - ``right_wrist_0_rgb``
     - Secondary view
     - ``extra_image_1`` (right standing camera)

.. note::

   If your camera layout differs, adjust the ``pi0_slot_keys`` config
   accordingly.

Launch Training
^^^^^^^^^^^^^^^

.. code-block:: bash

   cd <OPENPI_PATH>

   # Point to the dataset path (LeRobot format)
   export HF_LEROBOT_HOME="<REMOTE_RLINF_PATH>/dataset"
   export HF_HUB_OFFLINE=1

   # 8-GPU training
   uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 \
       scripts/train_pytorch.py pi0_realworld_pnp \
       --exp_name <EXPERIMENT_NAME>

.. note::

   **Hyperparameter alignment:** if you plan to compare with RLinf Option A,
   ensure hyperparameters are kept consistent.

**OpenPI training output:**

.. code-block:: text

   <OPENPI_PATH>/outputs/<EXPERIMENT_NAME>/trial/
   ├── checkpoint-1000/
   │   ├── model.safetensors
   │   └── ...
   ├── checkpoint-2000/
   │   └── ...
   └── checkpoint-30000/
       └── ...

Use the corresponding checkpoint directory path for evaluation.

----

.. _section-6-deployment:

6. Deployment & Evaluation
--------------------------

Deployment uses an integrated rollout worker architecture — model inference
runs directly on the 4090 server with no separate Policy Server or SSH tunnel
required. Only two machines are needed: the 4090 server (actor + rollout +
env worker) and the NUC (FrankaController).

6.1 Transfer Checkpoint to the 4090 Server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sync the trained checkpoint from the A100 to the 4090 server:

.. code-block:: bash

   rsync -avhzP <REMOTE_HOST>:<REMOTE_RLINF_PATH>/logs/<timestamp>/test_openpi/checkpoints/global_step_<N>/actor/model_state_dict/ \
       /path/to/RLinf/checkpoints/realworld_pnp/

6.2 Deployment Config
~~~~~~~~~~~~~~~~~~~~~

Config file:
``examples/embodiment/config/realworld_pi0_zed_robotiq_async.yaml``

This config places the actor (training/inference), rollout worker (policy
inference), and env worker (environment interaction + cameras) all on the
4090 server. The NUC runs only the FrankaController.

Key configuration items:

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Config Key
     - Default
     - Description
   * - ``runner.only_eval``
     - ``False``
     - Set ``True`` for evaluation only (no training)
   * - ``runner.ckpt_path``
     - ``null``
     - Path to ``.pt`` checkpoint file
   * - ``actor.model.model_path``
     - ``"/path/to/model"``
     - Pi0 pre-trained or fine-tuned model path
   * - ``rollout.model.model_path``
     - ``"/path/to/model"``
     - Same as actor
   * - ``actor.model.openpi.config_name``
     - ``"pi0_realworld_pnp"``
     - OpenPI config name
   * - ``env.train.task_description``
     - ``null``
     - Task description prompt (e.g. ``"pick up the chip"``)
   * - ``env.train.keyboard_reward_wrapper``
     - ``"single_stage"``
     - Keyboard reward labeling mode

6.3 Launch Deployment
~~~~~~~~~~~~~~~~~~~~~

**Step 1 — NUC side:**

.. code-block:: bash

   # 1. Source the ROS environment
   source /opt/ros/noetic/setup.bash

   # 2. Activate the RLinf virtual environment
   source ~/RLinf/.venv/bin/activate

   # 3. Join the Ray cluster as a worker
   RLINF_NODE_RANK=1 ray start --address=<4090_SERVER_IP>:6379

**Step 2 — 4090 server side:**

.. code-block:: bash

   # 1. Enter the RLinf directory
   cd /path/to/RLinf

   # 2. Activate the virtual environment
   source .venv/bin/activate

   # 3. Set environment variables
   export EMBODIED_PATH="$(pwd)/examples/embodiment"

   # 4. Start the Ray Head node
   RLINF_NODE_RANK=0 ray start --head --port=6379 --node-ip-address=<4090_SERVER_IP>

   # 5. Wait for the NUC to join the cluster
   ray status

   # 6a. Evaluation only: load checkpoint and run eval
   bash examples/embodiment/run_realworld_async.sh realworld_pi0_zed_robotiq_async \
       runner.only_eval=True \
       runner.ckpt_path=/path/to/checkpoints/realworld_pnp/full_weights.pt

   # 6b. Online RL: load SFT checkpoint and continue training (SAC)
   bash examples/embodiment/run_realworld_async.sh realworld_pi0_zed_robotiq_async \
       runner.ckpt_path=/path/to/checkpoints/realworld_pnp/full_weights.pt

6.4 Keyboard Controls (single_stage Mode)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

During evaluation or online training, manually label each episode's outcome:

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - Key
     - Action
   * - **a**
     - **Failure**: reward = −1, ends the current episode
   * - **b**
     - **Neutral**: reward = 0, does **not** end the episode
   * - **c**
     - **Success**: reward = +1, ends the current episode

Typical workflow:

1. After launch, the rollout worker performs local policy inference and the arm
   begins executing actions.
2. Observe the arm's behavior.
3. Press ``c`` on success or ``a`` on failure.
4. The system automatically advances to the next episode.

----

Appendix A: Keyboard Control Quick Reference
---------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - Scenario
     - a
     - b
     - c
   * - **Data Collection** (``collect_data_with_wrapper.sh``)
     - Start recording
     - Failure, end episode
     - Success, end episode
   * - **Deployment / Online RL** (``run_realworld_async.sh``, single_stage)
     - Failure (−1, end)
     - Neutral (0, no end)
     - Success (+1, end)

.. note::

   - The key meanings are **different** between data collection and evaluation
     — take care not to confuse them.
   - Keyboard controls apply **only** to ``collect_data_with_wrapper.sh``
     (wrapper version). The original ``collect_data.sh`` (Replay Buffer
     version) has no keyboard controls and records continuously.

----

Appendix B: Key File Index
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - File
     - Purpose
   * - ``examples/embodiment/config/realworld_collect_data.yaml``
     - Data collection config (Replay Buffer / RL training)
   * - ``examples/embodiment/config/realworld_collect_data_wrapper.yaml``
     - Data collection config (LeRobot wrapper / single-node base)
   * - ``examples/embodiment/config/realworld_collect_data_zed_robotiq.yaml``
     - Data collection config (LeRobot wrapper / dual-node ZED + Robotiq)
   * - ``examples/embodiment/config/realworld_pi0_zed_robotiq_async.yaml``
     - Real-world deployment / online RL config
   * - ``examples/embodiment/collect_data.sh``
     - Data collection launch script (Replay Buffer / ``.pt`` format)
   * - ``examples/embodiment/collect_real_data.py``
     - Data collection Python entry (Replay Buffer / RL training)
   * - ``examples/embodiment/collect_data_with_wrapper.sh``
     - Data collection launch script (LeRobot format + keyboard control)
   * - ``examples/embodiment/collect_real_data_with_wrapper.py``
     - Data collection Python entry (CollectEpisode wrapper)
   * - ``examples/embodiment/run_realworld_async.sh``
     - Real-world deployment / online RL launch script
   * - ``examples/embodiment/train_async.py``
     - Async training / deployment Python entry
   * - ``examples/sft/config/custom_sft_openpi.yaml``
     - RLinf Pi0 SFT training config (**contains placeholder paths** — must be
       replaced)
   * - ``toolkits/convert_stats_to_norm_stats.py``
     - ``stats.json`` → ``norm_stats.json`` conversion tool
   * - ``ray_utils/realworld/setup_before_ray.sh``
     - Pre-Ray environment setup template for real-world
   * - ``requirements/install.sh``
     - RLinf dependency installation script
   * - ``rlinf/models/embodiment/openpi/dataconfig/__init__.py``
     - RLinf-side ``pi0_realworld_pnp`` config definition
   * - ``openpi/src/openpi/policies/realworld_policy.py``
     - OpenPI-side RealworldInputs (must be aligned with RLinf)
   * - ``openpi/src/openpi/training/config.py``
     - OpenPI-side ``pi0_realworld_pnp`` training config (must be aligned with RLinf)
   * - ``openpi/scripts/train_pytorch.py``
     - OpenPI native PyTorch training entry

----

Appendix C: FAQ
---------------

Q1: Cannot connect to the Ray cluster?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Ensure the 4090 server and NUC are on the same LAN.
- Check that port 6379 is open in the firewall.
- The IP in ``ray start --address=<IP>:6379`` on the NUC must be the 4090
  server's IP.
- Run ``ray status`` to verify both nodes have joined the cluster.

Q2: ROS import errors on the NUC?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ray captures environment variables at ``ray start`` time. Make sure you source
ROS and the virtual environment **before** running ``ray start``:

.. code-block:: bash

   source /opt/ros/noetic/setup.bash
   source ~/RLinf/.venv/bin/activate
   RLINF_NODE_RANK=1 ray start --address=<IP>:6379

Q3: How to change the task or initial pose?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Modify ``env.train.override_cfg.target_ee_pose`` in the config file. The value
is in ``[x, y, z, rx, ry, rz]`` format (Euler angles).

Q4: How to do pure SFT without a value head?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set ``actor.model.add_value_head`` to ``False`` in the training config
``custom_sft_openpi.yaml``.
