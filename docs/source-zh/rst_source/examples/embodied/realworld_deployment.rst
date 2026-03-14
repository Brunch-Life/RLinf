真机部署：Franka + ZED + Robotiq (Pi0)
=======================================

本文档记录使用 RLinf 在 Franka Panda 真机环境下，基于 ZED 相机和 Robotiq 夹爪，完成 **数据采集 → SFT 训练 → 真机部署评估** 的完整流程。

- **SFT 训练阶段**：远端 A100 服务器（仅训练，完成后将 checkpoint 传输至 4090 服务器）
- **部署评估 / 在线 RL 阶段**：4090 服务器 + NUC 两节点部署

.. contents:: 目录
   :local:
   :depth: 2

----

1. 硬件架构
----------

本方案采用以下机器协同工作：

.. code-block:: text

   ┌──────────────────────────────────────────────────────────────┐
   │  远端 A100 服务器（仅 SFT 训练）                               │
   │   - 多卡 A100 GPU                                            │
   │   - Docker 容器内训练 RLinf SFT                               │
   │   - 训练完成后将 checkpoint 传输至 4090 服务器                  │
   └──────────────────────────────────────────────────────────────┘

   ┌──────────────────────────────────────────────────────────────┐
   │  4090 服务器 (node 0, Ubuntu 22.04)                           │
   │   - 3× ZED 相机                                              │
   │   - GPU 用于推理 + 训练 + env worker                          │
   │   - actor + rollout worker + env worker 全部运行在此           │
   │   - Ray Head 节点：RLINF_NODE_RANK=0                         │
   └───────────────────────────┬──────────────────────────────────┘
                               │ 局域网（直连 / 交换机）
   ┌───────────────────────────┴──────────────────────────────────┐
   │  NUC (node 1, Ubuntu 20.04)                                   │
   │   - Franka Panda 机械臂（robot_ip: 172.16.0.2）              │
   │   - Robotiq 夹爪（/dev/ttyUSB0, USB-RS485）                  │
   │   - GELLO 遥操作手柄                                         │
   │   - ROS Noetic                                               │
   │   - FrankaController 运行在此                                 │
   │   - Ray Worker 节点：RLINF_NODE_RANK=1                       │
   └──────────────────────────────────────────────────────────────┘

.. note::

   **请根据你自己的环境修改以下内容：**

   - ZED 相机序列号（当前：``10848563``, ``39651335``, ``34303972``）
   - Franka 机械臂 IP（当前：``172.16.0.2``）
   - Robotiq 夹爪串口（当前：``/dev/ttyUSB0``）
   - GELLO 遥操作串口（当前：``/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA0OUKN-if00-port0``）
   - NUC 上 ``python_interpreter_path``（配置文件中的 ``cluster.node_groups[].env_configs[].python_interpreter_path``）。在 NUC 上激活虚拟环境后运行 ``which python3`` 获取正确路径，替换配置文件中的默认值。**路径错误会导致 Ray worker 启动失败。**
   - 4090 服务器和 NUC 的 IP 地址
   - 远端 A100 服务器 SSH 地址和端口
   - 初始末端位姿 ``target_ee_pose``

----

2. 环境安装
----------

2.1 NUC 端（Franka 控制器节点）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NUC 需要安装 ROS Noetic、Franka 驱动以及 RLinf 的 franka 环境依赖。

.. code-block:: bash

   # 1. 克隆 RLinf（--recurse-submodules 会自动拉取 GELLO 遥操作工具包）
   git clone --recurse-submodules https://github.com/Brunch-Life/RLinf.git
   cd RLinf
   git checkout feature/real_sft
   # 若已克隆但缺少 submodule，补充初始化：
   # git submodule update --init --recursive

   # 2. 安装 franka 环境依赖（会安装 ROS、franka_ros 等）
   #    如果已有 ROS 等系统依赖，可加 SKIP_ROS=1 跳过系统包安装
   bash requirements/install.sh embodied --env franka --install-rlinf

   # 3. 激活虚拟环境
   source .venv/bin/activate

.. note::

   NUC 上不需要安装模型依赖，它只运行 FrankaController（机械臂底层控制）。

2.2 4090 服务器端（env worker 节点）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

4090 服务器同样安装 franka 环境依赖，并额外安装 openpi 相关依赖。

.. code-block:: bash

   # 1. 克隆 RLinf
   git clone --recurse-submodules https://github.com/Brunch-Life/RLinf.git
   cd RLinf
   git checkout feature/real_sft

   # 2. 安装 franka 环境 + RLinf
   bash requirements/install.sh embodied --env franka --install-rlinf

   # 3. 激活虚拟环境
   source .venv/bin/activate

2.3 远端 A100 服务器（训练节点）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # 1. 克隆 RLinf
   git clone https://github.com/Brunch-Life/RLinf.git
   cd RLinf
   git checkout feature/real_sft

推荐使用 Docker 镜像，无需手动安装依赖。

.. code-block:: bash

   # 拉取 Docker 镜像（国内可用 docker.1ms.run 加速）
   docker pull rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0

   # 启动容器
   docker run -it --gpus all \
       --shm-size 100g \
       --net=host \
       --name rlinf \
       -e NVIDIA_DRIVER_CAPABILITIES=all \
       -v /path/to/RLinf:/workspace/RLinf \
       rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0 /bin/bash

   cd /workspace/RLinf

.. note::

   如果不使用 Docker，也可以用 ``install.sh`` 安装（仅限 22.04）：

   .. code-block:: bash

      bash requirements/install.sh embodied --model openpi --env maniskill_libero

----

3. 数据采集
----------

数据采集使用两个节点：4090 服务器（Ray Head + env worker）和 NUC（FrankaController）。

RLinf 提供两种采集脚本，适用于不同训练流程：

- ``collect_data.sh`` — 使用原始 ``TrajectoryReplayBuffer`` 格式（``.pt``），适用于 RLPD / 在线 RL 训练流程。全程自动录制，无键盘控制。
- ``collect_data_with_wrapper.sh`` — 使用 ``CollectEpisode`` wrapper，输出 LeRobot v2.0 格式，支持键盘交互式录制（``a`` = 开始录制, ``b`` = 失败, ``c`` = 成功）。**本文档以此脚本为例。**

3.1 配置文件
~~~~~~~~~~~~

本节使用 ``collect_data_with_wrapper.sh``（LeRobot wrapper 版本），对应配置文件
``examples/embodiment/config/realworld_collect_data_zed_robotiq.yaml``。

.. note::

   如果需要采集 RLPD / 在线 RL 训练用的 ``.pt`` 格式数据，请使用 ``collect_data.sh`` 配合
   ``realworld_collect_data.yaml``，该流程无需以下 wrapper 相关参数。

关键配置项：

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - 配置项
     - 值
     - 说明
   * - ``cluster.num_nodes``
     - ``2``
     - 双节点：4090 + NUC
   * - ``env.eval.use_gello``
     - ``True``
     - 使用 GELLO 遥操作
   * - ``env.eval.use_spacemouse``
     - ``False``
     - 不使用 SpaceMouse
   * - ``runner.num_data_episodes``
     - ``50``
     - 采集 50 个 episode
   * - ``runner.export_format``
     - ``"lerobot"``
     - 输出 LeRobot v2.0 格式（wrapper 专用）
   * - ``runner.fps``
     - ``10``
     - 采集帧率 10Hz（wrapper 专用）
   * - ``runner.only_success``
     - ``True``
     - 只保存成功的 episode（wrapper 专用）
   * - ``camera_type``
     - ``"zed"``
     - ZED 相机
   * - ``gripper_type``
     - ``"robotiq"``
     - Robotiq 夹爪

根据实际情况修改配置文件中的 **相机序列号**、**机械臂 IP**、**夹爪串口**、**GELLO 串口**、**NUC Python 路径** 和 **初始末端位姿** ``target_ee_pose``。

3.2 启动步骤
~~~~~~~~~~~~

**Step 1：NUC 端启动**

.. warning::

   **关键：环境变量必须在 ray start 之前设置！**

   Ray 在 ``ray start`` 时 **一次性捕获** 当前 shell 的所有环境变量（包括 ``PATH``、``PYTHONPATH``、``ROS_*`` 等）。之后启动的 Ray worker 进程将继承这些值。如果在 ``ray start`` **之后** 才 source ROS 或 venv，NUC 上的 FrankaController 将找不到 ROS / Python 包，报出难以排查的 import 错误。

.. code-block:: bash

   # 1. 激活 ROS 环境（必须在 ray start 之前）
   source /opt/ros/noetic/setup.bash

   # 2. 激活 RLinf 虚拟环境（必须在 ray start 之前）
   source ~/RLinf/.venv/bin/activate

   # 3. 以 Worker 身份加入 Ray 集群
   RLINF_NODE_RANK=1 ray start --address=<4090服务器IP>:6379

**Step 2：4090 服务器端启动**

.. code-block:: bash

   # 1. 进入 RLinf 目录
   cd /path/to/RLinf

   # 2. 激活虚拟环境（必须在 ray start 之前）
   source .venv/bin/activate

   # 3. 启动 Ray Head 节点
   RLINF_NODE_RANK=0 ray start --head --port=6379 --node-ip-address=<4090服务器IP>

   # 4. 等待 NUC 加入集群（可选检查）
   ray status

   # 5. 启动数据采集（LeRobot 格式 + 键盘控制）
   #    脚本内部会自动设置 EMBODIED_PATH，无需手动 export
   bash examples/embodiment/collect_data_with_wrapper.sh realworld_collect_data_zed_robotiq

.. note::

   若需要采集 RLPD 训练用的 ``.pt`` 格式数据，改用 ``collect_data.sh``：

   .. code-block:: bash

      bash examples/embodiment/collect_data.sh realworld_collect_data_zed_robotiq

3.3 采集时键盘控制（仅 wrapper 版本）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

以下键盘控制仅在使用 ``collect_data_with_wrapper.sh`` 时生效。使用原始 ``collect_data.sh`` 时无键盘控制，全程自动录制。

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - 按键
     - 功能
   * - **a**
     - **开始录制**。按下前可用 GELLO 自由移动机械臂定位到初始位置，此阶段不记录数据
   * - **c**
     - **标记成功** 并结束当前 episode（reward = +1）
   * - **b**
     - **标记失败** 并结束当前 episode（reward = -1）

典型工作流：

1. 用 GELLO 移动机械臂到合适的初始位姿
2. 按 ``a`` 开始录制
3. 用 GELLO 操作机械臂完成任务
4. 任务成功按 ``c``，失败按 ``b``
5. 系统自动进入下一个 episode，重复上述步骤

3.4 数据输出
~~~~~~~~~~~~

采集完成后，数据以 LeRobot v2.0 格式保存：

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

4. 数据处理
-----------

4.1 传输数据到训练服务器
~~~~~~~~~~~~~~~~~~~~~~~~

将采集到的数据集同步到远端训练服务器：

.. code-block:: bash

   # 从本地 4090 服务器同步到远端 A100
   # <REMOTE_HOST> 替换为远端服务器 SSH 别名或 user@ip
   # <REMOTE_RLINF_PATH> 替换为远端 RLinf 仓库路径
   rsync -avhzP logs/<timestamp>/lerobot_dataset/ \
       <REMOTE_HOST>:<REMOTE_RLINF_PATH>/dataset/<DATASET_REPO_ID>/

.. note::

   **占位符说明：**

   - ``<REMOTE_HOST>``：远端服务器 SSH 别名或 ``user@ip:port`` 格式（如 ``my-server`` 或 ``user@192.168.1.100``）
   - ``<REMOTE_RLINF_PATH>``：远端 RLinf 仓库路径（如 ``/workspace/RLinf`` 或 ``/mnt/data/RLinf``）
   - ``<DATASET_REPO_ID>``：数据集标识（如 ``<user>/<dataset>``）

4.2 转换 norm_stats
~~~~~~~~~~~~~~~~~~~~

Pi0 模型需要 ``norm_stats.json`` 进行归一化。在训练服务器上使用工具脚本从 ``stats.json`` 生成：

.. code-block:: bash

   python toolkits/convert_stats_to_norm_stats.py \
       --stats-json dataset/<DATASET_REPO_ID>/meta/stats.json \
       --output-dir checkpoints/torch/pi0_base/<DATASET_REPO_ID> \
       --select-state-dims 4 5 6 7 8 9 0 \
       --action-dim 32

**具体示例：** 假设你的数据集标识为 ``alice/pick_place_chip``，Pi0 预训练权重在
``/workspace/RLinf/checkpoints/torch/pi0_base``：

.. code-block:: bash

   # 数据集路径：/workspace/RLinf/dataset/alice/pick_place_chip/meta/stats.json
   # norm_stats 输出：/workspace/RLinf/checkpoints/torch/pi0_base/alice/pick_place_chip/norm_stats.json

   python toolkits/convert_stats_to_norm_stats.py \
       --stats-json dataset/alice/pick_place_chip/meta/stats.json \
       --output-dir checkpoints/torch/pi0_base/alice/pick_place_chip \
       --select-state-dims 4 5 6 7 8 9 0 \
       --action-dim 32

.. note::

   **路径规范：** ``<DATASET_REPO_ID>`` 应采用 ``<用户名>/<数据集名>`` 格式（如
   ``alice/pick_place_chip``）。``--output-dir`` 必须是模型 checkpoint 路径 +
   ``<DATASET_REPO_ID>`` 子目录，OpenPI 通过这个路径约定自动查找 ``norm_stats.json``。

参数说明：

- ``--stats-json``：数据集的 ``stats.json`` 路径
- ``--output-dir``：输出 ``norm_stats.json`` 的目录，需与模型 checkpoint 路径下的数据集名称对应
- ``--select-state-dims``：从原始 state 向量中选取的维度索引（对应 ``pi0_realworld_pnp`` 配置）
- ``--action-dim``：Pi0 最大 action/state 维度（零填充至此维度）

.. note::

   如果使用 OpenPI 原生训练（方案 B），还需要将 norm_stats 复制到 OpenPI 的 assets 目录，详见 :ref:`5.3 节 <方案B>`.

----

5. 训练
-------

训练在远端 A100 服务器上进行。提供两种训练方案：

- **方案 A：RLinf SFT** — 使用 RLinf 框架的分布式训练流程，支持 value head、后续 RL 微调
- **方案 B：OpenPI 原生 PyTorch** — 使用 OpenPI 自带的 ``torchrun`` 训练脚本，更轻量

两种方案训练出的 checkpoint 均可用于后续的 `部署评估`_ 。

5.1 准备工作（通用）
~~~~~~~~~~~~~~~~~~~~

确保以下文件已就绪：

- 数据集：``<REMOTE_RLINF_PATH>/dataset/<DATASET_REPO_ID>/``
- Pi0 预训练权重：``<REMOTE_RLINF_PATH>/checkpoints/torch/pi0_base``
- norm_stats：``checkpoints/torch/pi0_base/<DATASET_REPO_ID>/norm_stats.json``

5.2 方案 A：使用 RLinf SFT 训练
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

配置文件：``examples/sft/config/custom_sft_openpi.yaml``

.. note::

   配置文件中的 ``train_data_paths`` 和 ``model_path`` 是 **占位符**（``/path/to/custom-data``、``/path/to/pi0-model``），必须替换为实际路径。

关键训练参数：

.. list-table::
   :header-rows: 1
   :widths: 28 20 15 37

   * - 参数
     - 配置文件默认值
     - 建议值（8卡）
     - 说明
   * - ``data.train_data_paths``
     - ``"/path/to/custom-data"``
     - 数据集绝对路径
     - 如 ``"/workspace/RLinf/dataset/<DATASET_REPO_ID>"``
   * - ``actor.model.model_path``
     - ``"/path/to/pi0-model"``
     - Pi0 预训练权重路径
     - 如 ``"/workspace/RLinf/checkpoints/torch/pi0_base"``
   * - ``actor.model.openpi.config_name``
     - ``"pi0_realworld_pnp"``
     - ``"pi0_realworld_pnp"``
     - OpenPI 配置名（ZED+Robotiq 真机场景）
   * - ``actor.micro_batch_size``
     - ``1``
     - ``16``
     - 每卡 batch size，根据 GPU 显存调整
   * - ``actor.global_batch_size``
     - ``16``
     - ``128``
     - 全局 batch size（8 卡 × 16 = 128）
   * - ``actor.optim.lr``
     - ``7.91e-6``
     - ``2.5e-5``
     - 学习率，可根据实际训练调整
   * - ``runner.max_steps``
     - ``-1`` （不限制）
     - ``30000``
     - 总训练步数，``-1`` 表示按 ``max_epochs`` 控制
   * - ``runner.save_interval``
     - ``10``
     - ``1000``
     - 每 N 步保存 checkpoint
   * - ``actor.model.add_value_head``
     - ``True``
     - ``True``
     - 添加 value head（纯 SFT 可设为 ``False``）

**启动训练：**

.. code-block:: bash

   # 进入 Docker 容器
   docker run -it --gpus all --shm-size 100g --net=host --name rlinf \
       -e NVIDIA_DRIVER_CAPABILITIES=all -v /mnt:/mnt \
       rlinf/rlinf:agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0 /bin/bash

   # 容器内
   cd /workspace/RLinf

   # 切换到 openpi 环境（Docker 镜像内）
   source switch_env openpi

   # 启动 Ray
   ray start --head

   # 设置环境变量
   export EMBODIED_PATH="$(pwd)/examples/embodiment"

   # 启动训练
   bash examples/sft/run_vla_sft.sh custom_sft_openpi

**RLinf 训练输出：**

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

评估时需要用到的是 ``actor/model_state_dict/`` 目录（包含 ``.pt`` 或 ``.safetensors`` 文件）。

.. _方案B:

5.3 方案 B：使用 OpenPI 原生 PyTorch 训练
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

如果不需要 RLinf 的分布式调度和 value head，或者需要对比，可以直接使用 OpenPI 的原生训练脚本。

前提条件
^^^^^^^^

1. 安装 OpenPI 仓库：

.. code-block:: bash

   git clone https://github.com/RLinf/openpi.git
   cd openpi
   pip install -e .

2. **对齐数据处理流程：** OpenPI 侧的 ``pi0_realworld_pnp`` 配置需要与 RLinf 的数据处理保持一致。主要涉及以下修改（请确保你的 OpenPI 版本已包含这些改动）：

   - ``src/openpi/policies/realworld_policy.py``：``RealworldInputs`` 中的 state 维度选择（19D→7D）、多相机 slot 映射
   - ``src/openpi/training/config.py``：``pi0_realworld_pnp`` 的 ``state_indices``、``extra_image_keys``、``pi0_slot_keys`` 等参数

3. 复制 norm_stats 到 OpenPI 的 assets 目录：

.. code-block:: bash

   cp <REMOTE_RLINF_PATH>/checkpoints/torch/pi0_base/<DATASET_REPO_ID>/norm_stats.json \
      <OPENPI_PATH>/assets/pi0_realworld_pnp/<DATASET_REPO_ID>/norm_stats.json

相机 Slot 映射
^^^^^^^^^^^^^^

Pi0 预训练模型有固定的相机语义 slot，需要将你的相机正确映射：

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Pi0 Slot
     - 预训练语义
     - 实际相机
   * - ``base_0_rgb``
     - 全局 / 第三人称视角
     - ``extra_image_0``（左侧站立相机）
   * - ``left_wrist_0_rgb``
     - 腕部 / 近距离视角
     - ``image``（腕部相机）
   * - ``right_wrist_0_rgb``
     - 第二视角
     - ``extra_image_1``（右侧站立相机）

.. note::

   如果你的相机布局不同，需要相应调整 ``pi0_slot_keys`` 配置。

启动训练
^^^^^^^^

.. code-block:: bash

   cd <OPENPI_PATH>

   # 指定数据集路径（LeRobot 格式）
   export HF_LEROBOT_HOME="<REMOTE_RLINF_PATH>/dataset"
   export HF_HUB_OFFLINE=1

   # 8 卡训练
   uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 \
       scripts/train_pytorch.py pi0_realworld_pnp \
       --exp_name <EXPERIMENT_NAME>

.. note::

   **超参数对齐：** 如果需要与 RLinf 方案 A 做对比实验，确保超参数一致。

**OpenPI 训练输出：**

.. code-block:: text

   <OPENPI_PATH>/outputs/<EXPERIMENT_NAME>/trial/
   ├── checkpoint-1000/
   │   ├── model.safetensors
   │   └── ...
   ├── checkpoint-2000/
   │   └── ...
   └── checkpoint-30000/
       └── ...

评估时使用对应的 checkpoint 目录路径。

----

.. _部署评估:

6. 部署评估
-----------

部署评估采用集成 rollout worker 架构，模型推理直接运行在 4090 服务器上，无需独立的 Policy Server 或 SSH 隧道。只需两台机器：4090 服务器（actor + rollout + env worker）和 NUC（FrankaController）。

6.1 传输 checkpoint 到 4090 服务器
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

将 A100 训练好的 checkpoint 同步到 4090 服务器：

.. code-block:: bash

   # 保留目录结构，rsync 整个 global_step 目录
   rsync -avhzP <REMOTE_HOST>:<REMOTE_RLINF_PATH>/logs/<timestamp>/test_openpi/checkpoints/global_step_<N>/ \
       /path/to/RLinf/checkpoints/realworld_pnp/

.. note::

   需要从 ``global_step_<N>/`` 层级开始 rsync（而非 ``actor/model_state_dict/``），以保留
   ``actor/model_state_dict/full_weights.pt`` 的目录结构。代码会按
   ``<dir>/full_weights.pt``、``<dir>/model_state_dict/full_weights.pt``、
   ``<dir>/actor/model_state_dict/full_weights.pt``、``<dir>/*.safetensors``
   的顺序查找权重文件。

6.2 部署配置
~~~~~~~~~~~~~

配置文件：``examples/embodiment/config/realworld_pi0_zed_robotiq_async.yaml``

该配置将 actor（训练/推理）、rollout worker（策略推理）、env worker（环境交互 + 相机）全部部署在 4090 服务器上，NUC 仅运行 FrankaController。

关键配置项：

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - 配置项
     - 默认值
     - 说明
   * - ``runner.only_eval``
     - ``False``
     - 设为 ``True`` 仅评估（不训练）
   * - ``runner.ckpt_path``
     - ``null``
     - 指向 ``.pt`` checkpoint 文件路径
   * - ``actor.model.model_path``
     - ``"/path/to/model"``
     - Pi0 预训练或微调后的模型路径
   * - ``rollout.model.model_path``
     - ``"/path/to/model"``
     - 与 actor 一致
   * - ``actor.model.openpi.config_name``
     - ``"pi0_realworld_pnp"``
     - OpenPI 配置名
   * - ``env.train.task_description``
     - ``null``
     - 任务描述 prompt（如 ``"pick up the chip"``）
   * - ``env.train.keyboard_reward_wrapper``
     - ``"single_stage"``
     - 键盘控制奖励模式

6.3 启动部署
~~~~~~~~~~~~~

**Step 1：NUC 端**

.. warning::

   **关键：** 与数据采集一样，必须在 ``ray start`` **之前** source ROS 和虚拟环境。如果 NUC 上已有旧的 Ray 进程，先执行 ``ray stop`` 清理。

.. code-block:: bash

   # 0. 清理旧的 Ray 进程（如果之前运行过数据采集等）
   ray stop

   # 1. 激活 ROS 环境（必须在 ray start 之前）
   source /opt/ros/noetic/setup.bash

   # 2. 激活 RLinf 虚拟环境（必须在 ray start 之前）
   source ~/RLinf/.venv/bin/activate

   # 3. 以 Worker 身份加入 Ray 集群
   RLINF_NODE_RANK=1 ray start --address=<4090服务器IP>:6379

**Step 2：4090 服务器端**

.. code-block:: bash

   # 0. 清理旧的 Ray 进程（如果之前运行过数据采集等）
   ray stop

   # 1. 进入 RLinf 目录
   cd /path/to/RLinf

   # 2. 激活虚拟环境（必须在 ray start 之前）
   source .venv/bin/activate

   # 3. 启动 Ray Head 节点
   RLINF_NODE_RANK=0 ray start --head --port=6379 --node-ip-address=<4090服务器IP>

   # 4. 等待 NUC 加入集群
   ray status

   # 5a. 仅评估模式：加载 checkpoint 运行评估
   #     脚本内部会自动设置 EMBODIED_PATH 和 PYTHONPATH，无需手动 export
   bash examples/embodiment/run_realworld_async.sh realworld_pi0_zed_robotiq_async \
       runner.only_eval=True \
       runner.ckpt_path=/path/to/checkpoints/realworld_pnp/full_weights.pt

   # 5b. 在线 RL 模式：加载 SFT checkpoint 继续在线训练（SAC）
   bash examples/embodiment/run_realworld_async.sh realworld_pi0_zed_robotiq_async \
       runner.ckpt_path=/path/to/checkpoints/realworld_pnp/full_weights.pt

6.4 评估时键盘控制（single_stage 模式）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

评估 / 在线训练过程中需要人工通过键盘标记每个 episode 的结果：

.. list-table::
   :header-rows: 1
   :widths: 10 90

   * - 按键
     - 功能
   * - **a**
     - **中性**：reward = 0，不结束 episode
   * - **b**
     - **失败**：reward = -1，结束当前 episode
   * - **c**
     - **成功**：reward = +1，结束当前 episode

典型工作流：

1. 启动后，rollout worker 在本地进行策略推理，机械臂开始执行动作
2. 观察机械臂执行情况
3. 任务成功按 ``c``，失败按 ``b``
4. 系统自动进入下一个 episode

----

附录 A：键盘控制速查表
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - 场景
     - a
     - b
     - c
   * - **数据采集** (``collect_data_with_wrapper.sh``)
     - 开始录制
     - 失败并结束 episode
     - 成功并结束 episode
   * - **部署评估 / 在线 RL** (``run_realworld_async.sh``, single_stage)
     - 中性 (0, 不结束)
     - 失败 (-1, 结束)
     - 成功 (+1, 结束)

.. note::

   - 数据采集时 ``b`` = 失败，评估时 ``b`` = 失败，含义一致；但数据采集时 ``a`` = 开始录制，评估时 ``a`` = 中性（无操作），请留意区别。
   - 键盘控制仅适用于 ``collect_data_with_wrapper.sh``（wrapper 版本）。原始 ``collect_data.sh``（Replay Buffer 版本）无键盘控制，全程自动录制。

----

附录 B：关键文件索引
---------------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - 文件
     - 作用
   * - ``examples/embodiment/config/realworld_collect_data.yaml``
     - 数据采集配置（Replay Buffer / RL 训练用）
   * - ``examples/embodiment/config/realworld_collect_data_wrapper.yaml``
     - 数据采集配置（LeRobot wrapper / 单节点基础版）
   * - ``examples/embodiment/config/realworld_collect_data_zed_robotiq.yaml``
     - 数据采集配置（LeRobot wrapper / 双节点 ZED+Robotiq）
   * - ``examples/embodiment/config/realworld_pi0_zed_robotiq_async.yaml``
     - 真机部署评估 / 在线 RL 配置
   * - ``examples/embodiment/collect_data.sh``
     - 数据采集启动脚本（Replay Buffer / ``.pt`` 格式）
   * - ``examples/embodiment/collect_real_data.py``
     - 数据采集 Python 入口（Replay Buffer / RL 训练用）
   * - ``examples/embodiment/collect_data_with_wrapper.sh``
     - 数据采集启动脚本（LeRobot 格式 + 键盘控制）
   * - ``examples/embodiment/collect_real_data_with_wrapper.py``
     - 数据采集 Python 入口（CollectEpisode wrapper）
   * - ``examples/embodiment/run_realworld_async.sh``
     - 真机部署 / 在线 RL 启动脚本
   * - ``examples/embodiment/train_async.py``
     - 异步训练 / 部署 Python 入口
   * - ``examples/sft/config/custom_sft_openpi.yaml``
     - RLinf Pi0 SFT 训练配置（含占位符路径，需替换）
   * - ``toolkits/convert_stats_to_norm_stats.py``
     - stats.json → norm_stats.json 转换工具
   * - ``ray_utils/realworld/setup_before_ray.sh``
     - 真机 Ray 启动前环境配置模板
   * - ``requirements/install.sh``
     - RLinf 依赖安装脚本
   * - ``rlinf/models/embodiment/openpi/dataconfig/__init__.py``
     - RLinf 侧 ``pi0_realworld_pnp`` 配置定义
   * - ``openpi/src/openpi/policies/realworld_policy.py``
     - OpenPI 侧 RealworldInputs（需与 RLinf 对齐）
   * - ``openpi/src/openpi/training/config.py``
     - OpenPI 侧 ``pi0_realworld_pnp`` 训练配置（需与 RLinf 对齐）
   * - ``openpi/scripts/train_pytorch.py``
     - OpenPI 原生 PyTorch 训练入口

----

附录 C：常见问题
-----------------

Q1: Ray 集群连不上怎么办？
~~~~~~~~~~~~~~~~~~~~~~~~~~

- 确保 4090 服务器和 NUC 在同一局域网
- 检查防火墙是否开放 6379 端口
- NUC 端 ``ray start --address=<IP>:6379`` 中的 IP 必须是 4090 服务器的 IP
- 使用 ``ray status`` 检查集群状态，确认两个节点都已加入

Q2: NUC 端报 ROS 相关 import 错误？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ray 在 ``ray start`` 时捕获当前环境变量。确保在执行 ``ray start`` 之前已经 source 了 ROS 和虚拟环境：

.. code-block:: bash

   source /opt/ros/noetic/setup.bash
   source ~/RLinf/.venv/bin/activate
   RLINF_NODE_RANK=1 ray start --address=<IP>:6379

Q3: 如何更换任务 / 初始位姿？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

修改配置文件中的 ``env.train.override_cfg.target_ee_pose``，该值为 ``[x, y, z, rx, ry, rz]`` 格式的初始末端位姿（欧拉角表示）。

Q4: 如何只做纯 SFT（不加 value head）？
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

将训练配置 ``custom_sft_openpi.yaml`` 中的 ``actor.model.add_value_head`` 设为 ``False``。
