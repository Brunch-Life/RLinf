# RLinf 真机部署文档：Franka + ZED + Robotiq (Pi0)

> 本文档记录使用 RLinf 在 Franka Panda 真机环境下，基于 ZED 相机和 Robotiq 夹爪，完成**数据采集 → SFT 训练 → 真机部署评估**的完整流程。

---

## 目录

- [1. 硬件架构](#1-硬件架构)
- [2. 环境安装](#2-环境安装)
- [3. 数据采集](#3-数据采集)
- [4. 数据处理](#4-数据处理)
- [5. 训练](#5-训练)
  - [5.2 方案 A：RLinf SFT](#52-方案-a使用-rlinf-sft-训练)
  - [5.3 方案 B：OpenPI 原生 PyTorch](#53-方案-b使用-openpi-原生-pytorch-训练)
- [6. 部署评估](#6-部署评估)
- [附录 A：键盘控制速查表](#附录-a键盘控制速查表)
- [附录 B：关键文件索引](#附录-b关键文件索引)
- [附录 C：常见问题](#附录-c常见问题)

---

## 1. 硬件架构

本方案采用三台机器协同工作：

```
┌──────────────────────────────────────────────────────────────┐
│  远端 A100 服务器（训练 + Policy Server）                      │
│   - 多卡 A100 GPU                                            │
│   - Docker 容器内训练 RLinf SFT                               │
│   - serve_policy.py 启动 policy server                       │
└───────────────────────────┬──────────────────────────────────┘
                            │ SSH -L 端口转发
┌───────────────────────────┴──────────────────────────────────┐
│  本地 4090 服务器 (node 0, Ubuntu 22.04)                      │
│   - 3× ZED 相机                                              │
│   - GPU 用于图像处理 + env worker                             │
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
```

> **请根据你自己的环境修改以下内容：**
>
> - ZED 相机序列号（当前：`10848563`, `39651335`, `34303972`）
> - Franka 机械臂 IP（当前：`172.16.0.2`）
> - Robotiq 夹爪串口（当前：`/dev/ttyUSB0`）
> - NUC 上 Python 虚拟环境路径（当前：`/home/franka/workspace/RLinf/.venv/bin/python3`）
> - 4090 服务器和 NUC 的 IP 地址
> - 远端 A100 服务器 SSH 地址和端口

---

## 2. 环境安装

### 2.1 NUC 端（Franka 控制器节点）

NUC 需要安装 ROS Noetic、Franka 驱动以及 RLinf 的 franka 环境依赖。

```bash
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
```

> **注意：** NUC 上不需要安装模型依赖，它只运行 FrankaController（机械臂底层控制）。

### 2.2 4090 服务器端（env worker 节点）

4090 服务器同样安装 franka 环境依赖，并额外安装 openpi 相关依赖。

```bash
# 1. 克隆 RLinf
git clone --recurse-submodules https://github.com/Brunch-Life/RLinf.git
cd RLinf
git checkout feature/real_sft

# 2. 安装 franka 环境 + RLinf
bash requirements/install.sh embodied --env franka --install-rlinf

# 3. 激活虚拟环境
source .venv/bin/activate
```

### 2.3 远端 A100 服务器（训练节点）

```bash
# 1. 克隆 RLinf
git clone https://github.com/Brunch-Life/RLinf.git
cd RLinf
git checkout feature/real_sft
```


推荐使用 Docker 镜像，无需手动安装依赖。

```bash
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
```

> 如果不使用 Docker，也可以用 `install.sh` 安装（仅限22.04）：
>
> ```bash
> bash requirements/install.sh embodied --model openpi --env maniskill_libero
> ```

---

## 3. 数据采集

数据采集使用两个节点：4090 服务器（Ray Head + env worker）和 NUC（FrankaController）。

### 3.1 配置文件

本节使用 `collect_data_with_wrapper.sh`（LeRobot wrapper 版本），对应配置文件 `examples/embodiment/config/realworld_collect_data_zed_robotiq.yaml`。

> 如果需要采集 RLPD / 在线 RL 训练用的 `.pt` 格式数据，请使用 `collect_data.sh` 配合 `realworld_collect_data.yaml`，该流程无需以下 wrapper 相关参数。

关键配置项：


| 配置项                        | 值           | 说明                            |
| -------------------------- | ----------- | ----------------------------- |
| `cluster.num_nodes`        | `2`         | 双节点：4090 + NUC                |
| `env.eval.use_gello`       | `True`      | 使用 GELLO 遥操作                  |
| `env.eval.use_spacemouse`  | `False`     | 不使用 SpaceMouse                |
| `runner.num_data_episodes` | `50`        | 采集 50 个 episode               |
| `runner.export_format`     | `"lerobot"` | 输出 LeRobot v2.0 格式（wrapper 专用） |
| `runner.fps`               | `10`        | 采集帧率 10Hz（wrapper 专用）          |
| `runner.only_success`      | `True`      | 只保存成功的 episode（wrapper 专用）     |
| `camera_type`              | `"zed"`     | ZED 相机                        |
| `gripper_type`             | `"robotiq"` | Robotiq 夹爪                    |


根据实际情况修改配置文件中的 **相机序列号**、**机械臂 IP**、**夹爪串口**、**GELLO 串口**、**NUC Python 路径** 和 **初始末端位姿** `target_ee_pose`。

### 3.2 启动步骤

> **两种采集脚本说明：**
>
> - `collect_data.sh` — 使用原始 `TrajectoryReplayBuffer` 格式（`.pt`），适用于 RLPD / 在线 RL 训练流程。全程录制，无键盘控制。
> - `collect_data_with_wrapper.sh` — 使用 `CollectEpisode` wrapper，输出 LeRobot v2.0 格式，支持键盘交互式录制。**本文档以此脚本为例。**

**Step 1：NUC 端启动**

```bash
# 1. 激活 ROS 环境
source /opt/ros/noetic/setup.bash

# 2. 激活 RLinf 虚拟环境
source ~/RLinf/.venv/bin/activate

# 3. 以 Worker 身份加入 Ray 集群
RLINF_NODE_RANK=1 ray start --address=<4090服务器IP>:6379
```

> **重要：** 必须在 `ray start` **之前** source ROS 和虚拟环境，因为 Ray 在启动时捕获环境变量。

**Step 2：4090 服务器端启动**

```bash
# 1. 进入 RLinf 目录
cd /path/to/RLinf

# 2. 激活虚拟环境
source .venv/bin/activate

# 3. 设置环境变量
export EMBODIED_PATH="$(pwd)/examples/embodiment"

# 4. 启动 Ray Head 节点
RLINF_NODE_RANK=0 ray start --head --port=6379 --node-ip-address=<4090服务器IP>

# 5. 等待 NUC 加入集群（可选检查）
ray status

# 6. 启动数据采集（LeRobot 格式 + 键盘控制）
bash examples/embodiment/collect_data_with_wrapper.sh realworld_collect_data_zed_robotiq
```

> 若需要采集 RLPD 训练用的 `.pt` 格式数据，改用 `collect_data.sh`：
>
> ```bash
> bash examples/embodiment/collect_data.sh realworld_collect_data_zed_robotiq
> ```

### 3.3 采集时键盘控制（仅 wrapper 版本）

> 以下键盘控制仅在使用 `collect_data_with_wrapper.sh` 时生效。使用原始 `collect_data.sh` 时无键盘控制，全程自动录制。


| 按键    | 功能                                           |
| ----- | -------------------------------------------- |
| **a** | **开始录制**。按下前可用 GELLO 自由移动机械臂定位到初始位置，此阶段不记录数据 |
| **c** | **标记成功**并结束当前 episode（reward = +1）           |
| **b** | **标记失败**并结束当前 episode（reward = -1）           |


典型工作流：

1. 用 GELLO 移动机械臂到合适的初始位姿
2. 按 `a` 开始录制
3. 用 GELLO 操作机械臂完成任务
4. 任务成功按 `c`，失败按 `b`
5. 系统自动进入下一个 episode，重复上述步骤

### 3.4 数据输出

采集完成后，数据以 LeRobot v2.0 格式保存：

```
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
```

---

## 4. 数据处理

### 4.1 传输数据到训练服务器

将采集到的数据集同步到远端训练服务器：

```bash
# 从本地 4090 服务器同步到远端 A100
# <REMOTE_HOST> 替换为远端服务器 SSH 别名或 user@ip
# <REMOTE_RLINF_PATH> 替换为远端 RLinf 仓库路径
rsync -avhzP logs/<timestamp>/lerobot_dataset/ \
    <REMOTE_HOST>:<REMOTE_RLINF_PATH>/dataset/<DATASET_REPO_ID>/
```

> **占位符说明：**
>
> - `<REMOTE_HOST>`：远端服务器 SSH 别名或 `user@ip:port` 格式（如 `my-server` 或 `user@192.168.1.100`）
> - `<REMOTE_RLINF_PATH>`：远端 RLinf 仓库路径（如 `/workspace/RLinf` 或 `/mnt/data/RLinf`）
> - `<DATASET_REPO_ID>`：数据集标识（如 `<user>/<dataset>`）

### 4.2 转换 norm_stats

Pi0 模型需要 `norm_stats.json` 进行归一化。在训练服务器上使用工具脚本从 `stats.json` 生成：

```bash
python toolkits/convert_stats_to_norm_stats.py \
    --stats-json dataset/<DATASET_REPO_ID>/meta/stats.json \
    --output-dir checkpoints/torch/pi0_base/<DATASET_REPO_ID> \
    --select-state-dims 4 5 6 7 8 9 0 \
    --action-dim 32
```

参数说明：

- `--stats-json`：数据集的 `stats.json` 路径
- `--output-dir`：输出 `norm_stats.json` 的目录，需与模型 checkpoint 路径下的数据集名称对应
- `--select-state-dims`：从原始 state 向量中选取的维度索引（对应 `pi0_custom` 配置）
- `--action-dim`：Pi0 最大 action/state 维度（零填充至此维度）

> 如果使用 OpenPI 原生训练（方案 B），还需要将 norm_stats 复制到 OpenPI 的 assets 目录，详见 [5.3 节](#53-方案-b使用-openpi-原生-pytorch-训练)。

---

## 5. 训练

训练在远端 A100 服务器上进行。提供两种训练方案：

- **方案 A：RLinf SFT** — 使用 RLinf 框架的分布式训练流程，支持 value head、后续 RL 微调
- **方案 B：OpenPI 原生 PyTorch** — 使用 OpenPI 自带的 `torchrun` 训练脚本，更轻量

两种方案训练出的 checkpoint 均可用于后续的 [部署评估](#6-部署评估)。

### 5.1 准备工作（通用）

确保以下文件已就绪：

- 数据集：`<REMOTE_RLINF_PATH>/dataset/<DATASET_REPO_ID>/`
- Pi0 预训练权重：`<REMOTE_RLINF_PATH>/checkpoints/torch/pi0_base`
- norm_stats：`checkpoints/torch/pi0_base/<DATASET_REPO_ID>/norm_stats.json`

### 5.2 方案 A：使用 RLinf SFT 训练

配置文件：`examples/sft/config/custom_sft_openpi.yaml`

配置文件中的 `train_data_paths` 和 `model_path` 是占位符，需替换为实际路径。关键训练参数：


| 参数                               | 配置文件默认值                  | 建议值（8卡）       | 说明                                                    |
| -------------------------------- | ------------------------ | -------------- | ----------------------------------------------------- |
| `data.train_data_paths`          | `"/path/to/custom-data"` | 数据集绝对路径        | 如 `"/workspace/RLinf/dataset/<DATASET_REPO_ID>"`       |
| `actor.model.model_path`         | `"/path/to/pi0-model"`   | Pi0 预训练权重路径    | 如 `"/workspace/RLinf/checkpoints/torch/pi0_base"`      |
| `actor.model.openpi.config_name` | `"pi0_custom"`           | `"pi0_custom"` | OpenPI 配置名                                            |
| `actor.micro_batch_size`         | `1`                      | `16`           | 每卡 batch size，根据 GPU 显存调整                             |
| `actor.global_batch_size`        | `16`                     | `128`          | 全局 batch size（8 卡 × 16 = 128）                         |
| `actor.optim.lr`                 | `7.91e-6`                | `2.5e-5`       | 学习率，可根据实际训练调整                                         |
| `runner.max_steps`               | `-1`（不限制）                | `30000`        | 总训练步数，`-1` 表示按 `max_epochs` 控制                        |
| `runner.save_interval`           | `10`                     | `1000`         | 每 N 步保存 checkpoint                                    |
| `actor.model.add_value_head`     | `True`                   | `True`         | 添加 value head（纯 SFT 可设为 `False`）                      |


**启动训练：**

```bash
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
```

**RLinf 训练输出：**

```
logs/<timestamp>/test_openpi/checkpoints/
├── global_step_1000/
│   └── actor/
│       └── model_state_dict/
│           └── full_weights.pt
├── global_step_2000/
│   └── ...
└── global_step_30000/
    └── ...
```

评估时需要用到的是 `actor/model_state_dict/` 目录（包含 `.pt` 或 `.safetensors` 文件）。

### 5.3 方案 B：使用 OpenPI 原生 PyTorch 训练

如果不需要 RLinf 的分布式调度和 value head，或者需要对比，可以直接使用 OpenPI 的原生训练脚本。

#### 前提条件

1. 安装 OpenPI 仓库：

```bash
git clone https://github.com/RLinf/openpi.git
cd openpi
pip install -e .
```

1. **对齐数据处理流程：** OpenPI 侧的 `pi0_custom` 配置需要与 RLinf 的数据处理保持一致。主要涉及以下修改（请确保你的 OpenPI 版本已包含这些改动）：
  - `src/openpi/policies/franka_policy.py`：`FrankaEEInputs` 中的 state/action padding（7D→32D）、多相机 slot 映射
  - `src/openpi/training/config.py`：`pi0_custom` 的 `select_state_dims`、`extra_image_keys`、`pi0_slot_keys` 等参数
2. 复制 norm_stats 到 OpenPI 的 assets 目录：

```bash
cp <REMOTE_RLINF_PATH>/checkpoints/torch/pi0_base/<DATASET_REPO_ID>/norm_stats.json \
   <OPENPI_PATH>/assets/pi0_custom/<DATASET_REPO_ID>/norm_stats.json
```

#### 相机 Slot 映射

Pi0 预训练模型有固定的相机语义 slot，需要将你的相机正确映射：


| Pi0 Slot            | 预训练语义       | 实际相机                    |
| ------------------- | ----------- | ----------------------- |
| `base_0_rgb`        | 全局 / 第三人称视角 | `extra_image_0`（左侧站立相机） |
| `left_wrist_0_rgb`  | 腕部 / 近距离视角  | `image`（腕部相机）           |
| `right_wrist_0_rgb` | 第二视角        | `extra_image_1`（右侧站立相机） |


> 如果你的相机布局不同，需要相应调整 `pi0_slot_keys` 配置。

#### 启动训练

```bash
cd <OPENPI_PATH>

# 指定数据集路径（LeRobot 格式）
export HF_LEROBOT_HOME="<REMOTE_RLINF_PATH>/dataset"
export HF_HUB_OFFLINE=1

# 8 卡训练
uv run torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    scripts/train_pytorch.py pi0_custom \
    --exp_name <EXPERIMENT_NAME>
```

> **超参数对齐：** 如果需要与 RLinf 方案 A 做对比实验，确保超参数一致。

**OpenPI 训练输出：**

```
<OPENPI_PATH>/outputs/<EXPERIMENT_NAME>/trial/
├── checkpoint-1000/
│   ├── model.safetensors
│   └── ...
├── checkpoint-2000/
│   └── ...
└── checkpoint-30000/
    └── ...
```

评估时使用对应的 checkpoint 目录路径。

---

## 6. 部署评估

部署评估涉及三台机器：远端 A100（Policy Server）、本地 4090（env worker）、NUC（FrankaController）。

### 6.1 远端 A100：启动 Policy Server

```bash
# 在 A100 服务器上
cd /workspace/RLinf

python examples/embodiment/serve_policy.py \
    --config pi0_custom \
    --checkpoint-dir <checkpoint_path>/actor/model_state_dict/ \
    --port 12345 \
    --num-steps 10
```

参数说明：

- `--config`：OpenPI 配置名，需与训练时一致（`pi0_custom`）
- `--checkpoint-dir`：checkpoint 目录路径（包含 `.safetensors` 或 `.pt`），也可以直接指向 `.pt` 文件
- `--port`：服务端口
- `--num-steps`：flow-matching 推理去噪步数（默认 10）
- `--default-prompt`：可选，设置默认 prompt（如 `"pick up the chip"`）

服务启动后会先进行 warmup（默认 3 次 dummy 推理），之后监听 WebSocket 连接。

### 6.2 SSH 隧道打洞

在本地 4090 服务器上建立到远端 A100 的端口转发：

```bash
ssh -L 12345:localhost:12345 <user>@<A100服务器IP> -p <SSH端口>
```

此命令将远端的 12345 端口映射到本地 localhost:12345，使 eval 脚本能通过 `localhost:12345` 访问 policy server。

> 保持此 SSH 连接不要断开，否则隧道会关闭。

### 6.3 评估配置

配置文件位于 `examples/embodiment/config/realworld_eval_zed_robotiq.yaml`，关键配置项：


| 配置项                                | 值                | 说明                     |
| ---------------------------------- | ---------------- | ---------------------- |
| `policy_server.host`               | `"localhost"`    | 通过 SSH 隧道访问            |
| `policy_server.port`               | `12345`          | 与 serve_policy.py 端口一致 |
| `policy_server.config_name`        | `"pi0_custom"`   | OpenPI 配置名             |
| `runner.action_type`               | `"delta"`        | 动作类型：增量                |
| `runner.num_eval_episodes`         | `20`             | 评估 episode 数           |
| `runner.dry_run`                   | `False`          | 设为 `True` 可干跑（不动机器人）   |
| `env.eval.keyboard_reward_wrapper` | `"single_stage"` | 键盘控制奖励的模式              |


### 6.4 启动评估

**Step 1：NUC 端**

```bash
# 1. 激活 ROS 环境
source /opt/ros/noetic/setup.bash

# 2. 激活 RLinf 虚拟环境
source ~/RLinf/.venv/bin/activate

# 3. 以 Worker 身份加入 Ray 集群
RLINF_NODE_RANK=1 ray start --address=<4090服务器IP>:6379
```

**Step 2：4090 服务器端**

```bash
# 1. 进入 RLinf 目录
cd /path/to/RLinf

# 2. 激活虚拟环境
source .venv/bin/activate

# 3. 设置环境变量
export EMBODIED_PATH="$(pwd)/examples/embodiment"

# 4. 启动 Ray Head 节点
RLINF_NODE_RANK=0 ray start --head --port=6379 --node-ip-address=<4090服务器IP>

# 5. 等待 NUC 加入集群
ray status

# 6. 启动评估
bash examples/embodiment/eval_realworld.sh realworld_eval_zed_robotiq
```

### 6.5 评估时键盘控制（single_stage 模式）

评估过程中需要人工通过键盘标记每个 episode 的结果：


| 按键    | 功能                              |
| ----- | ------------------------------- |
| **a** | **失败**：reward = -1，结束当前 episode |
| **b** | **中性**：reward = 0，不结束 episode   |
| **c** | **成功**：reward = +1，结束当前 episode |


典型工作流：

1. 评估脚本自动启动，policy server 返回动作，机械臂开始执行
2. 观察机械臂执行情况
3. 任务成功按 `c`，失败按 `a`
4. 系统自动进入下一个 episode

> **调试技巧：** 首次部署建议先将 `runner.dry_run` 设为 `True` 进行干跑测试，确认 policy server 连接正常、推理无误后再设为 `False` 实际执行。

---

## 附录 A：键盘控制速查表


| 场景                                                          | a           | b             | c             |
| ----------------------------------------------------------- | ----------- | ------------- | ------------- |
| **数据采集** (`collect_data_with_wrapper.sh`)                   | 开始录制        | 失败并结束 episode | 成功并结束 episode |
| **评估** (`eval_realworld.sh`, single_stage)                  | 失败 (-1, 结束) | 中性 (0, 不结束)   | 成功 (+1, 结束)   |


> **注意：**
> - 数据采集和评估的按键含义**不同**，请注意区分。
> - 键盘控制仅适用于 `collect_data_with_wrapper.sh`（wrapper 版本）。原始 `collect_data.sh`（Replay Buffer 版本）无键盘控制，全程自动录制。

---

## 附录 B：关键文件索引


| 文件                                                                   | 作用                                          |
| -------------------------------------------------------------------- | ------------------------------------------- |
| `examples/embodiment/config/realworld_collect_data.yaml`             | 数据采集配置（Replay Buffer / RL 训练用）             |
| `examples/embodiment/config/realworld_collect_data_wrapper.yaml`     | 数据采集配置（LeRobot wrapper / 单节点基础版）           |
| `examples/embodiment/config/realworld_collect_data_zed_robotiq.yaml` | 数据采集配置（LeRobot wrapper / 双节点 ZED+Robotiq） |
| `examples/embodiment/config/realworld_eval_zed_robotiq.yaml`         | 真机评估配置                                      |
| `examples/embodiment/collect_data.sh`                                | 数据采集启动脚本（Replay Buffer / `.pt` 格式）         |
| `examples/embodiment/collect_real_data.py`                           | 数据采集 Python 入口（Replay Buffer / RL 训练用）     |
| `examples/embodiment/collect_data_with_wrapper.sh`                   | 数据采集启动脚本（LeRobot 格式 + 键盘控制）              |
| `examples/embodiment/collect_real_data_with_wrapper.py`              | 数据采集 Python 入口（CollectEpisode wrapper）     |
| `examples/embodiment/eval_realworld.sh`                              | 真机评估启动脚本                                    |
| `examples/embodiment/eval_realworld.py`                              | 真机评估 Python 入口                              |
| `examples/embodiment/serve_policy.py`                                | Policy Server（远端部署）                         |
| `examples/sft/config/custom_sft_openpi.yaml`                         | RLinf Pi0 SFT 训练配置（含占位符路径，需替换）            |
| `toolkits/convert_stats_to_norm_stats.py`                            | stats.json → norm_stats.json 转换工具           |
| `ray_utils/realworld/setup_before_ray.sh`                            | 真机 Ray 启动前环境配置模板                            |
| `requirements/install.sh`                                            | RLinf 依赖安装脚本                                |
| `rlinf/models/embodiment/openpi/dataconfig/__init__.py`              | RLinf 侧 `pi0_custom` 配置定义                   |
| `openpi/src/openpi/policies/franka_policy.py`                        | OpenPI 侧 FrankaEEInputs（需与 RLinf 对齐）        |
| `openpi/src/openpi/training/config.py`                               | OpenPI 侧 `pi0_custom` 训练配置（需与 RLinf 对齐）     |
| `openpi/scripts/train_pytorch.py`                                    | OpenPI 原生 PyTorch 训练入口                      |


---

## 附录 C：常见问题

### Q1: Ray 集群连不上怎么办？

- 确保 4090 服务器和 NUC 在同一局域网
- 检查防火墙是否开放 6379 端口
- NUC 端 `ray start --address=<IP>:6379` 中的 IP 必须是 4090 服务器的 IP
- 使用 `ray status` 检查集群状态，确认两个节点都已加入

### Q2: NUC 端报 ROS 相关 import 错误？

Ray 在 `ray start` 时捕获当前环境变量。确保在执行 `ray start` 之前已经 source 了 ROS 和虚拟环境：

```bash
source /opt/ros/noetic/setup.bash
source ~/RLinf/.venv/bin/activate
RLINF_NODE_RANK=1 ray start --address=<IP>:6379
```

### Q3: SSH 隧道断了怎么办？

重新建立 SSH 隧道即可：`ssh -L 12345:localhost:12345 <user>@<server> -p <port>`。评估脚本会在下一次请求时自动重连 policy server。

### Q4: 如何更换任务 / 初始位姿？

修改配置文件中的 `env.eval.override_cfg.target_ee_pose`，该值为 `[x, y, z, rx, ry, rz]` 格式的初始末端位姿（欧拉角表示）。

### Q5: 如何调整安全边界？

评估配置中的 `ee_pose_limit_min` 和 `ee_pose_limit_max` 定义了末端执行器的安全活动范围。如果全部设为零，系统会根据 `random_*_range` 参数自动计算。建议根据实际场景手动设置合理的范围。

### Q6: 如何只做纯 SFT（不加 value head）？

将训练配置 `custom_sft_openpi.yaml` 中的 `actor.model.add_value_head` 设为 `False`。