# XSquare Turtle2 / x2robot 真机部署运行指南

本指南详细介绍了在硬件和软件依赖（XSquare Docker、RLinf 虚拟环境等）已就绪的前提下，如何逐步在 GPU 推理端与工控机（真机）端配置并启动 OpenPI 绝对位姿推理任务。

---

## 1. 物理参数标定与 YAML 配置文件更新

在正式上电运行前，必须标定目标与复位位姿，并将对应的参数写入配置文件。这是确保机械臂不发生物理碰撞的**最关键前置步骤**。

### A. 获取目标与复位位姿（欧拉角格式）
1. 使用 XSquare 官方提供的手动示教或控制界面，将机械臂分别移动至：
   *   **复位安全中心**（即复位后手臂停留的悬空位置）。
   *   **任务终点位置**（例如按键按钮表面）。
2. 在控制界面读取两手爪末端执行器当前的绝对位姿。
3. XSquare 约定的位姿格式为 6 维：`[x, y, z, rz, ry, rx]`（单位：米、弧度）。

### B. 校准环境配置文件
打开 `examples/embodiment/config/env/realworld_x2robot.yaml`，按标定数据更新以下字段：
```yaml
override_cfg:
  is_dummy: False                 # 确保物理运行前该值设为 False
  use_arm: dual                 # 双臂模式（或单臂 left/right）
  use_arm_ids: [0, 1]           # 0 为左臂，1 为右臂
  use_camera_ids: [0, 1, 2]     # 启用的机载摄像头 ID
  camera_names: [face_view, left_wrist_view, right_wrist_view]
  
  # [[左臂绝对位姿], [右臂绝对位姿]]
  target_ee_pose: 
    - [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]                     # 左臂目标位姿
    - [0.3, 0.0, 0.15, 0.0, 1.0, 0.0]                    # 右臂目标位姿
    
  reset_ee_pose: 
    - [0.3, 0.0, 0.1, 0.0, 0.0, 0.0]                     # 左臂复位安全位姿
    - [0.1, 0.0, 0.1, 0.0, 0.8, 0.0]                     # 右臂复位安全位姿
```

### C. 校准全局评估配置文件
打开 `examples/embodiment/config/realworld_x2robot_eval.yaml`，确保核心维度及路径与你转换的 Checkpoint 对应：
```yaml
runner:
  ckpt_path: "/path/to/x2robot_ckpt.pt"                  # RLinf 覆盖权重路径
  only_eval: True                                        # 必须为 True

algorithm:
  eval_rollout_epoch: 1                                  # 运行轮数（确保存在，避免缺失配置报错）

rollout:
  model:
    model_path: "/path/to/x2robot_model"                 # HuggingFace 格式模型及 norm_stats 路径

actor:
  model:
    num_action_chunks: 10                                # 动作 Horizon（必须与 Checkpoint 一致）
    state_dim: 14                                        # 双臂：14 维 (7 * 2)
    action_dim: 14                                       # 双臂：14 维 (7 * 2)
```

---

## 2. 集群组网与环境变量配置

RLinf 基于 Ray 进行分布式调度。推理节点（GPU PC）作为 Head，控制节点（Robot PC）作为 Worker 加入。

### A. 节点 1：机器人控制PC (Rank 1)
配置局域网网卡以及节点 Rank，使 Ray 能够正确将 Env 调度到工控机端：
```bash
# 激活虚拟环境
source .venv/bin/activate

# 设定节点配置（在工控机上执行）
export PYTHONPATH=/path/to/your/RLinf:$PYTHONPATH
export RLINF_NODE_RANK=1
export RLINF_COMM_NET_DEVICES=eth0    # 用于通信的局域网物理网卡
```

### B. 节点 0：GPU 推理 PC (Rank 0 / Head)
```bash
# 激活环境或进入 Docker 容器
source .venv/bin/activate

# 设定节点配置（在推理 PC 上执行）
export PYTHONPATH=/workspace/RLinf:$PYTHONPATH
export RLINF_NODE_RANK=0
export RLINF_COMM_NET_DEVICES=eth0    # 与工控机相连的物理网卡
```

---

## 3. 启动 Ray 集群

### A. 启动 Head 节点（在 GPU PC 执行）
```bash
ray start --head --port=6379 --node-ip-address=<gpu_pc_ip_address>
```

### B. 启动 Worker 节点（在控制 PC 执行）
```bash
ray start --address='<gpu_pc_ip_address>:6379'
```
*注：启动后可在任一节点执行 `ray status`，验证两台机器是否已经顺利组网。*

---

## 4. 前置安全自检 (Dummy 验证)

在正式让机械臂物理动起来之前，建议进行 **Dummy 闭环自检**，以验证两端通信与 VLA 推理速度是否达到预期：

1. 在 `examples/embodiment/config/env/realworld_x2robot.yaml` 中，临时将 `is_dummy: False` 修改为 `is_dummy: True`。
2. 在 GPU PC 终端执行评估命令：
   ```bash
   bash examples/embodiment/eval_embodiment.sh realworld_x2robot_eval
   ```
3. 观察输出日志，若两端顺利完成了 Ray Channel 的 Observation 和 Actions 数据包同步，代表分布式链路搭建完成。验证通过后，将 `is_dummy` 改回 `False`。

---

## 5. 闭环控制流与时序通信示意图

RLinf 并非采用异步高频队列，而是通过同步的 Ping-Pong 机制保证执行的安全与精准：

```
  [GPU/Inference 节点 Rank 0]               [Robot 控制 PC 节点 Rank 1]
           │                                          │
           │                                          │ 1. 读当前摄像头、关节状态
           │            Ray Channel: Get/Get          │    拼接出 14 维 TCP 位姿和 3 路图像
           │ <────────────────────────────────────────│
           │                                          │
 2. 运行 OpenPI/Pi0 前向推理                          │
    输入 obs 图像和 state                             │
    截断取前 10 帧 [10, 14] actions                    │
           │                                          │
           │            Ray Channel: Put/Put          │
           │ ────────────────────────────────────────>│
           │                                          │
           │                                          │ 3. 消费 10 帧 action chunk：
           │                                          │    在 20Hz 主时钟下
           │                                          │    每帧通过 nlerp 线性插值
           │                                          │    拆出 3 个 60Hz 绝对位姿子命令
           │                                          │    调用 move_abs().wait() 执行
           │                                          │
```

### 频率与时间单位解析
*   **动作 Chunk 的物理长度**：
    *   模型每轮推理输出包含 $10$ 帧绝对动作（即 Action Horizon = 10）。
    *   Env 按照 $20\text{ Hz}$（每帧间隔 $50\text{ ms}$）的节奏本地消费这 10 帧动作，因此一个完整 Action Chunk 在物理世界中执行时长为 $10 / 20\text{ Hz} = 0.5\text{ s}$。
*   **通信与推理周期**：
    *   Ray Channel 每轮在两端之间传递**一次**包含 10 帧动作的 Chunk（通信频率理想状况下约 $2\text{ Hz}$）。
    *   真机端在 $0.5\text{ s}$ 的 Chunk 执行完毕并刷新最新一帧图像/状态后，才会“拉动”并触发下一次 VLA 推理。
*   **真机执行精度**：
    *   控制节点的低层执行周期为 $60\text{ Hz}$。
    *   在 env 侧，每一个 $20\text{ Hz}$ 的动作帧会被分解为 $60 / 20 = 3$ 个绝对坐标子点。经过 Safety Box 范围裁剪后，最终通过 `Turtle2Controller.arms_control(left, right)` 同步阻塞执行，从而实现极佳的轨迹平滑度。
