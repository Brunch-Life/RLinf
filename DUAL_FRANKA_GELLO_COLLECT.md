# 双臂 Franka + GELLO 数采操作手册

简版，顺着跑就行。配置文件：
[examples/embodiment/config/realworld_collect_data_gello_joint_dual_franka.yaml](examples/embodiment/config/realworld_collect_data_gello_joint_dual_franka.yaml)。

## 1. 前置检查

- 左右 **Franka** 都已按黑键解锁、Desk 里 **FCI 激活**（指示灯绿）
- 两把 **Robotiq 2F-85** 通电，USB-RS485（`DAAIT8PU` / `DAAIR00N`）插好
- 两把 **GELLO**（FTDI `FTAJEDPC` / `FTAM5DHY`）通电，**trigger 手柄归中**
- 三相机插好：`ls /dev/v4l/by-id/` 能看到两条 `XVisio_vSLAM` + 一条 `RealSense_435i`

## 2. 节点分工

| 角色 | 主机 | 直连 IP | WiFi IP（备用） | 负责 |
|---|---|---|---|---|
| head (node 0) | `ubuntu-franka-slave` | **10.10.10.1** | 192.168.120.43 | DataCollector + 左臂 controller + 3 相机 + 2 GELLO |
| worker (node 1) | `sohu-dual-master` | **10.10.10.2** | 192.168.120.42 | 右臂 controller |

### 2.1 网线直连 + IP 设置（一次性）

两台机器之间拉一根 USB Ethernet 网线直连，Ray 跨节点 RPC（franky streamer、gripper、controller state）走这根线；共享 WiFi 有 ~25 ms mdev 抖动，会让 teleop 右臂一卡一卡。

接好线后确认两端都出现了 `enx<MAC>` 形式的网卡（`ip link` 查看），然后分别持久化配置：

```bash
# node 0（10.10.10.1）
nmcli connection add type ethernet ifname enx207bd232e224 con-name rlinf-direct \
    ipv4.addresses 10.10.10.1/24 ipv4.method manual ipv6.method ignore \
    connection.autoconnect yes
nmcli connection up rlinf-direct

# node 1（10.10.10.2） — 通过 ssh sohu-dual-master 过去执行
nmcli connection add type ethernet ifname enx00e04c364742 con-name rlinf-direct \
    ipv4.addresses 10.10.10.2/24 ipv4.method manual ipv6.method ignore \
    connection.autoconnect yes
nmcli connection up rlinf-direct
```

- **MAC / `enx...` 名字每台不同**，用 `ip link` 实际查；如果换了线或换了 USB 网口，重跑 `nmcli connection modify rlinf-direct connection.interface-name enx<新MAC>`。
- 验证：`ping -c3 10.10.10.2`（从 node 0）应拿到 sub-ms RTT；WiFi 的 ~25 ms 是参照物。

### 2.2 如果直连网线临时坏了

`ray_utils/realworld/start_ray_node0.sh` 和 `start_ray_node1.sh` 里把 `HEAD_IP` / `WORKER_IP` 改回 WiFi IP（`192.168.120.43` / `192.168.120.42`），就能退回旧链路跑（会卡，但能跑）。脚本顶部的注释保留了旧 IP 作为注记。

## 3. 启动 Ray（两台分别跑）

```bash
# node 0
bash ray_utils/realworld/start_ray_node0.sh
# node 1
bash ray_utils/realworld/start_ray_node1.sh
```

脚本自己 `ray stop --force` + `source .venv/bin/activate` + `export RLINF_NODE_RANK`，本 shell 不用预先激活。

## 4. 跑采集（在 node 0）

```bash
cd ~/cynws/RLinf
source .venv/bin/activate            # collect_data.sh 不会自动 source
# 编辑 examples/embodiment/config/realworld_collect_data_gello_joint_dual_franka.yaml
#   runner.num_data_episodes: 20     ← 想采多少个成功 episode 改这里
bash examples/embodiment/collect_data.sh realworld_collect_data_gello_joint_dual_franka
```

看到两臂 `Joint impedance tracking motion started` 就绪后开始下一步。

## 5. 键盘控制

进度条窗口**焦点要在终端**（evdev 直读 `/dev/input/event*`，实际上即使别的窗口激活也能触发，但通常把终端留在前台）。按键语义：

| 键 | 语义 |
|---|---|
| `a` | **开始**录制当前 episode；已在录制时按 `a` = 丢弃当前缓冲并重新开始 |
| `b` | **失败结束** — reward = -1, episode 丢弃 |
| `c` | **成功结束** — reward = +1, episode 保存为 parquet |

`only_success: True`（默认）下，`b` 的 episode 只进内存、不落盘。

### 5.1 实时反馈

wrapper 运行在 Ray worker 里，`print` 会被 Ray 的 log monitor 批处理，不够实时。现在改成按键边沿触发（`start` / `restart` / `end_fail` / `end_success`）时调 `self.log_info`，输出类似：

```
[INFO 14:23:07 DataCollector-Rank-0][collect_real_data.py:144] [keyboard] start
[INFO 14:23:42 DataCollector-Rank-0][collect_real_data.py:144] [keyboard] end_success
```

同一时刻也会落到 `logs/<timestamp>/worker_logs/DataCollector/rank_0.log`，可以 `tail -f` 观察：

```bash
tail -f logs/$(ls -t logs | head -1)/worker_logs/DataCollector/rank_0.log
```

如果按了键但终端 **5 秒内还没出现对应的 `[keyboard] ...` 行**，再排查：evdev 设备权限、`RLINF_KEYBOARD_DEVICE` 是否指到了正确的 `/dev/input/eventX`、或进程是否在 sleep。

## 6. 下一个 episode / 终止条件

- 按 `b` 或 `c` 后，两臂**自动 reset 到 home pose**，ready 后按 `a` 开始下一个
- reset 期间**GELLO 放稳**，daemon 还在 1 kHz 发关节目标，race 会导致 reset 偏
- 超时：`max_num_steps: 300` @ 10 Hz ≈ **30 s** 会 truncate，按丢弃处理

## 7. 数据落盘（全在 node 0）

每次采集新建一个时间戳目录：

```
logs/YYYYMMDD-HH:MM:SS/
├── run_embodiment.log                                   # 完整 stdout/stderr
├── demos/                                               # 实时 ReplayBuffer
└── collected_data/rank_0/id_0/                          # LeRobot dataset v2.1
    ├── meta/
    │   ├── info.json                                    # 总帧数 / fps / features
    │   ├── episodes.jsonl                               # 每 episode 长度 + task
    │   ├── episodes_stats.jsonl
    │   └── tasks.jsonl
    └── data/chunk-000/
        ├── episode_000000.parquet
        ├── episode_000001.parquet
        └── ...
```

## 8. 数据格式速查

**fps** = 10 Hz，**robot_type** = `dual_panda`，**图像** = 224×224×3 uint8。

### state（68 维，字母序 concat）

| slot | 字段 | size | 含义 |
|---|---|---|---|
| `[0:2]` | `gripper_position` | 2 | 左/右夹爪宽度（m, 0–0.085） |
| `[2:16]` | `joint_position` | 14 | 左 7 + 右 7 关节角（rad） |
| `[16:30]` | `joint_velocity` | 14 | 左 7 + 右 7 关节速度（rad/s） |
| `[30:36]` | `tcp_force` | 6 | 左 xyz + 右 xyz（N） |
| `[36:50]` | `tcp_pose` | 14 | 左 (xyz + quat) + 右 (xyz + quat) |
| `[50:56]` | `tcp_torque` | 6 | 左 xyz + 右 xyz（Nm） |
| `[56:68]` | `tcp_vel` | 12 | 左 (linear3+angular3) + 右 同 |

### action（16 维）

| slot | 含义 |
|---|---|
| `[0:7]` | 左臂 7 关节目标（rad，absolute） |
| `[7]` | 左夹爪 trigger，`-1..+1`（+1 = close） |
| `[8:15]` | 右臂 7 关节目标（rad） |
| `[15]` | 右夹爪 trigger |

### 相机键映射

| parquet 字段 | 物理相机 | 位置 |
|---|---|---|
| `image` | XVisio Lumos `...1218` | **左腕** |
| `extra_view_image-0` | Intel RealSense D435i (`941322072906`) | **base 第三人称** |
| `extra_view_image-1` | XVisio Lumos `...1173` | **右腕** |

（依据 `main_image_key: left_wrist_0_rgb` + 其余字母序排入 `extra_view_images`。）

### flags

- `done`：仅末帧 True
- `is_success`：整个 episode 每帧广播 True（LeRobot 惯例，失败 episode 被丢不进 dataset）
- `intervene_flag`：teleop 期间恒 True

## 9. 干净结束

```
Ctrl+C          # 在 collect_data.sh 的终端里，主 Python 进程退出
ray stop --force   # 想彻底清掉 Ray，可选；下次 start_ray_* 脚本会自动清
```

Data 已在 `logs/<timestamp>/collected_data/` 里，安全。
