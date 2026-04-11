# Franky backend for Franka Panda — install & RT tuning

这份文档是 `feat/pylibfranka-joint-control` 分支上把 Franka 控制后端从
pylibfranka 迁移到 [franky](https://github.com/TimSchneider42/franky)
的记录。`FrankyController`
（`rlinf/envs/realworld/franka/franky_controller.py`）是主路径，
`PylibfrankaController` 保留作为 fallback。

## 为什么要迁移

pylibfranka 0.19 的 `start_joint_position_control` + `readOnce/writeOnce`
streaming API 在 Python 里跑 1 kHz 控制循环有几个致命问题：

1. **pylibfranka bindings 完全不释放 GIL** —— `bind` 源文件里没有任何
   `py::call_guard<py::gil_scoped_release>()`。每次 `readOnce/writeOnce`
   在阻塞期间持有 GIL，任何别的 Python 线程（Ray actor dispatch、gripper
   I/O、logging、GC）都能让 RT 线程错过 1 ms cycle → libfranka
   有限差分看到 v/a 不连续 → internal impedance 补偿 → 可听嗡鸣。
2. **Python 侧 jerk-limiter 累积误差** —— 手写的 3-level profile
   对 Python 的 dt 抖动非常敏感，libfranka 动不动就报
   `acceleration_discontinuity`。
3. **10 Hz step-hold 输入** —— 上层 env 每个 step 都给 smoother 一个新的
   target，smoother retrigger ramp，抗不住噪声。
4. **GELLO 1 kHz 遥操被 env 10 Hz step 砍到 10 Hz**，99% 采样被丢弃。

franky 的架构直接解掉上面所有问题：

* **C++ `std::thread` 跑 `robot.control()` 回调**，真正的 1 kHz 循环在
  native 代码里，跟 Python 没关系。
* **所有 pybind11 bindings 都带 `py::call_guard<py::gil_scoped_release>()`**
  （见 `python/bind_robot.cpp:63,72,147,...,237`）—— Python 主线程完全
  不阻塞 C++ 控制线程。
* **内置 Ruckig OTG**，做 jerk-constrained online trajectory
  generation。调用 `robot.move(motion, asynchronous=True)` 立即返回，
  再次调用时 Ruckig 在线 re-plan，从当前状态/速度/加速度平滑过渡到新
  target —— 这就是 streaming preemption 的正确姿势。
* **PyPI 有预编译 wheel**（`pip install franky-control`）。

## 硬件和内核前提

这份控制器假设你已经按 Franka 官方建议搭好了环境：

1. **PREEMPT_RT 内核**。本项目验证过的版本：`5.15.133-rt69`。
   ```bash
   uname -a | grep -o PREEMPT_RT   # 必须打印 PREEMPT_RT
   ```
2. **直连的千兆网卡** 指向 Franka 的 FCI 口（通常 `172.16.0.2`），
   中间不要有交换机。`FrankyController` 假设 `/proc/cmdline`
   没有奇怪的 iommu/apic 干扰。
3. **`/etc/security/limits.d/99-realtime.conf`** 放开 rtprio 99 +
   memlock unlimited：
   ```
   *  -  rtprio    99
   *  -  memlock   unlimited
   ```
   改完 logout + login 一次让 PAM 重新读。

## 一次性安装

```bash
# 清掉老的 venv（可选）
rm -rf .venv

# 装系统依赖（rt-tests, ethtool, pinocchio 等）
bash requirements/embodied/franky_install.sh

# 装 franky-control 到 venv
bash requirements/install.sh embodied --env franka --franka-backend franky --use-mirror
```

`--use-mirror` 跟其它 env 一致。`--no-root` 可以跳过 `franky_install.sh`
（当系统依赖已经装好时），但第一次装完整跑一遍更保险。

## 每次开机必跑的 RT 调优

这些参数默认不是 franky 友好的，**每次开机都得手动跑一遍**
（或者写到 systemd oneshot / rc.local 里持久化）：

```bash
# 1) CPU governor → performance（防止 P-state 切换引入 µs 级抖动）
sudo bash -c 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > "$g"
done'
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor   # 期望: performance

# 2) 放开 RT bandwidth（默认 950000/1000000 会 throttle SCHED_FIFO 线程到 95%）
sudo sysctl -w kernel.sched_rt_runtime_us=-1
cat /proc/sys/kernel/sched_rt_runtime_us                    # 期望: -1

# 3) 关掉 Franka 链路的 NIC interrupt coalescing
sudo ethtool -C eno1 rx-usecs 0 tx-usecs 0 2>/dev/null || true
```

（把 `eno1` 换成你的 Franka 直连网卡名，用 `ip -br a` 确认。）

持久化 rt_runtime（可选）：

```bash
echo 'kernel.sched_rt_runtime_us = -1' | sudo tee /etc/sysctl.d/99-franka-rt.conf
```

## 验证

跑完调优之后，按这个清单验一遍，任何一项不达标都直接影响稳定性。

| 检查 | 命令 | 期望 |
|---|---|---|
| PREEMPT_RT 活跃 | `uname -a \| grep PREEMPT_RT` | 有输出 |
| Governor | `cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor` | `performance` |
| RT bandwidth | `cat /proc/sys/kernel/sched_rt_runtime_us` | `-1` |
| rtprio 配额 | `ulimit -r` | 99 或 unlimited |
| memlock 配额 | `ulimit -l` | unlimited |
| Cyclictest 抖动 | `sudo cyclictest -p 80 -t 4 -i 1000 -l 300000 -m` | Max < 150 µs |
| 直连网络抖动 | `sudo ping -c 1000 -i 0.001 172.16.0.2 \| tail -3` | avg < 0.5 ms, max < 2 ms |

`FrankyController.__init__` 会自动尝试 `mlockall`、`SCHED_FIFO
priority=80`、CPU affinity（把 Python 主线程 pin 到 0/1 + 4..N，把
CPU 2/3 留给 franky 的 C++ 控制线程和它的网卡中断）。如果 limits 没设好，
这些调用会抛 `PermissionError`，控制器会 warn 并继续跑 —— 不会崩，
但抖动会回来。查 logs 里的这几行：

```
[INFO] mlockall: memory pages pinned
[INFO] SCHED_FIFO priority 80 applied (policy=1)
[INFO] Python thread affinity set to [0, 1, 4, 5, ..., 31]; CPUs 2-3 reserved for franky RT thread
```

## Smoke test

```bash
FRANKA_ROBOT_IP=172.16.0.2 FRANKA_GRIPPER_TYPE=robotiq \
  FRANKA_GRIPPER_PORT=/dev/ttyUSB0 \
  python toolkits/realworld_check/test_franky_controller.py
```

交互命令（跟 `test_pylibfranka_controller.py` 同款，方便 a/b 对比）：

```
cmd> getjoint                    # 当前关节角
cmd> home                        # 复位到 HOME_JOINTS（franky Ruckig 插值）
cmd> hold 30                     # 静置 30 s，耳测有没有嗡鸣
cmd> nudge 4 0.3                 # J4 +0.3 rad 单次 move_joints
cmd> stream 4 0.001 500          # 以 1 kHz 频率发 500 条 J4 +0.001 rad（streaming preemption 压测）
cmd> impedance 300 300 300 300 150 80 30    # 降 impedance 再测
cmd> open                        # 夹爪开
cmd> close                       # 夹爪关
cmd> q
```

### Acceptance criteria

一次合格的验证跑（换算到人类能判断的程度）：

1. **静置 60 s 无可听嗡鸣**，`state.arm_joint_velocity` rms < 1e-3 rad/s。
2. **`stream 4 0.001 1000` 能跑满 ~1 kHz**（打印的 Hz 数 > 800），
   且运动过程中关节平滑无反射错误。
3. **`home` 从任意合法起始位姿成功复位**，无
   `start_pose_invalid` / `acceleration_discontinuity`。
4. **5 分钟 GELLO 遥操**（用 `realworld_collect_data_gello_joint.yaml`
   + `teleop_direct_stream: true`）期间 `state.control_command_success_rate`
   > 0.99。
5. **cyclictest** `-p 80 -t 4 -i 1000 -l 300000` Max latency < 150 µs。

## GELLO 遥操的 1 kHz 直通路径

`realworld_collect_data_gello_joint.yaml` 里开了
`teleop_direct_stream: true` 之后：

* `FrankaJointEnv.step()` **不再** 调 `controller.move_joints`。它只做
  action 解析、gripper 事件、`time.sleep` 维持 10 Hz 节拍、取 state
  算 reward。
* `GelloJointIntervention` 拉起一个 daemon 线程，以 ~1 kHz 的节奏从
  `GelloJointExpert.get_action()` 读遥操关节角，直接调
  `controller.move_joints(q)`。franky 的 Ruckig 会 preempt + re-plan，
  不会 race。
* Gripper 只在开/关状态翻转时触发一次 `open_gripper()` / `close_gripper()`，
  避免 serial/Modbus 通道被每毫秒的命令打爆。

这条路径只在 `direct_stream=True` 时启用。RL 训练 rollout 的
`step_frequency=10` 路径完全不受影响。

## 还没解决的问题

1. 如果 franky-control 的 PyPI wheel 跟你的 Python ABI / libfranka
   版本不匹配，`pip install` 会 fallback 到源码编译，需要
   `libfranka`、`pinocchio`、`Ruckig` 的 CMake 找得到。`franky_install.sh`
   装了 eigen/poco/fmt 基础依赖，但 libfranka 源码仍然需要你从
   pylibfranka 路径复用（`.venv/franka_catkin_ws/libfranka`）——见
   `install.sh: install_franka_pylibfranka_env` 的 libfranka build 步骤。
2. 首次 `move_joints` 调用前 franky 需要完成 `robot.recover_from_errors()`，
   这在 `__init__` 里已经做了；但如果机器启动时处于 user-stopped 状态，
   recover 会 no-op，后续 `move` 才会报错。手动按掉 user-stop 按钮再跑。
3. `teleop_direct_stream` 模式下，env reset 路径（Cartesian
   `_interpolate_move`）调 `move_arm` 不受 `direct_stream` gate 影响，
   还是会正常发。所以 reset 途中 GELLO daemon + env 有可能互相 race —
   如果 reset 期间 GELLO 线程还在动，franky 会收到两路 motion 然后 preempt
   到更新的那一条。建议 reset 时先把遥操托盘放稳。
