双 Franka 真机：GELLO 数据采集、π₀.₅ SFT 与部署
====================================================

本指南是 RLinf 中 **双臂 Franka** 真机的端到端流程：双节点环境搭建、
1 kHz GELLO 关节空间双臂数据采集、π₀.₅ 在 20 维 rot6d 动作空间上的
SFT 微调，以及通过脚踏开关将训练好的策略部署回真机。

阅读本页前请先阅读：

* :doc:`franka` — 单臂 Franka 基础、Ray cluster 搭建、RealSense +
  SpaceMouse 数据采集路径。如果尚不熟悉 ``FrankaController`` /
  ``FCI`` / ``RLINF_NODE_RANK``，请先完整阅读该页。
* :doc:`franka_gello` — GELLO 硬件安装、Dynamixel SDK、
  ``gello-teleop`` 包、USB-FTDI 权限。

本页只覆盖双臂 rig 的差异点：

* **franky** 底层后端（``franky-control`` 包封装的 libfranka），
  替代 :doc:`franka` 使用的 ROS / serl 路径，左右两台机械臂共用；
* 三个新双臂环境 —— ``DualFrankaEnv``\ （旧版 14 维 Cartesian
  delta）、``DualFrankaJointEnv``\ （16 维 joint，采集用）、
  ``DualFrankaRot6dEnv``\ （20 维 TCP-rot6d，SFT 与部署用）；
* 用 **rot6d / SE(3) body-frame delta** 替换 openpi 自带的
  component-wise ``DeltaActions``；
* 由 3 键脚踏驱动的多任务、可断点续采的数据采集流；
* 双物理节点 Ray cluster：每个节点拥有一条 Franka 的 controller，
  env worker 与 GPU 均运行在 node 0。


该 rig 的适用范围与非适用范围
--------------------------------

RLinf 的双 Franka rig 面向 **双臂操作的 SFT** —— 采集高质量的遥操作
数据，并在 π₀ / π₀.₅ 等 VLA 上进行微调。与 :doc:`franka` 的单臂
SAC / PPO 训练相比，该 rig：

* **面向模仿学习，不执行在线 RL。** 采集使用 GELLO 关节空间遥操作；
  部署使用 SFT 模型自主推理 + 脚踏控制 episode 边界。不包含
  reward 标注，也没有 RL 更新。
* **左右臂统一使用 ``franky-control`` 这一 libfranka 后端。**
  所有底层控制均在 ``franky`` 内部的 C++ 1 kHz 循环中运行，Python
  仅更新参考点。该设计规避了"纯 Python 控制循环 + ROS"路径下的
  GIL 抖动问题。
* **方向用 6D 表示，不再用 Euler。** Euler 状态/动作
  会向 π₀.₅ 引入 ±π wrap 不连续点（上一帧 roll = +3.14 rad，
  下一帧 roll = −3.14 rad ⇒ 一个 "−2π" 的伪 delta，被策略当作
  规律学习）。改用 rot6d + SE(3) body-frame delta 后可消除此类
  问题。
* **左右臂分到两台机器上。** 每个节点用一根专线直连一根 Franka
  （FCI 都是 ``172.16.0.2``）；两个 ``172.16.0.0/24`` 子网在物理
  上完全独立（一根线、一张 NIC、一台机器对一台机械臂），所以同 IP
  并不冲突。两个节点之间走另一张共享 LAN，仅用于 Ray 控制流和张量
  同步。

如需进行单臂 Franka 在线 RL（SAC / PPO），请参考 :doc:`franka`，
而非本页。


硬件拓扑
--------

.. list-table::
   :header-rows: 1
   :widths: 18 32 50

   * - 节点
     - 角色
     - 节点上的硬件
   * - **node 0**\ （head）
     - Ray head；env worker；左 ``FrankyController``；
       部署阶段的 actor / rollout；所有相机和 GELLO 采集
     - 1× GPU（如 RTX 4090，仅 SFT 与部署阶段使用）；
       左 Franka FR3 直连一张 NIC，FCI IP ``172.16.0.2``；
       左 Robotiq 2F-85（USB-RS485 Modbus）；
       **左右两台 GELLO** Dynamixel 链（USB-FTDI）；
       **三台相机全部在此**\ —— base RealSense D435i（第三人称）+
       左腕 Lumos USB-3 + 右腕 Lumos USB-3；
       PCsensor 3 键脚踏（可放在任一节点）
   * - **node 1**\ （worker）
     - Ray worker；只跑右 ``FrankyController``
     - 可选 GPU（推理不需要）；
       右 Franka FR3 直连自己的 NIC，FCI IP ``172.16.0.2``；
       右 Robotiq 2F-85

.. warning::

   两台 Franka 的 FCI IP 都是 ``172.16.0.2``，**这不是 IP 冲突**
   —— 两个 ``172.16.0.0/24`` 子网在物理上完全独立，每台机械臂接一
   根专线、各自一张 NIC。从 node 0 ``ping 172.16.0.2`` 只到左臂；
   从 node 1 ``ping 172.16.0.2`` 只到右臂。**不要**\ 为了避免“表面上
   冲突”而修改 FCI IP —— Franka Desk 只在标准子网上暴露 FCI；让
   两条 NIC 物理隔离才是保证两条控制环互不干扰的关键。

相机角色（wrapper 栈用 ``main_image_key: left_wrist_0_rgb``，
所以 π₀.₅ 的 ``observation/image`` 槽位是**左腕**相机；
``base_0_rgb`` 与 ``right_wrist_0_rgb`` 进
``observation/extra_view_image-{0,1}``）。三台相机的 USB 全部接到
**node 0**\ —— env worker 在 node 0，由它统一打开
``/dev/v4l/by-id/...`` 与 ``rs.pipeline()``。所以右腕 Lumos 的 USB
线虽然要拉回 node 0，机械臂本体仍然挂在 node 1：

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - 相机槽位
     - 后端
     - 用途
   * - ``base_0_rgb``
     - RealSense D435i
     - 第三人称视角，左右臂共用
   * - ``left_wrist_0_rgb``
     - Lumos USB 3（XVisio vSLAM）
     - 左臂腕相机，作为 π₀.₅ 主 ``image``
   * - ``right_wrist_0_rgb``
     - Lumos USB 3（XVisio vSLAM）
     - 右臂腕相机

脚踏：3 键 PCsensor FootSwitch，3 个踏板烧成键码 ``a`` / ``b`` /
``c`` （使用厂家提供的 Windows 工具刷写一次，键码写入固件，重启后保留）。
``KeyboardListener`` 在 wrapper 进程内直接打开本地 ``evdev`` 路径，
而 env worker 在仓库默认 placement 下被钉到 node 0
（``component_placement.env.placement: 0`` +
``DualFrankaConfig.node_rank: 0``）—— 因此**脚踏必须接在 node 0**。
如果将 env worker 重新 placement 到其他节点，脚踏也必须迁移到同一节点。
拥有脚踏的那台节点要在 ``ray start`` **之前** 导出
``RLINF_KEYBOARD_DEVICE=/dev/input/eventXX`` ，让 Ray 把这个变量
打包进 worker 环境。``KeyboardListener`` 直接使用 ``evdev`` ，
不依赖 ``DISPLAY`` / ``xev`` ，也不依赖终端焦点。


软件栈
------

**采集** 阶段的数据通路::

  GELLO 机械臂（Dynamixel）                  env worker (node 0)
        │                                          │
        ▼                                          ▼
  GelloJointExpert（1 kHz 读）              DualFrankaJointEnv.step
        │ ±2π 反 wrap                              │ 10 Hz
        ▼                                          │
  DualGelloJointIntervention                       │
   （direct_stream 守护线程，1 kHz）               │ （env.step 只读
        │                                          │  state + grippers，
        └─move_joints─► FrankyController(left)  ◄──┘  不发 motion）
        └─move_joints─► FrankyController(right)
                              │ C++ 1 kHz JointImpedanceTracker
                              ▼
                        Franka FR3

**部署** 阶段的数据通路::

  observation（state[20] + 3 个相机）
        │
        ▼
  DualFrankaRot6dInputs ─► RigidBodyDeltaActions ─► π₀ / π₀.₅
                                                       │
                                                       ▼
                            RigidBodyAbsoluteActions ◄┘  （T_abs = T_state @ T_delta）
                                       │
                                       ▼
                            DualFrankaRot6dOutputs（切回 20 维）
                                       │
                                       ▼
                  DualFrankaRot6dEnv.step（每台机械臂 move_tcp_pose）
                                       │ C++ 1 kHz CartesianImpedanceTracker
                                       ▼
                                 Franka FR3

``FrankyController`` 内部的两个 tracker
（``JointImpedanceTracker`` 和 ``CartesianImpedanceTracker``）
是**互斥**的 —— 从采集（关节阻抗）切到部署（笛卡尔阻抗）时会
自动停掉前一个，因此跨场景切换不需要重启 franky 进程。


安装（每个节点都执行）
----------------------

以下步骤需在 ``node 0`` 和 ``node 1`` 上**分别执行一次**。两个节点是
独立 checkout、独立 venv，只共享 LAN 网络。

1. PREEMPT_RT 内核与 rtprio 限额
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

franky 后端假设主机已经在 PREEMPT_RT 内核上运行。请按 Franka 官方文档
`Setting up the real-time kernel
<https://frankarobotics.github.io/docs/libfranka/docs/real_time_kernel.html>`_
编译并启动；本项目验证过的版本为 ``5.15.133-rt69``\ 。验证：

.. code-block:: bash

   uname -a | grep -o PREEMPT_RT   # 必须输出 PREEMPT_RT

直连的千兆网卡指向 Franka 的 FCI 口（通常 ``172.16.0.2``\ ），中间
不要有交换机；同时检查 ``/proc/cmdline`` 没有奇怪的 ``iommu`` /
``apic`` 选项干扰 RT 线程。

放置 ``/etc/security/limits.d/99-realtime.conf``\ ，让 PAM 给当前用户
开放 ``rtprio 99`` 和 ``memlock unlimited``：

.. code-block:: text

   *  -  rtprio    99
   *  -  memlock   unlimited

退出登录再重新登录让 PAM 重新读取限额；然后 ``ulimit -r`` 应当返回
``99`` 或 ``unlimited``\ ，``ulimit -l`` 应当返回 ``unlimited``\ 。
否则 ``FrankyController.__init__`` 会打印 ``SCHED_FIFO denied`` /
``mlockall failed`` 并 fallback 到默认调度——控制器仍能运行，但
RT 抖动会回来。

.. note::

   这些限额由
   ``rlinf/envs/realworld/franka/franky_controller.py`` 中的
   ``_apply_rt_hardening()`` 在启动时检查；如果 ``SCHED_FIFO``
   被拒绝或 ``mlockall`` 失败，控制器会以 best-effort 模式继续
   运行并打 warning，而不会直接退出，warning 文本里附带具体的
   修复指引。

2. 每次开机的 RT 调优
~~~~~~~~~~~~~~~~~~~~~~

下面这些参数每次重启都会被重置。每次启动会话跑一次，或者写到
systemd one-shot / ``rc.local`` 里持久化：

.. code-block:: bash

   # 1. CPU governor → performance（防止 P-state 切换引入 µs 级抖动）
   sudo bash -c 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
       echo performance > "$g"
   done'
   cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor   # 期望: performance

   # 2. 放开 SCHED_FIFO 95% throttle（默认 950000/1000000）
   sudo sysctl -w kernel.sched_rt_runtime_us=-1
   cat /proc/sys/kernel/sched_rt_runtime_us                    # 期望: -1

   # 3. 关掉 Franka 链路的 NIC interrupt coalescing
   sudo ethtool -C eno1 rx-usecs 0 tx-usecs 0                  # 把 eno1 换成你的网卡

用 ``ip -br a`` 确认实际网卡名。如果想让 ``rt_runtime`` 持久化：

.. code-block:: bash

   echo 'kernel.sched_rt_runtime_us = -1' | sudo tee /etc/sysctl.d/99-franka-rt.conf

.. note::

   ``requirements/embodied/franky_install.sh`` 在安装结束时会把以上
   三条命令打印出来。本节是这几条命令的权威版本。

3. RLinf + franky
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/RLinf/RLinf.git
   cd RLinf

   # franky 系统依赖（rt-tests, ethtool, eigen, pinocchio）
   bash requirements/embodied/franky_install.sh

   # RLinf Python 依赖 + PyPI 上的 franky-control wheel
   bash requirements/install.sh embodied --env franka-franky --use-mirror

   source .venv/bin/activate

``--env franka-franky`` 固定使用 franky 路径
（PyPI 的 ``franky-control >= 0.15.0``），**跳过**
:doc:`franka` 使用的 ``serl_franka_controllers`` ROS / catkin 编译流。
``--use-mirror`` 面向国内用户（自动切换 PyPI / GitHub /
HuggingFace 镜像）。

.. note::

   ``requirements/install.sh embodied --env franka-franky`` 负责安装
   RLinf venv 与 ``franky-control`` wheel。系统级依赖（``rt-tests``、
   ``ethtool``、构建 ``libfranka`` 所需的 header 等）由
   ``requirements/embodied/franky_install.sh`` 安装。请先跑 franky
   安装脚本，再跑 install.sh。

如果 PyPI 上的 franky-control wheel 不匹配当前 Python + libfranka
组合，``pip`` 会回退到源码编译，这时需要 libfranka 头文件 +
pinocchio。``franky_install.sh`` 已经预装好 eigen / poco / fmt /
pinocchio，因此源码编译路径通常无需额外操作；不过 PyPI wheel 默认能
覆盖 Python 3.11 + libfranka 0.15.x。

.. warning::

   **请避开 libfranka 0.18.0**。Franka 官方 0.18.0 release notes
   标注了阻抗 / 笛卡尔控制路径的回归 bug；在本文使用的 joint /
   Cartesian impedance tracker 下，这个 bug 表现为机械臂
   **严重出力不足** —— 无力、无法承载自身重力，甚至无法跟踪轻量级
   GELLO 动作。版本请按 Franka firmware 在官方
   `compatibility matrix
   <https://frankarobotics.github.io/docs/compatibility.html>`_
   中查找匹配版本，**不要选择 0.18.0**\ （0.18.x 后续 patch 如已发布，
   以及 0.17.x、0.15.x 均已有实际使用记录）。安装后可
   检查 ``franky.__libfranka_version__``。

4. GELLO（env worker 所在节点）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

两台 GELLO 的 USB-FTDI 都接到 env worker 所在节点（仓库默认 placement
下是 **node 0** ），整个数据采集过程都保持在那里。
``DualGelloJointIntervention`` 在 env worker 进程里直接打开两个
串口，以 ~1 kHz 读取 —— 跨 LAN 访问 node 1 上的 GELLO 会超出实时性
预算、丢失采样，并导致 impedance tracker 参考点抖动。

具体安装命令（``gello`` + ``gello-teleop`` + USB-FTDI 权限，以及
"为什么只 init ``DynamixelSDK`` 这个 submodule"的背景）见
:doc:`franka_gello`。在 **node 0 上单独执行** 这些命令，并安装到与
RLinf 同一个 venv —— ``DualGelloJointIntervention`` 在 env wrapper
栈构建时会 in-process 直接 import 这两个包。

5. 脚踏
~~~~~~~

PCsensor FootSwitch 通过厂家提供的 Windows 工具把 3 个踏板烧成
键码 ``a`` / ``b`` / ``c``\ （写入一次后进入固件，重启后保留）。验证 +
授权：

.. code-block:: bash

   ls -l /dev/input/by-id/*-event-kbd
   #  期望: usb-PCsensor_FootSwitch-event-kbd → ../eventXX

   sudo chmod 666 /dev/input/eventXX

   # 一定要在 `ray start` 之前 export，让 Ray 把变量打包进 worker
   export RLINF_KEYBOARD_DEVICE=/dev/input/eventXX

``KeyboardListener`` 直接使用 ``evdev``，支持 ``ENODEV`` 自动重连
（USB 短时断连后也可恢复），并用边沿触发的 press 队列保证短于轮询周期
的按压也不会丢失。也可以把 export 写到
``ray_utils/realworld/setup_before_ray.sh`` 里持久化。


硬件验证
--------

启动 Ray 之前，每个节点需先对各硬件执行单项测试。

相机
~~~~

.. code-block:: bash

   # RealSense：枚举总线，确认协商到 USB-3。
   rs-enumerate-devices | grep -E "Name|Serial|USB Type"

   # Lumos（XVisio vSLAM）：确认两个 /dev/v4l/by-id 节点都在。
   ls /dev/v4l/by-id/

   # USB 拓扑：Lumos 和 RealSense 应当协商到 5000M。
   # 任何掉到 "480M" 都是 USB-2 fallback（线缆或 hub 不行）。
   lsusb -t

GELLO
~~~~~

.. code-block:: bash

   python -m gello_teleop.gello_expert \
       --port /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_<id>-if00-port0

操作 GELLO 时应当看到关节读数实时变化。如果数值阻塞或者突然跳
±2π，请执行下一节的标定流程。

每台机械臂单独验
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   FRANKA_ROBOT_IP=172.16.0.2 \
   FRANKA_GRIPPER_TYPE=robotiq \
   FRANKA_GRIPPER_PORT=/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_<id>-if00-port0 \
       python toolkits/realworld_check/test_franky_controller.py

REPL 命令：

* ``getjoint`` —— 打印当前关节角
* ``home`` —— 同步复位到 ``HOME_JOINTS``
* ``hold 30`` —— 静置 30 s，听有没有嗡鸣
* ``stream 4 0.001 500`` —— 1 kHz 推 500 条 J4 += 0.001 rad
  （streaming preemption 压测）
* ``impedance 300 300 300 300 150 80 30`` —— 降低关节阻抗后再压测一次
* ``open`` / ``close`` —— gripper sanity

每节点对自己那台机械臂单独执行：静置无可听嗡鸣、``stream 4 0.001 1000``
能跑 ≥ 800 Hz、``home`` 从任意合法位姿都能干净复位即可。**两台机械臂
都通过验证之前不要启动 Ray。**


GELLO 标定
----------

GELLO 把 Dynamixel 电机位置映射到 Franka 关节角靠
``DynamixelRobotConfig``\ （关节符号 + 偏移）。每台 GELLO 单独标定。
3 个脚本覆盖整套流程：

1. **标定**\ （每台 GELLO 一次，更换电机后再标）：

   .. code-block:: bash

      bash examples/embodiment/gello_calibrate.sh

   脚本会将机器人安全地依次移动到两个已知姿态（``POSE_A`` =
   Franka 原点，``POSE_B`` = π/4 倍数），让操作员将 GELLO 各
   摆成相同姿态，然后从两次差值解出 ``joint_signs`` 和
   ``joint_offsets``，最后打印一段可直接粘贴到
   ``gello_software/gello/agents/gello_agent.py`` 的
   ``DynamixelRobotConfig`` 块::

       "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_<id>-if00-port0":
           DynamixelRobotConfig(
               joint_ids=(1, 2, 3, 4, 5, 6, 7),
               joint_offsets=(...),
               joint_signs=(...),
               gripper_config=(8, ..., ...),
               baudrate=1_000_000,
           ),

   ``gello`` 是 editable install，粘贴完成后不需要重新安装，只需重启下一个
   import ``gello`` 的进程即可。

2. **对齐**\ （观察到 GELLO leader 和机械臂位姿不一致时执行 ——
   例如手动移动过机械臂、长时间未使用、或采集会话开始前需要确认状态
   时）：

   .. code-block:: bash

      bash examples/embodiment/gello_align.sh

   脚本会将机器人移动到一个固定的对齐 HOME 位姿（J4 = −π/2、
   J6 = +π/2 等），然后逐关节 J1 → J7 引导操作员对齐，每个关节带
   live progress bar。一旦某关节连续 8 帧都在 ±0.10 rad 内就
   自动跳到下一个。

两个脚本都会用 glob ``/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_*-if00-port0``
自动找到本机 Robotiq 串口，因此无需关心当前位于左节点还是右节点。


硬件 YAML
---------

双 Franka 的硬件契约写在
``examples/embodiment/config/env/realworld_franka_joint_dual.yaml``
（采集）和
``examples/embodiment/config/env/realworld_franka_rot6d_dual.yaml``
（rot6d 部署）。cluster 部分两份完全一致，差异位于 override 段。

cluster 块（采集）：

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
             # 两台机械臂用同一个 FCI IP —— 见上文"硬件拓扑"警告
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
               left_gripper_connection:  "/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_<id-l>-if00-port0"  # 在 node 0
               right_gripper_connection: "/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_<id-r>-if00-port0"  # 在 node 1
               left_controller_node_rank:  0
               right_controller_node_rank: 1
               node_rank: 0  # env worker + 相机都在 node 0

``DualFrankaConfig`` 字段逐项说明
（``rlinf/scheduler/hardware/robots/dual_franka.py``）：

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - 字段
     - 含义
   * - ``left_robot_ip`` / ``right_robot_ip``
     - FCI IP（config 加载时会执行 IP 格式校验）。两台用同一个 IP
       是合法的，因为子网是物理隔离的。
   * - ``left_camera_serials`` / ``right_camera_serials`` / ``base_camera_serials``
     - 每种相机用自己的 SDK 报告的 serial：
       RealSense 用 ``librealsense`` serial（如
       ``<librealsense-serial>``）；
       Lumos 用 ``/dev/v4l/by-id`` 路径（``usb-XVisio_..._video-index0``）；
       ZED 用 ZED SDK 设备列表里的 serial。
   * - ``camera_type``
     - 默认后端（``realsense`` / ``zed`` / ``lumos``），当某槽位
       没单独指定 ``*_camera_type`` 时用它。
   * - ``base_camera_type`` / ``left_camera_type`` / ``right_camera_type``
     - 槽位级覆盖。支持在同一台 rig 上混用不同后端 —— 参考
       rig base 用 ``realsense``、两腕用 ``lumos``。
   * - ``left_gripper_type`` / ``right_gripper_type``
     - 该 Franky 路径只支持 ``robotiq`` （USB-FTDI 上运行 RS-485
       Modbus，无 ROS 依赖）。``common/gripper/franka_gripper.py``
       中的 Franka Hand 后端依赖 ROS controller，``FrankyController``
       中 **未接入该后端**，构造时会直接 raise。
   * - ``left_gripper_connection`` / ``right_gripper_connection``
     - 推荐用 ``/dev/serial/by-id`` 路径。**强烈** 不建议用
       ``/dev/ttyUSB*`` —— ``ttyUSB*`` 序号会随重启和热插拔变；
       ``by-id`` 锁的是 FTDI 芯片烧录的序列号。
   * - ``left_controller_node_rank`` / ``right_controller_node_rank``
     - 每台机械臂的 ``FrankyController`` Ray actor 落在哪个节点。
       标准拓扑设 ``0`` / ``1``。也可以两个都设 ``0``\ （单节点
       双臂调试 rig）。
   * - ``node_rank``
     - env worker（相机采集、wrapper、动作派发）所在节点。一般
       钉到拥有 base 相机那一台。

override 块设置每台机械臂的安全箱（``ee_pose_limit_min/max``）、
reset 位姿，还有 joint 模式下的 ``joint_action_mode`` 或 rot6d
模式下的字段。完整示例请参考仓库内 yaml。


Ray cluster 启动
-----------------

Ray 在 ``ray start`` 时会捕获当前的 Python 解释器和**已 export 的
环境变量**，worker actor 都继承这个快照。``ray start`` 之后再
``pip install`` 进 venv 的包，下次 import 时仍可见（Ray 不会
冻结 ``site-packages`` ）；但环境变量不会同步更新 —— 未 export 的变量，
worker 永远拿不到。顺序：

1. **每个节点上**：激活 venv，export ``RLINF_NODE_RANK``，
   （可选）export ``RLINF_COMM_NET_DEVICES`` ，如果脚踏在该节点
   就 export ``RLINF_KEYBOARD_DEVICE``。验证 ``franky``、
   ``gello``、``gello_teleop`` 都能 import。
2. **然后** ``ray start`` —— node 0 head，node 1 worker。

在每个节点上激活 venv，export rank 相关环境变量，再 ``ray start``。
``HEAD_IP`` / ``WORKER_IP`` 是两台机器相互通信用的局域网 IP（不是
``127.0.0.1``，也不是公网 IP）。

.. code-block:: bash

   # node 0（Ray head）
   source .venv/bin/activate
   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   export RLINF_NODE_RANK=0
   export RLINF_KEYBOARD_DEVICE=/dev/input/eventXX  # 若脚踏在这台

   ray stop --force
   ray start --head --port=6379 --node-ip-address=<HEAD_IP>

.. code-block:: bash

   # node 1（Ray worker）
   source .venv/bin/activate
   export PYTHONPATH=$PWD:${PYTHONPATH:-}
   export RLINF_NODE_RANK=1

   ray stop --force
   ray start --address=<HEAD_IP>:6379 --node-ip-address=<WORKER_IP>

在 node 0 验证：

.. code-block:: bash

   ray status
   # 期望：2 个节点都 ALIVE，cluster 里 GPU/CPU 资源对得上

.. warning::

   两个节点是**独立 checkout**。node 0 改完代码要 rsync 到 node 1
   （``rsync -av --delete RLinf/ <node1>:/path/to/RLinf/``）
   **并** 在该节点重启 Ray，让新代码进 Ray 的环境快照。忘了同步
   会出现"node 0 上可运行、node 1 上 ImportError"或者更隐蔽的
   "feature 在某些 worker 上行为不一致"问题。


数据采集（GELLO 关节空间）
--------------------------

采集路径是 ``DualFrankaJointEnv-v1`` + ``teleop_direct_stream:
true``。``DualGelloJointIntervention`` 内部的守护线程以 ~1 kHz
读 GELLO Dynamixel 电机位置，直接推到两个 ``FrankyController``
actor（再转发给 franky 的 ``JointImpedanceTracker``）。
``env.step`` 以 10 Hz 运行，只读取 state、在状态翻转时触发 gripper
开/合、采集相机帧 —— 它**不调用** ``move_joints``。

为什么使用 direct-stream 而不是 env-step gating？10 Hz 采样会把
高频腕部动作抹掉。1 kHz 守护线程按 GELLO 原生频率采样操作员实际
的手部运动，env.step 再以 10 Hz 读取**已经发生**的关节状态 ——
因此数据集记录的是操作员实际执行的动作，而不是被 100 ms 网格
截断后的轨迹。10 Hz 这个状态读取频率也正是 π₀.₅ 推理时接收的输入
频率。

配置
~~~~

用仓库里的
``examples/embodiment/config/realworld_collect_data_gello_joint_dual_franka.yaml``。
开采集前会改的字段：

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - YAML 字段
     - 含义
   * - ``runner.num_data_episodes``
     - 目标 episode 数。结合 ``data_collection.resume`` 之后这是
       *跨多次会话累积* 的总目标，不是单次会话目标。
   * - ``env.eval.left_gello_port`` / ``right_gello_port``
     - 这次会话临时换 GELLO 单元时在这里覆盖。否则继承自 env yaml。
   * - ``env.eval.override_cfg.task_description``
     - 写到每帧 ``task`` 字段的 prompt。多任务模式下会被每个
       task 自己的 prompt 覆盖。
   * - ``env.eval.override_cfg.joint_action_mode``
     - ``absolute``\ （采集用，1:1 映射 GELLO 关节）；``delta``
       用于同一 env 类的离线 RL。
   * - ``env.eval.override_cfg.teleop_direct_stream``
     - ``true`` 开 1 kHz 守护线程。设成 ``false`` 会回到 env.step
       gating 路径，**不是推荐的采集路径**。
   * - ``data_collection.save_dir``
     - 数据集根目录。默认每次会话写到
       ``${runner.logger.log_path}/collected_data``；命令行 override
       同一根目录，让多次会话累积。
   * - ``data_collection.resume``
     - ``true`` 时会从 ``save_dir/rank_0`` 下已有的 ``id_*``
       shard 累计 episode 数继续。
   * - ``data_collection.tasks``
     - 可选。一组 ``{name, prompt}`` 条目，开 round-robin 多任务
       采集。注释掉就是单任务模式。

启动
~~~~

Ray 启动后打开 3 个终端。

**终端 1** —— launcher（在 node 0）：

.. code-block:: bash

   bash examples/embodiment/collect_data.sh \
        realworld_collect_data_gello_joint_dual_franka 2>&1 \
        | tee logs/collect.log

   # 后面可接 Hydra override，例如：
   #   bash examples/embodiment/collect_data.sh \
   #        realworld_collect_data_gello_joint_dual_franka \
   #        env.eval.data_collection.save_dir=/data/dual_franka_v1 \
   #        env.eval.data_collection.resume=true \
   #        runner.num_data_episodes=200 \
   #        2>&1 | tee logs/collect.log

**终端 2** —— 实时进度条（在 node 0）：

.. code-block:: bash

   python toolkits/realworld_check/collect_monitor.py logs/collect.log

单独使用一个 monitor，是因为采集器作为 Ray worker 运行，stdout
被 Ray 的 log monitor 批量缓冲（~500 ms），会破坏 ``tqdm`` 的
``\r`` 原位刷新。Monitor 在自己的 TTY 里 tail 日志文件，渲染一
条干净的 tqdm 进度条，显示成功计数、最近脚踏事件、最后一次
reward。默认启动会 replay 已有日志，所以监视器开晚也能对齐进度
（``--no-replay`` 切到只 tail EOF）。``--source=worker``\ （默认
``auto``）会直接 tail Ray worker 的 stdout 文件
（``/tmp/ray/session_latest/logs/worker-*-<pid>.out``），完全
绕开 log monitor batching（快 1-2 分钟），找不到时回退到 tee 日志。

每个 episode 的工作流
~~~~~~~~~~~~~~~~~~~~~~

确认 ``gello_align.sh`` 报告 ``ALL JOINTS ALIGNED`` 之后：

1. **(pre)** 每次 reset 时，机械臂会跟着 GELLO 当前位姿对齐
   （``KeyboardStartEndWrapper`` + ``DualGelloJointIntervention``
   会通过 ``options["skip_reset_to_home"]=True`` 跳过 home 复位）。
   机械臂保持在操作员当前手部位置。
2. **踩下 ``a``** —— 从当前位姿开始记录第 0 帧。
3. **演示任务**。每一步数据进 buffer。机械臂以 1 kHz 跟踪 GELLO；
   相机以 10 Hz 抓帧。
4. **踩下 ``b``** —— 子任务边界：``segment_id`` +1
   （1 s 防抖；窗口内的二次按下被忽略）。用来标 "approach" /
   "grasp" / "transfer" / "place" 等阶段，方便下游策略按
   segment_id 条件化。
5. **踩下 ``c``** —— 标记成功：reward = 1.0、``terminated=True``、
   ``CollectEpisode`` 把 buffer 写到 LeRobot shard。
6. **录制中再次踩下 ``a``** —— 中止：丢弃 buffer，回到 pre 阶段。
   机械臂不复位 home，停在当前位置（便于操作员立即重试，不打断
   GELLO 跟踪）。

多任务采集
~~~~~~~~~~

设置 ``data_collection.tasks``\ （仓库里默认是注释状态）：

.. code-block:: yaml

   data_collection:
     tasks:
       - name: pour_water
         prompt: "Pour water from the cup into the bowl"
       - name: pick_cup
         prompt: "Pick up the cup from the table"

采集会 round-robin 切换任务：ep0 → ``tasks[0]``、ep1 →
``tasks[1]``、ep2 → ``tasks[0]``、… 每个任务有自己的 LeRobot
dataset，根目录在 ``<save_dir>/<name>/``，每帧 ``task`` 字段写
对应任务的 prompt。

中止（``a`` 中止录制）**不消耗** task slot —— 只有成功
（``c``）才推动轮转。这能让任务采集数量保持平衡，即使某个任务
比另一个难很多。

多任务模式下 ``resume`` 会被**忽略**：每次会话都从空 shard 开始
分别写每个任务子目录。

输出格式
~~~~~~~~

LeRobot v2.1，每次会话一个 shard，路径是
``<save_dir>/[<task>/]rank_0/id_{N}/``：

* ``meta/info.json`` —— feature schema。``state`` 固定 ``[68]``；
  ``actions`` 在 joint 模式下 ``[16]``，rot6d 模式下 ``[20]``。
* ``meta/episodes_stats.jsonl`` —— 每条 episode 的 ``state`` /
  ``actions`` min / max / mean / std。
* ``data/episode_NNNNNN.parquet`` —— 每一步一行。

每帧字段：

* ``state`` —— ``DualFrankaJointEnv.STATE_LAYOUT`` 拼接
  ``[gripper_position(2), joint_position(14), joint_velocity(14),
  tcp_force(6), tcp_pose(14), tcp_torque(6), tcp_vel(12)]`` = 68。
  前 2 个 slot 故意是 ``[L_grip, R_grip]``，匹配 rot6d 策略
  ``_rearrange_state`` 的切片假设。
* ``actions`` —— GELLO 守护线程那一步派发的动作（joint 模式
  16 维：``[L_jpos(7), L_grip, R_jpos(7), R_grip]``）。
* ``image`` —— ``left_wrist_0_rgb``\ （``main_image_key``）。
* ``wrist_image-0`` / ``wrist_image-1`` —— 通过
  ``CollectEpisode._expand_multi_view_images`` 展开的左右腕视图。
* ``extra_view_image-0`` / ``extra_view_image-1`` —— base + 右腕
  视图，**顺序锁死** ``("base_0_rgb", "right_wrist_0_rgb")``。
  这个顺序在 ``DualFrankaRot6dInputs._extract_extra_views`` 里被
  断言，rig 重命名时会显式报错，避免静默调换相机含义。
* ``task`` —— 该 episode 所属任务的 prompt。
* ``is_success`` —— sticky flag，整条 episode 都为 ``True``
  当且仅当 episode 由踩下 ``c`` 结束。
* ``done`` —— 只有 episode 最后一帧为 ``True``。
* ``intervene_flag`` —— 采集阶段始终 ``True``\ （GELLO 守护
  线程的命令就是 action）。
* ``segment_id`` —— uint8，踩下 ``b`` 时 +1。

断点续采
~~~~~~~~

``data_collection.resume: true`` 加上原 ``save_dir`` 重新执行：
``CollectEpisode._count_existing_lerobot_episodes`` 会扫
``id_*`` shard 累计 ``total_episodes``\ （恶意损坏的 shard 自动
跳过，避免上次中断留下的脏 shard 阻塞 resume），新会话写到
新建的 ``id_{N}`` shard，已 finalize 的数据保持不变。

进度条初始位置会以已有计数初始化 —— ``num_data_episodes: 200``
加上之前已存的 50 条 success，新会话还需要采集 150 条。

如需跨多次会话累积到同一根目录，可在命令行覆盖 ``save_dir``：

.. code-block:: bash

   bash examples/embodiment/collect_data.sh \
        realworld_collect_data_gello_joint_dual_franka \
        env.eval.data_collection.save_dir=/data/dual_franka_v1 \
        env.eval.data_collection.resume=true \
        2>&1 | tee logs/collect.log


回填 rot6d 与 norm_stats
-------------------------

采集得到的数据 **不能直接输入 π₀.₅**，需要离线改写：

* **state 是 68 维**\ （``DualFrankaJointEnv.STATE_LAYOUT`` 拼接：
  ``gripper_position(2) + joint_position(14) + joint_velocity(14) +
  tcp_force(6) + tcp_pose(14) + tcp_torque(6) + tcp_vel(12)``
  = 68）。
* **动作是 16 维 joint 目标**\ （GELLO 下发的是关节角，每台机械臂
  ``[j(7) + grip(1)]``）。

而 π₀.₅ SFT 路径要求 **20 维 rot6d**：state 和 action 都是
``[xyz(3) + rot6d(6) + grip(1)] × 2``。因此需要在 SFT 之前把
joint 数据集离线改写：动作 16 → 20 维；state 仍保留 68 维（让
parquet schema 保持不变），但前 20 维被 rot6d 前缀覆盖（π₀.₅ 的
``_rearrange_state`` 只切 ``[:20]`` ，后面 48 维不会被读取）。

.. code-block:: bash

   export PYTHONPATH=$(pwd)
   python toolkits/dual_franka/backfill_rot6d.py \
       --src $HF_LEROBOT_HOME/<repo_id>/joint_v1 \
       --dst $HF_LEROBOT_HOME/<repo_id>/rot6d_v1

``backfill_rot6d.py`` 做了什么：

* **state 前 20 维改写**。``state[:, 0:20]`` 改成
  ``[L_grip, R_grip, L_xyz(3), L_rot6d(6), R_xyz(3), R_rot6d(6)]``，
  来源是原 ``state[:, 36:43]`` （左 tcp_pose）和
  ``state[:, 43:50]`` （右 tcp_pose），quat 通过
  ``quat_xyzw_to_rot6d`` 转成 rot6d。``state[:, 20:68]`` 余下的
  字段（joint_velocity 尾部、tcp_force、整段 tcp_pose、tcp_torque、
  tcp_vel）原样保留 —— π₀.₅ 的 ``_rearrange_state`` 切到 ``:20``
  就停，剩余部分不会被读取。
* **actions 16 → 20**。每帧 20 维向量是
  ``[L_xyz, L_rot6d, L_grip, R_xyz, R_rot6d, R_grip]``，
  其中 xyz / rot6d 用**下一帧** ``tcp_pose``\ （这一帧 GELLO 操作员
  正在朝该目标运动的最佳代理；离线无法执行 FK，因为需要 ``franky.Model``
  绑定到活机器人）。gripper 槽位用原来 ``action[7]`` /
  ``action[15]`` 的触发信号。最后一帧重复当前 state（无运动）。
* **schema 修正**。parquet schema 里 HuggingFace 元数据的
  ``actions.length`` 从 16 升到 20；每条 episode 的
  ``state`` / ``actions`` stats 重新算。
* **幂等。** 如果 ``--src`` 指向已经回填过的数据集，脚本会清楚地
  报错，而不是静默再次写入。

回填完之后算 norm stats：

.. code-block:: bash

   export HF_LEROBOT_HOME=/path/to/lerobot_root
   python toolkits/lerobot/calculate_norm_stats.py \
       --config-name pi05_dualfranka_rot6d \
       --repo-id <repo_id>/rot6d_v1

脚本会按 SFT 数据 pipeline 执行一遍数据集
（``RepackTransform`` → ``DualFrankaRot6dInputs`` →
``RigidBodyDeltaActions`` ），把 ``norm_stats.json`` 写到
``<openpi_assets_dirs>/<data_config.repo_id>/`` 下。同一个
``<repo_id>`` 也是部署阶段 rollout worker 查找 norm_stats 的
key —— 详见下文"ckpt / norm_stats 锁步"完整路径解析规则。

norm stats 必须**在回填之后**计算，不能在之前计算 —— 它需要看到策略
真正会预测的 body-frame delta，而不是磁盘上的绝对目标。


SFT（π₀.₅，rot6d_v1）
---------------------

配置
~~~~

``examples/sft/config/dual_franka_rot6d_sft_openpi.yaml``。启动前
需要修改的字段：

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 字段
     - 设成
   * - ``data.train_data_paths``
     - 已完成回填的 rot6d_v1 数据集所在的 LeRobot 根目录。这个值
       会被 ``train_vla_sft.py`` 在 validate 之前 export 成
       ``HF_LEROBOT_HOME``，openpi 数据加载器会自动用。
   * - ``actor.model.model_path``
     - π₀ / π₀.₅ base ckpt（torch 转换后的 weights，例如
       ``checkpoints/torch/pi05_base/``）。
   * - ``actor.model.action_dim``
     - ``20``\ （必须匹配 rot6d 数据 layout）。
   * - ``actor.model.num_action_chunks``
     - ``20``\ （匹配 ``pi05_dualfranka_rot6d`` TrainConfig 里的
       ``action_horizon``）。
   * - ``actor.model.openpi.config_name``
     - ``pi05_dualfranka_rot6d``。
   * - ``actor.optim.lr``
     - π₀.₅ 在此类数据集上 ``7.91e-6`` 是一个合理默认值。
   * - ``actor.fsdp_config.sharding_strategy``
     - ``full_shard``\ （若 GPU 数量超过 8 张、希望使用跨副本
       all-reduce 而非 all-gather，则改为 ``hybrid_shard``）。
   * - ``runner.save_interval``
     - ``500`` 步保存一次 ckpt 到
       ``${runner.logger.log_path}/checkpoints/global_step_<N>/``。

启动
~~~~

.. code-block:: bash

   # 单节点、4 张 GPU —— cluster.num_nodes: 1，
   # component_placement.actor,env,rollout 用 GPU 0..3。
   bash examples/sft/run_vla_sft.sh dual_franka_rot6d_sft_openpi

Runner 每 ``runner.save_interval`` 步保存一次 ckpt，目录布局：

.. code-block:: text

   <log_path>/checkpoints/global_step_<N>/
   ├── actor/
   │   └── model_state_dict/
   │       └── full_weights.pt
   └── <asset_id>/                        # 例如 "<your-hf-user>/<your-dataset>"
       └── norm_stats.json                # 推理使用的 pinned norm stats

真机部署时，rollout worker 会从
``<model_path>/actor/model_state_dict/full_weights.pt`` 读策略
权重，从 ``<model_path>/<asset_id>/norm_stats.json`` 读 norm
stats。

Body-frame SE(3) delta 数学
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Transform 链（在 ``DualFrankaRot6dDataConfig.create`` 中构建）：

.. code-block:: text

   磁盘上：                  state[68]（前 20 维是 rot6d prefix） +
                             actions[H, 20] （绝对目标）
                                            │
                                            ▼
   DualFrankaRot6dInputs:    把 state 切到 [:20] 后重排成训练 layout
                             ([L_xyz, L_rot6d, L_grip, R_xyz, R_rot6d, R_grip])；
                             把 state 和 actions pad 到 action_dim；
                             提取 base / right_wrist 图像（顺序断言）
                                            │
                                            ▼
   RigidBodyDeltaActions:    对 DUAL_ARM_ROT6D_LAYOUT 里每个 pose6d slot：
                             T_state = pose_to_SE3(state.xyz, state.rot6d)
                             T_abs   = pose_to_SE3(actions.xyz, actions.rot6d)
                             T_delta = inv(T_state) @ T_abs
                             把 T_delta 写回 actions.xyz / .rot6d；
                             scalar_abs slot（gripper 在 idx 9 和 19）原样不动
                                            │
                                            ▼
                                        π₀ / π₀.₅
                                            │
                                            ▼ （推理）
   RigidBodyAbsoluteActions: T_abs = T_state @ T_delta
                             把 T_abs 写回 actions.xyz / .rot6d
                                            │
                                            ▼
   DualFrankaRot6dOutputs:   把预测切回 20 维
                                            │
                                            ▼
                              env.step（每台机械臂 move_tcp_pose）

为什么使用 SE(3) 而不是 openpi 自带的 component-wise
``DeltaActions``：rot6d 是 ``SO(3)`` 矩阵前两列展平
``[r1; r2]``。component-wise 减法
``rot6d_delta = rot6d_abs − rot6d_state`` 出来的 ``r1, r2`` 不
正交 —— Gram-Schmidt 还需要额外执行一步投影才能映射回 ``SO(3)``，而且
得到的旋转与原始的"body-frame delta"语义不一致。在 ``SE(3)`` 上
做合成可以让旋转部分始终精确正交，并保持几何意义。

Body-frame 约定（``T_delta = inv(T_state) @ T_abs``）的好处是：
delta 表达在**当前 end-effector 坐标系**下，所以部署时 reset 位姿
跟训练时不完全一致，预测的动作不会被拖到 manifold 之外。Action
slot 划分见
``rlinf/models/embodiment/openpi/transforms/rigid_body_delta.py``
里的 ``DUAL_ARM_ROT6D_LAYOUT``::

   ({"kind": "pose6d", "xyz": slice(0, 3),  "rot6d": slice(3, 9)}),
   ({"kind": "scalar_abs", "idx": 9}),                        # 左 gripper
   ({"kind": "pose6d", "xyz": slice(10, 13), "rot6d": slice(13, 19)}),
   ({"kind": "scalar_abs", "idx": 19})                        # 右 gripper

action slot ``9`` 和 ``19``\ （左右 gripper）是绝对的 [-1, 1]
触发信号；只有两个 pose6d slot 使用 SE(3) round-trip。

变换的 round-trip 性质有 ``tests/unit_tests/test_rigid_body_delta.py``
做回归：50 个随机 ``(state, abs chunk)``，``delta → absolute``
能把原 chunk 恢复到 ``atol=1e-5``；gripper 通道和 pad 尾部被
断言不变。


真机部署
--------

跟采集用同一套 Ray cluster，换入口脚本 + 配置。

配置
~~~~

``examples/embodiment/config/realworld_eval_dual_franka.yaml``。
占位符标注为 ``# Replace:``。最常修改的字段：

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - 字段
     - 设成
   * - ``rollout.model.model_path``
     - ``<sft_log>/checkpoints/global_step_<N>/`` —— 必须包含
       ``actor/model_state_dict/full_weights.pt`` 和
       ``<data_config.repo_id>/norm_stats.json`` （
       ``data_config.repo_id`` 怎么算见下文"ckpt / norm_stats
       锁步"）。
   * - ``actor.model.openpi_data.repo_id``
     - 作为 ``data_kwargs`` 传给 ``get_openpi_config`` ，会覆盖
       ``data_config.repo_id`` ；这个 ``repo_id`` 就是部署时
       ``norm_stats.json`` 的查找 key。和
       ``calculate_norm_stats.py --repo-id`` 时给的值保持一致。
   * - ``env.eval.override_cfg.task_description``
     - 跟训练 prompt 一致。
   * - ``env.eval.override_cfg.joint_reset_qpos``
     - 从 SFT 数据集首帧 joint 均值反算回来；用过期值会让
       reset 后第一帧 obs 在训练分布外。
   * - ``env.eval.override_cfg.target_ee_pose`` / ``reset_ee_pose``
     - 跟采集时的 workspace 对齐。
   * - ``cluster.node_groups[*].env_configs[0].python_interpreter_path``
     - node 0 上 openpi venv 的 python 路径（env worker / rollout
       actor 用这个起 worker 进程）。

硬件 ``configs`` 与采集 yaml 完全一致 —— 同 IP、同相机 serial、
同 gripper 串口。Wrapper 是按 ``env.eval.use_*`` flag 装的，所以
采集 vs 部署的 yaml 差别只有 3 个：

* ``use_gello_joint: false``\ （采集是 ``true``）
* ``keyboard_reward_wrapper: eval_control``\ （采集是 ``start_end``）
* ``use_relative_frame: false`` —— rot6d 部署必须，否则
  ``DualRelativeFrame`` 会破坏 rot6d state。

启动
~~~~

.. code-block:: bash

   bash examples/embodiment/run_realworld_eval.sh realworld_eval_dual_franka

   # Hydra override 示例：
   #   bash examples/embodiment/run_realworld_eval.sh realworld_eval_dual_franka \
   #        rollout.model.model_path=/sft/global_step_5000 \
   #        env.eval.override_cfg.task_description="pour water"

每个 episode 的部署工作流
~~~~~~~~~~~~~~~~~~~~~~~~~

``KeyboardEvalControlWrapper`` 把脚踏 wrapper 切成自主推理模式：

1. ``env.reset()`` 之后两台机械臂保持在 reset 位姿。``env.step()``
   被截到 **idle** 模式 —— 不向内层 env 转发（impedance 控制器
   保持上一次 reset 时的目标，机械臂原地静止），但 wrapper 仍会把
   最近一次 obs 返回，让策略的 chunked rollout 循环空转，不下发
   任何关节指令。
2. 踩下 ``a`` —— wrapper 切到 **running**。下一步 ``env.step``
   开始向内层 env 转发策略输出。
3. 踩下 ``c`` —— 成功：``terminated=True``、``reward=1.0``、
   ``info["eval_result"]="success"``。Wrapper 内部立刻调
   ``env.reset()`` 让机械臂回 home，然后回到 idle 等下一次 ``a``
   —— 这是脚踏可连续操作的关键，即使 eval ``env_worker``
   是 ``auto_reset=False``。
4. 踩下 ``b`` —— 失败：行为同 ``c``，但 ``reward=0.0``、
   ``info["eval_result"]="failure"``。
5. running 阶段，wrapper 强制把 ``terminated`` / ``truncated``
   置 False，除非脚踏触发 —— env 自己的 ``max_episode_steps``
   不会切断策略。把 ``max_episode_steps`` 设大一点（仓库 yaml
   是 ``10000``），让脚踏始终是边界 owner。

ckpt / norm_stats 锁步
~~~~~~~~~~~~~~~~~~~~~~~

部署时最常见的崩盘原因是 ``norm_stats`` 不匹配。Rollout worker
的 norm_stats 路径解析在
``rlinf/models/embodiment/openpi/__init__.py``::

   pinned_path = <model_path>/<data_config.asset_id>/norm_stats.json
   if pinned_path 存在:
       用它
   else:
       退回 data_config.norm_stats，并输出明确告警

``data_config.asset_id`` 是 SFT 阶段 ``DualFrankaRot6dDataConfig.create()``
解析出来的（继承自 ``AssetsConfig.asset_id``，没显式设的话会回退到
``data_config.repo_id`` ）。同一个 key 也被
``calculate_norm_stats.py`` 用来写出，路径是
``<openpi_assets_dirs>/<data_config.repo_id>/`` 下。所以
``<model_path>/...`` 下的路径必须与 SFT 实际使用的路径一致。

实际操作：

* 如果保留 ``actor.model.openpi_data.repo_id`` 的默认值
  （ ``<your-hf-user>/<your-dataset>`` ），norm_stats 在
  ``<model_path>/<your-hf-user>/<your-dataset>/norm_stats.json``。
* 如果将 ``actor.model.openpi_data.repo_id`` （以
  ``data_kwargs`` 形式被透传）覆盖成本地回填数据集，
  ``data_config.repo_id`` 会被替换，查找 key 也跟着变成新值。
  **务必用同一个 ``--repo-id`` 执行**
  ``calculate_norm_stats.py`` ，再把结果拷到
  ``<model_path>/<那个 repo_id>/norm_stats.json``。

启动前自检：

.. code-block:: bash

   # 直接 grep SFT 日志中 rollout worker 实际查找的路径：
   grep "norm_stats" <sft_log>/run_embodiment.log | tail
   # 或者直接确认 model_path 下能找到 norm_stats.json：
   find <model_path> -maxdepth 3 -name norm_stats.json
   ls <model_path>/actor/model_state_dict/full_weights.pt

不匹配的 stats 会静默产生分布外的 state；策略会塌缩成一个固定动作
（向角落移动、gripper 锁死在打开状态等），且不会显式报错。fallback 路径**会**
输出 ``"norm_stats fallback: ... verify they match training or
inference will be wrong"`` warning —— 在判定 rollout 健康之前请先
grep log。


配置速查表
----------

Joint 采集（``realworld_collect_data_gello_joint_dual_franka.yaml``）

* ``env.eval.max_episode_steps: null`` —— 让脚踏控制边界。
* ``env.eval.override_cfg.teleop_direct_stream: true`` —— 1 kHz
  GELLO 守护线程。
* ``env.eval.override_cfg.joint_action_mode: absolute`` —— 直接
  映射 GELLO。
* ``data_collection.fps: 10`` —— 采集步频。
* ``data_collection.only_success: true`` —— 丢弃 abort 的 episode。
* ``data_collection.finalize_interval: 100`` —— 每 100 条 flush
  ``info.json``。
* ``data_collection.export_format: lerobot``。
* ``data_collection.robot_type: dual_FR3`` —— 写入数据集元数据。

Rot6d 部署（``realworld_eval_dual_franka.yaml``）

* ``env.eval.use_relative_frame: false`` —— rot6d state 没有 euler
  框架，``DualRelativeFrame`` 会破坏。
* ``env.eval.use_gello: false`` / ``use_gello_joint: false`` /
  ``use_spacemouse: false`` —— 完全自主。
* ``env.eval.keyboard_reward_wrapper: eval_control``。
* ``env.eval.override_cfg.success_hold_steps: 1`` —— 脚踏触发
  立即终止。
* ``algorithm.eval_rollout_epoch: 100`` —— rollout 执行该数量的 epoch
  之后 runner 退出。
* ``rollout.backend: huggingface`` —— π₀ / π₀.₅ 推理路径。

每台机械臂的阻抗 override（环境变量，在 Ray worker 第一次 import
``franky_controller`` 时读到 module 级常量里 —— 所以要在 controller
节点 ``ray start`` **之前** export）

* ``RLINF_CART_K_T`` —— 平动刚度，默认 ``1000`` N/m。
* ``RLINF_CART_K_R`` —— 转动刚度，默认 ``50`` Nm/rad。
* ``RLINF_CART_MAX_DTAU`` —— 1 kHz 周期内最大 torque step。
* ``RLINF_CART_ERR_CLIP_M`` / ``RLINF_CART_ERR_CLIP_RAD`` ——
  跟踪误差饱和（impedance saturation）。
* ``RLINF_CART_GAINS_TC`` —— 增益时间常数。
* ``RLINF_CART_MAX_STEP_M`` —— 每帧目标平动跳变上限（默认
  ``0.03`` m，主要是让数据集的单帧大跳变变成"slew"而不是
  "step input"）。
* ``RLINF_CART_MAX_STEP_RAD`` —— 同上但是转动，默认 ``0.15``
  rad。

调试跟踪抖动时优先调整这两个 ``MAX_STEP`` 变量 —— 代价较低，并可将策略
诱发的尖峰跳变转换为平滑过渡。


故障排查
--------

**GELLO 守护线程未启动**
   ``DualGelloJointIntervention._start_stream_thread`` 只有当**两侧**
   ``GelloJointExpert`` 都 ``ready`` 时才拉起守护线程。
   ``GelloJointExpert.ready`` 在第一帧成功 Dynamixel 读时翻成
   True —— 连续 50 次错误后会回 False。GELLO 重新上电、FTDI
   重插，然后用 ``python -m gello_teleop.gello_expert
   --port /dev/...`` 验证。

**Ray worker 静默死在 import**
   通常是以下之一：(a) ``franky`` / ``gello`` / ``gello_teleop``
   未安装到 Ray 当前使用的 venv 里；(b) ``ray start`` 是从另一个
   venv 启动的，与执行 ``pip install`` 的 venv 不是同一个；(c) 缺少底层依赖
   （libfranka 共享库、Dynamixel SDK、Lumos 用的 X11 库等）。
   ``ray status`` 会显示 worker 消失；
   ``/tmp/ray/session_latest/logs/worker-*.err`` 里有
   ``ImportError``。在执行 ``ray start`` 的同一个 shell 里执行
   ``which python && python -c "import franky, gello, gello_teleop"``
   先确认 venv 和包是否一致。

**有一台机械臂 reset 时挂住**
   在 controller node 上 ``ping -c 100 172.16.0.2``。如果 FCI
   链路第一次 ``FrankyController.__init__`` 时丢包，
   ``recover_from_errors`` 会无操作，后续 ``move_*`` 调用会失败。
   重启该机械臂后再次执行。

**开机后 ``move_joints`` 一直报错**
   机械臂处于 user-stop 状态——白色急停按钮没释放，所以
   ``FrankyController.__init__`` 里的 ``robot.recover_from_errors()``
   只是空操作，但后续每个 motion 调用都会被拒绝。请释放急停按钮，
   在 Desk 网页（\ ``http://172.16.0.2/desk/``\ ）里点击
   *Activate FCI*\ ，等关节解锁（白色 LED → 蓝色），再启动。

**GELLO 守护线程和 env reset 互相 race**
   ``teleop_direct_stream: true`` 模式下，1 kHz GELLO 守护线程会和
   env 的 Cartesian ``_interpolate_move`` reset 并行跑。如果 reset
   还没完成而操作员就在动 GELLO leader，franky 会先后收到两条
   ``move_*`` 然后 preempt 到后到的那一条，机械臂会跟着已经过期的
   参考点跑。reset 期间把 GELLO leader 放稳（最好搁在支架上），等
   ``KeyboardStartEndWrapper`` 报告 reset 结束再继续。

**脚踏报 "Permission denied"**
   ``RLINF_KEYBOARD_DEVICE`` 指向 ``/dev/input/eventXX``，但
   重启之后 ``chmod`` 失效了。要么重新 ``sudo chmod 666``，要么
   写 udev rule（``KERNEL=="event*", SUBSYSTEM=="input",
   ATTRS{name}=="PCsensor FootSwitch", MODE="0666"``）。

**RealSense 退到 USB 2.x**
   ``rs-enumerate-devices`` 会打印 USB descriptor，``lsusb -t`` 会
   显示 ``480M`` 而不是 ``5000M``。更换线缆，切换到 root USB-3
   端口（蓝色端口），再次确认。

**Lumos 冷启动第一次失败**
   ``LumosCamera`` 内置 double-open + I420 buffer warmup 来处理
   ``STREAMON`` 冷启动 USB 带宽竞速。重启后仍失败说明驱动状态
   阻塞 —— 请重新插拔 USB 线。

**部署时 idle 一直不响应**
   ``KeyboardEvalControlWrapper`` 在等待踩下 ``a``。确认
   ``RLINF_KEYBOARD_DEVICE`` 指对了 ``/dev/input/eventXX``，
   ``chmod 666`` 还在生效。注意 wrapper 每 ``IDLE_POLL_S = 0.05``
   s 轮询一次，并且在 ``env.step`` 转发到内层 env **之前** 就
   截断 —— 因此 ``ray status`` 显示会很正常，即使 wrapper
   卡在 idle。

**部署阶段跟踪抖动**
   降 ``RLINF_CART_K_R``、提高 ``RLINF_CART_GAINS_TC``、把
   ``RLINF_CART_MAX_STEP_RAD`` 收得更紧一些。如果仍无法满足要求，则
   缩短策略 chunk 长度 —— 长 chunk 会放大"训练分布窄于部署分布"场景
   下 ``T_state`` 老化的影响。

**多任务 round-robin 不推进**
   只有踩下 ``c`` （成功）才推进；踩下 ``a`` 中止不消耗 slot。
   如果 ``data_collection.only_success: true`` （默认）但 episode
   执行到 ``c`` 的瞬间 ``reward != 1.0`` ，该 episode 会被静默
   丢弃，轮转也不推进。``KeyboardStartEndWrapper`` 在踩下 ``c``
   时始终将 ``reward`` 设为 ``1.0`` —— 因此该情况不应出现，
   除非 wrapper 被禁用或替换。

**部署时找不到 ``norm_stats.json``**
   Rollout worker 的查找路径是
   ``<model_path>/<data_config.asset_id>/norm_stats.json``。
   ``data_config.asset_id`` 没显式设时会回退到
   ``data_config.repo_id`` ；而 ``data_config.repo_id`` 就是 SFT
   yaml 里 ``actor.model.openpi_data.repo_id`` 的值。如果 SFT
   ckpt 目录里没有（例如只 rsync 了 ``actor/`` 子树），将
   ``calculate_norm_stats.py`` 写出的
   ``<openpi_assets_dirs>/<repo_id>/`` 复制回 SFT ckpt 目录即可。
   fallback 路径会输出 ``"norm_stats fallback"`` warning ——
   grep 之后再判断推理是否确实正常。

**collect_monitor 无进展**
   launcher 必须把 stdout 重定向到日志文件供 monitor 持续读取
   （``2>&1 | tee logs/collect.log`` ）。若省略 ``tee`` ，monitor
   就没有内容可读。env worker 在另一节点时需为 monitor 添加
   ``--source=worker``。

**controller 启动时输出 ``sched_setaffinity failed`` warning**
   主机 CPU 不到 6 核或者用户没有 ``CAP_SYS_NICE``。controller
   仍能运行 —— 但 RT 线程 pin 是 best-effort 的，主机负载高时抖动
   会更严重。要么换 6+ 核机器，要么对 venv 解释器 ``sudo setcap
   cap_sys_nice=eip $(which python)``。

**reset 时两台机械臂都动了，但之后只有一根跟踪 GELLO**
   ``DualGelloJointIntervention._start_stream_thread`` 因为某一
   侧的 expert 未 ready 而 early return。每台机械臂单独执行
   ``python toolkits/realworld_check/test_gello.py align-check`` 确认两台
   GELLO 都在持续产出关节读数，然后重启。


引用清单
--------

配置

* ``examples/embodiment/config/realworld_collect_data_gello_joint_dual_franka.yaml``
* ``examples/embodiment/config/realworld_eval_dual_franka.yaml``
* ``examples/embodiment/config/env/realworld_franka_joint_dual.yaml``
* ``examples/embodiment/config/env/realworld_franka_rot6d_dual.yaml``
* ``examples/embodiment/config/env/realworld_dual_franka.yaml`` ——
  旧版 14 维 Cartesian dual env（保留兼容性，rot6d SFT 路径不使用）。
* ``examples/sft/config/dual_franka_rot6d_sft_openpi.yaml``

工具脚本

* ``toolkits/dual_franka/backfill_rot6d.py`` —— joint 数据集 →
  rot6d_v1。
* ``toolkits/realworld_check/collect_monitor.py`` —— 进程外
  tqdm 监视器。
* ``toolkits/realworld_check/test_franky_controller.py`` —— 单臂
  REPL 验证。
* ``toolkits/realworld_check/test_gello.py`` —— GELLO 标定 / 对齐 / 复位
  统一入口。子命令：``align-check``、``align-sequential``、
  ``calibrate``、``reset-to-gello``。
* ``toolkits/lerobot/calculate_norm_stats.py`` —— π₀ / π₀.₅
  norm stats 计算。

代码（标准入口）

* ``rlinf/scheduler/hardware/robots/dual_franka.py`` ——
  ``DualFrankaConfig``。
* ``rlinf/envs/realworld/franka/franky_controller.py`` —— libfranka
  后端。
* ``rlinf/envs/realworld/franka/dual_franka_franky_env.py`` ——
  joint + rot6d 共享的 env scaffold。
* ``rlinf/envs/realworld/franka/dual_franka_joint_env.py`` —— 16 维
  joint env。
* ``rlinf/envs/realworld/franka/dual_franka_rot6d_env.py`` —— 20 维
  rot6d env。
* ``rlinf/envs/realworld/common/wrappers/dual_gello_joint_intervention.py``
  —— 1 kHz GELLO 守护线程。
* ``rlinf/envs/realworld/common/wrappers/keyboard_start_end_wrapper.py``
  —— 采集脚踏。
* ``rlinf/envs/realworld/common/wrappers/keyboard_eval_control_wrapper.py``
  —— 部署脚踏。
* ``rlinf/envs/realworld/common/wrappers/apply.py`` —— wrapper 组装。
* ``rlinf/utils/rot6d.py`` —— rot6d 数学 + SE(3) helper。
* ``rlinf/models/embodiment/openpi/transforms/rigid_body_delta.py``
  —— ``RigidBodyDeltaActions`` / ``RigidBodyAbsoluteActions``。
* ``rlinf/models/embodiment/openpi/policies/dual_franka_rot6d_policy.py``
  —— 数据 transform 输入/输出。
* ``rlinf/models/embodiment/openpi/dataconfig/dual_franka_rot6d_dataconfig.py``
  —— openpi data config wiring。
* ``rlinf/envs/wrappers/collect_episode.py`` —— 多任务 LeRobot
  writer + resume。
* ``examples/embodiment/collect_real_data.py`` —— 采集驱动入口。

相关文档

* :doc:`franka` —— 单臂 Franka 基础。
* :doc:`franka_gello` —— GELLO 硬件安装。
* :doc:`franka_pi0_sft_deploy` —— 单臂 π₀ SFT 部署示例。
* :doc:`sft_openpi` —— OpenPI 全量 / LoRA SFT pipeline 总览。
