Franka 真机关节空间控制：Franky + GELLO
========================================

本文介绍如何使用 `franky <https://github.com/TimSchneider42/franky>`_ 后端
对 Franka 机械臂进行 **关节空间控制** ，并搭配 **GELLO** 遥操作设备进行数据采集。

.. note::

   本页扩展了 :doc:`franka` 和 :doc:`franka_gello` 的内容。
   请先阅读这两份文档，了解 Franka 通用配置和 GELLO 安装流程。


为什么使用 Franky 进行关节空间控制？
--------------------------------------

RLinf 默认的 Franka 控制路径采用 **ROS + 笛卡尔阻抗控制** 。对于某些场景，
这存在两个局限：

1. **笛卡尔 IK 开销** ：GELLO 的关节读数经 FK 转为笛卡尔再转回关节，丢失信息
   且增加延迟。
2. **Python GIL 竞争** ：任何 Python 层面的 1 kHz 控制循环都与 Ray actor
   调度、相机 I/O、垃圾回收争夺 GIL，导致的抖动会触发 libfranka 安全反射。

**franky** 后端同时解决这两个问题：

- ``franky`` 将 1 kHz ``robot.control()`` 循环和 Ruckig 在线轨迹生成完全运行在
  C++ ``std::thread`` 中。
- 所有 pybind11 绑定通过 ``py::call_guard<py::gil_scoped_release>()`` 释放 GIL，
  Python 线程永远不会阻塞 RT 路径。
- GELLO 的关节目标 **直接** 映射到机器人关节——无需 FK/IK 往返转换。

最终实现了即使在高负载 Python/Ray 环境下也能 **无抖动实时控制** ，且关节空间
动作空间天然适合 RL 训练。


架构
-----

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - 组件
     - 职责
     - 关键类
   * - ``FrankyController``
     - Ray Worker：通过 libfranka 连接机器人，应用 RT 加固（mlockall、
       SCHED_FIFO），暴露 ``move_joints`` / ``get_state``
     - ``rlinf/envs/realworld/franka/franky_controller.py``
   * - ``FrankaJointEnv``
     - Gymnasium 环境：8D 动作空间 ``[j1..j7, gripper]``，关节限位和速度限位
       安全裁剪
     - ``rlinf/envs/realworld/franka/franka_joint_env.py``
   * - ``GelloJointIntervention``
     - ActionWrapper：用 GELLO 关节读数覆盖策略动作；可选 1 kHz 直接流模式
     - ``rlinf/envs/realworld/common/wrappers/gello_joint_intervention.py``
   * - ``GelloJointExpert``
     - 后台线程以 ~1 kHz 轮询 GELLO，暴露原始 7 自由度关节位置 + 夹爪状态
     - ``rlinf/envs/realworld/common/gello/gello_joint_expert.py``


GELLO 的两种流模式：

- **步进门控** （默认）：``GelloJointIntervention.action()`` 在环境的
  ``step_frequency`` (~10 Hz) 下返回 GELLO 关节位置。简单但丢弃大部分 GELLO 采样。
- **直接流** （ ``direct_stream=True`` ）：守护线程以 ~1 kHz 读取 GELLO 并直接
  推送关节目标到控制器，绕过环境步进频率限制。需配合环境配置
  ``teleop_direct_stream=True`` ，使 ``env.step()`` 仅做记录而不重复发送
  ``move_joints``——wrapper 拥有 1 kHz 硬件循环。


依赖安装
---------

1. 安装 franky 后端
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   # 方式 A：通过 install.sh
   bash requirements/install.sh embodied --env franka --franka-backend franky

   # 方式 B：手动 pip 安装（已有虚拟环境时）
   pip install "franky-control>=0.15.0"

franky 需要 PREEMPT_RT 内核、CPU 调速器和 rtprio/memlock 限制才能实现确定性行为，
详见仓库根目录下的 ``franky_install.md``。

2. 安装 GELLO 软件
^^^^^^^^^^^^^^^^^^^^^

按照 :doc:`franka_gello` 中的步骤安装 GELLO：

.. code-block:: bash

   # gello 驱动
   git clone https://github.com/wuphilipp/gello_software.git
   cd gello_software && git submodule init && git submodule update
   pip install -e . && pip install -e third_party/DynamixelSDK/python

   # gello-teleop
   pip install git+https://github.com/RLinf/gello-teleop.git

3. 每次开机前应用 RT 调优
^^^^^^^^^^^^^^^^^^^^^^^^^^^

运行控制器之前，执行系统调优：

.. code-block:: bash

   # CPU 性能模式
   sudo bash -c 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$g"; done'

   # 允许无限制 RT 调度
   sudo sysctl -w kernel.sched_rt_runtime_us=-1

验证一次性限制设置：

.. code-block:: bash

   ulimit -r   # 期望：99 或 unlimited
   ulimit -l   # 期望：unlimited


快速开始：数据采集
--------------------

使用预置配置进行 GELLO 关节空间数据采集：

.. code-block:: bash

   export EMBODIED_PATH=$(pwd)/examples/embodiment
   python examples/embodiment/train_embodied_agent.py \
       --config-name realworld_collect_data_gello_joint

**关键 YAML 设置** （ ``realworld_collect_data_gello_joint.yaml`` ）：

.. code-block:: yaml

   env:
     eval:
       use_gello_joint: True
       gello_port: "/dev/serial/by-id/your-gello-serial-port"
       override_cfg:
         joint_action_mode: "absolute"
         teleop_direct_stream: true   # 启用 1 kHz 流模式

   cluster:
     node_groups:
       - label: franka
         hardware:
           configs:
             - robot_ip: ROBOT_IP
               gripper_connection: /dev/ttyUSB0  # Robotiq 串口

.. list-table:: 关键配置字段
   :header-rows: 1
   :widths: 30 15 55

   * - 字段
     - 默认值
     - 说明
   * - ``use_gello_joint``
     - ``False``
     - 启用 GELLO **关节空间** 遥操作（与 ``use_gello`` 和 ``use_spacemouse``
       互斥）。
   * - ``joint_action_mode``
     - ``"absolute"``
     - ``"absolute"`` — GELLO 关节位置直接作为目标；
       ``"delta"`` — 关节增量，缩放系数为 ``joint_action_scale``。
   * - ``teleop_direct_stream``
     - ``False``
     - 为 ``True`` 时，wrapper 中的 1 kHz 守护线程直接将 GELLO 关节推送到
       控制器，``env.step()`` 跳过自身的 ``move_joints`` 调用。
   * - ``joint_velocity_limit``
     - ``0.5``
     - 每关节最大速度 (rad/s)，用于安全裁剪。
   * - ``gripper_connection``
     - ``null``
     - Robotiq 夹爪的串口（如 ``/dev/ttyUSB0``）。``FrankyController``
       **必填** 。


环境配置
---------

关节空间控制的环境 YAML 为
``examples/embodiment/config/env/realworld_franka_joint.yaml``：

.. code-block:: yaml

   env_type: realworld
   use_gello_joint: True
   use_relative_frame: False   # 关节空间不适用

   override_cfg:
     joint_action_mode: "absolute"
     joint_velocity_limit: 0.5
     gripper_type: robotiq
     gripper_connection: /dev/ttyUSB0

动作空间为 8D：``[j1, j2, j3, j4, j5, j6, j7, gripper]``。

- **绝对模式** 下，``j1..j7`` 为目标关节角度（弧度）。
- **增量模式** 下，``j1..j7`` 为缩放后的关节增量。


验证
-----

在运行数据采集前，可独立验证控制器：

.. code-block:: bash

   python toolkits/realworld_check/test_franky_controller.py \
       --robot-ip <ROBOT_IP> \
       --gripper-type robotiq \
       --gripper-connection /dev/ttyUSB0

该交互式工具支持 ``getjoint``、``home``、``nudge``、``open``/``close``、
``impedance``、``stream``（1 kHz 流测试）等命令。


常见问题
---------

**"SCHED_FIFO not granted" 警告**

- 未应用开机 RT 调优，或 ``/etc/security/limits.d/`` 未授予用户
  ``rtprio >= 80``。请参照 ``franky_install.md``。

**"gripper_connection must be specified" 错误**

- 在环境 YAML 的 ``override_cfg`` 或
  ``cluster.node_groups[].hardware.configs[]`` 中添加
  ``gripper_connection: /dev/ttyUSB0``。

**机器人在负载下嗡嗡作响或抖动**

- 确认 CPU 调速器设为 ``performance``。
- 确认 ``kernel.sched_rt_runtime_us=-1`` 已生效。
- 使用 ``cyclictest -p 80 -t 1 -n -i 1000 -l 10000`` 验证最坏延迟 < 100 μs。

**GELLO 读数停止更新**

- 检查 GELLO 设备已上电且串口可访问。
- ``GelloJointExpert`` 在连续 50 次读取错误后将自身标记为未就绪，此时 wrapper
  会回退到策略动作。

更多通用 Franka 问题排查，请参阅 :doc:`franka` 和
`FAQ <https://rlinf.readthedocs.io/zh-cn/latest/rst_source/faq.html>`_。
