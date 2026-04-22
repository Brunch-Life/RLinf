# Dual-Franka GELLO joint-space — 本分支所有修改

分支 `feature/dual-franka-gello-joint`。下表按提交顺序（旧 → 新），从 `995a2f9`（main 起点）之后开始。

| # | Commit | 标题 | 主要文件 |
|---|---|---|---|
| 1 | `160ae61` | `docs(franka-gello): only init DynamixelSDK submodule` | `docs/source-*/.../franka_gello.rst` |
| 2 | `dd919c6` | `chore(gitignore): ignore top-level datasets/ and models/` | `.gitignore` |
| 3 | `b514093` | `feat(logger): support wandb_entity in MetricLogger config` | `rlinf/utils/metric_logger.py` |
| 4 | `0d9b4a1` | `feat(realworld/franka): add franky-control RT backend` | `requirements/embodied/franky_install.{sh,md}`、`requirements/install.sh`（`--franka-backend`）、`rlinf/envs/realworld/franka/franky_controller.py` |
| 5 | `1989294` | `feat(realworld): Ray multi-node launch scripts for dual-Franka` | `ray_utils/realworld/start_ray_node{0,1}.sh` |
| 6 | `91d1c94` | `feat(realworld/camera): add LUMOS USB camera + realsense busy-retry` | `rlinf/envs/realworld/common/camera/{__init__,lumos_camera,realsense_camera}.py` |
| 7 | `a8d1197` | `feat(realworld/gello): dual-arm joint-space GELLO teleop` | `common/gello/gello_joint_expert.py`、`common/wrappers/{dual_gello_joint_intervention,keyboard_start_end_wrapper,apply,__init__,reward_done_wrapper}.py`、`examples/embodiment/gello_*.sh`、`toolkits/realworld_check/gello_*.py` |
| 8 | `80df3b0` | `feat(realworld): dual-Franka joint-space teleop data collection` | `franka/dual_franka_joint_env.py`、`franka_env.py`（gripper 非阻塞）、`realworld_env.py`（psutil 防御 + extra_view_image_names）、`common/gripper/robotiq_gripper.py`（FTDI 重试）、`common/keyboard/keyboard_listener.py`（无损 queue）、`common/wrappers/gripper_close.py`（N-D 泛化）、`data/{lerobot_writer,embodied_io_struct}.py`、`envs/wrappers/collect_episode.py`（record-gate）、`examples/embodiment/{collect_real_data.py, config/env/realworld_franka_joint_dual.yaml, config/realworld_collect_data_gello_joint_dual_franka.yaml}`、`tests/test_dual_arm_data_collection.py`、`toolkits/realworld_check/collect_monitor.py` |
| 9 | `99c9a0c` | `feat(sft): [UNTESTED] Dual-Franka rot6d SFT pipeline for pi0.5` | `rlinf/utils/rot6d.py`、`openpi/transforms/rigid_body_delta.py`、`openpi/dataconfig/dual_franka_rot6d_dataconfig.py` + `__init__.py`（注册 `pi05_dualfranka_rot6d`）、`openpi/policies/dual_franka_rot6d_policy.py`、`examples/sft/config/dual_franka_rot6d_sft_openpi.yaml`、`tests/test_{rot6d,rigid_body_delta}.py` |
| 10 | `57c53c5` | `feat(realworld/deploy): [UNTESTED] dual-Franka rot6d eval path` | `realworld_env.py`（`dispatch_chunk` hook）、`openpi/__init__.py`（norm_stats checkpoint-pinned 优先 + repo fallback）、`openpi/openpi_action_model.py`（`extra_view_image_names` 消费）、`examples/embodiment/config/realworld_eval_dual_franka.yaml`、`tests/test_dual_franka_dummy.py` |

## 关键设计决定

| 决定 | 为什么 |
|---|---|
| 控制后端从 ROS/serl 换成 franky-control | C++ RT 线程 + Ruckig + pybind11 释放 GIL，Ray actor 不再和 RT 回路抢 GIL |
| 位姿表示走 rot6d 而非 euler | 2π 不变、无 ±π roll 边界 wrap；SE(3) delta 走矩阵乘法而非逐通道减法 |
| GELLO 直推 1 kHz 旁路 `env.step` 节拍 | `env.step` 在 policy rate（~30 Hz）太慢，impedance 跟踪有延迟感；daemon 直推到 franky 就跟手 |
| Keyboard `a/b/c` 录制门 + 无损 press queue | 10 Hz 轮询会丢 <100 ms 的 tap；无损 queue 在 evdev 线程里抓首次按下事件 |
| `norm_stats` checkpoint-pinned 优先 | repo 资产漂移 → 推理 silently 出错；强制 pin，fallback 大声喊 |
| `[UNTESTED]` tag 在 commits 9 / 10 | 端到端训练和 eval 还没跑过，只过了 unit test |

## `[UNTESTED]` 清单

- SFT 训练：`calculate_norm_stats.py` → `dual_franka_rot6d_sft_openpi.yaml` 整条链 **未跑过**
- 部署 eval：pi0.5 checkpoint → `realworld_eval_dual_franka.yaml` 端到端 **未跑过**
- 第一次跑训练和 eval 时预期会有小 fixup
