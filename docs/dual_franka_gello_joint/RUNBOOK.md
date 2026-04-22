# Dual-Franka GELLO 数采 → 训练 → 部署 全流程

假设：仓库路径 `$REPO`，HF LeRobot 根 `$LEROBOT_ROOT`，GELLO 串口 by-id 已查到。

## 0. 一次性准备

### 0.1 环境安装（两个 node 都要）
```bash
cd $REPO
bash requirements/install.sh --target embodied --env franka --franka-backend franky
```
按 `requirements/embodied/franky_install.md` 做 **PREEMPT_RT + sched_rt_runtime_us + rtprio/memlock ulimit + cpu governor performance** 一次性系统调优。

### 0.2 GELLO 软件（单独，只在 env-worker 机器）
按 `docs/source-en/rst_source/examples/embodied/franka_gello.rst`，`git submodule update --init third_party/DynamixelSDK`（不要初始化 mujoco_menagerie）。

### 0.3 GELLO 标定（一次性，两只手各做一次）
```bash
bash $REPO/examples/embodiment/gello_calibrate.sh
bash $REPO/examples/embodiment/gello_align.sh       # 对齐检查
```

## 1. 数采

### 1.1 启动 Ray 集群
```bash
# head (node 0, 10.10.10.1)
bash $REPO/ray_utils/realworld/start_ray_node0.sh
# worker (node 1, 10.10.10.2)
bash $REPO/ray_utils/realworld/start_ray_node1.sh
```

### 1.2 编辑采集 YAML
- `examples/embodiment/config/env/realworld_franka_joint_dual.yaml` — 填 `left_gello_port` / `right_gello_port` by-id 路径
- `examples/embodiment/config/realworld_collect_data_gello_joint_dual_franka.yaml` — 填 `save_dir` / `num_data_episodes` / task_description

### 1.3 跑采集
```bash
cd $REPO
source .venv/bin/activate
python examples/embodiment/collect_real_data.py \
    --config-name realworld_collect_data_gello_joint_dual_franka
```
操作：`a` 开始录 / 重录（重新开始当前集），`c` 成功结束，`b` 失败结束。

### 1.4（可选）另一 terminal 监控
```bash
python toolkits/realworld_check/collect_monitor.py <collector_log_path>
```

采集完成后得到一个 LeRobot 数据集，路径 = `save_dir`，格式为 quat-layout（**还不是 rot6d**）。

## 2. 训练（[UNTESTED]）

### 2.1 把 quat-layout 数据集改写成 rot6d_v1
```bash
python toolkits/dual_franka/backfill_rot6d.py \
    --src  $LEROBOT_ROOT/<your_quat_repo_id> \
    --dst  $LEROBOT_ROOT/YinuoTHU/Dual-franka-rot6d \
    --new-repo-id YinuoTHU/Dual-franka-rot6d
```
产出：20-d state 前缀 + 20-d policy-facing action + 重写后的 `episodes_stats.jsonl`。

### 2.2 算 norm_stats（**必跑**，否则 SFT 起不来）
```bash
HF_LEROBOT_HOME=$LEROBOT_ROOT PYTHONPATH=$REPO \
    .venv/bin/python toolkits/lerobot/calculate_norm_stats.py \
    --config-name pi05_dualfranka_rot6d \
    --repo-id YinuoTHU/Dual-franka-rot6d
```

### 2.3 启 SFT
编辑 `examples/sft/config/dual_franka_rot6d_sft_openpi.yaml` 里的路径（`model_path` → pi05_base、`train_data_paths` → LeRobot 根、`logger` 按需）后：
```bash
python examples/sft/main.py --config-name dual_franka_rot6d_sft_openpi
```
Checkpoint 会落到 `runner.logger.log_path` 下。

## 3. 部署 / Eval（[UNTESTED]）

### 3.1 编辑 eval YAML
`examples/embodiment/config/realworld_eval_dual_franka.yaml`：
- `actor.model.model_path` → 2.3 的 checkpoint
- `actor.model.openpi.config_name` = `pi05_dualfranka_rot6d`（要和训练一致）
- GELLO 端口字段按需置空（eval 不需要 teleop）

### 3.2 起 Ray（同 1.1）。

### 3.3 跑 eval
```bash
python examples/embodiment/eval_real_data.py \
    --config-name realworld_eval_dual_franka
```
首次加载会从 checkpoint 里 `<checkpoint_dir>/<asset_id>/norm_stats.json` 读 norm stats；如果该文件不在，会回落到 repo 资产并**大声警告**—— 看到警告就去确认 repo 资产没漂移。

## 常见陷阱

| 现象 | 原因 | 解决 |
|---|---|---|
| 跑 `env.step` 时抖得厉害 | 跑在了 joint stream mode 但 franky 系统调优没做 | 按 `franky_install.md` 做 RT tuning，`ulimit -r` 应 ≥99 |
| LeRobot 写入时 action 维度报错 | 跑 SFT 前没做 `backfill_rot6d.py` | 先 backfill 再 norm_stats |
| Eval 动作方向看起来"小而怪" | `norm_stats` fallback 漂移了 | 看 stderr 有没有那条 `NORM STATS FALLBACK` banner，有就复制训练时的 `norm_stats.json` 到 checkpoint 下 |
| Robotiq 夹爪开合卡 600 ms | 用到了老代码的 `.wait()+sleep` 路径 | 确认跑在 `80df3b0` 之后的 `franka_env.py` |
| GELLO `a/b/c` 按键偶尔丢 | 用了 `get_key` 而不是 `pop_pressed_keys` | `KeyboardStartEndWrapper` 走的就是 queue，新 wrapper 栈已默认开 |
