# dual-Franka TCP-pose SFT — wrap-fix 远端部署说明

这份说明给在训练机上操作的 agent：把预处理 / norm_stats / SFT 启动三步走完，
并做一轮 PASS/FAIL 自检。

---

## 背景一句话

旧版 `preprocess_tcp_pose.py` 把每帧的 quat 独立转成 scipy XYZ-extrinsic
euler 后直接写入 `actions`；训练时 `openpi.transforms.DeltaActions` 做
`actions -= state`，没有 angle-wrap 处理。dual-Franka 夹爪朝下，L/R roll 长
期贴在 ±π，结果相邻帧真实只差 0.02 rad 时会被算出 ≈ ±2π 的假 delta，污染
`norm_stats.actions.q01/q99` 和 `std`，直接导致策略对 roll/yaw 方向信号学不
到（真机 eval 看到 ~82° yaw 误差）。

修复只在预处理层：

- `state_euler` 保持 canonical `[-π, π]`（和推理时 env 的输出一致）
- `action_euler[t] = state_euler[t] + wrap_to_pi(euler[t+1] - euler[t])`

`DeltaActions` 再做 `action - state` 得到的恰好是物理正确的小 delta，推理
侧的 `AbsoluteActions + euler→quat` 的 2π 周期性不受影响，所以模型下游
链路零改动。

---

## 前置

- 本机仓库根：`/mnt/public1/chenyinuo/RLinf`（`feature/dual-franka-gello-joint` 分支）
- LeRobot 根：`~/data/datasets/collected_data`（`HF_LEROBOT_HOME` 就指这里）
- 原始采集：`~/data/datasets/collected_data/rank_0/id_0/`（21 eps, 4055 frames）
- 旧的 preprocessed：`~/data/datasets/collected_data/YinuoTHU/Dual-franka-tcp/`
  （tcp_v1_14500 训练时用的，**不要覆盖**，留做 A/B）
- 旧 ckpt 的 norm_stats：`tcp_v1_14500` 下对应 `Dual-franka-tcp/norm_stats.json`，
  留做 Step 3 对比基线
- OpenPI venv：训练机上已装的那套（下面用 `$OPENPI_PY` 代称）

---

## Step 0：同步代码

```bash
cd /mnt/public1/chenyinuo/RLinf
git fetch origin
git pull --ff-only origin feature/dual-franka-gello-joint
```

本次拉下来的关键改动：

| 文件 | 动作 | 说明 |
|---|---|---|
| `toolkits/dual_franka/preprocess_tcp_pose.py` | M | wrap-aware euler action |
| `toolkits/dual_franka/verify_sft_on_dataset.py` | +A | SFT-on-dataset sanity 脚本（和预处理同步更新） |
| `toolkits/dual_franka/compare_norm_stats.py` | +A | Step 3 PASS/FAIL 工具 |
| `toolkits/dual_franka/TCP_POSE_INFERENCE_FLOW.md` | +A | 策略推理链路手册 |
| `rlinf/models/embodiment/openpi/dataconfig/__init__.py` | M | 新增 `pi05_dualfranka_wrapfix` TrainConfig |
| `examples/sft/config/dual_franka_tcp_wrapfix_sft_openpi.yaml` | +A | SFT 启动 yaml |

---

## Step 1：跑新预处理

**注意**：本地（真机采集机）上已经 rsync 了新预处理的产物到
`~/data/datasets/collected_data/YinuoTHU/Dual-franka-tcp-wrapfix/`，你可以
**跳过重跑**，直接到 Step 2 检查目录齐全即可：

```bash
ls ~/data/datasets/collected_data/YinuoTHU/Dual-franka-tcp-wrapfix/
# 应该看到: data/  meta/
ls ~/data/datasets/collected_data/YinuoTHU/Dual-franka-tcp-wrapfix/meta/
# episodes.jsonl  episodes_stats.jsonl  info.json  tasks.jsonl
ls ~/data/datasets/collected_data/YinuoTHU/Dual-franka-tcp-wrapfix/data/chunk-000/ | wc -l
# 期望: 21
```

**如果产物丢失或者不完整**，自己从原始数据重跑：

```bash
cd /mnt/public1/chenyinuo/RLinf
$OPENPI_PY toolkits/dual_franka/preprocess_tcp_pose.py \
    --src ~/data/datasets/collected_data/rank_0/id_0 \
    --dst ~/data/datasets/collected_data/YinuoTHU/Dual-franka-tcp-wrapfix \
    --new-repo-id YinuoTHU/Dual-franka-tcp-wrapfix
```

> 如果你确认训练集里除了 `rank_0/id_0` 还有其他 collect session（比如
> 旧的 `tcp_v1_14500` 是在更大的合并集上训的），请把所有 session 原始目录
> 列出来逐一跑 `preprocess_tcp_pose.py`，都指到同一个 `--dst`。meta 会被
> 最后一次覆盖，手动合并 `episodes.jsonl` / `tasks.jsonl` / `episodes_stats.jsonl`。

---

## Step 2：计算 norm_stats

```bash
cd /mnt/public1/chenyinuo/RLinf
export HF_LEROBOT_HOME=~/data/datasets/collected_data

$OPENPI_PY toolkits/lerobot/calculate_norm_stats.py \
    --config-name pi05_dualfranka_wrapfix \
    --repo-id YinuoTHU/Dual-franka-tcp-wrapfix
```

写出路径（根据 `calculate_norm_stats.py:145-147` 和 `pi05_dualfranka_wrapfix`
TrainConfig 的 `assets_dir="checkpoints/torch/pi05_base/assets"`）：

```
<config.assets_dirs>/YinuoTHU/Dual-franka-tcp-wrapfix/norm_stats.json
```

这个 `<config.assets_dirs>` 的具体解析结果看 `_override_with_model_path` 逻辑；
如果启动训练时报 `norm_stats.json not found`，手动把上面写出的文件 cp/ln
到 `actor.model.model_path` 下 `<repo_id>/norm_stats.json` 即可（和旧
`tcp_v1_14500/Dual-franka-tcp/norm_stats.json` 是同样的放法）。

---

## Step 3：PASS/FAIL 自检（不能省）

用新老 `norm_stats.json` 对比，直接跑：

```bash
cd /mnt/public1/chenyinuo/RLinf
OLD=<path/to>/tcp_v1_14500/Dual-franka-tcp/norm_stats.json
NEW=<path/to>/pi05_base/assets/YinuoTHU/Dual-franka-tcp-wrapfix/norm_stats.json

$OPENPI_PY toolkits/dual_franka/compare_norm_stats.py --old "$OLD" --new "$NEW"
```

**期望看到**（最底下的 verdict 部分全 PASS）：

- `L_roll`, `R_roll`, `R_yaw` 的 `std` **从 1-2 rad 掉到 <0.05 rad**
- euler 通道的 `q99-q01` 从 ~12 rad 掉到 <1 rad
- 其他已经干净的通道（L_pitch / L_yaw / R_pitch）基本无变化

如果 verdict 是 FAIL：
1. 确认 `--repo-id` 和实际数据目录名一致
2. 确认 `calculate_norm_stats.py` 是基于 `pi05_dualfranka_wrapfix`（不是
   老的 `pi05_dualfranka`）跑的 — 老 config 会指向老 dataset
3. 如果产物完整但 std 还是偏大，可能是数据集里除了 21 eps 还混了没预处理
   过的老数据。清库重跑 Step 1。

把 compare_norm_stats 的完整输出贴给本机 agent，我帮你终审一下。

---

## Step 4：启动 SFT

**不要从 tcp_v1_14500 resume**，从 `pi05_base` 开始：

```bash
cd /mnt/public1/chenyinuo/RLinf
bash examples/sft/run_vla_sft.sh dual_franka_tcp_wrapfix_sft_openpi
```

这个 yaml（`examples/sft/config/dual_franka_tcp_wrapfix_sft_openpi.yaml`）
和 `custom_sft_openpi.yaml` 的唯一区别是 `openpi.config_name` 指向了
`pi05_dualfranka_wrapfix`（即新 repo_id + 新 norm_stats）。启动前请再确认：

- `actor.model.model_path`（`/mnt/public1/chenyinuo/RLinf/models/pi05/torch`）
  下有 `YinuoTHU/Dual-franka-tcp-wrapfix/norm_stats.json`（Step 2 末尾的放置）
- `data.train_data_paths` 指向 LeRobot 根（`~/data/datasets/collected_data`）
- 新实验 wandb project 叫 `dual-franka-openpi`, experiment 叫
  `dual_franka_tcp_wrapfix_v1`（可改）

---

## Step 5：训练中回读（可选但建议）

第一个 ckpt（500 steps）落盘后，在真机采集机上跑 `verify_sft_on_dataset.py`
指到新 ckpt，对比预测 action 和 dataset GT action：

```bash
# 在真机端
/home/i-yinuo/cynws/RLinf/.venv-openpi/bin/python \
    toolkits/dual_franka/verify_sft_on_dataset.py \
    --episodes 0,5,10,15 --frames-per-ep 5
# (记得改脚本里的 MODEL_PATH 指向新 ckpt)
```

旧 ckpt 在这个脚本下典型会报 `wp0 ang L/R ≈ 几十度`；修复后应该回到 <10°。

---

## 常见坑

1. **两份 `norm_stats` 混了**：训练时 OpenPI loader 会按 `<model_path>/<repo_id>`
   读；如果你把旧 norm_stats 放在 `Dual-franka-tcp-wrapfix/` 下面了，loss
   会爆炸。启动前 `md5sum` 确认一下。
2. **venv 不是 openpi**：`preprocess_tcp_pose.py` 和 `compare_norm_stats.py`
   都只需要 numpy/scipy，理论上任何环境都能跑；但 `calculate_norm_stats.py`
   必须是 openpi venv（import openpi.*）。
3. **旧 ckpt 不要 resume**：`runner.resume_dir` 留空。硬要 resume，归一化
   空间错位，训练跟从头训没区别还更不稳。

---

## Rollback

如果有问题要回到修复前，保留的手段：

- 旧 repo_id `YinuoTHU/Dual-franka-tcp/` 没有被动
- 旧 TrainConfig `pi05_dualfranka` 没有被改（新 config 是平行的）
- 旧 yaml `custom_sft_openpi.yaml` 没有被改
- 旧 ckpt `tcp_v1_14500` 没有被动

直接还原到旧 config 跑就行，代码没破坏向后兼容。
