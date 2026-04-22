# Dual-Franka TCP-pose SFT → 真机推理数据流

本文档把训练、归一化、推理、动作下发的每一环代码路径**完整**理一遍，用以回答这个问题：
当前从策略输出到控制器下发的链路里，模型输出的 *delta action* 最终是**怎样**变成绝对 TCP pose 的？状态 *state* 是在**哪一步**加回去的？控制器有没有办法直接吃 delta？

> 所有结论都来自对当前仓库 + `.venv-openpi` 的代码阅读，**每一条都附带文件:行号以便复核**。

---

## 1. 数据集离线预处理：保留"绝对 TCP 目标"

文件：[toolkits/dual_franka/preprocess_tcp_pose.py](preprocess_tcp_pose.py)

原始 lerobot parquet 里每帧：
- `state`: 68 维 GELLO 采集格式，`[0]=L_grip, [1]=R_grip, [2:9]=L_joints, [9:16]=R_joints, [36:50]=L_tcp_xyz+L_quat+R_tcp_xyz+R_quat, …`。
- `actions`: 16 维关节动作 `[L7joints, L_grip_trig, R7joints, R_grip_trig]`。

预处理**改写**为 TCP-pose 格式（[preprocess_tcp_pose.py:81-98](preprocess_tcp_pose.py#L81-L98)）：

- `new_state[:16] = [L_grip, R_grip, L_xyz, L_euler_xyz, 0pad, R_xyz, R_euler_xyz, 0pad]` —— 每臂 7 维，euler 是 scipy 的 **extrinsic `xyz`**（小写）。
- `new_action[t, :16] = [L_xyz, L_euler_xyz, 0pad, L_grip_trig, R_xyz, R_euler_xyz, 0pad, R_grip_trig]`，其中每臂的 7 维 TCP 是 **`state[t+1]`** 即"下一帧"的 EE pose；末帧复制当前帧；grip 触发值原样保留。

**关键事实：dataset 里 actions 是"绝对 EE pose"，不是 delta**（见文件头注释 [preprocess_tcp_pose.py:10-17](preprocess_tcp_pose.py#L10-L17)）。delta 的计算**延迟到训练 pipeline 里**，由 `DeltaActions` 变换动态做。

---

## 2. 训练 DataConfig：挂上 DeltaActions / AbsoluteActions

文件：[rlinf/models/embodiment/openpi/dataconfig/dual_franka_dataconfig.py](../../rlinf/models/embodiment/openpi/dataconfig/dual_franka_dataconfig.py)

`DualFrankaDataConfig.create()` 里做两件事：

**(a) 数据 transforms：先绝对 → 再 delta 化**（[dual_franka_dataconfig.py:60-80](../../rlinf/models/embodiment/openpi/dataconfig/dual_franka_dataconfig.py#L60-L80)）：

```python
data_transforms = _transforms.Group(
    inputs=[DualFrankaInputs(action_dim=..., model_type=...)],
    outputs=[DualFrankaOutputs()],
)
if self.extra_delta_transform:  # True for pi05_dualfranka
    delta_action_mask = [True]*7 + [False] + [True]*7 + [False]
    data_transforms = data_transforms.push(
        inputs=[_transforms.DeltaActions(delta_action_mask)],
        outputs=[_transforms.AbsoluteActions(delta_action_mask)],
    )
```

**(b) `Group.push` 的顺序**（[.venv-openpi/openpi/transforms.py:49-59](../../.venv-openpi/lib/python3.11/site-packages/openpi/transforms.py)）：

```
inputs  : 追加到末尾   →  [DualFrankaInputs, DeltaActions]
outputs : 追加到开头   →  [AbsoluteActions, DualFrankaOutputs]
```

所以训练/推理时"数据变换"组的顺序是：

```
inputs  (训练/推理都走):
    DualFrankaInputs  →  DeltaActions
outputs (仅推理):
    AbsoluteActions   →  DualFrankaOutputs
```

**Delta mask**（16 维）：

| 维度 | 内容 | mask |
|---|---|---|
| 0-2 | L_xyz | True（delta） |
| 3-5 | L_euler_xyz | True（delta） |
| 6 | L 0pad | True（delta，但 state=action=0 → delta=0） |
| 7 | L_grip_trig | **False（绝对保留）** |
| 8-10 | R_xyz | True |
| 11-13 | R_euler_xyz | True |
| 14 | R 0pad | True |
| 15 | R_grip_trig | **False** |

### 2.1 DeltaActions / AbsoluteActions 的精确实现

文件 [.venv-openpi/openpi/transforms.py:204-244](../../.venv-openpi/lib/python3.11/site-packages/openpi/transforms.py)：

```python
class DeltaActions:
    def __call__(self, data):
        if "actions" not in data or self.mask is None:
            return data
        state, actions = data["state"], data["actions"]
        mask = np.asarray(self.mask); dims = mask.shape[-1]
        # actions shape (chunk, D); state shape (D,)
        actions[..., :dims] -= np.expand_dims(
            np.where(mask, state[..., :dims], 0), axis=-2
        )
        data["actions"] = actions
        return data

class AbsoluteActions:
    # 完全对称，把 -= 换成 +=
```

**两个要点钉死：**

1. **"action 没在 data 里就整个 no-op"**：inference 时 `"actions"` 不存在（那正是模型要预测的东西），`DeltaActions.__call__` 直接返回。所以 **inputs 里的 DeltaActions 在 inference 阶段不起作用**——它只在训练时对 dataset 里读进来的 actions 减 state。
2. **AbsoluteActions 的 state 是"当次调用时 data 里的 state"**：一次 `+=` 里，一个 16 维 state 被**broadcast 到整个 20 步的 chunk**（`np.expand_dims(..., axis=-2)`）。即**所有 20 步用的都是同一个 state**（inference 那一刻的 obs），不是每步一个实时 state。

---

## 3. Policy 推理链的完整拼装

文件：[rlinf/models/embodiment/openpi/__init__.py:130-151](../../rlinf/models/embodiment/openpi/__init__.py#L130-L151)

```python
model.setup_wrappers(
    transforms=[
        *repack_transforms.inputs,             # 空
        transforms.InjectDefaultPrompt(None),  # 注入默认 prompt（本例为 None）
        *data_config.data_transforms.inputs,   # DualFrankaInputs, DeltaActions(no-op at infer)
        transforms.Normalize(norm_stats, use_quantiles=True),  # pi05 用 quantile
        *data_config.model_transforms.inputs,  # InjectDefaultPrompt, ResizeImages, TokenizePrompt, PadStatesAndActions(32)
    ],
    output_transforms=[
        *data_config.model_transforms.outputs, # 空（pi05）
        transforms.Unnormalize(norm_stats, use_quantiles=True),
        *data_config.data_transforms.outputs,  # AbsoluteActions, DualFrankaOutputs
        *repack_transforms.outputs,            # 空
    ],
)
```

其中 `use_quantile_norm=True` 因为 pi05 ≠ pi0（见 [.venv-openpi/openpi/training/config.py:186](../../.venv-openpi/lib/python3.11/site-packages/openpi/training/config.py#L186)）。

### 3.1 Quantile Normalize 公式

文件 [.venv-openpi/openpi/transforms.py:141-145,175-181](../../.venv-openpi/lib/python3.11/site-packages/openpi/transforms.py)：

```python
# Normalize
x_norm = (x - q01) / (q99 - q01 + 1e-6) * 2 - 1
# Unnormalize
x_raw = (x_norm + 1) / 2 * (q99 - q01 + 1e-6) + q01
```

Round-trip 在 1e-6 数值精度内恒等。所以 Unnormalize 后的 state 就是 inference 时真实的那一份 16-d state（padding 到 32 的后 16 维保持 0）。

### 3.2 norm_stats 是对谁算的

训练时顺序是 `DualFrankaInputs → DeltaActions → Normalize`。所以：

- **state stats**：对 16-d 重排后的 state 计算（绝对 TCP 姿态）
- **action stats**：对**已经 delta 化**的 actions 计算（delta TCP 姿态 + 绝对 gripper 触发）

这一点直接影响对 norm_stats.json 的解读，也回答了上一轮我看到"L_roll action std=1.139 rad (65°)"——那是 **delta roll 的分布**，不是绝对 roll 的分布。

---

## 4. env 观测 → 模型输入 → 模型输出 → env 动作：逐步追踪

### 4.1 env 输出的 states 长什么样

文件 [rlinf/envs/realworld/franka/dual_franka_joint_env.py:538-600](../../rlinf/envs/realworld/franka/dual_franka_joint_env.py#L538-L600)：

`_get_observation()` 返回的字典里 `state["joint_position"]` **在 tcp_target 模式下被替换**（[dual_franka_joint_env.py:549-556](../../rlinf/envs/realworld/franka/dual_franka_joint_env.py#L549-L556)）：

```python
if self.config.joint_action_mode == "tcp_target":
    joint_position = self._tcp_euler_14d()   # [L_xyz, L_euler, 0pad, R_xyz, R_euler, 0pad]
```

`_wrap_obs`（[rlinf/envs/realworld/realworld_env.py:217-222](../../rlinf/envs/realworld/realworld_env.py#L217-L222)）把 `state` dict 按 key **字母序** concatenate：

```
gripper_position(2) | joint_position(14) | joint_velocity(14) | tcp_force(6) | tcp_pose(14) | tcp_torque(6) | tcp_vel(12)
```

所以 `env_obs["states"][:16]` = `[L_grip, R_grip, L_xyz, L_euler, 0pad, R_xyz, R_euler, 0pad]`（刚好对上 preprocess 里的 `new_state[:16]`）。

### 4.2 模型侧的 state 重排（DualFrankaInputs）

文件 [rlinf/models/embodiment/openpi/policies/dual_franka_policy.py:48-54,86-87](../../rlinf/models/embodiment/openpi/policies/dual_franka_policy.py#L48-L54)：

```python
def _rearrange_state(state):
    s = state[..., :16]
    return np.concatenate(
        [s[..., 2:9],   # L_xyz, L_euler, 0pad  → positions [0:7]
         s[..., 0:1],   # L_grip               → position  [7]
         s[..., 9:16],  # R_xyz, R_euler, 0pad  → positions [8:15]
         s[..., 1:2]],  # R_grip               → position  [15]
        axis=-1)
```

重排后 state 变成 **与 action 同布局**：`[L_xyz, L_euler, 0pad, L_grip, R_xyz, R_euler, 0pad, R_grip]`，这就是 delta mask 要对上的布局。

之后 `PadStatesAndActions(32)`（[.venv-openpi/openpi/transforms.py:328-336](../../.venv-openpi/lib/python3.11/site-packages/openpi/transforms.py)）把 state 从 16 维补 0 到 32 维，因为 pi05 的 `action_dim=32`（见 [rlinf/models/embodiment/openpi/dataconfig/__init__.py:403-406](../../rlinf/models/embodiment/openpi/dataconfig/__init__.py#L403-L406)）。

### 4.3 模型输出 & AbsoluteActions 加回 state

文件 [rlinf/models/embodiment/openpi/openpi_action_model.py:578-585](../../rlinf/models/embodiment/openpi/openpi_action_model.py#L578-L585)：

```python
outputs = self.sample_actions(observation, mode=mode, ...)
actions = self.output_transform(
    {"actions": outputs["actions"], "state": observation.state}
)["actions"]
```

`observation.state` 是 **归一化 + padding 后的 32 维 state**。`output_transform`（[openpi_action_model.py:303-317](../../rlinf/models/embodiment/openpi/openpi_action_model.py#L303-L317)）逐样本跑：

```
for i in range(B):
    sample = { "actions": normalized_delta_chunk (20, 32),
               "state":   normalized_padded_state (32,) }
    sample = Unnormalize(sample)   # state/actions 都反归一化
    sample = AbsoluteActions(sample)   # actions[:, :16] += state[:16] (mask 掉 grip)
    sample = DualFrankaOutputs(sample) # 截到 actions[:, :16]
```

**结果**：`actions.shape = (B, 20, 16)`，每步 k 的每个 mask=True 维 = `unnorm_delta[k] + unnorm_state_at_inference_time`；grip 两维不加 state（模型输出绝对 gripper 触发值）。

### 4.4 动作下发到控制器

runner 拿到 (B, 20, 16) 的 actions，调 `RealWorldEnv.chunk_step(actions)`。

在 [rlinf/envs/realworld/realworld_env.py:326-343](../../rlinf/envs/realworld/realworld_env.py#L326-L343) 里：

```python
if chunk_size > 0:
    actions_np = chunk_actions[0].detach().cpu().numpy()   # (20, 16)
    self.env.call("dispatch_chunk", actions_np)            # 派发到 DualFrankaJointEnv
```

然后 [dual_franka_joint_env.py:303-376](../../rlinf/envs/realworld/franka/dual_franka_joint_env.py#L303-L376) 的 `dispatch_chunk`：

1. 把 16-d 动作 reshape 成 `(20, 2, 8)` per-arm。
2. 每步把 `[xyz, euler, pad, grip]` 中的 xyz + euler 抽出来，clip 到 `ee_pose_limit_min/max`，euler→quat。
3. 得到 `left_poses (20, 7)` 和 `right_poses (20, 7)`（xyz + quat_xyzw）。
4. 对每条 pose 序列跑 `_finite_diff_twists` 生成 (20, 6) twist。
5. 分别调 `FrankyController.move_waypoints(poses, twists)` 把整段 20 waypoint 作为**一个** `CartesianWaypointMotion` 派发给 libfranka。

控制器侧 [franky_controller.py:703-795](../../rlinf/envs/realworld/franka/franky_controller.py#L703-L795)：把 poses 作为 `franky.Affine`、twists 作为 `franky.Twist` 包进 `franky.CartesianWaypoint`，用 `robot.move(motion, asynchronous=True)` 下发。motion 本身是绝对 6-DoF TCP 轨迹，**在进入 libfranka 那一刻它就是绝对目标**。

---

## 5. 钉结论：delta 到底在哪一步消失

> **在 `AbsoluteActions` 那一步，inference-time 的 state 被加回到模型输出上，`actions[..., :16] += state[..., :16]`，输出就已经是绝对 TCP 目标。** 之后 `DualFrankaOutputs` 只做 slice，`env.chunk_step → dispatch_chunk → move_waypoints` 都是按绝对 pose 处理的，不再碰 state。

以下几条在本次复核中逐一确认：

- ✅ **dataset 里 actions = 绝对 EE pose**（pred target = state_{t+1}）。
  Source: [preprocess_tcp_pose.py:89-98](preprocess_tcp_pose.py#L89-L98)。

- ✅ **训练时 DeltaActions 动态减去 state[t]**，模型学到的是 delta。
  Source: [dual_franka_dataconfig.py:73-80](../../rlinf/models/embodiment/openpi/dataconfig/dual_franka_dataconfig.py#L73-L80) + [.venv-openpi/transforms.py:219](../../.venv-openpi/lib/python3.11/site-packages/openpi/transforms.py#L219)。

- ✅ **norm_stats.json 里的 "actions" 分布 = delta 分布**（不是绝对 pose 分布）。因为 Normalize 在 DeltaActions **之后**执行（training pipeline），统计量是对 delta 算的。
  Source: [rlinf/models/embodiment/openpi/__init__.py:134-141](../../rlinf/models/embodiment/openpi/__init__.py#L134-L141) + Group.push 顺序。

- ✅ **inference 时 DeltaActions 在 inputs chain 里但是 no-op**（"actions" 不在 data 里）。
  Source: [.venv-openpi/transforms.py:212-214](../../.venv-openpi/lib/python3.11/site-packages/openpi/transforms.py#L212-L214)。

- ✅ **AbsoluteActions 加的 state = observation.state**（predict_action_batch 显式传入），即 obs 采集那一刻的 state，**不是** dispatch 时刻的 state。
  Source: [openpi_action_model.py:583-585](../../rlinf/models/embodiment/openpi/openpi_action_model.py#L583-L585)。

- ✅ **同一个 state 被 broadcast 到 20 步 chunk 的每一步**，不是每步独立加。
  Source: [.venv-openpi/transforms.py:241](../../.venv-openpi/lib/python3.11/site-packages/openpi/transforms.py#L241) 里的 `np.expand_dims(..., axis=-2)`。

- ✅ **mask 是 `[T]*7 + [F] + [T]*7 + [F]`**：xyz/euler/pad 被 delta 化，grip 触发值不被 delta 化（模型直接输出绝对 grip trigger）。

- ✅ **state 重排在 `_rearrange_state` 里完成**，把原始 `[L_grip, R_grip, L_tcp7, R_tcp7]` 变成与 action 同布局的 `[L_tcp7, L_grip, R_tcp7, R_grip]`，delta 才能按 mask 对位相减/相加。
  Source: [dual_franka_policy.py:48-54](../../rlinf/models/embodiment/openpi/policies/dual_franka_policy.py#L48-L54)。

- ✅ **控制器目前吃的是绝对 pose**，没有任何"相对" / "delta" 通道。
  Source: [franky_controller.py:703-795](../../rlinf/envs/realworld/franka/franky_controller.py#L703-L795) + libfranka `CartesianWaypointMotion` 本身就是绝对轨迹。

- ⚠️ **obs 采集时刻 ↔ dispatch 时刻之间的时间差没有被补偿**：
  - `predict_action_batch` 用的 state 来自 env_obs（采集发生在上一个 chunk_step 结束时）
  - `dispatch_chunk` 紧接着派发，几十到几百 ms 后 libfranka 真正开始执行
  - 中间机器人可能还在被上一个 chunk 尾巴带动，真实 pose ≠ obs 时刻的 state
  - AbsoluteActions 加的是 obs 时刻的 state，所以 **chunk 第一个 waypoint 的绝对目标是"基于旧 state 计算的"**，与真实 current_pose 有"被用 delta 放大"的误差

---

## 6. 直接让控制器吃 delta 有没有路径？

当前代码里**没有**。有两条可行路线：

### 路线 A：下发前把 delta 恢复出来，改由控制器现场加 current_pose

- 跳过 `AbsoluteActions` 或者在它之后"减回"策略传入的 state，得到 `delta_chunk (20, 16)`。
- 新增 `FrankyController.move_delta_waypoints(delta_poses, delta_twists)`，内部：
  1. `base_pose = robot.current_pose`（dispatch 那一刻的真实 pose）
  2. `absolute_pose[k] = base_pose ⊕ delta_pose[k]`（位置直接相加；旋转要用四元数乘法，**不能直接 euler 相加**）
  3. 用 `absolute_pose` 包 `CartesianWaypointMotion` 下发
- **物理意义**：每个 chunk 的起点绑定到 dispatch 时刻的真实 pose，不再受 obs→dispatch 之间的漂移影响；chunk-to-chunk 抢占时连续性也更好（因为新 chunk 的 wp0 是 current + small delta，不会"跳"）。
- **成本**：
  - 旋转 delta 不能在 euler 空间做（奇异点会炸），必须在四元数或 rotvec 空间做 `q_abs = q_delta * q_current`。
  - 训练 delta 是在 euler 空间算的（`DeltaActions` 是简单的 `-=`），所以从 euler-delta 恢复四元数 delta 要写一个**对称的** rotvec-delta；否则在 roll ≈ ±π 附近两边不等价。
  - 实际上，如果我们选择信任训练的 euler 空间 delta，就需要在控制器侧一样做 euler 相加：`euler_abs = euler_delta + euler_current`（然后 euler→quat）。这**和**现在 AbsoluteActions 的做法在代数上**一样**，唯一差别是"加哪一个 euler_current"—— 是 obs 时刻的还是 dispatch 时刻的。

### 路线 B：保留绝对 pose 派发，但在派发前**重新**基于 `robot.current_pose` 重构首个 waypoint

- 不改 policy 链路；dispatch_chunk 里：
  1. 读 `robot.current_pose_euler` 作为"真·现在 state"
  2. 从策略返回的 `absolute_chunk` 减去 policy 用过的 state（= env_obs 那份）得到"policy 意图的 delta 序列"
  3. 把 delta 序列重新基于 `robot.current_pose_euler` 加回去得到修正后的 `absolute_chunk'`
  4. 下发修正后的 `absolute_chunk'`
- 代码侧：需要在 `chunk_step` 把 env_obs 里的 state 一同传进 `dispatch_chunk`，或者在 env 里另存一份。
- 物理意义和路线 A 等价，只是改动局限在 env/dispatch 层，**不用碰 policy 包装器**。

### 路线 C（**推荐，最小变动**）：插 seed waypoint

- 保留所有现有链路。只在 `move_waypoints` 里最前面 **prepend** 一个额外 waypoint：`(robot.current_pose, robot.current_tcp_vel)`。
- Ruckig 拿到的就是"从真实状态出发 → 20 个策略 waypoint → 结束"共 21 个 waypoint。
- 消除 obs/dispatch 时间差引起的首段"瞬跳"，也让 chunk-to-chunk 抢占时 libfranka 看到的 commanded joint velocity 连续（reflex 触发的根因）。
- **不修正** policy 的 delta 方向/大小（这是 policy 的事），只保证轨迹起点连续。

路线 C 性价比最高，建议先上；如果跑起来 policy 的 delta 本身偏差过大（比如 reset pose OOD），再考虑路线 A/B 把"real-time current_pose"也作为 delta 的基准。

---

## 7. 术语表

- **delta mask**：一个布尔 16-d 向量，True 位置对应 "这一维走 delta 通道"。当前 = `[T,T,T,T,T,T,T,F,T,T,T,T,T,T,T,F]`。
- **训练期 delta 化**：`DeltaActions` 在 `state/actions` 都解 pack 后、Normalize 之前，对 True 位置执行 `actions -= state`。
- **推理期 absolutize**：`AbsoluteActions` 在 Unnormalize 之后，对 True 位置执行 `actions += state`（state 就是 policy 那一次 inference 看到的那一份）。
- **grip 保持绝对**：因为 gripper 触发值本身就是 [-1, 1] 的命令，不适合减 state（state 里存的是 gripper 开度 0-85 mm，量纲完全不同）。

---

## 8. 复现本次分析

真机 eval 的轨迹抓取工具（`_trajectory_debug.py` + `analyze_tcp_debug.py`）已
随 wrap-fix 一起清理掉。如需复跑这类诊断，可以用 `verify_sft_on_dataset.py`
在 dataset 上比对 policy 预测 vs GT action，或者自己临时加一段 JSONL hook
到 `dual_franka_joint_env._dispatch_chunk` 里记录 `pre_state` / `cmd_poses` /
`move_waypoints` 状态。关键观测量：
`start-jump = ‖cmd_first_waypoint_pose  ⊖  robot.current_tcp_pose_at_dispatch‖`
(policy 内部 delta + obs→dispatch 之间机器人实际漂移之和)。
