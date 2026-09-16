# Peg-Slot SAC 对手训练

入口：`bash examples/embodiment/run_peg_slot_sac.sh realagainst_peg_slot_adversary_sac`。
执行器固定使用未冻结 ResNet 训练得到的 SFT100k 检查点，本配置训练小型 SAC 对手。

## 下一阶段：训练执行策略

使用`realagainst_peg_slot_robot_sac`配置。固定对手加载已完成的dense SAC 50k
检查点，始终使用均值动作；执行器从未冻结双ResNet18的SFT100k权重初始化，
开始新的SAC阶段，critic、replay和更新计数从零开始。执行器扰动预算固定为对手
最终评估的2.30328170513957，`budget.minimum`和`budget.maximum`均设为该值；
每100步仍评估并录像，但不再调整预算。等值上下限也会在续训时覆盖检查点内的旧预算，
权重、优化器和replay照常恢复，旧预算下采集的replay由后续数据自然替换。
执行器配置`budget.zero_budget_eval: false`，预算降至下限后仍按该预算正常评估和训练，
不再额外进行零扰动评估或因此自动停止；设为`true`可恢复这项检查。

执行器奖励为首次成功+1、其余0。目标500,000步，前2,000步只更新critic。
新训练先按`algorithm.initial_collect_steps: 100`，让每个环境交互100个控制步，
期间不更新critic、actor或温度。复用25轮每轮4步的采集，在首次更新前取得至少
一回合的完整数据，共102,400条转移；成功回合会提前结束，环境照常自动reset。
随后恢复每轮采集4步、更新5次。2,000次critic预热另对应每环境1,600步交互。
预采仅在训练计数为0时执行，普通检查点续训跳过，预采步数不计入优化器步数。
预热是固定更新步数，不是实际成功回合计数门槛。
达到2,000步后，actor按8:1的更新周期开始训练。
GPU 0负责学习，GPU 0-1各采集512个环境，共1,024个；每轮每环境采集4步，
再从replay采样5个global batch 2,048（micro batch 2,048）进行5次critic更新，
每轮共10,240条抽样记录，允许重复采样，抽样条次与新增转移之比为2.5。
`algorithm.critic_actor_ratio: 8`控制预热结束后每8次critic更新才更新一次actor，
即每轮0–1次actor更新；critic UTD为5/4096，约0.00122。
这里UTD指critic更新次数除以新增转移数；抽样条次与新增转移之比等于UTD乘batch size。
执行器关闭`backup_entropy`，并将alpha固定为0，不使用Actor熵正则或自动温度调节。
RGB replay保留256个片段，共524,288条转移，对应每环境最近512个控制步，
确保初始100步数据不会在预采时被覆盖。有效RGB数据约294 GiB，运行时还需为
模型、Ray及临时缓存预留内存；保存检查点时也会保存当前replay。
`gamma: 0.98`，`sac_initial_log_std: -1.0`对应tanh前约0.368的初始标准差。
Critic学习率为`1e-3`，Actor学习率保持`1e-5`，均使用恒定学习率调度器。
全部命中缓存时直接gather到输出batch，减少一次图像复制，保持随机抽样顺序。
执行器的`learner`节点组将`OMP_NUM_THREADS`与`MKL_NUM_THREADS`设为8，
只作用于GPU 0上的学习进程，采集进程使用原配置。独立CPU采样基准中，
8线程优于4线程，16线程收益已饱和，32线程反而变慢；实际吞吐以训练日志为准。

评估使用2进程各48环境、3轮，共288回合，每100步评估并录像。
每1,000步保存检查点，权重每10步同步。所有这些步数均按critic更新次数计，
总步数上限保持500,000，critic预热单独设为2,000次更新。
TensorBoard和SwanLab并行记录。新实验日志目录以`YYYYMMDD-HHMMSS-`开头；
不能用对手检查点作为`runner.resume_dir`，该字段只用于同一训练角色续训。

## 默认设置

| 项目 | 设置 |
| --- | --- |
| GPU 分配 | GPU 0、1、2、3负责采集和推理，GPU 0同时负责SAC训练 |
| 训练环境 | 4 个采集进程，每卡256个环境，总计1,024个 |
| 动作采集 | 所有训练环境按 SAC 分布随机采样 |
| 初始随机性 | `actor.model.sac_initial_std: 0.1`，tanh 前的高斯标准差 |
| 一轮采集 | 每个环境走20个控制步，4段各5,120条，总计20,480条转移 |
| Replay buffer | 最近256段，满载1,310,720条转移；4段即可开始训练 |
| 训练批次 | global batch 和 micro batch 都是2,048，每轮更新50次 |
| 预热 | 从零训练时前100次更新仅训练critic；续训恢复原计数 |
| 权重同步 | 每50个训练更新步 |
| 评估与保存 | 每500个训练更新步；总目标50,000步 |
| 日志 | 本地TensorBoard与SwanLab云端同时记录 |
| 评估规模 | 4进程各24个环境、3轮，共288回合；使用均值动作 |

一轮产生20,480条新数据，训练从整个历史缓存随机抽取50批，共102,400条次。
四张GPU并行采集，完成后GPU 0进行参数更新。与单卡1,024环境方案相比，单段从
20,480条缩小为5,120条，因此缓存由64段恢复为256段，保持总容量一致。

两路图像供固定执行器推理；对手的replay保存19维特权状态、3维动作、奖励、
下一状态及结束标志等小向量。当前未启用`enable_preload`，因此`prefetch_size`
不会创建后台预取队列。

## 直接结果指标

TensorBoard 与 SwanLab 使用相同指标，评估间隔由各配置的
`runner.val_check_interval`控制：对手每500次、执行器每100次critic更新。
成功率是0–1之间的比例，统一查看`eval/robot_success_rate`与
`eval/adversary_success_rate`，兼容保留`eval/success_once`和`eval/fail_once`。
短采集窗口只返回已经结束的回合，成功回合可能先于100步超时回合结束，
因此不再上报`env/success_once`、`env/fail_once`、`env/adversary_timeout`和
`env/adversary_first_success`。其余`env/`回报、长度、预算和动作统计仍是
本轮已结束回合的诊断均值，`env/num_trajectories`是这些回合的数量。

| 指标 | 含义 |
| --- | --- |
| `eval/robot_success_rate` | 回合内机器人至少成功一次的比例，等于原有`eval/success_once` |
| `eval/adversary_success_rate` | 到100步机器人仍未成功的比例，来自`eval/adversary_timeout` |
| `eval/robot_success_count`、`eval/adversary_success_count` | 上述两类回合数 |
| `eval/num_trajectories` | 本次实际评估回合数，当前应为288 |
| `eval/return`、`eval/episode_len` | 当前训练角色的平均回报、平均回合长度 |
| `eval/slot_translation_mm`、`eval/slot_rotation_deg` | 每回合累计平移路程（毫米）、累计转角（度），均取回合均值 |
| `eval/adversary_budget_used`、`eval/budget_utilization` | 平均预算消耗、占本次评估预算的比例 |
| `eval/adversary_budget_exhausted` | 预算耗尽的回合比例 |
| `budget/evaluated`、`budget/next` | 本次评估使用的预算、评估后设定的下一阶段预算 |

累计路程和转角包含往返运动，不是终点相对起点的位移或偏航角。
原始`eval/`指标继续保留；旧版本已经把`eval/success_once`上传云端，只是命名不直观。
成功率比较应同时查看`budget/evaluated`；预算调整后，训练中已有回合仍使用各自
开始时的预算，新回合才使用`budget/next`。

## 对手的过程奖励

当前对手配置使用`adversary_reward_mode: dense`。每步结束后机器人尚未成功时，
发放`0.5 * 2^((t-1)/99) / sum(i=1..100, 2^((i-1)/99))`，时间从1开始。
100步过程奖励合计0.5，最后一步为第一步的2倍；100步仍未成功再给0.5，
因此完整超时回合的未折扣回报为1。机器人首次成功的当步奖励为0，回合结束，
此前过程奖励保留；终止后继续进行的批量评估步不再发奖。

三个参数位于`env.train.init_params`，评估通过YAML锚点共享：
`adversary_survival_reward_total: 0.5`、`adversary_survival_reward_ratio: 2.0`、
`adversary_timeout_bonus: 0.5`。单步幅度恢复为每轴2 mm、偏航1度；累计预算、
20 Hz控制频率和空间边界保持原设置。对手输入已包含回合进度。

`env/adversary_survival_reward_sum`与`eval/adversary_survival_reward_sum`记录
完成回合的累计过程奖励均值；`adversary_timeout_bonus`记录终局奖励，`return`
记录总回报。预算仍根据机器人成功率调整，不使用dense reward调整。
这个奖励同时鼓励拖延和最终阻止成功，训练回报不能直接当作对手成功率。

原来的`terminal`模式仍保留。切换奖励后从新对手、空replay开始独立实验，
避免继续使用旧奖励的缓存；执行器仍使用同一个未冻结ResNet的SFT100k模型。

## 初始动作噪声

`sac_initial_std`只控制初始化，标准差头仍参与训练。它与熵系数
`algorithm.entropy_tuning.initial_alpha`是两个不同参数。
当前`initial_alpha: 0.001`，继续使用自动熵调节。若要在已有训练中降低alpha，
需在从检查点恢复时额外传入`+algorithm.entropy_tuning.reset_on_resume=true`，
使alpha和其优化器状态重新初始化；策略、critic、replay和更新计数正常恢复。
仅修改`initial_alpha`而不设置这个覆盖项，续训仍会加载检查点内的alpha。
该覆盖项只用于主动调整alpha的这次续训，普通故障恢复不应重复重置。

0.1低于此前显式设置的0.5；此前对原始随机初始化在保存状态上的检查，三个
动作轴的平均标准差约为0.30–0.31。低噪声是否改善成功率，需要实际学习实验验证。

为检验初始化噪声，需要新开对手训练。加载`runner.resume_dir`或
`runner.ckpt_path`会恢复检查点中的标准差头，覆盖初始化值。执行器仍加载既有
SFT检查点。

## 可选：均值动作与随机探索混合采集

默认关闭：

```yaml
rollout:
  peg_slot_adversary:
    deterministic_train_fraction: 0.0
```

需要做消融或诊断时，可在新实验的启动命令后增加：

```bash
rollout.peg_slot_adversary.deterministic_train_fraction=0.5
```

0.5表示每个采集进程中，一半环境每步使用当前对手网络的均值动作，另一半
正常随机采样。两组共用同一套网络，环境分组保持固定，每一步仍重新预测动作。
评估始终全部使用均值动作，不受该参数影响。

这个选项曾帮助采集到持续干扰导致的终局失败样本，保留用于对照实验。
混合与非混合实验应分别保存配置和replay；启用选项时优先新开实验，以清楚
区分训练分布。

## 启动与短跑验证

首次使用SwanLab云端记录前，需在当前环境执行`swanlab login`完成登录。
日志后端在任务启动时初始化，修改配置不会热更新到已经运行的任务。
在仓库根目录激活环境，使用新的结果目录：

```bash
source .venv/bin/activate
bash examples/embodiment/run_peg_slot_sac.sh realagainst_peg_slot_adversary_sac \
  runner.logger.log_path=logs/peg_slot_sac_std01_env1024_run1
```

硬件与吞吐短跑可附加：

```bash
runner.max_steps=600 runner.max_epochs=600
```

短跑用于确认显存、吞吐、数据写入和实际更新次数；它不足以判断5k时的成功率。
降低初始噪声的效果应在相同评估设置下比较，并记录是否从头训练。

当前评估为4进程×24环境×3轮。此前单卡方案为1进程×32环境×9轮，更早的
三采集卡方案为3进程×32环境×3轮，均为288回合，但种子分组不同。比较检查点
时应统一评估拓扑、种子和预算，不能把不同分组的逐回合结果视为同一组样本。

## 四卡共置的初步验证（2026-09-12）

在4张A800 80GB上，从单卡试验的500步检查点继续，600–950步的8轮非评估
采集平均69.27秒，产生20,480条转移，即295.67条/秒；每轮50次参数更新平均
2.15秒。单卡1,024环境的参考值约169.80秒/轮、120.61条/秒，四卡方案约快2.45倍。

初期GPU 0显存峰值约18.24GiB，其他卡约17.1–17.2GiB。恢复后的更新计数与
500步检查点连续，SwanLab云端查询已确认收到950步的训练和计时指标。

结果位于
`logs/adversary_sac_4collectors_env256_actor0_std01_resume500_20260912_MlDtz8/throughput_validation.json`。
这是配置组合的短期吞吐对比，不是严格隔离所有变量的硬件消融。从单卡检查点
恢复时会暂时保留较大的历史片段，缓存的实际有效样本量和内部预分配内存可能
高于全部为256环境小片段时的稳态值。
