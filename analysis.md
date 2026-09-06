# Baseline 历史审计与 Fetch paired data 质量报告

> **2026-09-01 更新：** 本文第一部分记录的是已经移除的旧 baseline 实现，仅用于
> 追溯，不再描述当前代码。当前 active baseline 只有 GCRL、TT、DINO-WM；DreamerV3
> 和 TD-MPC2 的配置、launcher 与结果发现入口均已移除。
>
> 三个当前 baseline 都精确读取同一 `paired` transition prefix，并记录
> `training_environment_steps=0`。GCRL 已替换为 offline GC-IQL；TT 已在 2026-09-01
> 改为 VQ 图像 token Trajectory Transformer：预测离散 action、下一帧 token、原生
> reward 和 done，推理使用模型式 beam search，完全不读取 `control_state` 或 expert
> future；DINO-WM 保留官方 DINOv2 predictor+CEM，但训练 pair 改为读取项目 paired
> shard。下文旧 TT 的 loss 数字属于已废弃的 structured-state checkpoint，不能与新
> TT 比较；Fetch paired data 质量报告仍然有效。

审计日期：2026-08-31  
项目：`/home/ziyuan.zhao/fa-robotics-planner`

## 0. 审计边界与结论摘要

本报告只做了两类登录节点安全操作：

1. 静态阅读当前项目、固定上游源码和 YAML 配置；
2. 用 CPU 顺序读取 `.npz`/`manifest.json`，检查 shape、校验和、有限值、时序连续性和数据分布。

没有加载任何 checkpoint，没有创建或推理任何模型，没有启动环境 rollout、训练或 GPU 作业。因此本文中的网络结构来自源码，参数量来自公式或已有运行产物；数据结论来自离线数组扫描，不代表视觉语义或策略性能的 GPU 实测。

当前实际实现了 4 个 baseline：

| Baseline | 方法性质 | 训练观测 | 训练方式 | 推理方式 |
|---|---|---|---|---|
| GCRL / GC-SAC+HER | 无模型、在线 RL | GoalEnv 字典状态 | SAC + HER | actor 单步确定性动作 |
| TD-MPC2 | 状态空间世界模型、在线 RL | `control_state + goal` | 世界模型、Q 与 policy 联合训练 | latent-space MPPI/CEM 规划 |
| DreamerV3 | recurrent latent 世界模型、在线 RL | `control_state + goal` | RSSM + imagined actor-critic | RSSM belief 上 actor 单步动作 |
| DINO-WM | 视觉世界模型、先离线采集再训练 | RGB DINO patch + 状态/目标 token | 冻结 DINO，训练 causal predictor | 世界模型内 CEM 规划 |

最重要的 baseline 审计结论：

- 四者并非严格同输入、同奖励、同训练范式。GCRL 使用 GoalEnv 字典；TD-MPC2/Dreamer 用扁平结构状态；DINO-WM 额外使用 RGB。Windy 训练还分别使用 dense goal 或 goal-progress shaping，而最终评估统一记录原生环境 reward。
- GCRL 的实现是标准 SB3 SAC+HER，网络较小；TD-MPC2 是约 120 万至 147 万可训练参数的 `model_size=1`；DINO-WM 约 2293 万总参数但只有约 87–89 万参数参与训练；Dreamer `small` 的已有 summary 记录了约 5639 万个 checkpoint state 元素，但该数字包含训练状态，不能当成纯神经网络可训练参数。
- 当前 GCRL/TD-MPC2/Dreamer 是在线交互训练；DINO-WM 会自行重新采集 offline transitions，不会读取本报告第二部分审计的项目 paired data。这一点对 sample-efficiency 横向比较必须明确。

Fetch 数据的最重要结论：

- 共找到 8 个持久化 raw paired dataset，全部通过文件、schema、有限值、时序连续性和基本 RGB 完整性检查。
- `paper_v2` 与 `paper_v4` 的逐数组内容完全相同，合并二者不会增加独立样本量。
- `validated_seed0` 除 `action_is_expert` 外与 `paper_v4` 相同；它的 expert 标签将被选中的 50 步整段标真，包含失败轨迹和首次成功后的动作，是过期语义。当前 `train_adapters.py` 会根据 reward 在读取时重新过滤，因而训练路径目前能防住这个问题，但原始标签本身不应作为真值。
- 推荐把 `data/paper_v4/{fetch_slide,fetch_push}/paired` 作为唯一 canonical 版本。它们各有 10,000 transitions，但 Action Adapter 真正有效的 expert supervision 只有 Slide 1,828 步、Push 2,810 步。
- Push expert 数据质量明显好于 Slide：Push 被选 expert 轨迹中 163/164 曾成功；Slide 只有 86/164 曾成功。Slide 当前更像“状态动力学数据充足，但成功动作监督偏少”。

---

# 第一部分：所有 baseline 的实现、架构与训练细节

## 1. 共用入口、环境适配和实验产物

### 1.1 调用链

统一入口是 `scripts/train_baseline.py`：

1. 合并 `default.yaml`、环境配置、baseline 配置和命令行覆盖；
2. 根据 `environment_steps` 确定训练交互/采集预算；
3. 创建 run 目录、checkpoint 目录和元数据；
4. GCRL 在当前进程运行；其余三个通过 `ExternalBaselineRunner` 启动子进程；
5. 训练结束后做固定 seed 的评估，写 `metrics.jsonl`、`summary.json`、checkpoint 和首个 episode 的 GIF。

外部 runner 会把完整请求写入 `request.json`，把 stdout/stderr 分开保存，并要求子进程成功返回且生成 `summary.json`。命令中的 `python` 会替换为当前 `sys.executable`，因此这些 baseline 当前设计为共用 `planner` Conda 环境；Dreamer 的 GPU launcher 只是先选择同环境内的 cuDNN 9 runtime，再导入 JAX，不是第二个 Conda 环境。

### 1.2 共用扁平观测与动作适配

TD-MPC2、DreamerV3 和 DINO-WM 使用 `FlatBaselineEnvAdapter`：

\[
x_t = [\text{control\_state}_t,\;g_t]
\]

其中不会再次附加 proprio 或 state mask，避免与 `control_state` 重复。扁平维度为：

| 环境 | control state | goal | flat input | action | horizon |
|---|---:|---:|---:|---:|---:|
| Windy | 4 | 2 | 6 | 2 | 100 |
| Fetch Slide/Push | 64 | 3 | 67 | 4 | 50 |
| Humanoid Stand/Balance | 512 | 0 | 512 | 61 | 500 |
| Humanoid Reach/Push | 512 | 3 | 515 | 61 | 500 |

外部 baseline 的动作统一表示为 `[-1,1]^A`，adapter 用仿射变换映射到环境原生 action bounds，step 前后均做 clip。

训练 reward 有三种模式：

- `native`：直接使用环境 reward；
- `dense_goal`：使用 `-||achieved_goal-desired_goal||`；
- `goal_progress`：使用 `scale*(d_{t-1}-d_t)+native_reward`。

评估始终保存环境原生 reward，因此训练曲线与评估曲线不能在 Windy 上直接按绝对值比较。所有 seed 顺序执行，不会把多个训练环境混成一个 vector env。

### 1.3 公平性注意事项

| 项目 | GCRL | TD-MPC2 | DreamerV3 | DINO-WM |
|---|---|---|---|---|
| 是否在线训练 | 是 | 是 | 是 | 否；先采完整离线集 |
| 是否使用 RGB | 否 | 否 | 否 | 是 |
| 是否显式给 goal | 是 | 是 | 是 | 是 |
| Windy 训练 reward | dense goal | dense goal | goal progress × 50 + native | native/offline transition reward不参与规划目标 |
| Fetch 训练 reward | sparse | native sparse | native sparse | 结构化 goal distance 规划目标 |
| 推理是否规划 | 否 | 是 | 否 | 是 |

所以“相同 environment steps”只等价于环境交互预算相同，不等价于相同梯度步数、相同模型输入或相同计算量。

## 2. GCRL：Goal-conditioned SAC + HER

### 2.1 环境接口

实现位于 `src/fa_robotics_planner/baselines/gcrl.py`，使用 Stable-Baselines3 2.3.2。`GoalEnvAdapter` 输出：

```text
observation = {
  observation:     control_state,
  achieved_goal:   achieved_goal,
  desired_goal:    desired_goal
}
```

SB3 `CombinedExtractor` 只把三个向量展平并拼接，没有可训练参数。因 `control_state` 已包含与任务有关的 achieved-goal 信息，字典中再放一次 `achieved_goal` 存在信息重复，但符合 HER 的标准接口。

拼接后的 SAC 输入维度为：Windy 8、Fetch 70、Humanoid Stand/Balance 514、Humanoid Reach/Push 518。GCRL 保留环境原生 action bounds，由 SB3 的 squashed Gaussian 内部缩放。

### 2.2 网络架构

没有显式覆盖 SB3 SAC policy，故采用默认 `MultiInputPolicy`：

- Actor trunk：`F -> 256 -> 256`，ReLU；
- Actor heads：两个 `256 -> A` 线性头，分别输出 `mu` 和 `log_std`；
- `log_std` clamp 到 `[-20,2]`，经 reparameterized diagonal Gaussian + `tanh` 得到动作；
- Critic：两个独立 Q 网络，每个为 `(F+A) -> 256 -> 256 -> 1`，ReLU；
- 另有两个相同结构的 target Q 网络；
- 自动温度调节额外学习一个标量 `log(alpha)`，target entropy 为 `-A`。

不含 target critic、包含 entropy 标量的优化参数量为

\[
N=768F+1026A+198659.
\]

| 环境 | F | A | 优化参数量 |
|---|---:|---:|---:|
| Windy | 8 | 2 | 206,855 |
| Fetch Slide/Push | 70 | 4 | 256,523 |
| Humanoid Stand/Balance | 514 | 61 | 655,997 |
| Humanoid Reach/Push | 518 | 61 | 659,069 |

target critics 是冻结副本，计入内存但不计入上表 optimizer 参数。

### 2.3 replay、HER 与 reward

配置中的 HER tasks 为 Windy、Fetch Slide/Push、Humanoid Reach/Push；Stand/Balance 不使用 HER。HER 使用 `HerReplayBuffer`，goal selection strategy 为 future，每个真实样本产生 4 个 relabeled goals。

- Fetch/Humanoid goal task：训练 reward 为 sparse goal reward，成功为 0，失败为 -1；
- Windy：专门改用 dense goal reward，否则远距离初始状态下长期全零原生 reward 几乎没有学习信号；
- 评估：两类情况都记录原生环境 reward。

`learning_starts=max(config_value, episode_horizon+1)`，保证首次采样前至少存在完整 episode。默认 buffer size 被设为 `total_steps+1`，避免短预算实验中环形 buffer 覆盖后暂时没有完整 episode，引发 SB3 的 “Unable to sample before the end of the first episode”。这也是此前中途报错修复的核心。

### 2.4 损失与更新

SB3 默认超参数：Adam `lr=3e-4`，`gamma=0.99`，target soft update `tau=0.005`，每个环境步一次 gradient update，batch 256。

1. Critic target：

   \[
   y=r+\gamma(\min_i Q_i^{target}(s',a')-\alpha\log\pi(a'|s')).
   \]

2. 两个 critic 分别最小化 `MSE(Q_i,y)`；
3. actor 最小化 `alpha*log_pi - min(Q1,Q2)`；
4. entropy temperature 最小化自动温度目标，使策略熵接近 `-action_dim`；
5. HER batch 中重标后的 reward 用新 desired goal 重新计算。

### 2.5 推理

每一步把 GoalEnv dict 输入 actor，使用 deterministic mean action，经 `tanh` 和 action-space 映射后直接执行；没有 rollout search、recurrent state 或 test-time adaptation。评估 episode seed 为训练 seed 加 1000 起的顺序 seed。

## 3. TD-MPC2

### 3.1 代码版本与总体结构

入口是 `scripts/run_tdmpc2_subprocess.py`，实际算法来自固定 HumanoidBench checkout：

```text
/home/ziyuan.zhao/.tmp/humanoid-bench-cb1189
commit cb1189039151c8aadaaa987b442da54383c87fab
```

当前选择 `model_size=1`、latent dim 128、MLP dim 384、SimNorm group dim 8。该实现不处理 RGB，学习的是扁平状态世界模型。

### 3.2 模型架构

设输入维度 `D`、动作维度 `A`、latent `z in R^128`：

1. State encoder：`D -> 256 -> 128`。第一层是 NormedLinear + LayerNorm + Mish；输出经 SimNorm，以每 8 维为组做 simplex normalization。
2. Dynamics：`[z,a] -> 384 -> 384 -> 128`，Mish，输出再经 SimNorm，预测下一 latent。
3. Reward head：`[z,a] -> 384 -> 384 -> 101`，预测 `[-10,10]` 上 101 个离散 support 的 two-hot 分布。
4. Policy：`z -> 384 -> 384 -> 2A`，输出 mean/log-std；动作经 `tanh`。log-std 范围映射到 `[-10,2]`。
5. Q ensemble：两个独立 Q，每个 `[z,a] -> 384 -> 384 -> 101`，同样输出 two-hot value 分布；另有一套 frozen target Q。

不含 target Q 副本的可训练参数公式为：

\[
N=1,194,671+256D+2,306A.
\]

| 环境 | D | A | 可训练参数量 |
|---|---:|---:|---:|
| Windy | 6 | 2 | 1,200,819 |
| Fetch Slide/Push | 67 | 4 | 1,221,047 |
| Humanoid Stand/Balance | 512 | 61 | 1,466,409 |
| Humanoid Reach/Push | 515 | 61 | 1,467,177 |

### 3.3 replay 与在线训练节奏

Replay buffer 以完整 episode 存储，训练时采样长度 `horizon+1` 的连续片段。第一段最多 `episode_horizon` 的交互用于 seed data；随后每收集一个环境 transition 做一次 update。Linux 路径中的 buffer device 实际硬编码为 CUDA，因此这个 upstream 实现并不是真正的纯 CPU TD-MPC2。

任务配置：

| 环境 | batch | train horizon | planning iterations | samples | elites | policy trajectories |
|---|---:|---:|---:|---:|---:|---:|
| Windy | 64 | 5 | 3 | 64 | 8 | 8 |
| 其他任务 | 8 | 2 | 2 | 16 | 4 | 0 |

### 3.4 损失

从真实 `s_t` 编码 `z_t`，递归应用 dynamics。时间步损失按 `rho^t` 衰减，`rho=0.5`：

- consistency：预测 latent 与 stop-gradient 下一状态 encoder latent 的 MSE，权重 20；
- reward：reward head 的 two-hot soft cross entropy，权重 0.1；
- value：两个 Q 对 TD target 的 two-hot soft cross entropy，权重 0.1；
- actor：`entropy_coef*log_pi - Q`，entropy coefficient `1e-4`。

优化器 Adam `lr=3e-4`，encoder 学习率再乘 0.3，gradient norm clip 20；target Q 的 Polyak 系数为 0.01。discount 由 episode length 启发式决定并夹到 `[0.95,0.995]`：50 步 Fetch 为 0.95，100 步 Windy 为 0.95，500 步 Humanoid 为 0.99。

### 3.5 推理规划

1. encoder 将当前状态编码为 `z_t`；
2. 建立 horizon 长的高斯动作序列分布，上一时刻的均值序列左移作为 warm start；
3. 采样候选动作序列；Windy 还加入当前 policy rollout 的 8 条轨迹；
4. dynamics 展开 latent，累计 reward prediction，并在末端加 target Q；
5. 选择 elites，以温度 0.5 的指数权重更新每一时刻 mean/std；
6. 重复配置的 planning iterations，执行第一个动作。

`eval_mode` 会关闭最终额外探索噪声，但 elite 序列选择仍包含采样随机性，因此不是完全解析确定性的 planner。

## 4. DreamerV3

### 4.1 版本、输入与模型规格

入口是 `scripts/run_dreamerv3_subprocess.py`，上游同样固定在 HumanoidBench commit `cb1189...`。配置顺序为 `humanoid_proprio` 再覆盖 `small`，所以这是向量版 Dreamer，而不是图像 CNN Dreamer。

当前主要结构：

1. Vector encoder：两层 512-unit MLP，输入先做 symlog；
2. RSSM deterministic state：512 维 GRU state；
3. RSSM stochastic state：32 个 categorical variables，每个 32 类，即展平后 1024 维 one-hot；
4. posterior：由 deterministic state 和 observation embedding 得到 stochastic logits；
5. prior：只由 deterministic state 得到 stochastic logits；
6. model feature：`[deter(512), stoch(1024)]`，合计 1536 维；
7. Vector decoder：两层 512-unit MLP，symlog MSE 重建观测；
8. Reward head：两层 512-unit MLP，255-bin symexp two-hot 分布；
9. Continue head：两层 512-unit MLP，binary continuation；
10. Actor：两层 512-unit MLP，输出连续 Normal policy，`minstd=0.1`、`maxstd=1.0`；
11. Critic：两层 512-unit MLP，255-bin symexp two-hot value，并维护 slow critic 副本。

已有运行 `summary.json` 中的 `parameter_count` 为 56,385,330。源码中的统计对象来自 `agent.save()` 返回的完整 JAX params/state tree；该 tree 还包含 optimizer/slow-network 等训练状态，所以该数值应称为“checkpoint state 元素数”，不能严谨地称作纯可训练网络参数量。本次登录节点审计没有实例化模型来重新拆分统计。

### 4.2 世界模型损失

训练序列先由 encoder 得到 observation embedding，RSSM 在时间维上更新 posterior/prior。World model 总损失包括：

- observation reconstruction，scale 1；
- reward prediction，scale 1；
- continuation prediction，scale 1；
- dynamics KL `KL(stopgrad(posterior)||prior)`，scale 0.5；
- representation KL `KL(posterior||stopgrad(prior))`，scale 0.1；
- KL free nats = 1。

decoder/reward/continue heads 的梯度均可回到 world model feature。World model Adam `1e-4`，gradient clip 1000。

### 4.3 imagined actor-critic

从 posterior states 出发，在 RSSM 中 imagination 15 步：actor 采动作，dynamics prior 推进 latent，reward/continue/value heads 计算 return。使用 lambda return：`lambda=0.95`，discount 约 `1-1/333=0.9970`；continuous actor 使用 dynamics backprop，entropy scale `3e-4`。Actor/Critic Adam 均为 `3e-5`，gradient clip 100；slow critic 更新 fraction 0.02。

训练吞吐设置：

| 环境 | train ratio | batch | sequence length | train fill | 约每环境步 update 数 |
|---|---:|---:|---:|---:|---:|
| Windy | 64 | 8 | 16 | 100 | 0.5 |
| Fetch/Humanoid | 4 | 4 | 8 | 8 | 0.125 |

这里的 update 数约为 `train_ratio/(batch*length)`；单次 update 分别消费 128 或 32 个序列位置。Replay 容量为 `max(1000, environment_steps+1)`。

### 4.4 推理

每个 episode 的 `is_first` 会清空 recurrent carry。每步 encoder 处理当前向量观测，RSSM posterior 融合旧 latent、旧动作和新观测，actor 在当前 feature 上输出动作。评估采用 actor 的 mode，不做 sample；没有 CEM。Windy 使用训练时 goal-progress shaping，Fetch/Humanoid 使用 native reward，评估均保存原生 reward。

## 5. DINO-WM

### 5.1 版本与数据来源

入口是 `scripts/run_dino_wm_subprocess.py`，固定上游：

```text
/home/ziyuan.zhao/.tmp/dino_wm-official
commit 0a9492fa12044b852ae9e001cc74604b79c8bb0c
```

它只支持 Windy 和 Fetch；代码明确拒绝 Humanoid。原因是当前迁移层假定单张 RGB + 小型结构状态、固定 DINO patch token 和短时域 CEM，而 Humanoid 的 512 维状态、61 维动作和 500 步控制不在实现的输入/规划约束内。

训练前，子进程会自行与环境交互 `environment_steps` 步，生成 `(obs_t, action_t, obs_{t+1})` 离线数据。探索动作是 uniform normalized action 的指数平滑，平滑系数 0.5。它不会消费 `data/.../paired`。

### 5.2 tokenization 与 predictor

1. RGB 从 64/96 resize 到 224，并做 ImageNet normalization；
2. 官方 VWM 内部根据 ViT-S/14 patch size 将有效输入调整为 196×196；
3. frozen DINOv2 ViT-S/14 输出 `14×14=196` 个视觉 patch tokens，每 token 384 维；
4. `control_state+goal` 经过 1×1 Conv1d/线性投影成 1 个 384 维 proprio token；
5. action 经同类投影成 1 个 384 维 action token；
6. 每个时间点共 198 tokens；
7. predictor 是 1 层 causal Transformer：model dim 384、4 个 attention heads、head dim 64、FF dim 512、learned position embedding `[1,198,384]`，带 dropout；没有单独视觉 decoder。

DINO backbone 参数冻结。Windy 的已有 summary 给出总参数 22,926,464、可训练参数 869,888。可训练部分公式为：

\[
N_{train}=866048+384(D+1)+384(A+1).
\]

因此 Fetch (`D=67,A=4`) 可训练参数为 894,080，总参数约 22,950,656。绝大部分参数来自冻结 DINO encoder。

### 5.3 训练目标

batch 由相邻帧 pair 构成。当前动作 token 放在当前帧 token 后，下一帧位置放零动作 placeholder。Predictor 根据 causal context 预测下一 observation tokens：

- 官方基础 loss：预测 token 与 target visual+proprio tokens 的 MSE，不对 action token 求 reconstruction；
- 项目附加 loss：下一 proprio token MSE，权重 10，用来避免静态背景主导视觉 token loss；
- frozen DINO target 不反传；predictor、proprio projection、action projection 使用 AdamW，`lr=5e-4`，默认 weight decay 0.01；
- batch 8，训练 5 个完整 epochs，即约 `5*ceil(N/8)` 次更新；旧的固定 4 updates 行为已取消。

### 5.4 CEM 推理

当前统一配置：horizon 5、32 candidates、top 8、initial variance scale 1.5、3 次优化，上一动作序列左移 warm start。

每个候选动作序列在世界模型中自回归预测未来 tokens。`structured_goal` 目标通过 proprio projection 的伪逆从预测 token 解码结构状态，只在 achieved-goal slice 上计算目标距离；`planning_visual_weight=0`，所以当前最终排名不使用视觉 token 距离。配置中的 `proprio_alpha=25` 只服务于另一种 latent 目标形式，在当前 structured objective 中没有作用。官方 CEM 的 normalized Gaussian sample 本身不 clamp，执行前映射回 native bounds 时才 clip。

### 5.5 当前工程层面的记录问题

- `train_baseline.py` 初始化 run metadata 时给所有 baseline 写 `trainable_parameters: 0`，外部 summary 才可能给出真实统计；只看 metadata 会误判模型大小。
- TD-MPC2 某些 summary 字段仍叫 `debug_protocol_deviations`，内容是历史 smoke-test 说明，容易被误读成当前 paper run 设置。
- Dreamer 的 `parameter_count` 实际是 checkpoint state tree 元素数，应更名或拆分为 trainable/model/optimizer 三项。
- DINO-WM 的 `structured_goal + planning_visual_weight=0` 意味着 RGB 仍参与 dynamics representation 训练，但当前 planner 打分最终主要依赖解码出的结构状态；论文表述不应写成纯视觉目标匹配。

---

# 第二部分：Fetch Push / Slide paired data 质量报告

## 6. 搜索范围与检查方法

递归搜索了以下位置中的 `manifest.json`：项目目录、`/l/users/ziyuan.zhao/fa-robotics-planner` 以及 `/tmp`。共找到 8 个 Fetch raw paired dataset；`/tmp` 中没有额外匹配项。

对每个 episode 做了以下检查：

- manifest 文件存在、episode ID 连续、无重复/漏列文件、SHA256 与 manifest 一致；
- 所有字段 shape/dtype 与 schema 一致；
- float 数组无 NaN/Inf，action 在 bounds 内；
- `next_control_state[t] == control_state[t+1]`、proprio 和 RGB 同理；
- goal 在同一 episode 内不漂移；
- state mask 一致，padding 区域为零；
- 最后一步 truncation/termination 语义；
- RGB 像素方差、空白帧和相邻静态帧；
- success、首次成功、初末 goal distance、最小距离、物体位移、动作均值/方差/饱和率；
- `action_is_expert` 是否只覆盖成功 expert 的首次成功前缀；
- 不同目录的逐字段内容哈希去重。

“episode 曾成功”使用存储 reward `>=0`；当前 Fetch sparse reward 中成功为 0、失败为 -1。距离为 achieved goal 到 desired goal 的欧氏距离。所有百分比若未注明，分母是 episode 或 transition 总数。

## 7. 数据清单与重复关系

| 目录（前缀均为 `/l/users/ziyuan.zhao/fa-robotics-planner/data`） | episodes | transitions | 约大小 | expert 标签 |
|---|---:|---:|---:|---|
| `fetch_slide/paired` | 1,000 | 50,000 | 272 MiB | 无 |
| `fetch_push/paired` | 1,000 | 50,000 | 308 MiB | 无 |
| `paper_v2/fetch_slide/paired` | 200 | 10,000 | 55 MiB | 正确前缀标签 |
| `paper_v2/fetch_push/paired` | 200 | 10,000 | 65 MiB | 正确前缀标签 |
| `paper_v4/fetch_slide/paired` | 200 | 10,000 | 56 MiB | 正确前缀标签 |
| `paper_v4/fetch_push/paired` | 200 | 10,000 | 65 MiB | 正确前缀标签 |
| `validated_seed0/fetch_slide/paired` | 200 | 10,000 | 55 MiB | 过期整段标签 |
| `validated_seed0/fetch_push/paired` | 200 | 10,000 | 65 MiB | 过期整段标签 |

去重结果：

- Slide `paper_v2` 与 `paper_v4` 的逐数组语义哈希同为 `09f68…`；
- Push `paper_v2` 与 `paper_v4` 的逐数组语义哈希同为 `28272…`；
- `validated_seed0` 的 RGB、状态、动作、reward 等内容与 `paper_v4` 完全相同，只有 `action_is_expert` 不同。

所以纸面目录数量看似有 6 份新数据，实际上每个 Fetch task 只有一份独立的 10k mixed expert/random 数据。不要把 `paper_v2 + paper_v4 + validated_seed0` 拼接成 30k 后声称是 30k 独立 transitions。

## 8. 所有数据共有的结构质量

8 个数据集共同结果：

- checksum mismatch：0；
- schema error：0；
- manifest 外多余 shard、manifest 内缺失 shard：0；
- episode ID 重复/不连续：0；
- NaN/Inf：0；
- action 越界：0，实际覆盖完整 `[-1,1]`；
- episode 长度全部为 50；
- 最后一 transition 全部 `truncated=True`，全部 `terminated=False`；
- state/proprio/RGB 相邻 transition 连续性错误：0；
- 同 episode goal 变化：0；
- state mask 只有一种模式：64 维中 28 维有效，剩余 36 个 padding 维严格为 0；
- 空白 RGB 帧：0；最小单帧像素标准差仍大于 51；完全静止的相邻 RGB 比例为 0。

结论：文件和 transition 对齐质量很好，可以安全用于 state dynamics 训练。RGB 检查只能证明帧存在、非空且随状态变化，不能证明相机视角、物体可见性或 token reconstruction 语义正确；后者需要可视化抽样或模型重建，本次因登录节点约束没有做。

## 9. 旧版 50k random/OU 数据

### 9.1 Fetch Slide

| 指标 | 结果 |
|---|---:|
| 曾成功 episodes | 3 / 1,000 = 0.30% |
| success transitions | 55 / 50,000 |
| 初始 goal distance | 0.445 |
| 最终 goal distance | 0.468 |
| 平均 progress（初始−最终） | -0.0224 |
| episode 内最小 distance | 0.440 |
| 物体单步位移均值 | 0.001081 |
| 物体发生可测位移的 steps | 99.53% |
| 相邻 action L2 变化均值 | 0.479 |

动作四维 mean 约 `[-0.0021,-0.0017,0.0033,0.0041]`，std 约 `[0.4355,0.4352,0.4410,0.4398]`，边界饱和率约 2.1%–2.3%。它是覆盖平衡的 OU/random dynamics 数据，但基本没有成功行为。

### 9.2 Fetch Push

| 指标 | 结果 |
|---|---:|
| 曾成功 episodes | 71 / 1,000 = 7.10% |
| success transitions | 3,238 / 50,000 |
| 首次成功 step 均值 | 1.56（最小为 0） |
| 初始 goal distance | 0.173 |
| 最终 goal distance | 0.181 |
| 平均 progress | -0.0085 |
| episode 内最小 distance | 0.172 |
| 物体单步位移均值 | 0.000289 |
| 物体发生可测位移的 steps | 9.26% |
| 相邻 action L2 变化均值 | 0.479 |

Push 与 Slide 的 action 统计逐维相同，说明两者来自相同 seed 的 OU action sequence，而不是任务特定 expert。Push 的不少“成功”发生在 episode 开头附近，不能解读为 random policy 学会了推物体。

这两份旧数据没有 `action_is_expert`。当前训练代码对缺失标签先默认全真，再用 `successful_expert_only` 和首次成功 reward 重新筛选；因此不会把所有 50k 步都当 expert，但最终可用的动作前缀极少且多为偶然/初始成功。建议：可作为 state dynamics 的广覆盖补充，不应作为主要 Action Adapter 行为克隆数据。

## 10. canonical `paper_v4` 数据

两项任务均由 episode 级 Bernoulli 选择 expert，配置目标 `expert_fraction=0.8`、`expert_noise=0.02`。实际 200 个 episodes 中 164 个被选为 expert、36 个为 OU/random。失败 expert 的轨迹保留用于 State Adapter，但其 `action_is_expert` 全假；成功 expert 只标首次成功及以前的前缀。

### 10.1 Fetch Slide

| 指标 | 全部 200 | 被选 expert 164 | random 36 |
|---|---:|---:|---:|
| 曾成功 episodes | 86 (43.0%) | 86 (52.44%) | 0 |
| 初始 distance | 0.460 | — | — |
| 最终 distance | 0.391 | 0.373 | 0.475 |
| 平均 progress | +0.0682 | +0.0902 | -0.0317 |
| episode 内最小 distance | 0.129 | 0.0606 | — |
| 最终一步仍成功 | 5 (2.5%) | 5 (3.05%) | 0 |

Action supervision：

- 86 个 episode 有真 expert 标签；
- 共 1,828 / 10,000 transitions 标为 expert，即 18.28%；
- 首次成功 step 均值 20.26，范围 4–41；
- 首次成功后的 label violation 为 0；
- 在全部 164 条被选 expert 中平均有 1.91 个 success steps；只看 86 条曾成功 expert 则为 3.64 步，说明 slide puck 多数只是短暂穿过目标区；
- expert 轨迹中 goal distance 逐步下降的 transition 比例只有 36.78%，符合“先接近/击球、随后滑行和 overshoot”的非单调动力学，但也意味着简单逐步 goal-direction 行为克隆会很难。

动作 mean 为 `[0.524,0.0198,-0.0538,-0.0018]`，std 为 `[0.557,0.516,0.454,0.199]`；四维边界饱和率约 `[25.11%,8.29%,6.87%,0.41%]`，相邻 action L2 变化均值 0.266。第一维强正偏和 25% 饱和与当前 Slide expert 的击球几何一致，但会使 Action Adapter 的数据分布很窄，OOD 初始位姿下更容易失效。

状态覆盖方面，物体发生可测位移的 steps 为 96.73%，物体单步位移均值 0.0141，明显比旧 OU 数据更有任务相关动态。

质量判断：结构质量优秀，state dynamics 价值高；expert action 质量中等。最大瓶颈不是总 transitions，而是只有 86 条成功前缀、1,828 个有效 action targets，且 47.6% 被选 expert 从未成功。

### 10.2 Fetch Push

| 指标 | 全部 200 | 被选 expert 164 | random 36 |
|---|---:|---:|---:|
| 曾成功 episodes | 163 (81.5%) | 163 (99.39%) | 0 |
| 初始即成功 episodes | 11 (5.5%) | 11 | 0 |
| 非初始成功 | 152 / 189 (80.42%) | 152 / 153 (99.35%) | 0 |
| 初始 distance | 0.185 | — | — |
| 最终 distance | 0.114 | 0.0972 | 0.189 |
| 平均 progress | +0.0712 | +0.0894 | -0.0117 |
| episode 内最小 distance | 0.0394 | 0.00925 | — |
| 最终一步仍成功 | 60 (30.0%) | 60 (36.59%) | 0 |

Action supervision：

- 163 个 episode 有真 expert 标签；
- 共 2,810 / 10,000 transitions，即 28.10%；
- 首次成功 step 均值 16.24，范围 0–33；
- 首次成功后的 label violation 为 0；
- 在全部 164 条被选 expert 中平均有 15.04 个 success steps；只看 163 条曾成功 expert 则为 15.13 步。

动作 mean 为 `[0.026,0.0064,0.020,-0.0018]`，std 为 `[0.520,0.527,0.510,0.199]`，边界饱和率 `[8.44%,9.53%,7.28%,0.41%]`，相邻 action L2 变化均值 0.486。前三个控制维覆盖较均衡，没有 Slide 那样明显的单向偏置。

物体发生可测位移的 steps 为 65.70%，单步位移均值 0.00914。Push expert 几乎全部能至少一次到达目标，且排除初始成功后仍为 99.35%，因此它是高质量的 goal-conditioned action supervision。最终成功率低于 ever-success，说明长达 50 步且不在成功时终止会出现离开目标的情况；当前首次成功截断标签正确地避免了学习这些后续动作。

## 11. `validated_seed0` 标签问题

这两份数据把被选中的 164 个 expert episodes 的全部 50 步都标为 expert：

- 每个 task 都有 8,200 / 10,000 transitions 标真；
- Slide 包含 78 条失败 expert 的整段动作，并在 86 条成功轨迹中继续标注首次成功后的动作；
- Push 包含 1 条失败 expert，并在 163 条成功轨迹中继续标注首次成功后的动作。

因此 raw `action_is_expert` 不是当前定义下的有效监督。好消息是当前 `train_adapters.py` 在训练时会读取 `successful_expert_only=true` 和 `expert_until_first_success=true`，根据 stored reward 重新把失败轨迹清零并截到首次成功，所以现有主训练路径最终恢复成与 `paper_v4` 相同的有效 mask。

风险在于其他分析脚本、未来 dataloader 或外部代码若直接信任该字段，会错误报告 82% expert 数据并训练到失败/overshoot action。建议归档或重新写 manifest/shard，避免继续把 `validated_seed0` 作为“更干净”的版本使用。

## 12. paired token cache 附加检查

`paper_v4` 下还各有一个 `tokens/paired_seed0` cache，均为 200 episodes / 10,000 transitions、约 5.3 MiB、codebook size 512、每帧 12×12 tokens。它们的 checksum、schema 和 current/next 时序连续性均通过。

| task | 实际使用 codes | codebook 使用率 | perplexity | top-1 code 占比 |
|---|---:|---:|---:|---:|
| Fetch Slide | 58 / 512 | 11.33% | 11.97 | 28.37% |
| Fetch Push | 129 / 512 | 25.20% | 5.83 | 60.16% |

Push 虽使用的 code 种类更多，但 60.16% token 集中在单一 code，有明显背景主导/低有效熵现象；Slide 也只使用 58 个 code。由于 Fetch 图像大部分是静态桌面背景，这不必然是 codebook collapse，但应在 GPU 节点进一步看 object/gripper crop reconstruction 和 code occupancy，而不能只看全图 MSE。本次没有加载 VQ-VAE。

## 13. 最终质量评级与建议

| 维度 | Fetch Slide paper_v4 | Fetch Push paper_v4 |
|---|---|---|
| 文件/schema/连续性 | 优秀 | 优秀 |
| RGB 基本完整性 | 通过 | 通过 |
| State Adapter 动力学数据 | 良好 | 良好 |
| Action Adapter 成功监督 | 中等，偏少 | 优秀 |
| expert 分布覆盖 | 有明显单向/饱和偏置 | 较均衡 |
| 标签语义 | 正确 | 正确 |

建议按优先级执行：

1. 将 `paper_v4` 设为唯一 canonical paired data；`paper_v2` 只保留作历史记录，不参与合并计数；停止使用 `validated_seed0` 的 raw expert 标签。
2. 论文中分别报告“全部 paired transitions”和“有效 expert action transitions”。当前 10k 数据不能等价描述成 10k action supervision：Slide 是 1,828，Push 是 2,810。
3. 若只追加一类数据，应优先改善 Slide expert controller/初始状态覆盖，而不是继续增加同分布 random transitions。目标是提高被选 expert 的 success rate，并减少第一 action 维的饱和偏置。
4. State Adapter 可以使用失败 expert 和 random 的全部 10k transition；Action Adapter 必须继续使用 successful-prefix mask。这两个模块的数据有效量必须分开理解。
5. 旧 50k OU 数据可作为 dynamics coverage ablation，但不要与 expert 数据等权训练 Action Adapter，也不要把其偶然/初始 success 当作 expert performance。
6. 在 GPU 节点再做少量、可视化优先的检查：随机抽取 RGB trajectory GIF、VQ reconstruction 的物体局部 crop，以及 Slide/Push 各 20 条 state-adapter open-loop rollout。这里不需要先跑全量模型 eval。

总体判断：当前 Fetch paired data 没有损坏、错位、NaN 或明显渲染失败；真正的问题是版本重复和 action supervision 的有效量/成功率差异。Push 数据已足以支持一个合理的 action adapter 单 seed 验证，Slide 的 state data 足够但 successful expert action data 偏少，是更可能限制最终成功率的部分。
