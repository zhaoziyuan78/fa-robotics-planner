# 论文实验运行命令

当前版本使用 VQ-VAE、离散视频自回归 State Prior、observation 自回归 State
Prior，以及同时修正 observation/video 的 State Adapter。旧 CNN-State-Prior 和旧
Adapter checkpoint 与当前架构不兼容，必须使用新的 `paper_v4` 目录重新训练。

所有大文件写入 `/l/users/ziyuan.zhao/fa-robotics-planner/`。以下命令在 GPU 节点的
`planner` conda 环境运行。

推理计算量按任务固定，不随 seed 改变：

| task | horizon | candidates |
|---|---:|---:|
| Windy | 2 | 256 |
| FetchSlide | 1 | 256 |
| FetchPush | 2 | 256 |
| HumanoidStand | 2 | 1 |
| Humanoid Balance/Reach/Push | 2 | 64 |

训练数据量也固定，不再做 paired-data sweep：Windy、FetchSlide、FetchPush、
Humanoid Balance/Reach/Push 均使用 10k paired transitions；HumanoidStand 使用 2k。
`paired_steps` 只把这个固定训练数据量写入实验元数据，不控制推理规划量。代码会在
evaluation 开始时重新写入上表中的固定规划值，因此正式实验不做任何 budget sweep。

## 1. 环境和目录

```bash
cd /home/ziyuan.zhao/fa-robotics-planner
conda activate planner
set -euo pipefail

export PROJECT_ROOT=/home/ziyuan.zhao/fa-robotics-planner
export STORAGE_ROOT=/l/users/ziyuan.zhao/fa-robotics-planner
export DATA_ROOT="$STORAGE_ROOT/data/paper_v4"
export CKPT_ROOT="$STORAGE_ROOT/checkpoints/paper_v4"
export RUN_ROOT="$PROJECT_ROOT/runs/paper_v4"
export RESULT_ROOT="$PROJECT_ROOT/results/paper_v4"
export MUJOCO_GL=egl
export XDG_CACHE_HOME=/tmp/fa-xdg-cache
export MPLCONFIGDIR=/tmp/fa-mpl-cache

mkdir -p "$DATA_ROOT" "$CKPT_ROOT/tokenizers" "$CKPT_ROOT/priors" \
  "$CKPT_ROOT/adapters" "$RUN_ROOT" "$RESULT_ROOT"

nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m pytest -q
```

## 2. 生成原始数据

Windy 会自动使用原 `fa-planner` 的三种数据分布：action-only 是无风、朝目标方向的
恒定动作；state-only 是随机起点和零动作；paired 是 50% 风补偿 PD expert 与 50%
goal-directed Gaussian。Fetch/Humanoid 的生成器保持不变。

```bash
generate() {
  local ENV="$1" DATASET="$2" STEPS="$3" HORIZON="$4" OUTPUT_ENV="$5"
  local EPISODES=$((STEPS / HORIZON))
  python -m scripts.generate_data \
    --output "$DATA_ROOT/$OUTPUT_ENV/$DATASET" \
    env="$ENV" dataset="$DATASET" seed=0 \
    episodes="$EPISODES" horizon="$HORIZON"
}

generate windy state_prior 100000 100 windy
# Action-only stops as soon as the goal is reached, so request extra episodes;
# training still enforces the declared 25k-transition budget with max_transitions.
python -m scripts.generate_data \
  --output "$DATA_ROOT/windy/action_prior" \
  env=windy dataset=action_prior seed=0 episodes=1000 horizon=100
generate windy paired 10000 100 windy

for ENV in fetch_slide fetch_push
do
  generate "$ENV" state_prior 50000 50 "$ENV"
  generate "$ENV" action_prior 50000 50 "$ENV"
  generate "$ENV" paired 10000 50 "$ENV"
done

generate humanoid_shared state_prior 50000 100 humanoid_shared
generate humanoid_shared action_prior 50000 100 humanoid_shared
generate humanoid_stand paired 2000 500 humanoid_stand
for ENV in humanoid_balance humanoid_reach humanoid_push
do
  generate "$ENV" paired 10000 500 "$ENV"
done

for DIR in \
  "$DATA_ROOT/windy/state_prior" "$DATA_ROOT/windy/action_prior" "$DATA_ROOT/windy/paired" \
  "$DATA_ROOT/fetch_slide/state_prior" "$DATA_ROOT/fetch_slide/action_prior" "$DATA_ROOT/fetch_slide/paired" \
  "$DATA_ROOT/fetch_push/state_prior" "$DATA_ROOT/fetch_push/action_prior" "$DATA_ROOT/fetch_push/paired" \
  "$DATA_ROOT/humanoid_shared/state_prior" "$DATA_ROOT/humanoid_shared/action_prior" \
  "$DATA_ROOT/humanoid_stand/paired" "$DATA_ROOT/humanoid_balance/paired" \
  "$DATA_ROOT/humanoid_reach/paired" "$DATA_ROOT/humanoid_push/paired"
do
  python -m scripts.check_dataset "$DIR"
done
```

## 3. 训练 VQ-VAE 并 tokenize

每个 State Prior 数据域训练一个 tokenizer。State Prior 和 paired 数据使用同一个冻结
tokenizer；token cache 独立存储，不修改原始 shard。

```bash
for SEED in 0 1 2 3 4
do
  for ENV in windy fetch_slide fetch_push humanoid_shared
  do
    python -m scripts.train_vqvae \
      --data "$DATA_ROOT/$ENV/state_prior" \
      --output "$CKPT_ROOT/tokenizers/${ENV}_vqvae_seed${SEED}.pt" \
      env="$ENV" seed="$SEED" epochs=30 batch_size=64

    python -m scripts.tokenize_frames \
      --data "$DATA_ROOT/$ENV/state_prior" \
      --output "$DATA_ROOT/$ENV/tokens/state_prior_seed${SEED}" \
      --vqvae "$CKPT_ROOT/tokenizers/${ENV}_vqvae_seed${SEED}.pt" \
      env="$ENV" seed="$SEED" batch_size=128
  done

  for ENV in windy fetch_slide fetch_push
  do
    python -m scripts.tokenize_frames \
      --data "$DATA_ROOT/$ENV/paired" \
      --output "$DATA_ROOT/$ENV/tokens/paired_seed${SEED}" \
      --vqvae "$CKPT_ROOT/tokenizers/${ENV}_vqvae_seed${SEED}.pt" \
      env="$ENV" seed="$SEED" batch_size=128
  done

  for ENV in humanoid_stand humanoid_balance humanoid_reach humanoid_push
  do
    python -m scripts.tokenize_frames \
      --data "$DATA_ROOT/$ENV/paired" \
      --output "$DATA_ROOT/$ENV/tokens/paired_seed${SEED}" \
      --vqvae "$CKPT_ROOT/tokenizers/humanoid_shared_vqvae_seed${SEED}.pt" \
      env="$ENV" seed="$SEED" batch_size=128
  done
done

find "$DATA_ROOT" -path '*/tokens/*' -name manifest.json -print
```

如果 tokenizer 的 reconstruction 或 codebook perplexity 明显异常，应先处理 tokenizer，
不要继续训练 State Prior。

## 4. 训练 State/Action Prior

State Prior 的一个命令内部依次执行 observation warmup、video warmup、joint training。
`epochs` 是 joint 阶段最大 epoch；两个 warmup 的默认值均为 5。Humanoid joint 阶段
保留 plateau scheduler 和 early stopping。

```bash
train_prior_pair() {
  local ENV="$1" STATE_STEPS="$2" ACTION_STEPS="$3" JOINT_EPOCHS="$4" SEED="$5"

  python -m scripts.train_prior --prior state \
    --data "$DATA_ROOT/$ENV/state_prior" \
    --tokens "$DATA_ROOT/$ENV/tokens/state_prior_seed${SEED}" \
    --vqvae "$CKPT_ROOT/tokenizers/${ENV}_vqvae_seed${SEED}.pt" \
    --output "$CKPT_ROOT/priors/${ENV}_state_prior_seed${SEED}.pt" \
    env="$ENV" seed="$SEED" epochs="$JOINT_EPOCHS" batch_size=4 \
    max_transitions="$STATE_STEPS"

  python -m scripts.train_prior --prior action \
    --data "$DATA_ROOT/$ENV/action_prior" \
    --output "$CKPT_ROOT/priors/${ENV}_action_prior_seed${SEED}.pt" \
    env="$ENV" seed="$SEED" epochs=20 batch_size=64 \
    max_transitions="$ACTION_STEPS"
}

for SEED in 0 1 2 3 4
do
  train_prior_pair windy 100000 25000 30 "$SEED"
  train_prior_pair fetch_slide 50000 50000 30 "$SEED"
  train_prior_pair fetch_push 50000 50000 30 "$SEED"
  train_prior_pair humanoid_shared 50000 50000 300 "$SEED"
done
```

State Prior checkpoint 内已打包冻结的 VQ-VAE，因此 Adapter/evaluation 不需要额外的
`--vqvae` 参数。

## 5. 使用固定 paired data 训练 Adapter 和 ablation

```bash
train_adapter() {
  local ENV="$1" SEED="$2" STATE_ON="$3" ACTION_ON="$4" VARIANT="$5"
  local PRIOR_ENV STATE_STEPS ACTION_STEPS PAIRED_STEPS EPOCHS
  case "$ENV" in
    windy) PRIOR_ENV=windy; STATE_STEPS=100000; ACTION_STEPS=25000; PAIRED_STEPS=10000; EPOCHS=30 ;;
    fetch_*) PRIOR_ENV="$ENV"; STATE_STEPS=50000; ACTION_STEPS=50000; PAIRED_STEPS=10000; EPOCHS=60 ;;
    humanoid_stand) PRIOR_ENV=humanoid_shared; STATE_STEPS=50000; ACTION_STEPS=50000; PAIRED_STEPS=2000; EPOCHS=40 ;;
    humanoid_*) PRIOR_ENV=humanoid_shared; STATE_STEPS=50000; ACTION_STEPS=50000; PAIRED_STEPS=10000; EPOCHS=40 ;;
  esac

  python -m scripts.train_adapters \
    --data "$DATA_ROOT/$ENV/paired" \
    --tokens "$DATA_ROOT/$ENV/tokens/paired_seed${SEED}" \
    --state-prior "$CKPT_ROOT/priors/${PRIOR_ENV}_state_prior_seed${SEED}.pt" \
    --action-prior "$CKPT_ROOT/priors/${PRIOR_ENV}_action_prior_seed${SEED}.pt" \
    --output "$CKPT_ROOT/adapters/${ENV}_${VARIANT}_seed${SEED}.pt" \
    env="$ENV" seed="$SEED" epochs="$EPOCHS" max_transitions="$PAIRED_STEPS" \
    paired_steps="$PAIRED_STEPS" state_only_steps="$STATE_STEPS" action_only_steps="$ACTION_STEPS" \
    model.state_adapter.enabled="$STATE_ON" model.action_adapter.enabled="$ACTION_ON"
}

for ENV in windy fetch_slide fetch_push humanoid_stand humanoid_balance humanoid_reach humanoid_push
do
  for SEED in 0 1 2 3 4
  do
    train_adapter "$ENV" "$SEED" true true Full
    train_adapter "$ENV" "$SEED" true false StateAdapterOnly
    train_adapter "$ENV" "$SEED" false true ActionAdapterOnly
    train_adapter "$ENV" "$SEED" false false PriorsOnly
  done
done
```

训练日志分别记录 observation correction loss、video-token correction CE 和 Action
Adapter loss。诊断目录还会报告 passive/adapted video-token accuracy。

## 6. State Adapter 开环诊断

该诊断从 held-out paired trajectory 的真实历史分叉，在完全相同的 action 下分别递推
纯 State Prior 和 State Prior + State Adapter。分叉之后，两条路径只使用各自预测的
observation 和 video token，不会把未来真实状态喂回模型。

```bash
eval_state_adapter_rollout() {
  local ENV="$1" SEED="$2" PRIOR_ENV
  case "$ENV" in
    windy) PRIOR_ENV=windy ;;
    fetch_*) PRIOR_ENV="$ENV" ;;
    humanoid_*) PRIOR_ENV=humanoid_shared ;;
  esac

  python -m scripts.eval_state_adapter_rollout \
    --data "$DATA_ROOT/$ENV/paired" \
    --tokens "$DATA_ROOT/$ENV/tokens/paired_seed${SEED}" \
    --state-prior "$CKPT_ROOT/priors/${PRIOR_ENV}_state_prior_seed${SEED}.pt" \
    --adapters "$CKPT_ROOT/adapters/${ENV}_Full_seed${SEED}.pt" \
    --output "$RUN_ROOT/state_adapter_rollout/${ENV}_seed${SEED}" \
    --split val --episodes 10 --history-lengths 0 1 4 8 \
    --max-rollout-steps 0 \
    env="$ENV" seed="$SEED"
}

for ENV in windy fetch_slide fetch_push humanoid_stand humanoid_balance humanoid_reach humanoid_push
do
  for SEED in 0 1 2 3 4
  do
    eval_state_adapter_rollout "$ENV" "$SEED"
  done
done
```

主要看 `summary.json` 中以下对照：

- `passive_state_rmse` 与 `adapted_state_rmse`；
- `state_rmse_improvement_fraction`，正值表示 Adapter 改善；
- `passive_video_token_accuracy` 与 `adapted_video_token_accuracy`；
- Windy 的 position/velocity error，Fetch/Humanoid goal task 的 achieved-goal error。

每个可视化样本都会预测到该 episode 结束，并生成：按预测 state 渲染的三栏主 GIF
（State Prior / + State Adapter / Simulator）、真实 future 与两条预测轨迹的主 PNG、
独立的 video-token GIF/PNG，以及包含完整预测数组的 NPZ。非零
`--max-rollout-steps` 仅用于临时 smoke test。若只想检查 expert action 区间，在命令中
加入 `--expert-only`。

Windy 的两个预测栏使用与 simulator 相同的 world renderer，agent 由预测 position 和
velocity 直接绘制。Fetch/Humanoid 的 `control_state` 不能唯一恢复 MuJoCo 私有的完整
qpos/qvel，因此两个预测栏使用明确的 task-space 移动轨迹画布；对应的视觉模型输出放在
独立的 `*_video_tokens.gif`，避免把 token reconstruction 冒充为 state rendering。

## 7. 固定推理预算的 ID/OOD evaluation

```bash
evaluate_main() {
  local ENV="$1" SEED="$2" STATE_ON="$3" ACTION_ON="$4" VARIANT="$5"
  local PRIOR_ENV STATE_STEPS ACTION_STEPS PAIRED_STEPS OOD
  case "$ENV" in
    windy) PRIOR_ENV=windy; STATE_STEPS=100000; ACTION_STEPS=25000; PAIRED_STEPS=10000; OOD="high_wind slope_reversal unseen_quadrant" ;;
    fetch_*) PRIOR_ENV="$ENV"; STATE_STEPS=50000; ACTION_STEPS=50000; PAIRED_STEPS=10000; OOD="heavy_low_friction light_high_friction" ;;
    humanoid_stand) PRIOR_ENV=humanoid_shared; STATE_STEPS=50000; ACTION_STEPS=50000; PAIRED_STEPS=2000; OOD="low_friction_humanoid high_damping_humanoid weak_actuator_humanoid" ;;
    humanoid_*) PRIOR_ENV=humanoid_shared; STATE_STEPS=50000; ACTION_STEPS=50000; PAIRED_STEPS=10000; OOD="low_friction_humanoid high_damping_humanoid weak_actuator_humanoid" ;;
  esac

  local COMMON=(
    --state-prior "$CKPT_ROOT/priors/${PRIOR_ENV}_state_prior_seed${SEED}.pt"
    --action-prior "$CKPT_ROOT/priors/${PRIOR_ENV}_action_prior_seed${SEED}.pt"
    --adapters "$CKPT_ROOT/adapters/${ENV}_${VARIANT}_seed${SEED}.pt"
    env="$ENV" seed="$SEED" paired_steps="$PAIRED_STEPS"
    state_only_steps="$STATE_STEPS" action_only_steps="$ACTION_STEPS"
    model.state_adapter.enabled="$STATE_ON" model.action_adapter.enabled="$ACTION_ON"
    eval.fixed_planner_budget=true eval.episodes=20 eval.bootstrap_samples=10000
    run_root="$RUN_ROOT"
  )

  python -m scripts.evaluate "${COMMON[@]}" eval=id \
    experiment_id="fa_${ENV}_${VARIANT}_seed${SEED}_id"
  for CONDITION in $OOD
  do
    python -m scripts.evaluate "${COMMON[@]}" eval=ood condition="$CONDITION" \
      experiment_id="fa_${ENV}_${VARIANT}_seed${SEED}_ood_${CONDITION}"
  done
}

for ENV in windy fetch_slide fetch_push humanoid_stand humanoid_balance humanoid_reach humanoid_push
do
  for SEED in 0 1 2 3 4
  do
    evaluate_main "$ENV" "$SEED" true true Full
    evaluate_main "$ENV" "$SEED" true false StateAdapterOnly
    evaluate_main "$ENV" "$SEED" false true ActionAdapterOnly
    evaluate_main "$ENV" "$SEED" false false PriorsOnly
  done
done
```

`videos/eval.gif` 只包含环境原始画面；reward、success、seed、OOD 参数写入 metrics 和
`summary.json`，不再以黑底白字覆盖 GIF。

FetchPush 在 32 GiB RTX 5000 Ada 上默认用单个 256-candidate BF16 batch；这不改变
H=2、候选总数或打分方法。显存较小的 GPU 只修改 microbatch，例如：

```bash
eval.candidate_batch_size=64
```

Evaluation 会跨 candidate microbatch 复用真实历史 context，跨 rollout horizon 续用
video K/V cache，复用每帧固定的 observation/adapter condition projection，并跳过不会被
后续状态或打分消费的 terminal video generation。FetchSlide 的 horizon 为 1，因此自动
使用单个 256-candidate state-only batch。

## 8. Baselines

baseline 只包括 GCRL、Trajectory Transformer（命令名 `tt`）和 DINO-WM，全部使用
同一 `planner` Conda 环境。三者严格从已经生成的 `paired` shard 离线训练；训练期间不
构造环境、不调用 `env.step`，summary 和 metadata 均记录
`training_environment_steps: 0`。环境只在 checkpoint 保存完毕后用于 evaluation。

三种方法使用完全相同的 `DATA_ROOT/$ENV/paired` 和 transition cap。这里不混入
state/action prior data：state-only 没有 action，action-only 没有 state，直接混合无法
形成对齐 transition。GCRL 在 goal task 上只用公开 achieved/desired goal 计算 dense
训练 reward。TT 读取第 3 节生成的
`$DATA_ROOT/$ENV/tokens/paired_seed$SEED`，并从 token manifest 自动找到完全一致的
VQ-VAE。它会在 Windy 第一次 success 处结束其训练 trajectory，Fetch 则保留固定
horizon；它忽略 `action_is_expert` 采集器标签，再计算原生 return-to-go。这样分别与
各自 eval 的终止规则一致，并避免 Windy 持续采集产生的重复 success reward。
DINO-WM 的 dynamics 只看 RGB、
control state 和 candidate action，desired goal 仅作为 CEM target，不进入动力学输入。
所有 evaluation 和 reward curve 仍只保存原生环境 reward。

本项目 TT 现在与论文式视觉 TT 协议一致：输入当前/历史 VQ 图像 token、goal、
return-to-go 及过去 action/reward/done；训练离散 action、下一帧 VQ token、reward 和
done 的交叉熵；推理时以预测 reward 和轨迹似然进行 beam search，只执行最佳 beam 的
第一步动作。它不读取 `control_state`，也不读取 expert future。对 Fetch/Humanoid 的
多维动作采用 factorized action logits，在各维 top bins 的 Cartesian product 上做
k-best 联合候选生成，避免逐维组合数量指数爆炸。旧的 continuous structured-state TT
checkpoint 与新架构不兼容，必须重训。

DINO-WM 默认用 3 帧连续窗口训练 3 层 predictor，并冻结可逆的 structured-state
projection，CEM 内对 action bound 做裁剪且复用 RGB encoding。Windy 使用模型预测
速度的积分位置做 goal score；Fetch 仍直接比较 predicted achieved goal。已有的
DINO-WM checkpoint 缺少这些结构设置，不能由新代码兼容加载，也需要重训。

```bash
run_baseline() {
  local BASELINE="$1" ENV="$2" SEED="$3" OFFLINE_TRANSITIONS
  case "$ENV" in
    humanoid_stand) OFFLINE_TRANSITIONS=2000 ;;
    *) OFFLINE_TRANSITIONS=10000 ;;
  esac
  python -m scripts.train_baseline \
    baseline="$BASELINE" env="$ENV" seed="$SEED" \
    offline_transitions="$OFFLINE_TRANSITIONS" baseline_eval_episodes=100 \
    data_root="$DATA_ROOT" checkpoint_root="$CKPT_ROOT" run_root="$RUN_ROOT" \
    experiment_id="baseline_${BASELINE}_${ENV}_seed${SEED}"
}

for BASELINE in gcrl tt
do
  for ENV in windy fetch_slide fetch_push humanoid_stand humanoid_balance humanoid_reach humanoid_push
  do
    for SEED in 0 1 2 3 4
    do
      run_baseline "$BASELINE" "$ENV" "$SEED"
    done
  done
done

for ENV in windy fetch_slide fetch_push
do
  for SEED in 0 1 2 3 4
  do
    run_baseline dino_wm "$ENV" "$SEED"
  done
done
```

DINO-WM 不运行 Humanoid，因为当前任务没有与 vector goal 等价且无信息泄漏的
goal-image 协议。

三种 baseline 的 evaluation 都会把每个 episode 的原生逐步 reward 写入对应 run 的
`metrics.jsonl`。同一个 `experiment_id` 重跑时覆盖旧 evaluation，避免 reward curve
重复统计。画图和汇总脚本只接纳当前 GCRL、TT、DINO-WM，已有的 DreamerV3、
TD-MPC2 和在线 GC-SAC 历史 run 会被明确跳过。

## 9. 绘制 reward curves

主方法、三个 adapter ablation 和三个 active baseline 会按 `env` 自动发现。同一训练
seed 内先平均 evaluation episodes，再对不同 seed 等权平均；阴影是 seed 间的 95%
置信区间。默认绘制累计原生环境 reward，并在 PNG 旁写出同名 CSV。默认包含
baseline；加入 `--exclude-baselines` 时只绘制主方法和 ablation，PNG 与 CSV 会使用
同一筛选结果。也可以显式写 `--include-baselines`。

```bash
for ENV in windy #fetch_slide fetch_push #humanoid_stand humanoid_balance humanoid_reach humanoid_push
do
  python -m scripts.plot_reward_curve \
    --env "$ENV" --runs "$RUN_ROOT" \
    --baseline-runs "$PROJECT_ROOT/runs" \
    --out "$RESULT_ROOT/reward_curve_${ENV}_id.png" 
done
```

瞬时 reward 或单个 OOD condition：

```bash
python -m scripts.plot_reward_curve \
  --env windy --runs "$RUN_ROOT" --baseline-runs "$PROJECT_ROOT/runs" \
  --metric reward \
  --out "$RESULT_ROOT/reward_curve_windy_instantaneous.png"

python -m scripts.plot_reward_curve \
  --env windy --runs "$RUN_ROOT" --baseline-runs "$PROJECT_ROOT/runs" \
  --eval ood --condition high_wind \
  --out "$RESULT_ROOT/reward_curve_windy_ood_high_wind.png"
```

旧版 evaluation 只保存 episode return，无法无损恢复逐 step 曲线；脚本会明确跳过并
提示重新运行对应 eval，而不会伪造中间 reward。

从 paired dataset 中按固定规则选择一个优质 rollout，并为两个 Fetch 和四个 Humanoid
任务各导出一个无文字遮挡的 GIF：

```bash
python -m scripts.visualize_dataset_rollouts \
  --data-root "$DATA_ROOT" \
  --output "$PROJECT_ROOT/results/viz"
```

每个任务依次查找 `paper_v4`、`paper_v2` 和未分版本的数据；可用
`--dataset humanoid_push=/path/to/paired` 覆盖单个任务。Humanoid 长轨迹会均匀抽帧，
但保留首尾并覆盖完整 rollout。确切数据源、episode、return 和帧数写入
`results/viz/selection.json`。该命令只读取数据中已有的 RGB，不启动模拟器或加载模型。

## 10. 汇总与完整性检查

```bash
python -m fa_robotics_planner.experiments.aggregate --runs "$RUN_ROOT" --output "$RESULT_ROOT"
python -m scripts.fairness_report \
  --runs "$RUN_ROOT" --output "$RESULT_ROOT/fairness_report.md"

python -c "from pathlib import Path; r=Path('$RUN_ROOT'); print('\n'.join(str(p) for p in sorted(r.iterdir()) if p.is_dir() and not (p/'summary.json').exists()))"
find "$RUN_ROOT" -name baseline_status.json -type f -print
find "$RUN_ROOT" -name summary.json -type f | wc -l
```

推荐先跑每个任务 seed 0 的完整链路，确认 tokenizer reconstruction、video-token
accuracy、State Adapter state/video 指标和 eval 显存，再扩展到其余四个 seed。
