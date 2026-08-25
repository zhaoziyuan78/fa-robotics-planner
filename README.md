# FunctionAlignmentWM robotics planner

This repository turns the original WindyNav experiment in `~/fa-planner` into a
configuration-driven robotics experiment framework. The main method is named
`FunctionAlignmentWM`; its Action Prior is strictly action-only, its State Prior
is action-free, and both adapters are residual modules that can be independently
disabled for the required 2x2 ablation.

The original repository is not modified. Its saved Windy evaluations include
90–95% success configurations; interfaces, losses, planner behavior, checkpoint
shapes, and migration decisions are recorded in
[`docs/CURRENT_IMPLEMENTATION_AUDIT.md`](docs/CURRENT_IMPLEMENTATION_AUDIT.md).

## What is implemented

- Unified `ObservationBundle`, `StepResult`, and `UnifiedControlEnv` contracts.
- Legacy-compatible Windy dynamics plus explicitly configured drifting/OOD wind.
- Gymnasium-Robotics FetchSlide-v4 and FetchPush-v4 wrappers, including mass,
  friction, size, initial-velocity, and actuator OOD controls.
- HumanoidBench `h1hand` Stand, Balance Simple, Reach, and Push wrappers with a
  shared 151-dimensional proprioception schema, normalized 61-dimensional
  residual action, one frozen nominal controller, and physics OOD controls.
- Isolated sharded-NPZ datasets with checksummed manifests. Action-only shards
  physically reject state, goal, reward, and task fields.
- Strict causal Transformer Action/State Priors, latent RGB encoder, residual
  State/Action Adapters, frozen-prior checks, and parameter reports.
- Prior-sampling/random shooting and continuous CEM with terminal-aware batched
  rollout and receding-horizon execution.
- Deterministic Windy, Fetch, and H1 task scorers.
- ID/OOD evaluation, bootstrap confidence intervals, candidate ranking/contact
  metrics, run metadata, CSV/JSON/LaTeX aggregation, missing-run reporting, and
  standardized plots/videos.
- A real GC-SAC/GC-SAC+HER implementation through Stable-Baselines3.
- Official HumanoidBench DreamerV3 and TD-MPC2 workers, plus the official
  DINO-WM model/CEM migrated onto the modern unified environments. Every worker
  trains, saves, reloads, evaluates, and keeps independent logs.

The external algorithm sources and paper checkpoints are not vendored. Their
audited commits, smoke budgets, and remaining paper-scale work are explicit in
[`docs/IMPLEMENTATION_STATUS.md`](docs/IMPLEMENTATION_STATUS.md).

## Planner environment setup

The tested environment is the existing Conda environment `planner`. HumanoidBench
pins Gymnasium 0.29.1, MuJoCo 3.1.6, Torch 2.3.1, and torchvision 0.18.1. Fetch v4
comes from Gymnasium-Robotics 1.4.2. Its declared Gymnasium lower bound is newer,
but the runtime difference is confined to `MujocoRenderer(width, height)`; the
Fetch wrapper contains a narrow compatibility shim and both v4 tasks are tested.

```bash
conda activate planner

python -m pip install -e '.[data,test,gcrl,external-baselines]'
python -m pip install --no-deps \
  gymnasium==0.29.1 mujoco==3.1.6 \
  stable-baselines3==2.3.2 pettingzoo==1.24.3 \
  torchvision==0.18.1 gymnasium-robotics==1.4.2
```

HumanoidBench's packaging omits `dmc_deps`, `envs`, and XML assets from a normal
wheel. Install its full source at the audited commit:

```bash
git clone https://github.com/carlosferrazza/humanoid-bench.git \
  /home/ziyuan.zhao/.tmp/humanoid-bench-cb1189
git -C /home/ziyuan.zhao/.tmp/humanoid-bench-cb1189 \
  checkout cb1189039151c8aadaaa987b442da54383c87fab
python -m pip install --no-deps -e \
  /home/ziyuan.zhao/.tmp/humanoid-bench-cb1189
python -m pip install -e \
  /home/ziyuan.zhao/.tmp/humanoid-bench-cb1189/dreamerv3
python -m pip install -e \
  /home/ziyuan.zhao/.tmp/humanoid-bench-cb1189/tdmpc2

git clone https://github.com/gaoyuezhou/dino_wm.git \
  /home/ziyuan.zhao/.tmp/dino_wm-official
git -C /home/ziyuan.zhao/.tmp/dino_wm-official \
  checkout 0a9492fa12044b852ae9e001cc74604b79c8bb0c
```

The current machine has already been configured this way. Because
Gymnasium-Robotics' metadata does not describe this tested compatibility setup,
`pip check` reports its Gymnasium constraint even though both Fetch v4 smoke
tests pass.

## Storage paths

`configs/default.yaml` keeps large artifacts off the home filesystem by default:

- datasets: `/l/users/ziyuan.zhao/fa-robotics-planner/data`
- checkpoints: `/l/users/ziyuan.zhao/fa-robotics-planner/checkpoints`

Checkpoint types are separated into `priors/`, `adapters/`, `infrastructure/`,
and `baselines/`. Override `data_root=...` or `checkpoint_root=...` on any command
when a different location is needed. Explicit `data=...`, `output=...`, or
checkpoint-file arguments still take precedence.

## Environment smoke tests

Windy and Fetch need no additional arguments:

```bash
python -m scripts.smoke_env --env windy
python -m scripts.smoke_env --env fetch_slide
python -m scripts.smoke_env --env fetch_push
```

The repository contains a 512-step debug nominal controller for interface smoke
tests. It is not a paper-quality standing policy.

```bash
python -m scripts.smoke_env --env humanoid_stand
python -m scripts.smoke_env --env humanoid_balance
python -m scripts.smoke_env --env humanoid_reach
python -m scripts.smoke_env --env humanoid_push
```

Train the shared infrastructure controller with a research budget before paper
experiments:

```bash
python -m scripts.train_nominal_controller \
  --steps 1000000 \
  --seed 0 \
  --device cuda \
  nominal_controller_checkpoint=/l/users/ziyuan.zhao/fa-robotics-planner/checkpoints/infrastructure/h1hand_stand_ppo_paper.zip
```

The metadata beside the checkpoint records its independent training cost.

## Data generation

The three accepted dataset values are `state_prior`, `action_prior`, and
`paired`. Each episode is a lazy shard and every write updates `manifest.json`.

```bash
python -m scripts.generate_data \
  env=fetch_slide dataset=state_prior seed=0 episodes=1000

python -m scripts.generate_data \
  env=fetch_push dataset=paired seed=0 episodes=1000

python -m scripts.generate_data \
  env=windy dataset=paired seed=0 episodes=1000

python -m scripts.generate_data \
  env=humanoid_shared dataset=state_prior seed=0 episodes=1000 \
  nominal_controller_checkpoint=/l/users/ziyuan.zhao/fa-robotics-planner/checkpoints/infrastructure/h1hand_stand_ppo_paper.zip

python -m scripts.generate_data \
  env=humanoid_shared dataset=action_prior seed=0 episodes=1000 \
  nominal_controller_checkpoint=/l/users/ziyuan.zhao/fa-robotics-planner/checkpoints/infrastructure/h1hand_stand_ppo_paper.zip
```

Humanoid shared generation samples Stand/Balance/Reach/Push with probabilities
`0.25` each. State-only rollouts execute zero residual action while the nominal
controller remains active. Action-only shards contain no task identifier.

Validate any generated dataset:

```bash
python -m scripts.check_dataset \
  /l/users/ziyuan.zhao/fa-robotics-planner/data/fetch_slide/state_prior
```

## Training priors and adapters

```bash
python -m scripts.train_prior \
  prior=state env_group=humanoid_shared

python -m scripts.train_prior \
  prior=action env_group=humanoid_shared
```

Only task adapters are trainable for each H1 task:

```bash
python -m scripts.train_adapters \
  env=humanoid_push state_adapter=true action_adapter=true \
  state_prior_checkpoint=/l/users/ziyuan.zhao/fa-robotics-planner/checkpoints/priors/humanoid_shared_state_prior.pt \
  action_prior_checkpoint=/l/users/ziyuan.zhao/fa-robotics-planner/checkpoints/priors/humanoid_shared_action_prior.pt
```

Every adapter run prints trainable parameter names/count and frozen
names/count. Disabled adapters are removed from the optimizer.

## Main-method evaluation

```bash
python -m scripts.evaluate \
  method=prior_adapter env=fetch_slide eval=id planner=shooting

python -m scripts.evaluate \
  method=prior_adapter env=humanoid_push eval=ood condition=weak_actuator_humanoid \
  nominal_controller_checkpoint=/l/users/ziyuan.zhao/fa-robotics-planner/checkpoints/infrastructure/h1hand_stand_ppo_paper.zip \
  state_prior_checkpoint=/l/users/ziyuan.zhao/fa-robotics-planner/checkpoints/priors/humanoid_shared_state_prior.pt \
  action_prior_checkpoint=/l/users/ziyuan.zhao/fa-robotics-planner/checkpoints/priors/humanoid_shared_action_prior.pt
```

Select CEM with `planner=cem`. It is initialized from the corrected Action Prior
distribution; it does not replace either prior or adapter.

Validate every environment's configured OOD ranges and seed separation:

```bash
python -m scripts.validate_ood env=fetch_slide eval=ood
python -m scripts.validate_ood env=humanoid_push eval=ood
```

## Baselines

GC-SAC uses the same wrapper, bounds, reward, horizon, seeds, and interaction
budget. HER is enabled only on relabelable tasks.

```bash
python -m scripts.train_baseline \
  baseline=gcrl env=fetch_slide profile=debug environment_steps=1000
```

The three official workers are configured by default. These dependency smokes
use short budgets and are intended to verify training, checkpoint reload, and
five-episode evaluation—not to produce meaningful success rates:

```bash
python -m scripts.train_baseline \
  baseline=dreamerv3 env=windy profile=debug \
  environment_steps=100 experiment_id=smoke_dreamerv3

python -m scripts.train_baseline \
  baseline=tdmpc2 env=windy profile=debug \
  environment_steps=128 experiment_id=smoke_tdmpc2

python -m scripts.train_baseline \
  baseline=dino_wm env=windy profile=debug env.episode_horizon=10 \
  environment_steps=100 experiment_id=smoke_dino_wm_windy

python -m scripts.train_baseline \
  baseline=dino_wm env=fetch_slide profile=debug env.episode_horizon=10 \
  environment_steps=100 experiment_id=smoke_dino_wm_fetch_slide
```

Each subprocess receives `--request <json> --output <run_dir>` and writes a
checkpoint, resolved config, metrics, summary, stdout, and stderr. DreamerV3
runs its JAX debug model on CPU because this environment has CPU-only jaxlib;
TD-MPC2 and DINO-WM run on CUDA. DINO-WM uses integration strategy A: only its
official DINOv2/action-conditioned ViT/CEM code is imported, while its Gym 0.23
and `mujoco-py` environment stack is not. For an experiment that truly needs
the legacy stack, `scripts.run_dino_wm_isolated` launches an explicitly named
separate Conda environment.

## Ablation, aggregation, and plots

Generate the exact four adapter combinations for five seeds:

```bash
python -m scripts.run_adapter_ablation \
  --env windy --seeds 0,1,2,3,4
```

The command writes resolved YAML files and a manifest under
`runs/ablation_configs`; those resolved configs are the queue inputs for the
cluster launcher.

Aggregate all completed runs:

```bash
python -m experiments.aggregate --runs runs/ --output results/
python -m scripts.fairness_report \
  --runs runs/ --output results/fairness_report.md
```

This creates CSV, JSON, LaTeX, a missing-run report, and these standard figures:

- `success_vs_paired_data.png`
- `success_vs_planning_candidates.png`
- `id_vs_ood_success.png`
- `adapter_ablation_heatmap.png`
- `passive_rollout_error.png`
- `intervention_error.png`
- `candidate_ranking.png`

## Tests

```bash
XDG_CACHE_HOME=/tmp/fa-xdg-cache \
MPLCONFIGDIR=/tmp/fa-mpl-cache \
pytest -q
```

The suite includes real reset/step/render/OOD tests for Windy, FetchSlide,
FetchPush, and all four HumanoidBench tasks; prior information-isolation tests;
adapter freeze/bypass tests; variable-length masks; toy planner correctness;
dataset leakage/integrity checks; and external baseline protocol checks.

## Run layout

Every evaluation produces:

```text
runs/<experiment_id>/
  config.yaml
  metadata.json
  metrics.jsonl
  summary.json
  checkpoint/
  videos/
  plots/
  stdout.log
  stderr.log
```

`metadata.json` discloses state-only frames, action-only steps, paired/online
steps, planner budget, OOD parameters, and trainable parameter count so the
additional prior pretraining data is never hidden.
