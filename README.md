# FunctionAlignmentWM robotics planner

This repository implements FunctionAlignmentWM on WindyNav, Gymnasium-Robotics
FetchSlide/FetchPush, and HumanoidBench H1 tasks. The method separates action
regularity, passive world dynamics, and task-specific intervention:

```text
action history ── Action Prior ── Action Adapter(state, goal) ── candidate action

video-token history ── Video Prior ──┐
                                     ├─ State Adapter(action) ─ next state + video tokens
observation history ─ Observation Prior ─┘
```

The current architecture is checkpoint version 2. Checkpoints produced by the
older CNN visual encoder and single multimodal State Prior are intentionally
rejected instead of being partially loaded.

## Main architecture

### Frame tokenizer

RGB frames are tokenized by a separately trained VQ-VAE:

```text
RGB → 3 stride-2 convolutions → codebook lookup → discrete token grid
```

The default codebook contains 512 vectors of dimension 128. A `64×64` Windy
frame becomes `8×8=64` tokens; a `96×96` Fetch/H1 frame becomes `12×12=144`
tokens. VQ-VAE training uses reconstruction, codebook, and commitment losses.
The trained tokenizer is frozen before State Prior training.

### Coupled State Prior

The action-free State Prior has two independent causal branches:

- a raster-order video Transformer operating only on discrete video tokens;
- an observation GRU/Transformer operating only on `control_state` and its
  validity mask.

There is no separate proprioception branch or proprio encoder in the State
Prior: it consumes only video tokens plus the unified environment observation
(`control_state`). Windy uses a small GRU for its four-dimensional observation;
Fetch and H1 use causal Transformers.
Two zero-initialized directional MLPs connect the branches:

- the latest observation hidden conditions the next-frame token distribution;
- the latest video-frame summary conditions the next observation prediction.

Training has three explicit phases: observation warmup, video warmup, then joint
cross-modal training. Observation values are normalized with masked train-split
statistics stored in the checkpoint.

### Joint State Adapter

The State Adapter receives the current observation, passive next observation,
candidate action, observation hidden, and video summary. It produces:

- an additive next-observation correction;
- a shared spatially conditioned residual over VQ codebook logits for every
  next-frame token.

Both output heads are zero initialized. A newly constructed adapter is therefore
exactly the frozen passive prior. During imagined rollout, the corrected state
and corrected discrete video frame are both appended to the candidate history.

The Action Prior and Action Adapter remain strictly action-only and
state/goal-conditioned respectively.

## Environments and data isolation

The data values accepted by `scripts.generate_data` are:

- `state_prior`: RGB, observation and masks under passive/zero task action;
- `action_prior`: actions and action bounds only; state, goal and reward fields
  are physically rejected by schema validation;
- `paired`: action-conditioned transitions used by both adapters.

Windy generation matches the original `~/fa-planner` behavior:

- action-only: zero wind and one constant maximum action toward the goal;
- state-only: random initial position, regional static wind, zero action;
- paired: an episode-level 50/50 mixture of wind-compensating PD expert and
  goal-directed Gaussian actions.

Fetch and Humanoid data generation is unchanged. Raw shards stay immutable;
VQ tokens are written to separate checksummed token-cache datasets.

## Installation

The tested environment is the existing `planner` Conda environment:

```bash
conda activate planner
python -m pip install -e '.[data,test,gcrl,external-baselines]'
```

Large artifacts default to:

- data: `/l/users/ziyuan.zhao/fa-robotics-planner/data`
- checkpoints: `/l/users/ziyuan.zhao/fa-robotics-planner/checkpoints`

Both roots are controlled by `configs/default.yaml` and can be overridden with
`data_root=...` and `checkpoint_root=...`.

HumanoidBench and official external baseline sources are not vendored. The
complete installation and paper commands are in [command.md](command.md).

## End-to-end training

The required order is:

1. generate raw state/action/paired data;
2. train one VQ-VAE per prior data domain;
3. tokenize state-prior and paired RGB frames with that frozen VQ-VAE;
4. train the three-stage coupled State Prior and the Action Prior;
5. train State/Action Adapters;
6. evaluate with the fixed task planning budget.

Minimal Windy examples follow. Paper-scale paths and all tasks are covered in
[command.md](command.md).

### Generate data

```bash
python -m scripts.generate_data env=windy dataset=state_prior episodes=100
python -m scripts.generate_data env=windy dataset=action_prior episodes=100
python -m scripts.generate_data env=windy dataset=paired episodes=100
```

### Train and apply the tokenizer

```bash
python -m scripts.train_vqvae \
  --data /l/users/ziyuan.zhao/fa-robotics-planner/data/windy/state_prior \
  --output /l/users/ziyuan.zhao/fa-robotics-planner/checkpoints/tokenizers/windy_vqvae_seed0.pt \
  env=windy seed=0 epochs=30 batch_size=64

python -m scripts.tokenize_frames \
  --data /l/users/ziyuan.zhao/fa-robotics-planner/data/windy/state_prior \
  --output /l/users/ziyuan.zhao/fa-robotics-planner/data/windy/tokens/state_prior_seed0 \
  --vqvae /l/users/ziyuan.zhao/fa-robotics-planner/checkpoints/tokenizers/windy_vqvae_seed0.pt \
  env=windy seed=0

python -m scripts.tokenize_frames \
  --data /l/users/ziyuan.zhao/fa-robotics-planner/data/windy/paired \
  --output /l/users/ziyuan.zhao/fa-robotics-planner/data/windy/tokens/paired_seed0 \
  --vqvae /l/users/ziyuan.zhao/fa-robotics-planner/checkpoints/tokenizers/windy_vqvae_seed0.pt \
  env=windy seed=0
```

### Train priors

```bash
python -m scripts.train_prior --prior state \
  --data /l/users/ziyuan.zhao/fa-robotics-planner/data/windy/state_prior \
  --tokens /l/users/ziyuan.zhao/fa-robotics-planner/data/windy/tokens/state_prior_seed0 \
  --vqvae /l/users/ziyuan.zhao/fa-robotics-planner/checkpoints/tokenizers/windy_vqvae_seed0.pt \
  --output /l/users/ziyuan.zhao/fa-robotics-planner/checkpoints/priors/windy_state_prior_seed0.pt \
  env=windy seed=0 epochs=30 batch_size=4

python -m scripts.train_prior --prior action \
  --data /l/users/ziyuan.zhao/fa-robotics-planner/data/windy/action_prior \
  --output /l/users/ziyuan.zhao/fa-robotics-planner/checkpoints/priors/windy_action_prior_seed0.pt \
  env=windy seed=0 epochs=20 batch_size=64
```

The State Prior checkpoint bundles the frozen VQ-VAE, so downstream commands do
not require a separate tokenizer checkpoint argument.

### Train adapters

```bash
python -m scripts.train_adapters \
  --data /l/users/ziyuan.zhao/fa-robotics-planner/data/windy/paired \
  --tokens /l/users/ziyuan.zhao/fa-robotics-planner/data/windy/tokens/paired_seed0 \
  --state-prior /l/users/ziyuan.zhao/fa-robotics-planner/checkpoints/priors/windy_state_prior_seed0.pt \
  --action-prior /l/users/ziyuan.zhao/fa-robotics-planner/checkpoints/priors/windy_action_prior_seed0.pt \
  --output /l/users/ziyuan.zhao/fa-robotics-planner/checkpoints/adapters/windy_Full_seed0.pt \
  env=windy seed=0 epochs=30 max_transitions=10000 paired_steps=10000
```

Adapter diagnostics report passive/adapted observation RMSE and passive/adapted
video-token accuracy.

### Evaluate State Adapter rollouts

The migrated open-loop diagnostic evaluates the frozen State Prior and the
trained State Adapter under identical held-out actions. After the selected real
history, both branches recursively consume only their own predicted observation
and video tokens:

```bash
python -m scripts.eval_state_adapter_rollout \
  --data /l/users/ziyuan.zhao/fa-robotics-planner/data/windy/paired \
  --tokens /l/users/ziyuan.zhao/fa-robotics-planner/data/windy/tokens/paired_seed0 \
  --state-prior /l/users/ziyuan.zhao/fa-robotics-planner/checkpoints/priors/windy_state_prior_seed0.pt \
  --adapters /l/users/ziyuan.zhao/fa-robotics-planner/checkpoints/adapters/windy_Full_seed0.pt \
  --output runs/state_adapter_rollout/windy_seed0 \
  --split val --episodes 10 --history-lengths 0 1 4 8 \
  --max-rollout-steps 0 env=windy seed=0
```

Zero rollout limit evaluates the complete remaining episode. The main PNG
directly overlays the real future, passive-prior rollout, and adapted rollout;
the main GIF renders predicted states rather than decoded VQ tokens. Separate
`*_video_tokens.gif`/PNG files retain the video-token diagnostic. Add
`--expert-only` to evaluate only contiguous transitions explicitly labelled as
expert actions.

Windy predictions use the physical world renderer. Fetch/Humanoid observations
do not uniquely reconstruct private MuJoCo qpos/qvel, so their model panels use
an explicit task-space trajectory canvas instead of a misleading pseudo-render.

## Evaluation

Both paired training data and planning compute are fixed; this project does not
run a data-budget or planner-budget sweep. Windy, Fetch, and Humanoid
Balance/Reach/Push use 10k paired transitions, while HumanoidStand uses 2k.
Planning uses the following task-specific constants:

| task | horizon | candidates |
|---|---:|---:|
| Windy | 2 | 256 |
| FetchSlide | 1 | 256 |
| FetchPush | 2 | 256 |
| HumanoidStand | 2 | 1 |
| Humanoid Balance/Reach/Push | 2 | 64 |

```bash
python -m scripts.evaluate \
  --state-prior /l/users/ziyuan.zhao/fa-robotics-planner/checkpoints/priors/windy_state_prior_seed0.pt \
  --action-prior /l/users/ziyuan.zhao/fa-robotics-planner/checkpoints/priors/windy_action_prior_seed0.pt \
  --adapters /l/users/ziyuan.zhao/fa-robotics-planner/checkpoints/adapters/windy_Full_seed0.pt \
  env=windy eval=id seed=0 eval.fixed_planner_budget=true
```

Evaluation uses:

- Action Prior incremental cache;
- one cached real-history State Prior context shared by all candidate batches;
- incremental projected video Transformer K/V cache across rollout horizons;
- preallocated candidate K/V suffixes without per-token history copies;
- fused Q/K/V projection and no redundant single-token attention masks;
- one-time observation/adapter conditioning projection per generated frame;
- no unused video-frame generation at the terminal planning horizon;
- a single 256-candidate BF16 batch for FetchPush on the 32 GiB RTX 5000 Ada;
- memory-bounded candidate microbatches and BF16 on H1;
- tqdm step/episode progress.

Changing `eval.candidate_batch_size` changes only peak memory, not the total
candidate count. FetchPush defaults to 256; reduce it to 128 or 64 only on a
smaller GPU. `eval.gif` contains only raw environment frames. Reward,
success, seed, timestep and OOD annotations are written to JSON/JSONL files and
never cover the video.

Per-step native rewards are stored in `metrics.jsonl` for every main-method and
baseline evaluation. Rerunning the same experiment replaces its evaluation
records instead of appending duplicate episodes; the plotter still reads the
legacy baseline-only `baseline_eval_metrics.jsonl` filename as a fallback.
Plot the ID reward curve for one environment with:

```bash
python -m scripts.plot_reward_curve \
  --env windy --runs runs/paper_v4 --baseline-runs runs \
  --out results/reward_curve_windy_id.png
```

The plot automatically discovers Function Alignment, the three adapter
ablations, and all available baselines. Episodes are averaged within each
training seed before seeds are averaged with equal weight; the shaded region is
a 95% confidence interval across seeds. The default curve is cumulative native
reward. Pass `--metric reward` for instantaneous reward, or use
`--eval ood --condition high_wind` for one OOD condition. A CSV containing the
plotted means and intervals is written beside the PNG. Runs produced before
per-step rewards were introduced must be evaluated again; an episode return
alone cannot reconstruct a step curve.
`--baseline-runs` may be repeated when legacy baseline jobs live outside the
main paper run root. Only baseline-family records are imported from these
additional roots, and duplicate method/seed pairs keep the newest complete run.
Baselines are included by default; pass `--exclude-baselines` to plot only the
main method and adapter ablations (or `--include-baselines` explicitly). The PNG
and its CSV always contain the same selected method families.

To export one deterministic high-quality paired-data rollout for every Fetch
and Humanoid task without loading an environment or checkpoint:

```bash
python -m scripts.visualize_dataset_rollouts --output results/viz
```

The script prefers `paper_v4`, then `paper_v2`, then the unversioned dataset for
each task. Long Humanoid episodes are sampled uniformly over their full time
extent, and `results/viz/selection.json` records the exact episode, source path,
return, and frame indices used. Use `--dataset TASK=/path/to/paired` to override
one source dataset.

## Baselines

The active comparison set is **GCRL, TT, and DINO-WM**. All three are trained
strictly offline from the same generated `paired` dataset. Training constructs
no environment and records `training_environment_steps: 0`; the environment is
created only after the checkpoint has been written, for evaluation.

- GCRL is offline goal-conditioned IQL with twin critics, an expectile value
  network, advantage-weighted actor regression, and future-goal relabeling.
- TT reads the precomputed VQ image-token trajectories. Its causal Transformer
  predicts discretized actions and, conditioned on each candidate action, the
  next VQ tokens, native reward, and termination flag. Evaluation encodes only
  the live RGB frame and uses model-predictive beam search; it never reads the
  environment's structured state or an expert future trajectory. For arbitrary
  action dimension (including 61-D Humanoid), a heap performs k-best search over
  the retained per-dimension bins without an exponential Cartesian-product
  enumeration. TT ignores `action_is_expert`,
  computes all statistics from training shards only, and on Windy crops its
  private trajectory view at first success. Fetch retains its recorded horizon.
- DINO-WM retains the audited official DINOv2 encoder/world-model code and CEM
  planner, but reads consecutive windows from the shared paired shards. The
  dynamics input is RGB plus `control_state`; desired goal is target-only and
  cannot shortcut action dynamics. Its structured projection is frozen
  full-rank, direct state prediction is supervised, and CEM candidates are
  bounded to actions the environment can actually execute. Windy scores the
  model's predicted velocity integral because its one-step position contains a
  hidden-wind displacement; Fetch keeps direct achieved-goal scoring.

```bash
python -m scripts.train_baseline \
  baseline=gcrl env=fetch_slide offline_transitions=10000 \
  data_root=/l/users/ziyuan.zhao/fa-robotics-planner/data/paper_v4

python -m scripts.train_baseline \
  baseline=tt env=windy offline_transitions=10000 \
  data_root=/l/users/ziyuan.zhao/fa-robotics-planner/data/paper_v4

python -m scripts.train_baseline \
  baseline=dino_wm env=windy offline_transitions=10000 \
  data_root=/l/users/ziyuan.zhao/fa-robotics-planner/data/paper_v4
```

DINO-WM is run on Windy and Fetch only. H1 tasks do not have a vector-goal to
goal-image protocol equivalent to the official DINO-WM objective.

All baseline launchers use the existing `planner` Conda environment; no JAX or
second baseline environment is required. `offline_transitions` is an exact
static-data cap. The legacy `environment_steps` override is accepted only as a
backward-compatible alias for this cap and never enables online collection.
State-only and action-only prior datasets are intentionally not mixed by
default because neither independently contains aligned transitions.

Windy exposes only a sparse 0/1 success reward. GCRL may use dense public
goal-distance reward during offline optimization, while evaluation and every
reward curve always report native environment rewards. TT uses native
return-to-go conditioning and its learned native-reward head to score beam
rollouts after Windy's first-success terminalization (Fetch keeps its recorded
horizon). The TT checkpoint bundles the frozen VQ encoder recorded by the token
manifest, so evaluation cannot accidentally use a different tokenizer.
DINO-WM does not train on reward; its CEM score uses only public goal
coordinates decoded from the predicted structured token.

## Tests

```bash
XDG_CACHE_HOME=/tmp/fa-xdg-cache \
MPLCONFIGDIR=/tmp/fa-mpl-cache \
python -m pytest -q
```

The test suite covers data isolation and manifests, VQ token shapes, both causal
prior branches, directional fusion, state/video adapter corrections, prior
freezing, cache equivalence, planner batching, raw rollout GIF generation,
environment wrappers, OOD controls, and baseline protocols.

## Run layout

```text
runs/<experiment_id>/
  config.yaml
  metadata.json
  metrics.jsonl
  summary.json
  videos/eval.gif
  plots/
```

`metadata.json` records state/action/paired training amounts, the fixed planning
budget, OOD parameters, parameter counts and evaluation optimizations.
