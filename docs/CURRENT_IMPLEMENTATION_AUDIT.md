# Current implementation audit

Audit date: 2026-08-18  
Reference repository: `/home/ziyuan.zhao/fa-planner`  
Reference commit: `249831863fdc4db2fc2264cf7d8904b6bfd6d0ce` with local edits

## Scope and reproducibility baseline

The reference repository contains a runnable WindyNav experiment, checkpoints,
and saved evaluation arrays. It is not clean: tracked files are modified, one
training script is deleted, and the newer offline-baseline modules are
untracked. The migration therefore treats the files actually present on disk as
the source of truth and records deliberate behavior changes below.

The saved evaluations establish the requested regression baseline without
rerunning training:

| Evaluation | Saved success rate |
| --- | ---: |
| `eval/full_2/eval_full.npz` | 0.91 |
| `eval/full_2_ap_aad/eval_full.npz` | 0.90 |
| `eval/full_2_ap_only/eval_full.npz` | 0.94 |
| `eval/full_4/eval_full.npz` | 0.95 |
| `eval/full_sparse/eval_full.npz` | 0.91 |

This confirms that the existing checkpoints have produced at least 90% success
under several saved configurations. Exact training reproduction has not been
attempted because the original datasets and training budgets are large.

## Reference directory structure

- `fa_planner/envs/windynav.py`: simulator and success definition.
- `fa_planner/models/`: VQ-VAE, priors, adapters, and older baselines.
- `fa_planner/data/episodes.py`: eager NPZ episode readers.
- `fa_planner/utils/`: action quantization, sampling, seeding, and config loading.
- `scripts/generate_data.py`: `D_action`, `D_state`, and `D_align` generation.
- `scripts/train_{state,action}_prior.py`: prior training.
- `scripts/train_{adapter,action_adapter}.py`: adapter training.
- `scripts/eval_policy.py`: model construction, planning, evaluation, and result
  serialization in a single 990-line module.
- `models/`: existing checkpoints and loss curves.
- `eval/`: saved rollouts and success rates.

## WindyNav environment

`WindyNavEnv` has a four-dimensional state `[x, y, vx, vy]`, two-dimensional
bounded acceleration action, and two-dimensional goal. The default values are:

- `dt=0.1`, `T=100`, velocity decay `gamma=0.90`;
- `v_max=1.0`, `a_max=1.0`, `w_max=0.6`, `k_max=0.015`;
- boundary bounce coefficient `0.5`;
- success when Euclidean position-to-goal distance is at most `0.1`;
- 64x64 RGB world image and an ego crop.

The transition order is: decay and accelerate velocity, clip velocity, select
the wind band from current `y`, integrate position using `(velocity + wind)`,
then apply boundary bounce. `step` returns `(state, success, done, info)` rather
than the Gymnasium five-tuple. `success` is not itself terminal inside the
environment; callers stop on `done or success`.

The configuration and README describe linearly drifting wind, but `_wind_at`
currently ignores `region_k` and time. This is a clear inconsistency, but the
legacy Windy configuration keeps that behavior by default. Time-varying and OOD
wind are enabled only through explicit new configuration.

## Model interfaces and checkpoint shapes

### Visual tokenizer

The VQ-VAE maps RGB frames to an 8x8 grid of discrete codes from a vocabulary of
512. Its checkpoint has 1,267,139 parameters.

### State Prior

Interface: `StatePrior.forward(tokens) -> (logits, hidden)`, where tokens are
`int64[B,L,8,8]` or `int64[B,L,64]`. Default `L=16`, `d_model=512`, eight decoder
layers and eight attention heads. The prediction loss is cross entropy over the
64 tokens in the last frame. The prior receives no action. Its checkpoint has
25,784,832 parameters.

The incremental cache stores transformer K/V tensors, last logits, and a mean
hidden summary. Adapter training and planning append true or predicted frames to
this cache.

### Action Prior

Legacy interface: `ActionPrior.forward(input_tokens, direction, hidden=None)`.
Actions are quantized independently into 21 bins per dimension, giving 441
tokens plus a start token. The network is a two-layer GRU with width 256 and its
checkpoint has 1,017,273 parameters.

Important migration issue: despite being called an Action Prior, the legacy
model embeds the normalized direction from current position to goal at every
time step. Its training dataset reads state and goal to construct this input.
This violates the new action-only definition. New checkpoints use a strictly
action-only interface; legacy checkpoint loading remains a separately labelled
compatibility path and is never silently presented as the new Action Prior.

### State Adapter (`Adapter` in the reference code)

Legacy interface: `Adapter(state_summary, action) -> predicted_next_state`, with
shapes `[B,512]`, `[B,2]`, and `[B,4]`. It concatenates the State Prior hidden
summary with an action embedding and minimizes one-step MSE directly against
`s_{t+1}`. The frozen State Prior cache contains the current true frame. The
checkpoint has 3,419,140 parameters.

The implementation does not explicitly receive current physical state or a
decoded passive next-state prediction and does not explicitly output a
residual. The unified adapter adds this explicit residual contract. A legacy
mode retains the original direct-next-state loss for regression.

### Action Adapter

Legacy interface: `ActionAdapter(state, action_prior_dist) -> corrected_logits`,
with `[B,4]`, `[B,441]`, and `[B,441]`. It returns
`log(normalized_prior) + residual_scale * clipped_residual`; the final residual
layer is zero-initialized. The checkpoint has 6,160,825 parameters.

Training freezes the Action Prior and uses a Gaussian-smoothed expert token
cross entropy plus an annealed conservative KL from the adapted distribution to
the base distribution. Only transitions marked `action_source == expert` train
this module.

## Data flow

The legacy generator writes compressed per-episode NPZ files:

- `D_action`: zero wind, straight-line actions; nevertheless stores frames,
  states, goal, reward-related metadata, and wind alongside actions.
- `D_state`: zero task action under sampled wind, with state and frames.
- `D_align`: expert or goal-directed Gaussian action rollouts.

The tokenizer adds `world_tokens`/`ego_tokens`. Transition alignment is
`states[t]=s_t`, `actions[t]=a_t`, `next_states[t]=s_{t+1}`; `sim_state` is a
legacy alias for `next_states`. Dataset sampling is file based but NPZ arrays are
loaded per item. There is no manifest, integrity checker, episode-level split,
or schema enforcement. The new action-only writer physically omits state,
goal, reward, and task identifiers.

## Inference and planning

For the full method, each real observation is tokenized and appended to the
State Prior cache. The planner samples candidate first actions from either a
goal-directed Gaussian or the legacy goal-conditioned Action Prior, optionally
corrects their distribution with the Action Adapter, and predicts their next
states with the State Adapter. For horizon greater than one it renders predicted
physical states back to images, tokenizes those images, appends them to a copied
State Prior cache, and repeats. Only the first selected action is executed and
the next real frame replaces the predicted context.

The exact legacy planning score is the undiscounted final Euclidean distance
between predicted position `rollout_state[:2]` and `env.goal`; `argmin` selects
the action sequence. Intermediate distances, action cost, and model likelihood
are not used. The default configuration has horizon 1 and 32 samples, while the
saved high-success evaluations include horizon 2 and 4 variants.

## Frozen modules

Both adapter training scripts set the relevant prior to evaluation mode and set
every prior parameter's `requires_grad` to `False`. The unified implementation
retains this rule and additionally verifies optimizer membership and reports
trainable/frozen names and counts for every ablation.

## Known technical debt

- Environment, model loading, planning, evaluation, and serialization are
  tightly coupled in one evaluation script.
- Global NumPy seeding in `reset` mutates process-wide RNG state.
- Wind slope metadata is generated but ignored by dynamics.
- The legacy Action Prior leaks goal/state information.
- The state adapter name and mathematical behavior are ambiguous.
- Config is one monolithic YAML file and scripts duplicate construction logic.
- Action-only files contain forbidden observation and goal fields.
- No standard `terminated`/`truncated` distinction, schemas, manifests, OOD
  validation, confidence intervals, or unified run metadata.
- Baselines currently called goal-conditioned RL and trajectory transformer are
  offline IQL and a trajectory transformer, not the four requested baselines.
- There are no repository tests.

## Preservation and generalization plan

Preserve exactly under the legacy Windy profile: dynamics ordering, bounds,
rendering, state/action definitions, success radius, transition alignment,
token vocabulary, legacy adapter loss, and final-distance planner score.

Generalize behind explicit interfaces: observations, environment termination,
continuous/discrete action distributions, padded control state, task scorers,
random shooting/CEM, data schemas, adapter toggles, evaluation metrics, OOD
parameters, and run layout. Behavior changes—strict action-only priors, explicit
state residuals, multi-step loss, drifting wind, or alternate scorers—must be
selected in YAML and recorded in run metadata.

## 2026-08-22 engineering reconciliation

The continuous unified Action Adapter now preserves the stable parts of the
reference implementation without restoring its forbidden goal-conditioned
Action Prior or goal-directed Gaussian planner:

- zero-initialized, clipped, `0.5`-scaled distribution-parameter residual;
- Gaussian soft action targets and small label smoothing;
- annealed `KL(adapted || frozen action prior)` and cosine learning-rate decay;
- expert-only supervision computed after the complete causal action history;
- complete goal displacement rather than a unit direction that discarded
  stopping distance.

Because the unified Action Prior is strictly action-only, it needs more adapter
freedom than the reference goal-conditioned prior. The conservative KL was
therefore calibrated to `0.02 -> 0.001` rather than copied mechanically from
the reference `0.10 -> 0.02` schedule.

The old unified Windy paired-data controller was also found to succeed on only
about 36% of audited seeds while marking all of its trajectories as expert. A
state/goal-observable PD data oracle (`kp=10`, `kd=1`, no wind input) succeeds
on 98.6% of 500 real-environment seeds. A fresh 10k-transition-prefix adapter
trained from those demonstrations reached 18/20 (90%) ID success on seed 0.
This changes only offline expert quality; no expert state, dense reward, or
hand-coded controller is available at evaluation.

Fetch checkpoints from before bounded residuals must retain their original
unit residual scale, no clipping, normalized goal direction, and hard-MLE
training semantics. Restoring those checkpoint-local semantics recovers
FetchPush from 0/10 to 10/10 in the current compatibility audit (84/100 in the
stored paper-v2 run). FetchSlide benefits from discarding failed expert
episodes and all post-success actions, but remains difficult: the filtered
10k adapter reaches 3/20 at the audited H=1 setting. H=1 is both faster and
more reliable than longer imagined contact rollouts in independent 20- and
100-episode comparisons.

The former online GC-SAC/HER implementation has been retired. The active GCRL
baseline is goal-conditioned IQL trained only from paired shards; its future
goal relabeling never queries an environment. Trajectory Transformer and
DINO-WM use the same exact offline transition cap. DreamerV3 and TD-MPC2 are no
longer active baselines, so historical runs from those methods are excluded by
the plotting and aggregation code.

Training now writes loss histories as JSON, NPZ, and PNG. Full adapter runs
also write a state prediction GIF, an action target/prior/adapter plot, and
RMSE JSON. Main and baseline evaluation each write a rollout GIF and can
compose them into one side-by-side comparison.
