# Windy migration notes

The original working tree at `/home/ziyuan.zhao/fa-planner` remains untouched.
This repository is a clean implementation informed by the audit rather than a
copy of its 990-line evaluation script.

## Preserved by the legacy Windy profile

- State `[x, y, vx, vy]`, two-dimensional acceleration, integration order,
  clipping, boundary bounce, image semantics, goal distance, and success radius.
- Transition alignment `s_t, a_t, s_{t+1}`.
- Action discretization remains available conceptually, while new robotics
  checkpoints use bounded continuous distributions.
- `planner=shooting`, `horizon=2` uses final predicted goal distance, matching
  the high-success saved legacy configuration.
- A direct-next-state adapter objective can be selected in configuration for a
  legacy regression, but the default new checkpoint uses an explicit residual.

## Deliberate, configured changes

1. The old Action Prior embedded goal direction. It cannot satisfy the strict
   action-only definition, so it is labelled legacy and is not silently loaded
   into the new `CausalActionPrior`.
2. The new State Adapter receives current state, passive prediction, action, and
   State Prior hidden state, then returns an explicit delta and next state.
3. Legacy wind slope metadata was ignored by `_wind_at`. `legacy_static_wind`
   defaults to true for compatibility; slope/reversal requires explicit config.
4. `success` is now a proper `terminated` condition and time limit is
   `truncated`. This distinction propagates to TD-MPC2/external requests.
5. Action-only data physically omits all non-action fields instead of relying on
   the training loop to ignore them.
6. The resource-aware paper profile replaces the original generic 10k--2M
   paired-data ladder with task-scaled ladders (Windy 1k--30k, Fetch
   1k--10k, Stand 1k--2k, and other H1 tasks 1k--10k). This is an explicit experimental-protocol
   change requested for the available compute budget, not a change to the
   environment, loss, model, or success definition. Training can cap an
   existing larger manifest with `max_transitions`, so no generated data needs
   to be deleted.
7. Visual State Prior latents are now L2-normalized at both encoder and
   prediction outputs. This fixes an unbounded moving-target feedback loop that
   made Fetch loss explode and, together with invalid-episode filtering, made
   H1 loss non-finite. State Prior checkpoints made before this change are
   rejected by adapter training/evaluation; retrain State Priors and all
   dependent adapters. Datasets and Action Prior checkpoints remain reusable.
8. The State Prior now predicts a residual from the current state and all
   State/Action Adapter gradients are clipped. One-step residual fitting is the
   stable default; the configurable multi-step loss remains available as an
   ablation. This fixes the Fetch epoch-8 explosion and H1 NaNs.
9. Windy paired labels use only observable position/velocity/goal information;
   hidden wind is no longer leaked into the expert action. Fetch uses a staged
   collision-aware controller and explicit `action_is_expert` labels. Legacy H1
   OU trajectories fail a conservative RMS test and are not silently treated
   as expert demonstrations.
10. Manipulation Action Adapters train with corrupted action histories to
    reduce closed-loop exposure bias. Relabelled on-policy data can additionally
    store executed `actions` separately from `expert_actions`; the MLE objective
    itself is unchanged.
11. Random shooting retains the world-model rollout generated during
    state-conditioned proposal sampling instead of evaluating it twice. It also
    includes a deterministic proposal-mean anchor. Evaluation uses a causal
    Action Prior KV cache, vectorized H1 scoring, TF32/BF16 where appropriate,
    and a step-level progress bar.
12. The unstable generic H1 PPO nominal is replaced by HumanoidBench's released
    19-actuator reaching controller plus the official fixed hand pose. Stand
    uses a task-independent fixed target offset. Reach/Push use residual scale
    1.0 because 0.25 clipped away the offline expert action; this is configured
    per environment.

Existing reference checkpoints are architecture-specific and continue to run
from `~/fa-planner`. New checkpoints include the resolved config and cannot be
confused with legacy files.
