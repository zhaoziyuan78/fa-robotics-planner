# Seed-0 engineering validation

Updated: 2026-08-20. These are development checks on unseen environment seeds,
not the final five-training-seed paper table.

| Environment | Paired data | Evaluation | Success | Notes |
| --- | ---: | --- | ---: | --- |
| Windy | 30k | H=5, K=256, std×0.1, 20 episodes | 70% | Direct proposal mean is 75%. Strict Action Prior is action-only; the legacy repository's 90%+ checkpoint conditioned its prior on goal direction and used roughly 5M prior/500k paired transitions. |
| FetchPush | 10k | H=2, K=256, std×0.1, 20 episodes | 90% | Robust action-history training fixes the phase-transition feedback loop. |
| FetchSlide | 10k | H=2, K=256, std×0.1, 20 episodes | 30% | Improved from zero, but remains below the 90% data controller. One 4,024-transition on-policy relabel iteration did not improve the planner result beyond 30%. |
| HumanoidStand | 2k | proposal mean, 5 episodes | 60% | All episodes survive 500 steps; mean return 404.82. |
| HumanoidStand | 2k | H=2, deterministic mean anchor, 3 episodes | 66.7% | All survive 500 steps; mean return 400.75. |
| HumanoidReach | 10k | proposal mean, 3 episodes | 0% | Mean return 1398.88. The released expert is workspace-limited and BC errors destabilize the 61-D residual policy. Not paper-ready. |

## Root causes fixed

- Fetch/H1 State Prior instability was an unbounded direct-next-state feedback
  problem. Residual prediction, normalized visual latents, one-step default
  training, and gradient clipping remove the epoch-8 explosion and H1 NaNs.
- FetchPush failed because teacher-forced action histories never exposed the
  adapter to its own phase-transition errors. Noise/dropout history augmentation
  raises the verified result to 18/20.
- Old H1 paired manifests contain OU exploration rather than expert actions.
  Explicit expert flags and the legacy RMS guard prevent these labels from
  training the Action Adapter.
- The generic 61-D PPO nominal falls quickly. HumanoidBench's released reaching
  controller provides a stable 19-actuator body policy and official hand pose;
  a fixed task-independent target offset keeps Stand upright.
- Random shooting previously performed duplicate world-model rollouts and did
  not include the proposal mean. It now reuses proposal rollouts, uses the
  Action Prior KV cache, and anchors search with a deterministic mean candidate.

## Remaining blockers

- FetchSlide needs a stronger closed-loop imitation/data-aggregation result;
  30% is an improvement but not a satisfactory final task score.
- HumanoidBalance has no stable task-specific nominal/expert: the released reach
  nominal and the old PPO fall in roughly 20--30 steps.
- HumanoidReach targets can be outside the released controller's effective
  workspace. HumanoidPush's current reach-based heuristic does not reliably move
  the object. New task controllers or demonstrations are required before more
  adapter training is scientifically useful.

Validated artifacts are stored under:

- `/l/users/ziyuan.zhao/fa-robotics-planner/checkpoints/validated_seed0/`
- `/l/users/ziyuan.zhao/fa-robotics-planner/data/validated_seed0/`

Exact full experiment commands and reduced budgets are in `command.md`.
