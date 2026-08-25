# Implementation status

Updated: 2026-08-22

## Phase 0 — audit and regression

- [x] Audited Windy environment, training scripts, priors, adapters, planner,
  data alignment, checkpoints, and saved evaluation results.
- [x] Confirmed saved 0.90–0.95 Windy success results.
- [x] Added legacy dynamics regression tests and migration notes.
- [ ] Full original 50k-episode retraining was not rerun.

## Phase 1 — unified framework

- [x] Unified observations/transitions/environment interface.
- [x] Strict prior and residual adapter interfaces.
- [x] Shooting and CEM planners.
- [x] Lazy dataset schemas, manifests, deterministic episode splits, integrity
  checker, evaluator, run layout, metrics, and visualization package.

## Phase 2 — Fetch

- [x] FetchSlide-v4 and FetchPush-v4 reset/step/render/action bounds.
- [x] Passive initial impulse, mass/friction randomization, and zero-action data.
- [x] ID/OOD mutation, scorers, data/training/evaluation CLI paths.
- [x] Real simulator smoke tests.
- [x] Seed-0 low-data validation: FetchPush reaches 10/10 in the current
  compatibility audit (the stored 100-episode result is 84%). FetchSlide's
  filtered 10k adapter reaches 3/20 at H=1; a prior development checkpoint
  reached 6/20, so Slide remains the harder open item.

## Phase 3/4 — Humanoid infrastructure and adapters

- [x] All four requested H1 IDs installed and smoke-tested.
- [x] Common 151-D proprioception, 512-D padded state, 61-D normalized residual
  action, shared-space assertion, and task goal extraction.
- [x] Shared nominal-controller training script and frozen residual wrapper.
- [x] Integrated HumanoidBench's released low-level reach controller and
  normalization assets as the stable default nominal backend.
- [x] Shared balanced prior dataset path and task-specific adapter path.
- [x] Physics OOD mutations and deterministic seed compatibility.
- [x] Residual H1 State Prior uses a 1e-3 initial learning rate, held-out
  ReduceLROnPlateau scheduling, gradient clipping, and validation early
  stopping with best-weight restoration. A real-data GPU smoke run remained
  finite and reduced validation loss across all five test epochs.
- [x] HumanoidStand survives 500/500 steps on all tested seeds; the 2k paired
  adapter reaches 3/5 direct and 2/3 mean-anchor success.
- [ ] Balance falls with both available nominal controllers. Reach is limited by
  the released controller's target workspace; learned Reach and heuristic Push
  are not paper-ready and must not be reported as solved.

## Phase 5 — baselines

- [x] GC-SAC/GC-SAC+HER implementation.
- [x] HER warm-up and replay-capacity guards verified across the 500-step
  Humanoid boundary; real and relabelled samples share one -1/0 sparse reward.
- [x] 128-step train/save/load/5-episode Windy smoke run.
- [x] DreamerV3 uses the configured official model size (small by default),
  selects the available JAX platform, and respects the requested eval count.
- [x] HumanoidBench-bundled TD-MPC2: official 1M model, replay/update/MPPI,
  128 Windy steps, 28 gradient updates, CUDA checkpoint reload, and five
  evaluation episodes.
- [x] Official DINO-WM commit `0a9492f`: frozen DINOv2 ViT-S/14,
  action-conditioned ViT predictor, latent objective, and official CEM migrated
  onto the unified modern wrapper. Full 100-step save/load/5-episode smokes pass
  on both Windy and FetchSlide-v4 with CUDA.
- [x] DINO-WM covers the complete offline dataset for five epochs instead of
  four total minibatches; Fetch goal RGB places the object at the desired goal
  and restores simulator state exactly after rendering.
- [x] Independent JSON subprocess protocol, stdout/stderr/failure recording,
  resolved configs, and a separate legacy-Conda launcher.
- [ ] DINO-WM Humanoid evaluation remains unsupported because a reproducible H1
  goal-image protocol has not been defined; the worker fails explicitly.
- [x] Current GCRL was run through the 500-step Humanoid HER boundary and for
  the full 10k Windy development budget. The latter performs 9,898 updates but
  still has 0/20 sparse successes, now reported honestly with final distance
  rather than hidden behind a zero-only return.
- [ ] Full five-seed paper-budget training has not been run for every baseline.

No external baseline is substituted with a toy algorithm. The default baseline
configs now point to the verified official workers.

## Phase 6 — ablation and reporting

- [x] Exact Full/StateAdapterOnly/ActionAdapterOnly/PriorsOnly configuration
  matrix and five-seed manifest.
- [x] ID/OOD validation, bootstrap summaries, CSV/JSON/LaTeX, fairness helper,
  missing-run report, videos, and standard plot names.
- [x] Evaluator has tqdm progress, causal Action Prior KV cache, BF16/TF32,
  vectorized H1 scoring, and reused proposal rollouts.
- [x] Prior/adapter training writes JSON/NPZ/PNG loss histories. Full adapter
  training writes state-prediction GIFs and action-comparison plots; main and
  baseline evals write GIFs that can be combined side by side.
- [x] Audited Windy paired expert plus soft-target Action Adapter reaches 18/20
  seed-0 ID success using a 10k-transition prefix.
- [ ] Paper-scale 5-seed/100-episode sweeps and videos have not been executed.

## Verified commands

- The latest full suite passes 59/59 tests.
- Windy, FetchSlide, FetchPush, and all four H1 smoke commands pass.
- Three Windy dataset types pass checksum/schema validation.
- Tiny Windy Action Prior, State Prior, both adapters, shooting evaluation, and
  aggregation pass end to end.
- GC-SAC+HER debug training, save, reload, and evaluation pass.
- DreamerV3 and TD-MPC2 train/save/reload/five-episode Windy smokes pass.
- DINO-WM train/save/reload/CEM/five-episode smokes pass on Windy and
  FetchSlide-v4.
- Aggregate accepts scalar/bootstrap summaries, skips failed runs into the
  missing-run report, and regenerated six complete-run rows and all plots.
- CUDA forward/backward passes on RTX 5000 Ada with Torch 2.3.1+cu121;
  `torch.cuda.is_available()` is true on the allocated GPU node.
