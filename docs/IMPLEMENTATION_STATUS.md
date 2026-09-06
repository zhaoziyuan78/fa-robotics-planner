# Implementation status

Updated: 2026-08-28

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

- [x] Active set reduced to GCRL, Trajectory Transformer, and DINO-WM.
- [x] All three consume the same exact-prefix paired dataset and record zero
  training environment interactions.
- [x] GCRL is offline GC-IQL with twin Q networks, expectile value regression,
  advantage-weighted actor regression, and future-goal relabeling.
- [x] TT is a causal return/goal-conditioned trajectory model with action,
  next-state, reward, and done heads.
- [x] Official DINO-WM commit `0a9492f`: frozen DINOv2 ViT-S/14,
  action-conditioned ViT predictor, latent objective, and official CEM migrated
  onto the unified modern wrapper. Full 100-step save/load/5-episode smokes pass
  on both Windy and FetchSlide-v4 with CUDA.
- [x] DINO-WM covers the complete offline dataset for five epochs instead of
  four total minibatches; Fetch goal RGB places the object at the desired goal
  and restores simulator state exactly after rendering.
- [x] DINO-WM's independent JSON subprocess records stdout/stderr/failures and
  reads paired RGB/state/action transitions instead of collecting online data.
- [ ] DINO-WM Humanoid evaluation remains unsupported because a reproducible H1
  goal-image protocol has not been defined; the worker fails explicitly.
- [ ] Full five-seed paper-budget training has not been run for every baseline.

No external baseline is substituted with a toy algorithm. The default baseline
configs now point to the verified official workers.

## Phase 6 — ablation and reporting

- [x] Exact Full/StateAdapterOnly/ActionAdapterOnly/PriorsOnly configuration
  matrix and five-seed manifest.
- [x] ID/OOD validation, bootstrap summaries, CSV/JSON/LaTeX, fairness helper,
  missing-run report, videos, and standard plot names.
- [x] Evaluator has tqdm progress, causal Action Prior KV cache, BF16/TF32,
  vectorized H1 scoring, reused proposal rollouts, shared real-history context,
  preallocated video K/V suffixes, terminal-video elimination, fused incremental
  QKV projection, and one-time per-frame condition projections. FetchPush uses
  one 256-candidate BF16 batch on the 32 GiB RTX 5000 Ada.
- [x] Prior/adapter training writes JSON/NPZ/PNG loss histories. Full adapter
  training writes state-prediction GIFs and action-comparison plots; main and
  baseline evals write GIFs that can be combined side by side.
- [x] Main and baseline evaluations persist native per-step rewards. The reward
  curve tool discovers methods by environment, prefers the canonical
  `metrics.jsonl` while retaining a legacy-filename fallback, averages episodes
  within each seed and seeds with equal weight, and exports PNG plus
  paper-auditable CSV.
- [x] Audited Windy paired expert plus soft-target Action Adapter reaches 18/20
  seed-0 ID success using a 10k-transition prefix.
- [ ] Paper-scale 5-seed/100-episode sweeps and videos have not been executed.

## Verified commands

- The latest full suite passes 77/77 tests.
- Windy, FetchSlide, FetchPush, and all four H1 smoke commands pass.
- Three Windy dataset types pass checksum/schema validation.
- Tiny Windy Action Prior, State Prior, both adapters, shooting evaluation, and
  aggregation pass end to end.
- RTX 5000 offline checks pass: GCRL Q loss 0.698 to 0.342 with a successful
  checkpoint rollout; TT total loss 0.308 to 0.055 with a successful rollout;
  DINO-WM loss 6.07 to 4.28 over the first/last 20 updates.
- Aggregate accepts scalar/bootstrap summaries, skips failed runs into the
  missing-run report, and regenerated six complete-run rows and all plots.
- CUDA forward/backward passes on RTX 5000 Ada with Torch 2.3.1+cu121;
  `torch.cuda.is_available()` is true on the allocated GPU node.
