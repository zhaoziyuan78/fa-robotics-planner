from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from fa_robotics_planner.envs import make_env
from fa_robotics_planner.evaluation import evaluate_policy, write_episode_metrics
from fa_robotics_planner.experiments.run import RunDirectory
from fa_robotics_planner.models.builders import build_method
from fa_robotics_planner.models.distributions import TanhNormal
from fa_robotics_planner.planning import CEMPlanner, RandomShootingPlanner
from fa_robotics_planner.planning.scorers import (
    fetch_score,
    final_goal_distance,
    humanoid_score_torch,
)
from fa_robotics_planner.utils import seed_everything

from ._common import checkpoint_path, config_from_unknown
from .train_adapters import _load_priors


FIXED_TASK_PLANNER_BUDGETS = {
    "windy": (2, 256),
    "fetch_slide": (1, 256),
    "fetch_push": (2, 256),
    "humanoid_stand": (2, 1),
    "humanoid_balance": (2, 64),
    "humanoid_reach": (2, 64),
    "humanoid_push": (2, 64),
}


def apply_fixed_task_planner_budget(config) -> None:
    """Make inference compute independent of training-data budget/CLI sweeps."""

    if not bool(config.get("eval", {}).get("fixed_planner_budget", True)):
        return
    fixed = FIXED_TASK_PLANNER_BUDGETS.get(config["env"]["name"])
    if fixed is not None:
        config["planner"]["horizon"], config["planner"]["num_candidates"] = fixed


class MethodPolicy:
    def __init__(self, method, env, config, device):
        self.method, self.env, self.config, self.device = method, env, config, device
        eval_config = config.get("eval", {})
        self.action_selection = str(eval_config.get("action_selection", "planner"))
        if self.action_selection not in {"planner", "proposal_mean"}:
            raise ValueError(
                "eval.action_selection must be 'planner' or 'proposal_mean'"
            )
        self.use_action_kv_cache = bool(eval_config.get("use_action_kv_cache", True))
        self.proposal_std_scale = float(eval_config.get("proposal_std_scale", 1.0))
        if self.proposal_std_scale <= 0:
            raise ValueError("eval.proposal_std_scale must be positive")
        dtype_name = str(eval_config.get("mixed_precision_dtype", "bfloat16"))
        if dtype_name not in {"bfloat16", "float16"}:
            raise ValueError("eval.mixed_precision_dtype must be bfloat16 or float16")
        self.autocast_dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
        mixed_precision = eval_config.get("mixed_precision", "auto")
        if str(mixed_precision).lower() == "auto":
            # Large H1 states are memory-bound. FetchPush is video-rollout
            # bound instead: BF16 speeds its 144-token H=2 decode on Ada while
            # leaving the faster FP32 Windy/FetchSlide paths unchanged.
            mixed_precision = (
                int(config["env"]["state_size"]) >= 128
                or str(config["env"]["name"]) == "fetch_push"
            )
        self.mixed_precision = device.type == "cuda" and bool(mixed_precision)
        self.state_context_length = max(
            1, int(eval_config.get("state_context_length", 8))
        )
        planner = config["planner"]
        if planner["name"] == "cem":
            self.planner = CEMPlanner(
                planner["horizon"], planner["num_candidates"], planner["num_elites"],
                planner["iterations"], planner["momentum"], planner["min_std"], planner["execute_steps"],
            )
        else:
            self.planner = RandomShootingPlanner(
                planner["horizon"],
                planner["num_candidates"],
                planner["execute_steps"],
                planner["discount"],
                include_mean_candidate=bool(
                    planner.get("include_mean_candidate", False)
                ),
            )
        candidate_batch_size = eval_config.get("candidate_batch_size", "auto")
        if str(candidate_batch_size).lower() == "auto":
            state_size = int(config["env"]["state_size"])
            # State-Prior activations dominate evaluation memory. Keep small
            # problems fully vectorized, but bound the large Fetch/H1 batches.
            # A one-step planner never needs candidate video generation, so it
            # can safely evaluate every candidate in one state-only batch.
            if self.planner.horizon == 1:
                candidate_batch_size = self.planner.num_candidates
            elif state_size >= 256:
                candidate_batch_size = 16
            elif state_size >= 64:
                candidate_batch_size = 64
            else:
                candidate_batch_size = self.planner.num_candidates
        elif candidate_batch_size is None:
            candidate_batch_size = self.planner.num_candidates
        self.candidate_batch_size = int(candidate_batch_size)
        if self.candidate_batch_size <= 0:
            raise ValueError("eval.candidate_batch_size must be positive or 'auto'")
        self.candidate_batch_size = min(
            self.candidate_batch_size, self.planner.num_candidates
        )
        self.reset()

    def _clear_rollout_context(self):
        """Release candidate-sized tensors as soon as planning is finished."""

        self._rollout_states = None
        self._rollout_masks = None
        self._rollout_tokens = None
        self._rollout_video_cache = None
        self._rollout_video_summary = None
        self._rollout_step = 0

    def reset(self):
        action_dim = int(self.config["env"]["action_size"])
        self.history = torch.empty(1, 0, action_dim, device=self.device)
        self._action_prior_output = None
        self._observed_states = None
        self._observed_masks = None
        self._observed_tokens = None
        self._real_prior_context = None
        self._episode_step = 0
        self._clear_rollout_context()

    def _prior_output(self):
        if self.use_action_kv_cache:
            if self._action_prior_output is None:
                self._action_prior_output = self.method.action_prior.build_kv_cache(self.history)
            return self._action_prior_output
        distribution, hidden = self.method.action_prior.next_distribution(self.history)
        return distribution, hidden, None

    def _adapted_proposal(self, base, hidden, state, goal):
        distribution = self.method.adapt_action_distribution(
            base, hidden, state, goal
        )
        if self.proposal_std_scale == 1.0:
            return distribution
        return TanhNormal(
            distribution.loc,
            distribution.log_scale + math.log(self.proposal_std_scale),
            distribution.low,
            distribution.high,
        )

    def _sample_shooting_candidate_batch(
        self, state, goal, candidates, include_mean_candidate
    ):
        """Generate one memory-bounded batch of state-conditioned proposals."""

        self._clear_rollout_context()
        horizon = self.planner.horizon
        state_batch = state.reshape(1, -1).expand(candidates, -1)
        goal_batch = goal.reshape(1, -1).expand(candidates, -1)
        candidate_history = self.history.expand(candidates, -1, -1)
        base, hidden, branch_cache = self._prior_output()
        distribution = self._adapted_proposal(
            base, hidden, state.reshape(1, -1), goal.reshape(1, -1)
        )
        actions = []
        predicted_states = []
        previous = None
        try:
            for offset in range(horizon):
                if offset:
                    if (
                        branch_cache is not None
                        and branch_cache.token_count
                        < self.method.action_prior.max_length
                    ):
                        base, hidden, branch_cache = (
                            self.method.action_prior.append_kv_cache(
                                previous, branch_cache
                            )
                        )
                        distribution = self._adapted_proposal(
                            base, hidden, state_batch, goal_batch
                        )
                    else:
                        branch_cache = None
                        distribution = self.method.propose_actions(
                            candidate_history, state_batch, goal_batch
                        )
                    action = distribution.sample()
                else:
                    action = distribution.sample(torch.Size([candidates]))
                    if action.ndim == 3 and action.size(1) == 1:
                        action = action[:, 0]
                if include_mean_candidate:
                    action[0] = distribution.mean.reshape(-1, action.size(-1))[0]
                if action.shape != (
                    candidates,
                    self.method.action_prior.action_dim,
                ):
                    raise RuntimeError(
                        "Action Prior returned an invalid candidate action shape"
                    )
                actions.append(action)
                previous = action
                candidate_history = torch.cat(
                    (candidate_history, action[:, None]), dim=1
                )
                state_batch = self._dynamics(state_batch, action)
                predicted_states.append(state_batch)
            return torch.stack(actions, dim=1), torch.stack(
                predicted_states, dim=1
            )
        finally:
            self._clear_rollout_context()

    def _sample_shooting_candidates(self, state, goal):
        """Sample candidates while conditioning each step on its predicted state.

        The Action Adapter is state conditioned.  Reusing the real initial
        state at every planning offset silently turns an H-step proposal into
        an open-loop repetition of the first-step policy, which is especially
        damaging for staged manipulation.  We therefore advance the frozen
        world model between proposal steps. The resulting predicted states
        are retained and passed directly to the planner for ranking, avoiding
        a duplicate world-model rollout.
        """
        action_batches = []
        state_batches = []
        total_candidates = self.planner.num_candidates
        for start in range(0, total_candidates, self.candidate_batch_size):
            batch_size = min(
                self.candidate_batch_size, total_candidates - start
            )
            actions, predicted = self._sample_shooting_candidate_batch(
                state,
                goal,
                batch_size,
                self.planner.include_mean_candidate and start == 0,
            )
            action_batches.append(actions)
            state_batches.append(predicted)
        return torch.cat(action_batches, dim=0), torch.cat(state_batches, dim=0)

    def _advance_action_cache(self, action):
        if self.use_action_kv_cache and self._action_prior_output is not None:
            _, _, cache = self._action_prior_output
            if cache.token_count < self.method.action_prior.max_length:
                self._action_prior_output = self.method.action_prior.append_kv_cache(
                    action.reshape(1, -1), cache
                )
            else:
                self._action_prior_output = None
        action = action.reshape(1, 1, -1).to(self.history.dtype)
        self.history = torch.cat((self.history, action), dim=1)
        maximum = self.method.action_prior.max_length - 1
        if self.history.size(1) > maximum:
            self.history = self.history[:, -maximum:]

    def _dynamics(self, state, action):
        batch = state.size(0)
        if self._observed_states is None or self._observed_masks is None:
            raise RuntimeError("Planner observation context was not initialized")
        starting_rollout = (
            self._rollout_step == 0
            or self._rollout_step >= self.planner.horizon
        )
        if starting_rollout:
            self._rollout_step = 0
            # Every candidate shares the real observation prefix.  Evaluate
            # that prefix once; it is expanded only after the first action has
            # created genuinely different candidate states.
            self._rollout_states = self._observed_states
            self._rollout_masks = self._observed_masks
            if self._real_prior_context is None:
                valid = torch.ones(
                    self._observed_states.size(0),
                    self._observed_states.size(1),
                    dtype=torch.bool,
                    device=state.device,
                )
                self._real_prior_context = self.method.state_prior.encode_context(
                    self._observed_states,
                    self._observed_masks,
                    self._observed_tokens,
                    valid,
                )
            prior = self._real_prior_context
            passive = prior.passive_next
            hidden = prior.observation_hidden
            video_summary = prior.video_summary
            video_cache = prior.video_cache
        else:
            valid = torch.ones(
                self._rollout_states.size(0),
                self._rollout_states.size(1),
                dtype=torch.bool,
                device=state.device,
            )
            passive, hidden = (
                self.method.state_prior.encode_observation_context(
                    self._rollout_states,
                    self._rollout_masks,
                    self._rollout_video_summary,
                    valid,
                )
            )
            video_summary = self._rollout_video_summary
            video_cache = self._rollout_video_cache
        if passive.size(0) == 1 and batch != 1:
            passive = passive.expand(batch, -1)
            hidden = hidden.expand(batch, -1)
            video_summary = video_summary.expand(batch, -1)
        terminal_prediction = self._rollout_step + 1 >= self.planner.horizon
        if self.method.use_state_adapter:
            predicted, _, condition = self.method.state_adapter(
                state, passive, action, hidden, video_summary
            )
            projected_condition = (
                None
                if terminal_prediction
                else self.method.state_adapter.condition_to_video(condition)
            )

            def adapt(logits, token_hidden, spatial_index):
                return self.method.state_adapter.adapt_video_logits(
                    logits,
                    token_hidden,
                    spatial_index,
                    condition,
                    projected_condition=projected_condition,
                )[0]
        else:
            predicted = passive
            adapt = None
        predicted_tokens = None
        if not terminal_prediction:
            if starting_rollout:
                video_cache = self.method.state_prior.video_prior.repeat_cache(
                    video_cache,
                    batch,
                    additional_tokens=(
                        self.planner.horizon - self._rollout_step - 1
                    )
                    * self.method.state_prior.tokens_per_frame,
                )
            predicted_tokens, next_video_summary, video_cache = (
                self.method.state_prior.generate_next_video(
                    video_cache,
                    hidden,
                    logit_adapter=adapt,
                )
            )
            self._rollout_video_cache = video_cache
            self._rollout_video_summary = next_video_summary
        rollout_states = self._rollout_states
        rollout_masks = self._rollout_masks
        if rollout_states.size(0) == 1 and batch != 1:
            rollout_states = rollout_states.expand(batch, -1, -1)
            rollout_masks = rollout_masks.expand(batch, -1, -1)
        next_mask = rollout_masks[:, -1:]
        self._rollout_states = torch.cat((rollout_states, predicted[:, None]), 1)
        self._rollout_masks = torch.cat((rollout_masks, next_mask), 1)
        self._rollout_step += 1
        return predicted

    def _append_observation_context(self, state, mask, video_tokens):
        # Candidate rollout tensors are never part of the real observation
        # history and must not survive into the next environment step.
        self._clear_rollout_context()
        self._real_prior_context = None

        def append(current, value):
            value = value.reshape(1, 1, -1)
            combined = value if current is None else torch.cat((current, value), 1)
            return combined[:, -self.state_context_length :]

        self._observed_states = append(self._observed_states, state)
        self._observed_masks = append(self._observed_masks, mask)
        token_value = video_tokens.reshape(1, 1, *video_tokens.shape[-2:])
        self._observed_tokens = (
            token_value
            if self._observed_tokens is None
            else torch.cat((self._observed_tokens, token_value), 1)
        )[:, -self.state_context_length :]

    def _score(self, states, goal, actions):
        name = self.config["env"]["name"]
        if name == "windy":
            return final_goal_distance(states, goal, actions, slice(0, 2))
        if name.startswith("fetch_"):
            return fetch_score(name, states, goal, actions, self.config["env"])
        if name.startswith("humanoid_"):
            return humanoid_score_torch(
                name, states[:, -1], goal, actions, self.config["env"].get("scorer", {})
            )
        final_states = states[:, -1].detach().cpu().numpy()
        goal_array = goal.detach().cpu().numpy()
        action_array = actions.detach().cpu().numpy()
        scores = [
            self.env.compute_score(final_states[index], goal_array, action_array[index])
            for index in range(states.size(0))
        ]
        return torch.as_tensor(scores, device=states.device, dtype=states.dtype)

    @torch.inference_mode()
    def __call__(self, observation):
        state = torch.as_tensor(observation.control_state, device=self.device).float()
        goal = torch.as_tensor(observation.goal, device=self.device).float()
        state_mask = torch.as_tensor(
            observation.state_mask, dtype=torch.bool, device=self.device
        )
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
            enabled=self.mixed_precision,
        ):
            rgb = torch.tensor(np.asarray(observation.rgb).copy(), device=self.device)
            video_tokens = self.method.tokenizer.encode(rgb.unsqueeze(0)).squeeze(0)
            self._append_observation_context(state, state_mask, video_tokens)
            if self.action_selection == "proposal_mean":
                base, hidden, _ = self._prior_output()
                action = self._adapted_proposal(
                    base,
                    hidden,
                    state.reshape(1, -1),
                    goal.reshape(1, -1),
                ).mean.squeeze(0)
                self._advance_action_cache(action)
                self._episode_step += 1
                return action.reshape(-1).float().cpu().numpy(), {
                    "model_forward_calls": 0,
                    "sampled_action_sequences": 1,
                }
            if isinstance(self.planner, CEMPlanner):
                base, hidden, _ = self._prior_output()
                initial_distribution = self._adapted_proposal(
                    base, hidden, state.reshape(1, -1), goal.reshape(1, -1)
                )
                low = torch.as_tensor(self.env.action_low, device=self.device)
                high = torch.as_tensor(self.env.action_high, device=self.device)
                try:
                    result = self.planner.plan(
                        state,
                        goal,
                        self._dynamics,
                        self._score,
                        low,
                        high,
                        initial_mean=initial_distribution.mean.squeeze(0),
                        initial_std=0.25 * (high - low),
                        candidate_batch_size=self.candidate_batch_size,
                    )
                finally:
                    self._clear_rollout_context()
            else:
                candidate_actions, predicted_states = self._sample_shooting_candidates(
                    state, goal
                )
                batch_count = math.ceil(
                    self.planner.num_candidates / self.candidate_batch_size
                )
                result = self.planner.rank_candidates(
                    candidate_actions,
                    predicted_states,
                    goal,
                    self._score,
                    model_forward_calls=self.planner.horizon * batch_count,
                )
            action = result.first_action
            self._advance_action_cache(action)
            self._episode_step += 1
        return action.reshape(-1).float().cpu().numpy(), {
            "model_forward_calls": result.model_forward_calls,
            "sampled_action_sequences": result.candidates.size(0),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-prior")
    parser.add_argument("--action-prior")
    parser.add_argument("--adapters")
    args, unknown = parser.parse_known_args()
    config = config_from_unknown(["model=prior_adapter", "planner=shooting", *unknown])
    apply_fixed_task_planner_budget(config)
    seed_everything(int(config.get("seed", 0)))
    env = make_env(config)
    method = build_method(config, env.action_low, env.action_high)
    env_name = config["env"]["name"]
    _load_priors(
        method,
        Path(
            args.state_prior
            or config.get(
                "state_prior_checkpoint",
                checkpoint_path(config, "priors", f"{env_name}_state_prior.pt"),
            )
        ),
        Path(
            args.action_prior
            or config.get(
                "action_prior_checkpoint",
                checkpoint_path(config, "priors", f"{env_name}_action_prior.pt"),
            )
        ),
    )
    adapters = torch.load(
        args.adapters
        or config.get(
            "adapter_checkpoint",
            checkpoint_path(config, "adapters", f"{env_name}_adapters.pt"),
        ),
        map_location="cpu",
        weights_only=False,
    )
    if int(adapters.get("architecture_version", 0)) != int(
        method.state_prior.architecture_version
    ):
        raise ValueError(
            "Adapter checkpoint predates joint state/video correction; retrain adapters"
        )
    if adapters.get("state_adapter") is not None:
        method.state_adapter.load_state_dict(adapters["state_adapter"])
    if adapters.get("action_adapter") is not None:
        method.action_adapter.load_state_dict(adapters["action_adapter"])
        checkpoint_action_config = (
            adapters.get("config", {}).get("model", {}).get("action_adapter", {})
        )
        # Checkpoints predating bounded residuals were trained with unit scale
        # and no clipping. Preserve their exact inference semantics; newly
        # trained checkpoints carry the stabilized settings explicitly.
        method.action_adapter.residual_scale = float(
            checkpoint_action_config.get("residual_scale", 1.0)
        )
        checkpoint_clip = checkpoint_action_config.get("residual_clip")
        method.action_adapter.residual_clip = (
            None if checkpoint_clip is None else float(checkpoint_clip)
        )
        method.action_adapter.normalize_goal_direction = bool(
            checkpoint_action_config.get(
                "normalize_goal_direction",
                method.action_adapter.normalize_goal_direction,
            )
        )
        method.action_adapter.goal_feature_scale = float(
            checkpoint_action_config.get("goal_feature_scale", 1.0)
        )
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    eval_config = config.get("eval", {"name": "id", "episodes": 5, "seeds": [100]})
    allow_tf32 = device.type == "cuda" and bool(eval_config.get("allow_tf32", True))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        if allow_tf32:
            torch.set_float32_matmul_precision("high")
    method.to(device).eval()
    policy = MethodPolicy(method, env, config, device)
    condition_name = str(config.get("condition", "id"))
    if eval_config.get("name") == "ood":
        conditions = eval_config.get("ood_conditions", {})
        if condition_name == "id":
            condition_name = next(iter(conditions))
        env.set_ood_parameters(conditions[condition_name])
    seeds = eval_config.get("seeds", [100])
    episodes_per_seed = int(eval_config.get("episodes", 100))
    expanded_seeds = [int(seed) * 100000 + episode for seed in seeds for episode in range(episodes_per_seed)]
    report = method.parameter_report()
    experiment_id = str(config.get("experiment_id", f"{env_name}_{eval_config.get('name', 'id')}_{condition_name}_seed{config.get('seed', 0)}"))
    run = RunDirectory(config.get("run_root", "runs"), experiment_id)
    ood_config = (
        {}
        if condition_name == "id"
        else eval_config["ood_conditions"][condition_name]
    )
    run.initialize(
        config,
        {
            "seed": int(config.get("seed", 0)),
            "env_id": env_name,
            "method": "FunctionAlignmentWM",
            "state_adapter": method.use_state_adapter,
            "action_adapter": method.use_action_adapter,
            "observation_mode": "vq_tokens+control_state",
            "paired_steps": int(config.get("paired_steps", 0)),
            "state_only_steps": int(config.get("state_only_steps", 0)),
            "action_only_steps": int(config.get("action_only_steps", 0)),
            "planner_horizon": int(config["planner"]["horizon"]),
            "num_candidates": int(config["planner"]["num_candidates"]),
            "ood_config": ood_config,
            "trainable_parameters": report["trainable_parameters"],
            "eval": eval_config.get("name", "id"),
        },
    )
    episodes, summary = evaluate_policy(
        env,
        policy,
        expanded_seeds,
        int(config["env"].get("episode_horizon", 100)),
        int(eval_config.get("bootstrap_samples", 10000)),
        bool(eval_config.get("show_progress", True)),
        int(eval_config.get("progress_update_interval", 10)),
        f"{env_name}:{eval_config.get('name', 'id')}:{condition_name}",
        run.path / "videos" / "eval.gif",
        "Function Alignment",
        env_name,
        ood_config,
    )
    summary["parameter_count"] = sum(parameter.numel() for parameter in method.parameters())
    summary["trainable_parameters"] = report["trainable_parameters"]
    summary["peak_gpu_memory"] = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    summary["eval_optimizations"] = {
        "action_kv_cache": policy.use_action_kv_cache,
        "real_state_context_cache": True,
        "incremental_rollout_video_cache": True,
        "preallocated_rollout_video_kv_suffix": True,
        "skip_terminal_video_generation": True,
        "candidate_batch_size": policy.candidate_batch_size,
        "mixed_precision": policy.mixed_precision,
        "mixed_precision_dtype": str(eval_config.get("mixed_precision_dtype", "bfloat16")),
        "allow_tf32": allow_tf32,
        "vectorized_humanoid_scorer": env_name.startswith("humanoid_"),
        "fixed_task_planner_budget": bool(
            eval_config.get("fixed_planner_budget", True)
        ),
    }
    baseline_name = str(eval_config.get("comparison_baseline", "gcrl"))
    baseline_video_override = eval_config.get("comparison_baseline_video")
    if baseline_video_override:
        baseline_video = Path(str(baseline_video_override)).expanduser()
    else:
        baseline_experiment = (
            f"baseline_{baseline_name}_{env_name}_seed{int(config.get('seed', 0))}"
        )
        baseline_video = (
            Path(config.get("run_root", "runs"))
            / baseline_experiment
            / "videos"
            / "eval.gif"
        )
        # Read already-completed runs from the former data-budget naming scheme.
        if not baseline_video.exists():
            legacy_experiment = (
                f"baseline_{baseline_name}_{env_name}_steps"
                f"{int(config.get('paired_steps', 0))}_seed{int(config.get('seed', 0))}"
            )
            legacy_video = (
                Path(config.get("run_root", "runs"))
                / legacy_experiment
                / "videos"
                / "eval.gif"
            )
            if legacy_video.exists():
                baseline_video = legacy_video
    if baseline_video.exists():
        from fa_robotics_planner.visualization import save_side_by_side

        comparison_path = run.path / "videos" / f"ours_vs_{baseline_name}.gif"
        save_side_by_side(
            run.path / "videos" / "eval.gif",
            baseline_video,
            comparison_path,
            "Function Alignment",
            baseline_name,
        )
        summary["comparison_video"] = str(comparison_path)
    else:
        summary["comparison_video_pending"] = str(baseline_video)
    write_episode_metrics(run.path / "metrics.jsonl", episodes)
    summary["evaluation_metrics_file"] = "metrics.jsonl"
    summary["per_step_rewards"] = True
    run.write_json("summary.json", summary)
    env.close()
    print(f"Saved evaluation to {run.path}")


if __name__ == "__main__":
    main()
