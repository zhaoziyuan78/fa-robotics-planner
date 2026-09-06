import numpy as np
import torch
import imageio.v3 as iio
import pytest

from fa_robotics_planner.envs.unified import ObservationBundle, StepResult
from fa_robotics_planner.evaluation import evaluate_policy, write_episode_metrics
from fa_robotics_planner.models.action_adapter import ActionAdapter
from fa_robotics_planner.models.action_prior import CausalActionPrior
from fa_robotics_planner.models.method import FunctionAlignmentWM
from fa_robotics_planner.models.state_adapter import StateAdapter
from fa_robotics_planner.models.state_prior import CausalStatePrior
from fa_robotics_planner.models.vqvae import VQVAE
from scripts.evaluate import MethodPolicy, apply_fixed_task_planner_budget
from fa_robotics_planner.visualization import save_side_by_side


class TinyEnv:
    action_low = np.asarray([-1.0], np.float32)
    action_high = np.asarray([1.0], np.float32)

    def reset(self, seed):
        self.steps = 0
        return self._observation()

    def _observation(self):
        return ObservationBundle(
            np.zeros((2, 2, 3), np.uint8),
            control_state=np.asarray([self.steps], np.float32),
            state_mask=np.asarray([True]),
        )

    def step(self, action):
        self.steps += 1
        done = self.steps == 2
        return StepResult(
            self._observation(),
            1.0,
            done,
            False,
            {"success": done},
        )


def test_evaluation_progress_and_early_termination(capsys):
    episodes, summary = evaluate_policy(
        TinyEnv(),
        lambda observation: np.zeros(1, np.float32),
        [1, 2],
        5,
        bootstrap_samples=10,
        show_progress=True,
        progress_update_interval=1,
        progress_desc="tiny-eval",
    )
    assert [episode["episode_length"] for episode in episodes] == [2, 2]
    assert [episode["rewards"] for episode in episodes] == [[1.0, 1.0], [1.0, 1.0]]
    assert summary["episodes"] == 2
    assert summary["evaluation_wall_seconds"] >= 0
    assert summary["evaluation_steps"] == 4
    assert summary["evaluation_steps_per_second"] > 0
    assert summary["planning_steps_per_second"] > 0
    assert "tiny-eval" in capsys.readouterr().err


def test_episode_metrics_writer_requires_rewards_and_replaces_old_eval(tmp_path):
    target = tmp_path / "metrics.jsonl"
    write_episode_metrics(
        target,
        [
            {
                "seed": 1,
                "return": 2.0,
                "rewards": [1.0, 1.0],
                "episode_length": 2,
            },
            {
                "seed": 2,
                "return": 1.0,
                "rewards": [1.0],
                "episode_length": 1,
            },
        ],
    )
    assert len(target.read_text(encoding="utf-8").splitlines()) == 2

    write_episode_metrics(
        target,
        [
            {
                "seed": 3,
                "return": -1.0,
                "rewards": [-1.0],
                "episode_length": 1,
            }
        ],
    )
    assert len(target.read_text(encoding="utf-8").splitlines()) == 1
    with pytest.raises(ValueError, match="missing.*rewards"):
        write_episode_metrics(target, [{"seed": 4, "return": 0.0}])


def test_environment_can_finalize_survival_success():
    class SurvivalEnv(TinyEnv):
        def step(self, action):
            self.steps += 1
            return StepResult(self._observation(), 1.0, False, False, {})

        def episode_success(
            self, success, episode_return, steps_taken, episode_horizon, terminated, truncated
        ):
            return success or (not terminated and steps_taken >= episode_horizon)

    episodes, _ = evaluate_policy(
        SurvivalEnv(),
        lambda observation: np.zeros(1, np.float32),
        [1],
        3,
        bootstrap_samples=10,
        show_progress=False,
    )
    assert episodes[0]["success"] is True
    assert episodes[0]["time_to_success"] == 3


def test_evaluation_writes_rollout_and_side_by_side_gifs(tmp_path):
    left = tmp_path / "ours.gif"
    right = tmp_path / "baseline.gif"
    combined = tmp_path / "combined.gif"
    for output in (left, right):
        evaluate_policy(
            TinyEnv(),
            lambda observation: np.zeros(1, np.float32),
            [3],
            3,
            bootstrap_samples=5,
            show_progress=False,
            video_output=output,
            video_method="test",
            video_task="tiny",
        )
        assert output.exists() and output.stat().st_size > 0
        rendered = iio.imread(output)
        assert rendered.shape[-3:-1] == (2, 2)
        assert not rendered.any(), "eval.gif must not contain a text overlay"
    save_side_by_side(left, right, combined)
    assert combined.exists() and combined.stat().st_size > 0


def test_method_policy_bounds_candidate_batch_and_releases_rollout_context():
    action_prior = CausalActionPrior(
        2, d_model=16, n_layers=1, n_heads=2, dropout=0, max_length=8
    )
    state_prior = CausalStatePrior(
        4, codebook_size=16, tokens_per_frame=1,
        video_d_model=16, video_layers=1, video_heads=2, video_d_ff=32,
        d_model=16, n_layers=1, n_heads=2, dropout=0, max_length=8
    )
    method = FunctionAlignmentWM(
        action_prior,
        state_prior,
        StateAdapter(4, 2, 16, 16, 16, 16, 1),
        ActionAdapter(2, 4, 2, 16, 16),
        VQVAE(hidden_dim=16, codebook_size=16, code_dim=8),
    ).eval()
    observed_batch_sizes = []
    generated_video_batch_sizes = []
    original_forward = state_prior.encode_context
    original_generate = state_prior.generate_next_video

    def recording_forward(*args, **kwargs):
        observed_batch_sizes.append(args[0].size(0))
        return original_forward(*args, **kwargs)

    state_prior.encode_context = recording_forward

    def recording_generate(cache, hidden, **kwargs):
        generated_video_batch_sizes.append(hidden.size(0))
        return original_generate(cache, hidden, **kwargs)

    state_prior.generate_next_video = recording_generate

    class PlanningEnv:
        action_low = np.full(2, -1.0, np.float32)
        action_high = np.full(2, 1.0, np.float32)

    config = {
        "env": {"name": "windy", "state_size": 4, "action_size": 2},
        "planner": {
            "name": "shooting",
            "horizon": 2,
            "num_candidates": 5,
            "execute_steps": 1,
            "discount": 0.99,
            "include_mean_candidate": True,
        },
        "eval": {
            "candidate_batch_size": 2,
            "mixed_precision": False,
            "state_context_length": 2,
        },
    }
    policy = MethodPolicy(method, PlanningEnv(), config, torch.device("cpu"))
    observation = ObservationBundle(
        np.zeros((8, 8, 3), np.uint8),
        control_state=np.zeros(4, np.float32),
        state_mask=np.ones(4, bool),
        goal=np.ones(2, np.float32),
    )
    action, diagnostics = policy(observation)

    assert action.shape == (2,)
    assert diagnostics["sampled_action_sequences"] == 5
    assert diagnostics["model_forward_calls"] == 6
    assert observed_batch_sizes == [1]
    assert generated_video_batch_sizes == [2, 2, 1]
    assert policy._rollout_states is None
    assert policy._rollout_masks is None
    assert policy._rollout_tokens is None

    observed_batch_sizes.clear()
    generated_video_batch_sizes.clear()
    one_step_config = {
        **config,
        "planner": {**config["planner"], "horizon": 1},
        "eval": {**config["eval"], "candidate_batch_size": "auto"},
    }
    one_step_policy = MethodPolicy(
        method, PlanningEnv(), one_step_config, torch.device("cpu")
    )
    _, one_step_diagnostics = one_step_policy(observation)
    assert one_step_policy.candidate_batch_size == 5
    assert observed_batch_sizes == [1]
    assert generated_video_batch_sizes == []
    assert one_step_diagnostics["model_forward_calls"] == 1


def test_task_planner_budget_is_fixed_independent_of_paired_steps():
    config = {
        "env": {"name": "fetch_slide"},
        "planner": {"horizon": 99, "num_candidates": 3},
        "eval": {"fixed_planner_budget": True},
        "paired_steps": 1000,
    }
    apply_fixed_task_planner_budget(config)
    assert config["planner"] == {"horizon": 1, "num_candidates": 256}
    config["paired_steps"] = 10000
    apply_fixed_task_planner_budget(config)
    assert config["planner"] == {"horizon": 1, "num_candidates": 256}
