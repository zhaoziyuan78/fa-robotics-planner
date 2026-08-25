import numpy as np
import torch

from fa_robotics_planner.envs.unified import ObservationBundle, StepResult
from fa_robotics_planner.evaluation import evaluate_policy
from fa_robotics_planner.models.action_adapter import ActionAdapter
from fa_robotics_planner.models.action_prior import CausalActionPrior
from fa_robotics_planner.models.method import FunctionAlignmentWM
from fa_robotics_planner.models.state_adapter import StateAdapter
from fa_robotics_planner.models.state_prior import CausalStatePrior
from scripts.evaluate import MethodPolicy
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
    assert summary["episodes"] == 2
    assert summary["evaluation_wall_seconds"] >= 0
    assert "tiny-eval" in capsys.readouterr().err


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
    save_side_by_side(left, right, combined)
    assert combined.exists() and combined.stat().st_size > 0


def test_method_policy_bounds_candidate_batch_and_releases_rollout_context():
    action_prior = CausalActionPrior(
        2, d_model=16, n_layers=1, n_heads=2, dropout=0, max_length=8
    )
    state_prior = CausalStatePrior(
        4, d_model=16, n_layers=1, n_heads=2, dropout=0, max_length=8
    )
    method = FunctionAlignmentWM(
        action_prior,
        state_prior,
        StateAdapter(4, 2, 16, 16),
        ActionAdapter(2, 4, 2, 16, 16),
    ).eval()
    method.visual_encoder = None
    observed_batch_sizes = []
    original_forward = state_prior.forward

    def recording_forward(*args, **kwargs):
        observed_batch_sizes.append(args[0].size(0))
        return original_forward(*args, **kwargs)

    state_prior.forward = recording_forward

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
        np.zeros((2, 2, 3), np.uint8),
        control_state=np.zeros(4, np.float32),
        state_mask=np.ones(4, bool),
        goal=np.ones(2, np.float32),
    )
    action, diagnostics = policy(observation)

    assert action.shape == (2,)
    assert diagnostics["sampled_action_sequences"] == 5
    assert diagnostics["model_forward_calls"] == 6
    assert max(observed_batch_sizes) <= 2
    assert policy._rollout_states is None
    assert policy._rollout_masks is None
    assert policy._rollout_proprio is None
    assert policy._rollout_visual is None
