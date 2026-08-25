from pathlib import Path

import numpy as np
import pytest

from fa_robotics_planner.config import compose_config
from fa_robotics_planner.envs.humanoid import HumanoidControlEnv, assert_shared_humanoid_action_space


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "infrastructure" / "h1hand_stand_ppo.zip"
TASKS = ("humanoid_stand", "humanoid_balance", "humanoid_reach", "humanoid_push")


@pytest.mark.integration
@pytest.mark.parametrize("task", TASKS)
def test_humanoid_real_reset_step_render_and_seed(task):
    if not CHECKPOINT.exists():
        pytest.skip("debug nominal-controller checkpoint is absent")
    config = compose_config([f"env={task}"], config_root=ROOT / "configs")["env"]
    config.update(
        nominal_controller=str(CHECKPOINT),
        image_size=[64, 64],
        episode_horizon=2,
    )
    env = HumanoidControlEnv(task, config)
    first = env.reset(11)
    repeated = env.reset(11)
    assert np.array_equal(first.control_state, repeated.control_state)
    assert first.rgb.shape == (64, 64, 3)
    assert first.proprio.shape == (151,)
    assert first.control_state.shape == first.state_mask.shape == (512,)
    assert env.action_low.shape == env.action_high.shape == (61,)
    transition = env.step(np.zeros(61, np.float32))
    assert isinstance(transition.terminated, bool)
    assert isinstance(transition.truncated, bool)
    base_friction = env.env.unwrapped.model.geom_friction.copy()
    env.set_ood_parameters({"friction_multiplier": 0.6})
    env.reset(11)
    assert np.allclose(env.env.unwrapped.model.geom_friction, base_friction * 0.6)
    env.close()


@pytest.mark.integration
def test_humanoid_action_spaces_are_shared():
    if not CHECKPOINT.exists():
        pytest.skip("debug nominal-controller checkpoint is absent")
    envs = []
    try:
        for task in TASKS:
            config = compose_config([f"env={task}"], config_root=ROOT / "configs")["env"]
            config["nominal_controller"] = str(CHECKPOINT)
            envs.append(HumanoidControlEnv(task, config))
        assert_shared_humanoid_action_space(envs)
    finally:
        for env in envs:
            env.close()
