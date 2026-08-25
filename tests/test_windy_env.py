import numpy as np

from fa_robotics_planner.envs.windy import WindyControlEnv
from fa_robotics_planner.data.generate import _windy_expert_action


def config(**updates):
    base = {
        "dt": 0.1,
        "episode_horizon": 3,
        "gamma": 0.9,
        "v_max": 1.0,
        "a_max": 1.0,
        "w_max": 0.6,
        "bounce_beta": 0.5,
        "success_radius": 0.1,
        "region_count": 4,
        "region_w_scale": 0.6,
        "image_size": [64, 64],
        "legacy_static_wind": True,
    }
    base.update(updates)
    return base


def test_shapes_seed_and_bounds():
    env = WindyControlEnv(config())
    first = env.reset(7)
    second = env.reset(7)
    assert np.array_equal(first.control_state, second.control_state)
    assert first.rgb.shape == (64, 64, 3)
    assert first.rgb.dtype == np.uint8
    assert first.proprio.shape == (2,)
    assert first.control_state.shape == first.state_mask.shape == (4,)
    transition = env.step(np.array([10, -10], np.float32))
    assert transition.observation.control_state.shape == (4,)
    assert isinstance(transition.terminated, bool)
    assert isinstance(transition.truncated, bool)


def test_legacy_transition_order_regression():
    env = WindyControlEnv(config())
    env.reset(0)
    env.state = np.array([0.0, 0.0, 0.2, -0.1], np.float32)
    env.goal = np.array([0.9, 0.9], np.float32)
    env.region_w0[:] = np.array([0.1, 0.2], np.float32)
    result = env.step(np.array([0.5, -0.5], np.float32))
    velocity = 0.9 * np.array([0.2, -0.1]) + np.array([0.5, -0.5]) * 0.1
    position = (velocity + np.array([0.1, 0.2])) * 0.1
    assert np.allclose(result.observation.control_state, np.r_[position, velocity])


def test_termination_truncation_and_ood_slope():
    env = WindyControlEnv(config(legacy_static_wind=False, success_radius=0.0))
    env.set_ood_parameters({"slope": 0.02})
    env.reset(3)
    first = env.current_wind().copy()
    env.step(np.zeros(2))
    second = env.current_wind().copy()
    assert not np.array_equal(first, second)
    env.step(np.zeros(2))
    final = env.step(np.zeros(2))
    assert final.truncated and not final.terminated


def test_paired_data_expert_is_high_success():
    env = WindyControlEnv(
        config(
            episode_horizon=100,
            paired_data={"expert_kp": 10.0, "expert_kd": 1.0, "expert_kw": 0.0},
        )
    )
    successes = []
    for seed in range(20):
        observation = env.reset(seed)
        success = False
        for _ in range(100):
            transition = env.step(_windy_expert_action(env, observation))
            observation = transition.observation
            success = success or bool(transition.info["success"])
            if transition.done:
                break
        successes.append(success)
    assert np.mean(successes) >= 0.95
