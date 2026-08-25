import os

import numpy as np
import pytest


@pytest.mark.integration
@pytest.mark.parametrize("task", ("fetch_slide", "fetch_push"))
def test_fetch_real_smoke(task):
    os.environ.setdefault("MUJOCO_GL", "egl")
    from fa_robotics_planner.envs.fetch import FetchControlEnv

    try:
        env = FetchControlEnv(
            task,
            {"image_size": [64, 64], "state_size": 64, "episode_horizon": 2, "mujoco_gl": "egl"},
        )
    except RuntimeError as exc:
        pytest.skip(str(exc))
    observation = env.reset(0)
    transition = env.step(np.zeros(4, np.float32))
    assert observation.rgb.shape == (64, 64, 3)
    assert observation.control_state.shape == observation.state_mask.shape == (64,)
    assert transition.observation.goal.shape == (3,)
    qpos_before = env.unwrapped.data.qpos.copy()
    goal_frame = env.render_goal(transition.observation.goal)
    assert goal_frame.shape == observation.rgb.shape
    assert np.array_equal(env.unwrapped.data.qpos, qpos_before)
    object_ids = env._object_body_ids()
    base_mass = env.unwrapped.model.body_mass[object_ids].copy()
    env.set_ood_parameters({"mass_multiplier": 1.4, "friction_multiplier": 0.5})
    env.reset(0)
    assert np.allclose(env.unwrapped.model.body_mass[object_ids], base_mass * 1.4)
    env.close()
