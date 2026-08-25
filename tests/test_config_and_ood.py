from pathlib import Path

from fa_robotics_planner.config import compose_config
from fa_robotics_planner.envs import make_env
from fa_robotics_planner.envs.ood import validate_ood_config
from scripts._common import checkpoint_path, data_path


ROOT = Path(__file__).resolve().parents[1]


def test_config_composition_and_scalar_override():
    config = compose_config(
        ["env=windy", "model=prior_adapter", "planner=cem", "planner.horizon=5"],
        config_root=ROOT / "configs",
    )
    assert config["env"]["name"] == "windy"
    assert config["planner"]["name"] == "cem"
    assert config["planner"]["horizon"] == 5


def test_default_storage_paths_are_centralized():
    config = compose_config(["env=windy"], config_root=ROOT / "configs")
    storage_root = Path("/l/users/ziyuan.zhao/fa-robotics-planner")
    assert Path(config["data_root"]) == storage_root / "data"
    assert Path(config["checkpoint_root"]) == storage_root / "checkpoints"
    assert data_path(config, "windy", "paired") == storage_root / "data/windy/paired"
    assert checkpoint_path(config, "priors", "windy_state_prior.pt") == (
        storage_root / "checkpoints/priors/windy_state_prior.pt"
    )


def test_humanoid_nominal_controller_comes_from_central_config(monkeypatch):
    config = compose_config(["env=humanoid_stand"], config_root=ROOT / "configs")
    monkeypatch.setattr(
        "fa_robotics_planner.envs.humanoid.HumanoidControlEnv",
        lambda name, env_config: (name, env_config),
    )
    name, env_config = make_env(config)
    assert name == "humanoid_stand"
    assert Path(env_config["nominal_controller"]) == Path(
        config["humanoid_nominal_controller"]["checkpoint"]
    )
    assert env_config["nominal_controller_type"] == "reach"
    assert Path(env_config["nominal_controller_mean"]).parent == (
        Path(config["checkpoint_root"]) / "infrastructure"
    )


def test_humanoid_env_override_is_loaded_after_generic_planner():
    config = compose_config(
        [
            "model=prior_adapter",
            "planner=shooting",
            "env=humanoid_balance",
            "eval=id",
        ],
        config_root=ROOT / "configs",
    )
    assert config["planner"]["num_candidates"] == 64
    assert config["eval"]["candidate_batch_size"] == "auto"


def test_humanoid_shared_state_prior_uses_plateau_schedule_and_early_stop():
    config = compose_config(
        ["model=prior_adapter", "env=humanoid_shared"],
        config_root=ROOT / "configs",
    )
    state_prior = config["model"]["state_prior"]
    assert state_prior["learning_rate"] == 1e-3
    assert state_prior["scheduler"]["name"] == "reduce_on_plateau"
    assert state_prior["early_stopping"]["enabled"] is True


def test_ood_ranges_and_seed_overlap():
    errors = validate_ood_config(
        {"mass": [0.8, 1.2]},
        {"heavy": {"mass": 1.4}},
        [0, 1],
        [100, 101],
    )
    assert errors == []
    errors = validate_ood_config(
        {"mass": [0.8, 1.2]},
        {"not_ood": {"mass": 1.0}},
        [0, 1],
        [1, 2],
    )
    assert len(errors) == 2
