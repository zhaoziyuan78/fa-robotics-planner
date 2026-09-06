import inspect

import pytest
import torch

from fa_robotics_planner.models.action_adapter import ActionAdapter
from fa_robotics_planner.models.action_prior import CausalActionPrior, DiscreteCausalActionPrior
from fa_robotics_planner.models.method import FunctionAlignmentWM
from fa_robotics_planner.models.state_adapter import StateAdapter
from fa_robotics_planner.models.state_prior import CausalStatePrior
from fa_robotics_planner.models.vqvae import VQVAE
from fa_robotics_planner.training.parameters import make_adapter_optimizer
from fa_robotics_planner.models.distributions import TanhNormal
from fa_robotics_planner.training.losses import (
    masked_state_loss,
    soft_action_adapter_loss,
    tanh_normal_kl,
)
from scripts.train_adapters import _load_priors


def modules(use_state=True, use_action=True):
    action_prior = CausalActionPrior(2, d_model=16, n_layers=1, n_heads=2, dropout=0, max_length=8)
    state_prior = CausalStatePrior(
        4, codebook_size=16, tokens_per_frame=4,
        video_d_model=16, video_layers=1, video_heads=2, video_d_ff=32,
        d_model=16, n_layers=1, n_heads=2, dropout=0, max_length=8,
    )
    state_adapter = StateAdapter(4, 2, 16, 16, 16, 16, 4)
    action_adapter = ActionAdapter(2, 4, 2, 16, 16)
    return FunctionAlignmentWM(
        action_prior,
        state_prior,
        state_adapter,
        action_adapter,
        use_state_adapter=use_state,
        use_action_adapter=use_action,
    )


def test_prior_signatures_cannot_accept_forbidden_inputs():
    action_parameters = inspect.signature(CausalActionPrior.forward).parameters
    state_parameters = inspect.signature(CausalStatePrior.forward).parameters
    assert not {"state", "goal", "reward", "task_id"} & set(action_parameters)
    assert not {"action", "actions", "task_action"} & set(state_parameters)


def test_adapter_output_shapes_and_prior_freezing():
    method = modules()
    history = torch.zeros(3, 2, 2)
    state = torch.zeros(3, 4)
    goal = torch.zeros(3, 2)
    distribution = method.propose_actions(history, state, goal)
    assert distribution.mean.shape == (3, 2)
    predicted = method.predict_transition(
        torch.zeros(3, 1, 4),
        torch.ones(3, 1, 4, dtype=torch.bool),
        torch.zeros(3, 2),
        torch.zeros(3, 1, 2, 2, dtype=torch.long),
        valid_steps=torch.ones(3, 1, dtype=torch.bool),
    )
    assert predicted[0].shape == (3, 4)
    assert predicted[1].shape == (3, 4)
    assert all(not parameter.requires_grad for parameter in method.action_prior.parameters())
    assert all(not parameter.requires_grad for parameter in method.state_prior.parameters())


def test_state_adapter_preprojected_video_condition_is_equivalent():
    torch.manual_seed(9)
    adapter = StateAdapter(4, 2, 16, 16, 16, 16, 4).eval()
    logits = torch.randn(3, 16)
    token_hidden = torch.randn(3, 16)
    condition = torch.randn(3, 16)
    expected, _ = adapter.adapt_video_logits(
        logits, token_hidden, 2, condition
    )
    projected = adapter.condition_to_video(condition)
    actual, _ = adapter.adapt_video_logits(
        logits,
        token_hidden,
        2,
        condition,
        projected_condition=projected,
    )
    assert torch.equal(actual, expected)


def test_adapter_off_bypasses_modules_and_optimizer_membership():
    method = modules(False, False)

    def forbidden(*args, **kwargs):
        raise AssertionError("disabled adapter was called")

    method.action_adapter.forward = forbidden
    method.state_adapter.forward = forbidden
    method.propose_actions(torch.zeros(1, 0, 2), torch.zeros(1, 4), torch.zeros(1, 2))
    method.predict_transition(
        torch.zeros(1, 1, 4),
        torch.ones(1, 1, 4, dtype=torch.bool),
        torch.zeros(1, 2),
        torch.zeros(1, 1, 2, 2, dtype=torch.long),
        valid_steps=torch.ones(1, 1, dtype=torch.bool),
    )
    assert method.parameter_report()["trainable_parameters"] == 0


def test_variable_length_mask_matches_individual_result():
    prior = CausalActionPrior(2, d_model=16, n_layers=1, n_heads=2, dropout=0, max_length=8).eval()
    short = torch.randn(1, 2, 2)
    long = torch.randn(1, 4, 2)
    padded = torch.cat((short, torch.zeros(1, 2, 2)), dim=1)
    batch = torch.cat((padded, long), dim=0)
    valid = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    batch_dist, _ = prior.next_distribution(batch, valid)
    short_dist, _ = prior.next_distribution(short)
    assert torch.allclose(batch_dist.loc[0], short_dist.loc[0], atol=1e-5)


def test_action_prior_kv_cache_matches_full_prefix_recomputation():
    torch.manual_seed(4)
    prior = CausalActionPrior(
        2, d_model=16, n_layers=2, n_heads=2, dropout=0, max_length=8
    ).eval()
    history = torch.randn(1, 3, 2)
    expected, expected_hidden = prior.next_distribution(history)
    cached, cached_hidden, cache = prior.build_kv_cache(history)
    assert torch.allclose(cached.loc, expected.loc, atol=1e-6)
    assert torch.allclose(cached.log_scale, expected.log_scale, atol=1e-6)
    assert torch.allclose(cached_hidden, expected_hidden, atol=1e-6)

    actions = torch.randn(5, 2)
    histories = torch.cat((history.expand(5, -1, -1), actions[:, None]), dim=1)
    expected, expected_hidden = prior.next_distribution(histories)
    cached, cached_hidden, _ = prior.append_kv_cache(actions, cache)
    assert torch.allclose(cached.loc, expected.loc, atol=1e-5)
    assert torch.allclose(cached.log_scale, expected.log_scale, atol=1e-5)
    assert torch.allclose(cached_hidden, expected_hidden, atol=1e-5)


def test_discrete_action_distribution_interface():
    table = torch.tensor([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    prior = DiscreteCausalActionPrior(3, d_model=8, n_layers=1, action_table=table)
    distribution, hidden = prior.next_distribution(torch.tensor([[0, 1]]))
    assert distribution.sample().shape == (1, 2)
    assert distribution.mean.shape == (1, 2)
    assert distribution.log_prob(torch.tensor([[0.0, 0.0]])).shape == (1,)
    assert hidden.shape == (1, 8)


def test_frozen_prior_parameters_do_not_update():
    method = modules()
    before = {
        name: parameter.detach().clone()
        for name, parameter in method.named_parameters()
        if name.startswith(("action_prior.", "state_prior."))
    }
    optimizer = make_adapter_optimizer(method, 1e-2)
    distribution = method.propose_actions(
        torch.zeros(2, 1, 2), torch.zeros(2, 4), torch.zeros(2, 2)
    )
    predicted = method.predict_transition(
        torch.zeros(2, 1, 4),
        torch.ones(2, 1, 4, dtype=torch.bool),
        torch.zeros(2, 2),
        torch.zeros(2, 1, 2, 2, dtype=torch.long),
        valid_steps=torch.ones(2, 1, dtype=torch.bool),
    )
    loss = distribution.loc.sum() + predicted[0].sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    for name, expected in before.items():
        assert torch.equal(dict(method.named_parameters())[name], expected)


def test_vqvae_tokenizes_frames_and_state_prior_is_dual_autoregressive():
    tokenizer = VQVAE(hidden_dim=16, codebook_size=16, code_dim=8)
    images = torch.randint(0, 256, (3, 16, 16, 3), dtype=torch.uint8)
    output = tokenizer(images)
    assert output.tokens.shape == (3, 2, 2)
    assert output.reconstruction.shape == (3, 3, 16, 16)
    assert torch.isfinite(output.loss)

    prior = CausalStatePrior(
        4, codebook_size=16, tokens_per_frame=4,
        video_d_model=16, video_layers=1, video_heads=2, video_d_ff=32,
        d_model=16, n_layers=1, n_heads=2, dropout=0, max_length=8,
    )
    prediction = prior(
        torch.zeros(3, 2, 4),
        torch.ones(3, 2, 4, dtype=torch.bool),
        output.tokens[:, None].expand(-1, 2, -1, -1),
        torch.ones(3, 2, dtype=torch.bool),
    )
    assert prediction.passive_next.shape == (3, 1, 4)
    assert prediction.video_logits.shape == (3, 1, 4, 16)


def test_state_adapter_zero_initialization_and_joint_corrections():
    method = modules()
    states = torch.zeros(2, 2, 4)
    masks = torch.ones_like(states, dtype=torch.bool)
    tokens = torch.zeros(2, 2, 2, 2, dtype=torch.long)
    prior = method.state_prior(states, masks, tokens)
    predicted, delta, condition = method.state_adapter(
        states[:, :-1],
        prior.passive_next,
        torch.zeros(2, 1, 2),
        prior.hidden,
        prior.video_summary[:, :-1],
    )
    logits, logit_delta = method.state_adapter.adapt_video_sequence(
        prior.video_logits,
        prior.video_predictor_hidden,
        condition,
    )
    assert torch.equal(predicted, prior.passive_next)
    assert not delta.any()
    assert torch.equal(logits, prior.video_logits)
    assert not logit_delta.any()
    with torch.no_grad():
        method.state_adapter.state_head.bias.fill_(0.1)
        method.state_adapter.video_head[-1].bias[0] = 0.2
    predicted, delta, condition = method.state_adapter(
        states[:, :-1], prior.passive_next, torch.zeros(2, 1, 2),
        prior.hidden, prior.video_summary[:, :-1]
    )
    logits, logit_delta = method.state_adapter.adapt_video_sequence(
        prior.video_logits, prior.video_predictor_hidden, condition
    )
    assert delta.abs().sum() > 0
    assert logit_delta.abs().sum() > 0
    assert not torch.equal(predicted, prior.passive_next)
    assert not torch.equal(logits, prior.video_logits)


def test_video_prior_projected_kv_cache_matches_full_recomputation():
    torch.manual_seed(8)
    prior = modules().state_prior.eval()
    history = torch.randint(0, 16, (2, 2, 2, 2))
    appended = torch.randint(0, 16, (2, 4))
    cache = prior.video_prior.build_cache(history)
    cache, incremental = prior.video_prior.append_to_cache(cache, appended)
    full = prior.video_prior(
        torch.cat((history, appended.reshape(2, 1, 2, 2)), dim=1)
    )
    assert torch.allclose(incremental, full[:, -1], atol=1e-5)


def test_reserved_video_cache_matches_full_recomputation_without_reallocation():
    torch.manual_seed(18)
    prior = modules().state_prior.eval()
    history = torch.randint(0, 16, (1, 2, 2, 2))
    appended = torch.randint(0, 16, (3, 8))
    shared = prior.video_prior.build_cache(history)
    cache = prior.video_prior.repeat_cache(
        shared, batch_size=3, additional_tokens=8
    )
    storage = [layer["key"].data_ptr() for layer in cache["layers"]]
    cache, first = prior.video_prior.append_to_cache(cache, appended[:, :4])
    cache, second = prior.video_prior.append_to_cache(cache, appended[:, 4:])
    full_tokens = torch.cat(
        (
            history.expand(3, -1, -1, -1),
            appended.reshape(3, 2, 2, 2),
        ),
        dim=1,
    )
    full = prior.video_prior(full_tokens)
    assert torch.allclose(first, full[:, -2], atol=1e-5)
    assert torch.allclose(second, full[:, -1], atol=1e-5)
    assert storage == [layer["key"].data_ptr() for layer in cache["layers"]]
    assert shared["length"] == 8


def test_state_context_cache_matches_single_full_video_encoding():
    torch.manual_seed(9)
    prior = modules().state_prior.eval()
    states = torch.randn(2, 3, 4)
    masks = torch.ones_like(states, dtype=torch.bool)
    tokens = torch.randint(0, 16, (2, 3, 2, 2))
    valid = torch.ones(2, 3, dtype=torch.bool)
    normalized = prior._normalize(states) * masks.to(states.dtype)
    observation_hidden = prior.observation_prior(normalized, masks, valid)[:, -1]
    video_summary = prior.video_prior(tokens, valid)[:, -1].mean(1)
    expected = prior._predict_state(
        states[:, -1], observation_hidden, video_summary, True
    )
    cached = prior.encode_context(states, masks, tokens, valid)
    assert torch.allclose(cached.observation_hidden, observation_hidden, atol=1e-6)
    assert torch.allclose(cached.video_summary, video_summary, atol=1e-5)
    assert torch.allclose(cached.passive_next, expected, atol=1e-5)


def test_masked_state_loss_ignores_non_finite_padding():
    predicted = torch.tensor([[1.0, float("nan")]])
    target = torch.zeros_like(predicted)
    mask = torch.tensor([[True, False]])
    loss = masked_state_loss(predicted, target, mask)
    assert torch.isfinite(loss)
    assert loss.item() == 1.0


def test_action_adapter_starts_as_prior_and_bounds_parameter_residuals():
    adapter = ActionAdapter(
        2, 4, 2, 16, 16, residual_scale=0.5, residual_clip=0.25
    )
    base = TanhNormal(
        torch.zeros(3, 2),
        torch.zeros(3, 2),
        torch.full((2,), -1.0),
        torch.full((2,), 1.0),
    )
    initial, delta_mu, delta_log_scale = adapter(
        base, torch.zeros(3, 16), torch.zeros(3, 4), torch.zeros(3, 2)
    )
    assert torch.equal(initial.loc, base.loc)
    assert torch.equal(initial.log_scale, base.log_scale)
    with torch.no_grad():
        adapter.network[-1].bias.fill_(100.0)
    _, delta_mu, delta_log_scale = adapter(
        base, torch.zeros(3, 16), torch.zeros(3, 4), torch.zeros(3, 2)
    )
    assert delta_mu.abs().max().item() <= 0.125
    assert delta_log_scale.abs().max().item() <= 0.125


def test_soft_action_objective_has_zero_kl_for_unchanged_prior():
    base = TanhNormal(
        torch.zeros(5, 2),
        torch.zeros(5, 2),
        torch.full((2,), -1.0),
        torch.full((2,), 1.0),
    )
    assert torch.equal(tanh_normal_kl(base, base), torch.zeros(5))
    torch.manual_seed(3)
    total, soft_nll, kl = soft_action_adapter_loss(
        base,
        base,
        torch.zeros(5, 2),
        target_sigma=0.05,
        target_samples=3,
        label_smoothing=0.02,
        kl_weight=0.1,
    )
    assert torch.isfinite(total)
    assert torch.allclose(total, soft_nll)
    assert kl.item() == 0.0


def test_adapter_training_rejects_legacy_continuous_visual_checkpoint(tmp_path):
    method = modules()
    state_path = tmp_path / "state.pt"
    action_path = tmp_path / "action.pt"
    torch.save(
        {
            "config": {"model": {"state_prior": {}}},
            "state": {"state_prior": {}, "visual_encoder": None},
        },
        state_path,
    )
    torch.save({"state": {}}, action_path)
    with pytest.raises(ValueError, match="Retrain VQ-VAE"):
        _load_priors(method, state_path, action_path)
