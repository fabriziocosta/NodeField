from pathlib import Path

import pytest
import torch

from conditional_node_field_graph_generator.conditional_node_field_generator import (
    ConditionalNodeFieldGenerator,
    ConditionalNodeFieldModule,
)
from conditional_node_field_graph_generator.recurrent_interventions import (
    RecurrentIntervention,
    apply_recurrent_intervention,
)


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def model(**kw):
    args = dict(
        number_of_rows_per_example=3,
        input_feature_dimension=3,
        condition_feature_dimension=2,
        latent_embedding_dimension=8,
        number_of_transformer_layers=1,
        transformer_attention_head_count=2,
        transformer_dropout=0.0,
        max_degree=2,
        node_field_mode="recurrent_energy",
        recurrent_training_steps=3,
    )
    args.update(kw)
    return ConditionalNodeFieldModule(**args)


def inputs():
    return (
        torch.randn(2, 3, 3),
        torch.randn(2, 2),
        torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool),
        torch.ones(2, 3, dtype=torch.long),
    )


def test_baseline_mode_matches_existing_behavior():
    reference = torch.load(
        Path(__file__).parent / "fixtures/nodefield_baseline.pt", weights_only=True
    )
    for mode in ({}, dict(node_field_mode="baseline")):
        torch.manual_seed(1701)
        m = ConditionalNodeFieldModule(**reference["kwargs"], **mode).eval()
        assert m.state_dict().keys() == reference["state"].keys()
        for name, value in m.state_dict().items():
            torch.testing.assert_close(value, reference["state"][name], rtol=0, atol=0)
        torch.manual_seed(99)
        losses, _ = m._node_field_loss(
            reference["x"],
            reference["c"],
            reference["mask"],
            reference["deg"],
            reference["labels"],
            create_graph=True,
        )
        losses["total"].backward()
        for key, value in losses.items():
            torch.testing.assert_close(value, reference["loss"][key])
        for key, p in m.named_parameters():
            if p.grad is not None:
                torch.testing.assert_close(p.grad, reference["grads"][key])
        score, phi, _ = m._compute_score_field(
            reference["x"].detach().requires_grad_(True),
            reference["c"],
            reference["mask"],
            create_graph=False,
        )
        torch.testing.assert_close(score, reference["score"])
        torch.testing.assert_close(phi, reference["phi"])
        torch.manual_seed(100)
        sample = m.generate(reference["c"], total_steps=2, use_heads_projection=True)
        torch.testing.assert_close(sample, reference["sample"])
        for key, value in reference["heads"].items():
            if value is not None:
                torch.testing.assert_close(getattr(m, key), value)


def test_recurrent_hidden_shape():
    m = model(recurrent_hidden_dimension=5)
    assert m._initialize_recurrent_state(2, 3, torch.device("cpu"), torch.float64).shape == (
        2,
        3,
        5,
    )


def test_recurrent_hidden_padded_nodes_zero():
    m = model()
    x, c, mask, _ = inputs()
    h = m._initialize_recurrent_state(2, 3, x.device, x.dtype)
    for _ in range(3):
        _, _, _, h = m._compute_recurrent_score_field(
            x.detach().requires_grad_(True), h, c, mask, create_graph=True
        )
        assert torch.count_nonzero(h[~mask]) == 0


def test_recurrent_parameters_shared_across_steps():
    m = model()
    ids = [id(p) for p in m.parameters()]
    m.generate(torch.randn(2, 2), total_steps=5)
    assert ids == [id(p) for p in m.parameters()]


def test_score_gradient_only_wrt_x():
    m = model().double().eval()
    x = torch.randn(1, 3, 3, dtype=torch.float64, requires_grad=True)
    c = torch.randn(1, 2, dtype=torch.float64)
    h = torch.randn(1, 3, 8, dtype=torch.float64, requires_grad=True)
    score, phi, _, _ = m._compute_recurrent_score_field(x, h, c, create_graph=True)
    delta = torch.zeros_like(x)
    delta[0, 1, 2] = 1e-5
    plus = m.recurrent_readout(x + delta, h, c)["phi"]
    minus = m.recurrent_readout(x - delta, h, c)["phi"]
    torch.testing.assert_close(score[0, 1, 2], -(plus - minus).sum() / 2e-5, rtol=1e-4, atol=1e-7)
    assert h.grad is None
    score.square().sum().backward()
    assert m.recurrent_hidden_projection.weight.grad.abs().sum() > 0


def test_recurrent_generate_shapes():
    m = model()
    assert m.generate(torch.randn(2, 2), total_steps=2).shape == (2, 3, 3)
    m.generate(torch.randn(2, 2), total_steps=2, use_heads_projection=True)
    assert m._last_node_presence_mask.shape == (2, 3)


def test_hidden_reset_intervention():
    m = model()
    _, t = m.generate_recurrent(
        torch.randn(2, 2),
        total_steps=3,
        intervention=RecurrentIntervention("reset_hidden", step=1),
        return_trajectory=True,
    )
    assert torch.count_nonzero(t.evaluated_h[1]) == 0
    torch.testing.assert_close(t.evaluated_x[1], t.x[1])


def test_hidden_shuffle_intervention():
    x = torch.randn(2, 5, 3)
    h = torch.arange(2 * 5 * 2).reshape(2, 5, 2).float()
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
    h[~mask] = 0
    item = RecurrentIntervention("shuffle_hidden_nodes", step=1, seed=4)
    xx, hh = apply_recurrent_intervention(x, h, item, 1, mask)
    torch.testing.assert_close(xx, x)
    for b in range(2):
        torch.testing.assert_close(hh[b].sort(dim=0).values, h[b].sort(dim=0).values)
    assert torch.count_nonzero(hh[~mask]) == 0
    assert not torch.equal(hh, h)


def test_fresh_x_intervention():
    m = model()
    c = torch.randn(2, 2)
    torch.manual_seed(44)
    _, normal = m.generate_recurrent(c, total_steps=3, return_trajectory=True)
    torch.manual_seed(44)
    _, changed = m.generate_recurrent(
        c,
        total_steps=3,
        intervention=RecurrentIntervention("fresh_x_noise", step=1, seed=8),
        return_trajectory=True,
    )
    torch.testing.assert_close(normal.evaluated_h[1], changed.evaluated_h[1])
    assert not torch.equal(normal.evaluated_x[1], changed.evaluated_x[1])


def test_sigma_schedule_monotonic():
    m = model()
    s = m._build_recurrent_sigma_schedule(8, "cpu", torch.float32)
    assert torch.all(s[:-1] > s[1:])
    assert s[0].item() == pytest.approx(0.2)
    assert s[-1].item() == pytest.approx(0.02)
    assert m._build_recurrent_sigma_schedule(1, "cpu", torch.float32).item() == pytest.approx(0.2)


def test_constant_sigma_schedule():
    m = model(recurrent_corruption_schedule="constant")
    assert torch.all(m._build_recurrent_sigma_schedule(4, "cpu", torch.float32) == 0.2)


def test_recurrent_trajectory_capture():
    m = model()
    c = torch.randn(2, 2)
    torch.manual_seed(1)
    plain = m.generate(c, total_steps=3)
    torch.manual_seed(1)
    x, t = m.generate_recurrent(c, total_steps=3, return_trajectory=True)
    torch.testing.assert_close(plain, x)
    assert len(t.x) == len(t.h) == 4 and len(t.score) == len(t.phi) == len(t.diagnostics) == 3
    assert all(not v.requires_grad and v.device.type == "cpu" for v in t.x + t.h + t.score + t.phi)
    assert t.diagnostics[0]["cosine_score_consecutive"] is None


@pytest.mark.parametrize("schedule", ["annealed", "constant", "none"])
@pytest.mark.parametrize("all_steps", [True, False])
def test_recurrent_loss_gradients(schedule, all_steps):
    m = model(recurrent_corruption_schedule=schedule, recurrent_supervise_all_steps=all_steps)
    x, c, mask, deg = inputs()
    loss, _ = m._recurrent_node_field_loss(x, c, mask, deg, create_graph=True)
    loss["total"].backward()
    assert torch.isfinite(loss["total"])
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    assert m.recurrent_state_head[-1].weight.grad.abs().sum() > 0
    if schedule == "none":
        assert loss["node_field"] == 0


def test_readout_does_not_advance_memory():
    m = model().eval()
    x, c, mask, _ = inputs()
    h = torch.randn(2, 3, 8)
    before = h.clone()
    a = m.recurrent_readout(x, h, c, mask)
    b = m.recurrent_readout(x, h, c, mask)
    torch.testing.assert_close(h, before)
    torch.testing.assert_close(a["phi"], b["phi"])
    score, _, _, _ = m._compute_recurrent_score_field(x, h, c, mask, create_graph=False)
    changed, _, _, _ = m._compute_recurrent_score_field(
        x, h + torch.randn_like(h), c, mask, create_graph=False
    )
    assert not torch.allclose(score, changed)


def test_cfg_and_classifier_separate():
    m = model()
    c = torch.randn(2, 2)
    with pytest.raises(ValueError):
        m.generate_recurrent(
            c, global_condition_unconditional=c, classifier_guidance_fn=lambda x: x
        )
    x, t = m.generate_recurrent(
        c, total_steps=2, global_condition_unconditional=torch.zeros_like(c), return_trajectory=True
    )
    assert t.metadata["field_evaluations"] == 4
    assert torch.isfinite(x).all()


@pytest.mark.parametrize(
    "kw",
    [
        {"node_field_mode": "bad"},
        {"recurrent_training_steps": 0},
        {"recurrent_hidden_dimension": 0},
        {"recurrent_detach_interval": 0},
        {"recurrent_loss_discount": 0},
        {"recurrent_sigma_min": 0.3},
        {"recurrent_initial_state": "learned"},
    ],
)
def test_validation(kw):
    with pytest.raises(ValueError):
        model(**kw)


def test_public_generator_options():
    g = ConditionalNodeFieldGenerator(
        node_field_mode="recurrent_energy", recurrent_hidden_dimension=17
    )
    assert g.recurrent_hidden_dimension == 17


def test_partial_score_with_shared_x_hidden_ancestry():
    m = model().double().eval()
    x = torch.randn(1, 3, 3, dtype=torch.double, requires_grad=True)
    c = torch.randn(1, 2, dtype=torch.double)
    h = torch.cat([x, x, x[..., :2]], -1)
    score, _, _, _ = m._compute_recurrent_score_field(x, h, c, create_graph=True)
    reference, _, _, _ = m._compute_recurrent_score_field(x, h.detach(), c, create_graph=True)
    torch.testing.assert_close(score, reference)


def test_detach_interval_controls_hidden_history():
    for interval, reaches_first in [(None, True), (1, False), (2, True)]:
        m = model(recurrent_detach_interval=interval, recurrent_supervise_all_steps=False)
        states = []
        original = m._update_recurrent_hidden

        def capture(*args, **kwargs):
            h = original(*args, **kwargs)
            h.retain_grad()
            states.append(h)
            return h

        m._update_recurrent_hidden = capture
        x, c, mask, degree = inputs()
        loss, _ = m._recurrent_node_field_loss(x, c, mask, degree, create_graph=True)
        loss["total"].backward()
        assert (states[0].grad is not None) == (reaches_first and interval is None)
        # For interval 2 the boundary just before the final step truncates both earlier updates.


def test_discounted_aggregation():
    m = model(recurrent_loss_discount=0.5)
    original = m._node_structural_losses
    values = []

    def capture(*args):
        loss = original(*args)
        values.append(loss["total"])
        return loss

    m._node_structural_losses = capture
    x, c, mask, degree = inputs()
    loss, _ = m._recurrent_node_field_loss(x, c, mask, degree, create_graph=True)
    torch.testing.assert_close(
        loss["total"], sum(w * v for w, v in zip([0.25, 0.5, 1.0], values)) / 1.75
    )


def test_combined_reset_and_rng_isolation():
    m = model(langevin_noise_scale=0.1)
    c = torch.randn(2, 2)
    items = [
        RecurrentIntervention("fresh_x_noise_every_step", seed=5),
        RecurrentIntervention("reset_hidden", every_step=True),
    ]
    torch.manual_seed(10)
    _, t = m.generate_recurrent(c, total_steps=2, intervention=items, return_trajectory=True)
    after = torch.get_rng_state()
    torch.manual_seed(10)
    m.generate_recurrent(c, total_steps=2)
    torch.testing.assert_close(after, torch.get_rng_state())
    assert all(torch.count_nonzero(h) == 0 for h in t.evaluated_h)


def test_checkpoint_round_trip(tmp_path):
    m = model().eval()
    path = tmp_path / "model.pt"
    torch.save(m.state_dict(), path)
    restored = model().eval()
    restored.load_state_dict(torch.load(path, weights_only=True))
    c = torch.randn(2, 2)
    torch.manual_seed(7)
    x = m.generate(c, total_steps=2)
    torch.manual_seed(7)
    torch.testing.assert_close(x, restored.generate(c, total_steps=2))


def test_all_structural_heads_train_at_each_step():
    m = model(
        use_locality_supervision=True,
        use_auxiliary_locality_supervision=True,
        use_edge_label_head=True,
        num_edge_label_classes=2,
        use_node_label_head=True,
        num_node_label_classes=2,
        lambda_edge_count_importance=0.5,
        lambda_node_count_importance=0.5,
        lambda_degree_edge_consistency_importance=0.5,
        node_count_condition_index=0,
        edge_count_condition_index=1,
    )
    x, c, mask, degree = inputs()
    c = torch.tensor([[2.0, 1.0], [3.0, 2.0]])
    pairs = torch.tensor([[0, 0, 1], [1, 0, 1], [1, 1, 2]])
    targets = torch.ones(3)
    labels = torch.zeros(2, 3, dtype=torch.long)
    losses, _ = m._recurrent_node_field_loss(
        x,
        c,
        mask,
        degree,
        labels,
        create_graph=True,
        pair_targets=(pairs, targets, pairs, targets.long(), pairs, targets),
    )
    losses["total"].backward()
    for head in (
        m.edge_head,
        m.edge_label_head,
        m.auxiliary_edge_head,
        m.degree_head,
        m.exist_head,
        m.node_label_head,
    ):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
    assert {"edge_count_loss", "node_count_loss", "degree_edge_consistency_loss"} <= losses.keys()
