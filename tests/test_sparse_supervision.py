import pytest
import torch

from conditional_node_field_graph_generator.conditional_node_field_generator import (
    ConditionalNodeFieldGenerator,
    ConditionalNodeFieldModule,
)


def test_node_field_score_mask_expands_presence_mask_without_sparse_supervision():
    module = ConditionalNodeFieldModule(
        number_of_rows_per_example=2,
        input_feature_dimension=3,
        condition_feature_dimension=3,
        latent_embedding_dimension=4,
        number_of_transformer_layers=1,
        transformer_attention_head_count=1,
        max_degree=1,
    )
    input_examples = torch.zeros((1, 2, 3), dtype=torch.float32)
    node_presence_mask = torch.tensor([[True, False]])

    score_mask = module._build_node_field_score_mask(
        input_examples,
        node_presence_mask,
        apply_sparse_supervision=False,
    )

    expected = torch.tensor([[[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]])
    torch.testing.assert_close(score_mask, expected)


def test_node_field_sparse_score_mask_keeps_at_least_one_valid_coordinate(monkeypatch):
    module = ConditionalNodeFieldModule(
        number_of_rows_per_example=2,
        input_feature_dimension=3,
        condition_feature_dimension=3,
        latent_embedding_dimension=4,
        number_of_transformer_layers=1,
        transformer_attention_head_count=1,
        max_degree=1,
        sparse_supervision_mask_ratio=0.9,
    )
    input_examples = torch.zeros((1, 2, 3), dtype=torch.float32)
    node_presence_mask = torch.tensor([[True, False]])
    monkeypatch.setattr(torch, "rand_like", lambda tensor: torch.ones_like(tensor))

    score_mask = module._build_node_field_score_mask(
        input_examples,
        node_presence_mask,
        apply_sparse_supervision=True,
    )

    assert score_mask[:, 0, :].sum().item() == pytest.approx(1.0)
    assert score_mask[:, 1, :].sum().item() == pytest.approx(0.0)


def test_sparse_supervision_mask_ratio_must_be_less_than_one():
    with pytest.raises(ValueError, match="sparse_supervision_mask_ratio"):
        ConditionalNodeFieldModule(
            number_of_rows_per_example=2,
            input_feature_dimension=3,
            condition_feature_dimension=3,
            latent_embedding_dimension=4,
            number_of_transformer_layers=1,
            transformer_attention_head_count=1,
            max_degree=1,
            sparse_supervision_mask_ratio=1.0,
        )


def test_generator_accepts_sparse_supervision_mask_ratio():
    generator = ConditionalNodeFieldGenerator(sparse_supervision_mask_ratio=0.8)

    assert generator.sparse_supervision_mask_ratio == pytest.approx(0.8)
