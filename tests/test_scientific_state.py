import pytest

from conditional_node_field_graph_generator.scientific_state import (
    ScientificStateRepository,
    apply_operations,
    new_state,
    validate_candidate,
)


def test_state_repository_round_trips_and_applies_typed_operations(tmp_path):
    repository = ScientificStateRepository(tmp_path / "state.yaml")
    repository.initialize(new_state(campaign_id="unit", domain="molecules"))
    repository.apply_operations(
        [
            {
                "operation": "create_entity",
                "entity_type": "hypotheses",
                "entity_id": "hyp_0001",
                "value": {"title": "A test hypothesis", "status": "active"},
            }
        ]
    )
    state = repository.load()
    assert state["schema_version"] == 1
    assert state["entities"]["hypotheses"]["hyp_0001"]["title"] == "A test hypothesis"

    repository.apply_operations(
        [
            {
                "operation": "update_entity",
                "entity_type": "hypotheses",
                "entity_id": "hyp_0001",
                "path": "status",
                "old_value": "active",
                "new_value": "supported",
            }
        ]
    )
    assert repository.load()["entities"]["hypotheses"]["hyp_0001"]["status"] == "supported"


def test_state_rejects_stale_updates_and_historical_edits():
    state = new_state(campaign_id="unit", domain="molecules")
    state["entities"]["experiments"]["exp_0001"] = {"status": "completed"}
    with pytest.raises(ValueError, match="Historical entity collection"):
        apply_operations(
            state,
            [
                {
                    "operation": "update_entity",
                    "entity_type": "experiments",
                    "entity_id": "exp_0001",
                    "path": "status",
                    "old_value": "completed",
                    "new_value": "failed",
                }
            ],
        )


def test_candidate_validation_enforces_budget_and_allowlist():
    candidate = {
        "design": {
            "fixed": {"model": {"fixed": {}}},
            "varied": {"model": {"fixed": {"learning_rate": {"type": "real", "low": 1e-4, "high": 2e-4}}}},
            "seeds": [1, 2],
        },
        "expected_outcomes": {"if_hypothesis": {"ranking": ["low", "high"]}},
        "cost": {"estimated_gpu_hours": 2.0},
        "risks": [{"name": "seed_variance", "mitigation": "paired seeds"}],
    }
    validate_candidate(
        candidate,
        allowed_paths=["model.fixed"],
        remaining_gpu_hours=3.0,
        maximum_single_experiment_gpu_hours=4.0,
    )
    with pytest.raises(ValueError, match="remaining campaign budget"):
        validate_candidate(
            candidate,
            allowed_paths=["model.fixed"],
            remaining_gpu_hours=1.0,
            maximum_single_experiment_gpu_hours=4.0,
        )
