import json

from conditional_node_field_graph_generator.extensions.demo.campaign_dashboard import (
    _build_payload,
    discover_latest_campaign_state,
    render_graphviz,
)


def _state():
    entities = {
        name: {}
        for name in (
            "components",
            "datasets",
            "experiments",
            "observations",
            "hypotheses",
            "beliefs",
            "questions",
            "candidate_experiments",
        )
    }
    entities["experiments"]["exp_0001"] = {
        "status": "completed",
        "purpose": {"description": "Test experiment"},
        "execution": {"run_dir": "run-1"},
    }
    entities["observations"]["obs_0001"] = {
        "type": "validation_plateau",
        "statement": "Validation stopped improving.",
        "source_experiment": "exp_0001",
    }
    entities["hypotheses"]["hyp_0001"] = {
        "title": "A lower learning rate may help.",
        "status": "active",
    }
    entities["candidate_experiments"]["candidate_0001"] = {
        "status": "approved",
        "design": {"fixed": {}, "varied": {}},
        "cost": {"estimated_gpu_hours": 2},
    }
    return {
        "schema_version": 1,
        "project": {
            "id": "molecules_small",
            "domain": "molecules",
            "objective": "Improve validity",
            "primary_metric": "average_num_violations",
        },
        "entities": entities,
        "relations": [{"type": "supports", "source": "obs_0001", "target": "hyp_0001"}],
        "controller_state": {
            "active_run": {"run_dir": "run-1", "candidate_experiment_id": "candidate_0001"},
            "budgets": {"remaining_gpu_hours": 12},
        },
    }


def test_dashboard_discovers_latest_state_and_uses_stable_short_ids(tmp_path):
    state_dir = tmp_path / "artifact" / "molecules"
    state_dir.mkdir(parents=True)
    state_path = state_dir / "molecules_small_state.yaml"
    state_path.write_text(json.dumps(_state()))
    (state_dir / "molecules_small_campaign_state.json").write_text(
        json.dumps({"status": "running", "run_dir": "run-1"})
    )

    found_path, state, campaign_state = discover_latest_campaign_state(tmp_path)
    assert found_path == state_path
    payload = _build_payload(found_path, state, campaign_state)
    assert payload["current_experiment"]["short_id"] == "E001"
    assert payload["latest_observations"][0]["short_id"] == "O001"
    assert payload["active_hypotheses"][0]["short_id"] == "H001"
    assert payload["current_candidate"]["short_id"] == "C001"


def test_dashboard_graph_uses_graphviz_and_short_labels(tmp_path):
    state_path = tmp_path / "state.yaml"
    state_path.write_text(json.dumps(_state()))
    svg = render_graphviz(_state(), current_key="exp_0001")
    assert "<svg" in svg
    assert "E001" in svg
    assert "O001" in svg
