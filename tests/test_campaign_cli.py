import json
import os
from pathlib import Path

from conditional_node_field_graph_generator.campaign_cli import main


def _write_campaign_files(tmp_path):
    workflow_path = tmp_path / "workflow.json"
    workflow_path.write_text(
        json.dumps(
            {
                "experiment": {"name": "unit", "n_trials": 1, "random_state": 1, "verbose": 0},
                "dataset": {"num_graphs": 2, "min_size": 3, "max_size": 4, "random_state": 1},
                "model": {
                    "fixed": {"batch_size": 2},
                    "search_space": {
                        "trial_quality": {"type": "real", "low": 0.0, "high": 1.0}
                    },
                },
                "generation": {
                    "n_samples": 2,
                    "feasibility_effort": 1,
                    "feasibility_filter": "strict",
                },
                "outputs": {
                    "artifact_root": str(tmp_path / "artifact"),
                    "results_csv": "results.csv",
                },
            }
        )
    )
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(
        json.dumps(
            {
                "campaign": {"id": "molecules", "domain": "molecules", "prefix": "molecules"},
                "artifacts": {"root": str(tmp_path / "artifact")},
                "logbook": {"path": str(tmp_path / "LOGBOOK.md")},
                "runner": {"config_path": str(workflow_path)},
                "random_search": {"batch_size": 2, "random_state": 1},
                "agent": {
                    "allowed_paths": ["model.search_space.trial_quality"],
                    "max_search_leaf_count": 1,
                    "default_trial_patch_space": {
                        "model": {
                            "search_space": {
                                "trial_quality": {"type": "real", "low": 0.0, "high": 1.0}
                            }
                        }
                    },
                },
            }
        )
    )
    return campaign_path


def test_campaign_cli_list_and_status(capsys):
    assert main(["list"]) == 0
    output = capsys.readouterr().out
    assert "molecules-small" in output
    assert "molecules-large" in output
    assert "artificial-graphs-small" in output
    assert "artificial-graphs-large" in output

    assert main(["status", "molecules"]) == 0
    status_output = capsys.readouterr().out
    assert "campaign: molecules" in status_output
    assert "status:" in status_output


def test_campaign_cli_dry_run_with_config_override(tmp_path, capsys, monkeypatch):
    campaign_path = _write_campaign_files(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    assert main(["run", "molecules", "--dry-run", "--config", str(campaign_path)]) == 0
    output = capsys.readouterr().out

    assert "status: dry_run" in output
    assert "queued_trials: -" in output
    assert ".artifacts" not in output
    assert not any(Path(tmp_path / "artifact").glob("molecules/molecules_*"))
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_campaign_cli_internal_mini_batch_dry_run_samples_trials(tmp_path, capsys):
    campaign_path = _write_campaign_files(tmp_path)

    assert (
        main(
            [
                "run-mini-batch",
                "molecules",
                "--dry-run",
                "--config",
                str(campaign_path),
                "--run-timestamp",
                "20260625_091011",
                "--run-id",
                "dry001",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out

    assert "status: dry_run" in output
    assert "queued_trials: 2" in output
    assert "molecules_20260625_091011_dry001" in output


def test_campaign_cli_once_flag_is_accepted_for_compatibility(tmp_path, capsys):
    campaign_path = _write_campaign_files(tmp_path)

    assert main(["run", "molecules", "--once", "--dry-run", "--config", str(campaign_path)]) == 0
    output = capsys.readouterr().out

    assert "status: dry_run" in output


def test_campaign_cli_cuda_device_policy_can_be_requested(tmp_path, capsys, monkeypatch):
    campaign_path = _write_campaign_files(tmp_path)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    assert (
        main(
            [
                "run",
                "molecules",
                "--dry-run",
                "--device",
                "cuda",
                "--config",
                str(campaign_path),
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert "CUDA_VISIBLE_DEVICES" not in os.environ
