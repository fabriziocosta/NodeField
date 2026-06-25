from datetime import datetime
from pathlib import Path

from conditional_node_field_graph_generator.runtime_paths import (
    make_timestamped_run_dir,
    resolve_campaign_artifact_root,
    resolve_artifact_root,
    resolve_checkpoint_root,
    resolve_notebook_data_root,
    resolve_pubchem_data_root,
    resolve_repo_root,
    resolve_saved_generator_dir,
    resolve_zinc_data_root,
)


def test_runtime_paths_resolve_repo_and_artifact_roots_from_nested_start():
    start = Path(__file__).resolve().parent

    repo_root = resolve_repo_root(start)
    artifact_root = resolve_artifact_root(repo_root=repo_root)

    assert repo_root.name == "NodeField"
    assert artifact_root == repo_root / ".artifacts"
    assert resolve_notebook_data_root(repo_root=repo_root) == repo_root / "notebooks" / "datasets"
    assert resolve_checkpoint_root(repo_root=repo_root) == artifact_root / "checkpoints" / "node_field"
    assert resolve_saved_generator_dir(repo_root=repo_root) == artifact_root / "saved_generators"
    assert resolve_campaign_artifact_root(repo_root=repo_root) == repo_root / "artifact"


def test_runtime_paths_resolve_dataset_roots_with_explicit_overrides(tmp_path):
    assert resolve_pubchem_data_root(tmp_path / "PUBCHEM") == (tmp_path / "PUBCHEM").resolve()
    assert resolve_zinc_data_root(tmp_path / "zinc") == (tmp_path / "zinc").resolve()


def test_make_timestamped_run_dir_uses_campaign_prefix_and_short_id(tmp_path):
    run_dir = make_timestamped_run_dir(
        tmp_path,
        "molecules",
        now=datetime(2026, 6, 25, 9, 10, 11),
        short_id="abc123",
    )

    assert run_dir == (tmp_path / "molecules_20260625_091011_abc123").resolve()
    assert run_dir.is_dir()
