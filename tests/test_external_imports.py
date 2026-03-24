from pathlib import Path

from _external_imports import build_optional_dependency_candidates, resolve_source_checkout


def test_build_optional_dependency_candidates_includes_workspace_roots():
    candidates = build_optional_dependency_candidates()
    repo_root = Path(__file__).resolve().parents[1]

    assert candidates
    assert all(isinstance(path, Path) for path in candidates)
    assert any(path.name == "NodeField" for path in candidates)
    assert repo_root.resolve() in candidates
    assert repo_root.parent.resolve() in candidates
    assert any(path.name == "repos" for path in candidates)
    assert any(path.name == "abstractgraph-ecosystem" for path in candidates)


def test_resolve_source_checkout_uses_supplied_candidate_bases(tmp_path):
    base = tmp_path / "workspace"
    target = base / "repos" / "abstractgraph" / "src" / "abstractgraph"
    target.mkdir(parents=True)

    path = resolve_source_checkout(
        "repos/abstractgraph/src/abstractgraph",
        candidate_bases=[base],
    )

    assert path == target.resolve()
