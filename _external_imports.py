"""Helpers for resolving local source checkouts of optional dependencies."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def _dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def build_optional_dependency_candidates() -> list[Path]:
    """Return plausible local workspace roots for sibling source checkouts."""
    repo_root = Path(__file__).resolve().parent
    repo_parent = repo_root.parent
    candidates = [
        repo_parent,
        repo_parent / "abstractgraph-ecosystem" / "repos",
        repo_parent / "abstractgraph_ecosystem" / "repos",
        repo_parent / "abstractgraph-ecosystem",
        repo_parent / "abstractgraph_ecosystem",
        repo_root,
    ]
    return _dedupe_paths(candidates)


def resolve_source_checkout(*relative_roots: str, candidate_bases: Iterable[Path] | None = None) -> Path | None:
    """Resolve the first existing path under the known local workspace roots."""
    bases = build_optional_dependency_candidates() if candidate_bases is None else _dedupe_paths(candidate_bases)
    for base in bases:
        for relative_root in relative_roots:
            candidate = base / relative_root
            if candidate.exists():
                return candidate
    return None
