"""Compatibility shim for notebook bootstrap helpers."""

from __future__ import annotations

import warnings

from conditional_node_field_graph_generator.notebooks import (
    configure_notebook,
    ensure_nsppk_on_syspath,
    ensure_repo_on_syspath,
    find_repo_root,
    import_nsppk,
)

warnings.warn(
    "_notebook_bootstrap is deprecated; import notebook helpers from "
    "'conditional_node_field_graph_generator.notebooks' instead.",
    DeprecationWarning,
    stacklevel=2,
)
