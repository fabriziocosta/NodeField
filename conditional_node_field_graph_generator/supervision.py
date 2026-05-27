"""Supervision planning helpers for the graph generator orchestrator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np

from .runtime_utils import verbose_log


@dataclass(frozen=True)
class SupervisionChannelPlan:
    """Description of how one prediction channel should be handled during training."""

    name: str
    mode: str
    reason: str
    constant_value: Optional[Any] = None
    horizon: Optional[int] = None
    enabled: bool = False


@dataclass(frozen=True)
class SupervisionPlan:
    """Single source of truth for supervision decisions in the orchestration layer."""

    node_labels: SupervisionChannelPlan
    edge_labels: SupervisionChannelPlan
    direct_edges: SupervisionChannelPlan
    auxiliary_locality: SupervisionChannelPlan

    def as_dict(self) -> Dict[str, SupervisionChannelPlan]:
        return {
            "node_labels": self.node_labels,
            "edge_labels": self.edge_labels,
            "direct_edges": self.direct_edges,
            "auxiliary_locality": self.auxiliary_locality,
        }


@dataclass
class SupervisionPlanner:
    owner: Any

    def build_supervision_plan(
        self,
        graphs: List[nx.Graph],
        node_label_targets: List[np.ndarray],
        edge_label_targets: Optional[np.ndarray],
    ) -> SupervisionPlan:
        del graphs

        flat_node_labels = [
            label
            for labels in node_label_targets
            for label in np.asarray(labels, dtype=object).tolist()
        ]
        if len(flat_node_labels) == 0:
            node_label_mode = "disabled"
            node_label_reason = "No node labels were provided."
            node_label_constant = None
        else:
            unique_node_labels = np.unique(np.asarray(flat_node_labels, dtype=object))
            if len(unique_node_labels) == 1:
                node_label_mode = "constant"
                node_label_reason = "All training nodes share one label."
                node_label_constant = unique_node_labels[0]
            else:
                node_label_mode = "learned"
                node_label_reason = f"{len(unique_node_labels)} node labels detected."
                node_label_constant = None

        if edge_label_targets is None:
            edge_label_mode = "disabled"
            edge_label_reason = "No usable edge labels were provided."
            edge_label_constant = None
        else:
            unique_edge_labels = np.unique(np.asarray(edge_label_targets, dtype=object))
            if len(unique_edge_labels) == 1:
                edge_label_mode = "constant"
                edge_label_reason = "All labelled edges share one label."
                edge_label_constant = unique_edge_labels[0]
            else:
                edge_label_mode = "learned"
                edge_label_reason = f"{len(unique_edge_labels)} edge labels detected."
                edge_label_constant = None

        aux_enabled = bool(self.owner.locality_horizon > 1)
        auxiliary_reason = (
            f"Use horizon-{self.owner.locality_horizon} locality as auxiliary regularization."
            if aux_enabled
            else "No auxiliary locality is needed when locality_horizon=1."
        )

        return SupervisionPlan(
            node_labels=SupervisionChannelPlan(
                name="node_labels",
                mode=node_label_mode,
                reason=node_label_reason,
                constant_value=node_label_constant,
                enabled=node_label_mode != "disabled",
            ),
            edge_labels=SupervisionChannelPlan(
                name="edge_labels",
                mode=edge_label_mode,
                reason=edge_label_reason,
                constant_value=edge_label_constant,
                enabled=edge_label_mode != "disabled",
            ),
            direct_edges=SupervisionChannelPlan(
                name="direct_edges",
                mode="learned",
                reason="Generator should learn horizon-1 edge presence for the decoder.",
                horizon=1,
                enabled=True,
            ),
            auxiliary_locality=SupervisionChannelPlan(
                name="auxiliary_locality",
                mode="learned" if aux_enabled else "disabled",
                reason=auxiliary_reason,
                horizon=self.owner.locality_horizon if aux_enabled else None,
                enabled=aux_enabled,
            ),
        )

    def log_supervision_plan(self, supervision_plan: SupervisionPlan) -> None:
        if not self.owner.verbose:
            return
        verbose_log(self.owner, "Supervision plan:")
        for channel in supervision_plan.as_dict().values():
            enabled_text = "enabled" if channel.enabled else "disabled"
            horizon_text = f", horizon={channel.horizon}" if channel.horizon is not None else ""
            verbose_log(
                self.owner,
                f"  {channel.name}: mode={channel.mode}, {enabled_text}{horizon_text}. {channel.reason}",
            )

    def plan_channel(self, channel_name: str) -> Optional[SupervisionChannelPlan]:
        plan = getattr(self.owner, "supervision_plan_", None)
        if plan is None:
            return None
        return getattr(plan, channel_name, None)
