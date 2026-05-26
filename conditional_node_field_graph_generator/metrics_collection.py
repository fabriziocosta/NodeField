"""Metric collection callbacks for Conditional Node Field training."""

from pathlib import Path
from typing import Dict
import time

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import networkx as nx
import pytorch_lightning as pl
import torch

from .runtime_utils import get_runtime_logger, verbose_log

logger = get_runtime_logger(__name__)


class MetricsLogger(pl.callbacks.Callback):
    """Collect end-of-epoch metrics into the module's history lists."""

    def on_fit_start(self, trainer, pl_module):
        pl_module._fit_start_time = time.time()
        pl_module._ema_metrics = {}

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total_seconds = max(0, int(round(float(seconds))))
        hours, remainder = divmod(total_seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"

    @staticmethod
    def _format_metric_value(value: float) -> str:
        magnitude = abs(float(value))
        if magnitude >= 1000.0:
            return f"{value:>9.1f}"
        if magnitude >= 100.0:
            return f"{value:>9.2f}"
        if magnitude >= 10.0:
            return f"{value:>9.3f}"
        if magnitude >= 1.0:
            return f"{value:>9.4f}"
        if magnitude == 0.0:
            return f"{value:>9.4f}"
        return f"{value:>9.5f}"

    @staticmethod
    def _component_summary(pl_module, metrics: Dict[str, torch.Tensor], prefix: str):
        component_specs = [
            ("node_field", "node_field", 1.0),
            ("deg", "deg_ce", float(getattr(pl_module, "lambda_degree_importance", 1.0))),
            ("exist", "exist", float(getattr(pl_module, "lambda_node_exist_importance", 1.0))),
            ("node_count", "node_count_loss", float(getattr(pl_module, "lambda_node_count_importance", 0.0))),
            ("node_label", "node_label_ce", float(getattr(pl_module, "lambda_node_label_importance", 1.0))),
            ("edge_label", "edge_label_ce", float(getattr(pl_module, "lambda_edge_label_importance", 1.0))),
            (
                "edge",
                "edge_ce",
                float(getattr(pl_module, "lambda_direct_edge_importance", 1.0)),
            ),
            ("edge_count", "edge_count_loss", float(getattr(pl_module, "lambda_edge_count_importance", 0.0))),
            (
                "deg_edge_consistency",
                "degree_edge_consistency_loss",
                float(getattr(pl_module, "lambda_degree_edge_consistency_importance", 0.0)),
            ),
            (
                "aux",
                "aux_locality_ce",
                float(getattr(pl_module, "lambda_auxiliary_edge_importance", 1.0)),
            ),
        ]

        components = []
        total = 0.0
        for label, metric_name, scale in component_specs:
            key = f"{prefix}_{metric_name}"
            if key not in metrics:
                continue
            raw_value = float(metrics[key].item())
            weighted_value = raw_value * scale
            total += weighted_value
            components.append((label, raw_value, weighted_value))

        if not components:
            return 0.0, [], None, 0.0

        denominator = total if total > 0 else 1.0
        dominant_label, *_rest, dominant_weighted = max(components, key=lambda item: item[2])
        normalized_components = [
            (label, raw, weighted, weighted / denominator)
            for label, raw, weighted in components
        ]
        return total, normalized_components, dominant_label, dominant_weighted / denominator

    @staticmethod
    def _update_ema_metric(trainer, pl_module, metric_name: str, metric_value: float) -> float:
        alpha = float(getattr(pl_module, "early_stopping_ema_alpha", 0.3))
        if not 0.0 < alpha <= 1.0:
            alpha = 0.3
        previous = pl_module._ema_metrics.get(metric_name)
        ema_value = metric_value if previous is None else alpha * metric_value + (1.0 - alpha) * previous
        pl_module._ema_metrics[metric_name] = float(ema_value)
        ema_key = f"{metric_name}_ema"
        ema_tensor = torch.tensor(float(ema_value), dtype=torch.float32)
        trainer.callback_metrics[ema_key] = ema_tensor
        if hasattr(trainer, "logged_metrics") and isinstance(trainer.logged_metrics, dict):
            trainer.logged_metrics[ema_key] = ema_tensor
        return float(ema_value)

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        pl_module.train_losses.append(m.get("train_total", torch.tensor(0.0)).item())
        pl_module.train_deg_ce.append(m.get("train_deg_ce", torch.tensor(0.0)).item())
        pl_module.train_node_field.append(
            m.get("train_node_field", torch.tensor(0.0)).item()
        )
        if hasattr(pl_module, "train_exist"):
            pl_module.train_exist.append(m.get("train_exist", torch.tensor(0.0)).item())
        if hasattr(pl_module, "train_node_label_ce"):
            pl_module.train_node_label_ce.append(m.get("train_node_label_ce", m.get("train_label_ce", torch.tensor(0.0))).item())
        elif hasattr(pl_module, "train_label_ce"):
            pl_module.train_label_ce.append(m.get("train_label_ce", m.get("train_node_label_ce", torch.tensor(0.0))).item())
        if hasattr(pl_module, "train_edge_label_ce"):
            pl_module.train_edge_label_ce.append(m.get("train_edge_label_ce", torch.tensor(0.0)).item())
        if getattr(pl_module, "use_locality_supervision", False):
            pl_module.train_edge_loss.append(m.get("train_edge_ce", m.get("train_edge_loss", torch.tensor(0.0))).item())
            pl_module.train_edge_acc.append(m.get("train_edge_acc", torch.tensor(0.0)).item())
        if getattr(pl_module, "use_auxiliary_locality_supervision", False):
            pl_module.train_aux_edge_loss.append(m.get("train_aux_locality_ce", m.get("train_aux_edge_loss", torch.tensor(0.0))).item())
            pl_module.train_aux_edge_acc.append(m.get("train_aux_edge_acc", torch.tensor(0.0)).item())

    def on_validation_epoch_start(self, trainer, pl_module):
        current_epoch = int(getattr(trainer, "current_epoch", -1)) + 1
        setattr(pl_module, "_validation_epoch_started_at", time.time())
        verbose_log(pl_module, f"epoch {current_epoch}: starting validation", level=2)

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        pl_module.val_losses.append(m.get("val_total", torch.tensor(0.0)).item())
        pl_module.val_deg_ce.append(m.get("val_deg_ce", torch.tensor(0.0)).item())
        pl_module.val_node_field.append(
            m.get("val_node_field", torch.tensor(0.0)).item()
        )
        if hasattr(pl_module, "val_exist"):
            pl_module.val_exist.append(m.get("val_exist", torch.tensor(0.0)).item())
        if hasattr(pl_module, "val_node_label_ce"):
            pl_module.val_node_label_ce.append(m.get("val_node_label_ce", m.get("val_label_ce", torch.tensor(0.0))).item())
        elif hasattr(pl_module, "val_label_ce"):
            pl_module.val_label_ce.append(m.get("val_label_ce", m.get("val_node_label_ce", torch.tensor(0.0))).item())
        if hasattr(pl_module, "val_edge_label_ce"):
            pl_module.val_edge_label_ce.append(m.get("val_edge_label_ce", torch.tensor(0.0)).item())
        if getattr(pl_module, "use_locality_supervision", False):
            pl_module.val_edge_loss.append(m.get("val_edge_ce", m.get("val_edge_loss", torch.tensor(0.0))).item())
            pl_module.val_edge_acc.append(m.get("val_edge_acc", torch.tensor(0.0)).item())
        if getattr(pl_module, "use_auxiliary_locality_supervision", False):
            pl_module.val_aux_edge_loss.append(m.get("val_aux_locality_ce", m.get("val_aux_edge_loss", torch.tensor(0.0))).item())
            pl_module.val_aux_edge_acc.append(m.get("val_aux_edge_acc", torch.tensor(0.0)).item())
        self._update_ema_metric(trainer, pl_module, "val_total", pl_module.val_losses[-1])
        self._update_ema_metric(
            trainer,
            pl_module,
            "val_node_field",
            pl_module.val_node_field[-1],
        )

        verbose_level = 0
        try:
            verbose_level = int(getattr(pl_module, "verbose", 0))
        except (TypeError, ValueError):
            verbose_level = 1 if getattr(pl_module, "verbose", False) else 0
        if verbose_level >= 2:
            started_at = getattr(pl_module, "_validation_epoch_started_at", None)
            if started_at is not None:
                verbose_log(
                    pl_module,
                    f"epoch {int(getattr(trainer, 'current_epoch', -1)) + 1}: finished validation in "
                    f"{max(0.0, time.time() - float(started_at)):.2f}s",
                    level=2,
                )
            interval = int(getattr(pl_module, "verbose_epoch_interval", 10))
            current_epoch = int(getattr(trainer, "current_epoch", -1)) + 1
            if interval > 0 and (current_epoch % interval == 0):
                train_total, train_components, train_dominant, train_dominant_share = self._component_summary(pl_module, m, "train")
                val_total, val_components, val_dominant, val_dominant_share = self._component_summary(pl_module, m, "val")
                max_epochs = getattr(trainer, "max_epochs", None)
                fit_start_time = getattr(pl_module, "_fit_start_time", None)
                eta_label = None
                if (
                    isinstance(max_epochs, int)
                    and max_epochs > 0
                    and fit_start_time is not None
                    and current_epoch > 0
                ):
                    elapsed_seconds = max(0.0, time.time() - float(fit_start_time))
                    average_epoch_seconds = elapsed_seconds / float(current_epoch)
                    remaining_epochs = max(0, max_epochs - current_epoch)
                    eta_seconds = remaining_epochs * average_epoch_seconds
                    eta_label = self._format_duration(eta_seconds)
                epoch_label = (
                    f"Epoch {current_epoch}/{max_epochs}"
                    if isinstance(max_epochs, int) and max_epochs > 0
                    else f"Epoch {current_epoch}"
                )
                if eta_label is not None:
                    epoch_label += f" | ETA {eta_label}"
                ordered_labels = []
                for label, *_ in train_components + val_components:
                    if label not in ordered_labels:
                        ordered_labels.append(label)

                def _components_to_map(components):
                    return {
                        label: (raw, weighted, share)
                        for label, raw, weighted, share in components
                    }

                train_map = _components_to_map(train_components)
                val_map = _components_to_map(val_components)

                def _format_row(prefix_label, total_value, component_map, dominant_label, dominant_share):
                    def _format_share(value: float) -> str:
                        if value <= 0:
                            return "0%"
                        if value < 0.001:
                            return "<0.1%"
                        return f"{value:.1%}"

                    first_row_width = 4
                    continuation_row_width = 5
                    chunks = []
                    if ordered_labels:
                        chunks.append(ordered_labels[:first_row_width])
                        remaining_labels = ordered_labels[first_row_width:]
                        chunks.extend(
                            remaining_labels[index:index + continuation_row_width]
                            for index in range(0, len(remaining_labels), continuation_row_width)
                        )
                    rows = []
                    total_prefix = f"{prefix_label:<5} total={MetricsLogger._format_metric_value(total_value)}"
                    continuation_prefix = " " * len(total_prefix)
                    for chunk_index, labels_chunk in enumerate(chunks):
                        row = f"{prefix_label:<5}"
                        if chunk_index == 0:
                            row += total_prefix[len(f"{prefix_label:<5}"):]
                        else:
                            row += continuation_prefix[len(f"{prefix_label:<5}"):]
                        for label in labels_chunk:
                            if label in component_map:
                                _, weighted, share = component_map[label]
                                row += (
                                    f" | {label:>10} "
                                    f"{MetricsLogger._format_metric_value(weighted)} "
                                    f"[{_format_share(share)}]"
                                )
                            else:
                                row += f" | {label:>10} {'-':>9} [{' - '}]"
                        rows.append(row)
                    if not rows:
                        rows.append(total_prefix)
                    if dominant_label is not None:
                        rows[-1] += f" | dominant={dominant_label} [{_format_share(dominant_share)}]"
                    return rows

                logger.info("%s:", epoch_label)
                train_rows = _format_row("train", train_total, train_map, train_dominant, train_dominant_share)
                val_rows = _format_row("val", val_total, val_map, val_dominant, val_dominant_share)
                block_count = max(len(train_rows), len(val_rows))
                for block_index in range(block_count):
                    if block_index < len(train_rows):
                        logger.info("  %s", train_rows[block_index])
                    if block_index < len(val_rows):
                        logger.info("  %s", val_rows[block_index])


class GraphGeneratorEpochSnapshotCallback(pl.callbacks.Callback):
    """Persist a usable full graph-generator snapshot after each completed epoch."""

    def __init__(self, owner_graph_generator):
        self.owner_graph_generator = owner_graph_generator

    def on_validation_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "sanity_checking", False):
            return
        if hasattr(trainer, "is_global_zero") and not bool(trainer.is_global_zero):
            return
        owner = self.owner_graph_generator
        model_name = getattr(owner, "model_name", None)
        if model_name is None:
            return
        from .persistence import save_graph_generator

        epoch_label = int(getattr(trainer, "current_epoch", -1)) + 1
        previous_fit_state = bool(getattr(owner, "is_fitted_", False))
        owner.is_fitted_ = True
        try:
            verbose_log(owner, f"epoch {epoch_label}: saving generator snapshot", level=2)
            snapshot_started_at = time.time()
            save_graph_generator(
                owner,
                model_name=model_name,
                model_dir=getattr(owner, "model_dir", None),
                log=False,
            )
            verbose_log(
                owner,
                f"epoch {epoch_label}: finished generator snapshot in {max(0.0, time.time() - snapshot_started_at):.2f}s",
                level=2,
            )
        finally:
            owner.is_fitted_ = previous_fit_state


class GraphGeneratorTrainingSampleCallback(pl.callbacks.Callback):
    """Render direct-vs-ILP generation samples after selected validation epochs."""

    def __init__(
        self,
        owner_graph_generator,
        *,
        n_samples: int,
        every_n_epochs: int,
        output_path,
    ):
        if int(n_samples) < 1:
            raise ValueError("sample_training_progress_n_samples must be >= 1.")
        if int(every_n_epochs) < 1:
            raise ValueError("sample_training_progress_every_n_epochs must be >= 1.")
        self.owner_graph_generator = owner_graph_generator
        self.n_samples = int(n_samples)
        self.every_n_epochs = int(every_n_epochs)
        self.output_path = Path(output_path)
        self.epoch_samples = []

    @staticmethod
    def _node_color(label):
        if label is None:
            return "#d1d5db"
        label_hash = abs(hash(str(label)))
        cmap = plt.get_cmap("tab20")
        return cmap(label_hash % 20)

    def _draw_graph(self, ax, graph, title):
        ax.axis("off")
        ax.set_title(str(title), fontsize=9)
        if graph is None:
            ax.text(0.5, 0.5, "None", ha="center", va="center")
            return
        if graph.number_of_nodes() == 0:
            ax.text(0.5, 0.5, "empty", ha="center", va="center")
            return
        pos = nx.kamada_kawai_layout(graph)
        labels = {
            node: str(attrs.get("label", ""))
            for node, attrs in graph.nodes(data=True)
        }
        node_colors = [
            self._node_color(attrs.get("label"))
            for _, attrs in graph.nodes(data=True)
        ]
        nx.draw_networkx_edges(graph, pos, width=1.5, ax=ax)
        nx.draw_networkx_nodes(
            graph,
            pos,
            ax=ax,
            node_color=node_colors,
            edgecolors="black",
            linewidths=1.2,
            node_size=420,
        )
        if any(labels.values()):
            nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8, ax=ax)

    def _write_pdf(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        n_rows = max(1, 2 * len(self.epoch_samples))
        n_cols = max(1, self.n_samples)
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(3.0 * n_cols, 2.8 * n_rows),
            squeeze=False,
        )
        for row_idx, (epoch_label, mode_label, graphs) in enumerate(
            row
            for epoch_record in self.epoch_samples
            for row in (
                (epoch_record["epoch"], "direct", epoch_record["direct"]),
                (epoch_record["epoch"], "ILP", epoch_record["ilp"]),
            )
        ):
            for col_idx in range(n_cols):
                graph = graphs[col_idx] if col_idx < len(graphs) else None
                title = f"epoch {epoch_label} {mode_label} #{col_idx + 1}"
                self._draw_graph(axes[row_idx, col_idx], graph, title)
        fig.tight_layout()
        with PdfPages(self.output_path) as pdf:
            pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

    def on_validation_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "sanity_checking", False):
            return
        if hasattr(trainer, "is_global_zero") and not bool(trainer.is_global_zero):
            return
        epoch_label = int(getattr(trainer, "current_epoch", -1)) + 1
        if epoch_label < 1 or (epoch_label % self.every_n_epochs) != 0:
            return
        owner = self.owner_graph_generator
        previous_fit_state = bool(getattr(owner, "is_fitted_", False))
        owner.is_fitted_ = True
        try:
            verbose_log(
                owner,
                f"epoch {epoch_label}: sampling training progress graphs",
                level=2,
            )
            direct_graphs = owner.sample(
                n_samples=self.n_samples,
                apply_feasibility_filtering=False,
                use_ilp_decoder=False,
            )
            ilp_graphs = owner.sample(
                n_samples=self.n_samples,
                apply_feasibility_filtering=False,
                use_ilp_decoder=True,
            )
            self.epoch_samples.append(
                {
                    "epoch": epoch_label,
                    "direct": list(direct_graphs),
                    "ilp": list(ilp_graphs),
                }
            )
            try:
                self._write_pdf()
            except Exception as exc:
                logger.warning("Unable to render training sample PDF: %s", exc)
            else:
                verbose_log(
                    owner,
                    f"epoch {epoch_label}: wrote training sample PDF to {self.output_path}",
                    level=2,
                )
        finally:
            owner.is_fitted_ = previous_fit_state


class GraphGeneratorBatchSnapshotCallback(pl.callbacks.Callback):
    """Persist a usable full graph-generator snapshot after each completed train batch."""

    def __init__(self, owner_graph_generator):
        self.owner_graph_generator = owner_graph_generator

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if getattr(trainer, "sanity_checking", False):
            return
        if hasattr(trainer, "is_global_zero") and not bool(trainer.is_global_zero):
            return
        owner = self.owner_graph_generator
        model_name = getattr(owner, "model_name", None)
        if model_name is None:
            return
        snapshot_every_n_batches = max(1, int(getattr(owner, "stream_snapshot_every_n_batches", 10)))
        if ((int(batch_idx) + 1) % snapshot_every_n_batches) != 0:
            return
        from .persistence import save_graph_generator

        previous_fit_state = bool(getattr(owner, "is_fitted_", False))
        owner.is_fitted_ = True
        try:
            verbose_log(
                owner,
                f"epoch {int(getattr(trainer, 'current_epoch', -1)) + 1}: saving batch snapshot after batch {int(batch_idx) + 1}",
                level=2,
            )
            snapshot_started_at = time.time()
            save_graph_generator(
                owner,
                model_name=model_name,
                model_dir=getattr(owner, "model_dir", None),
                log=False,
            )
            verbose_log(
                owner,
                f"epoch {int(getattr(trainer, 'current_epoch', -1)) + 1}: finished batch snapshot in {max(0.0, time.time() - snapshot_started_at):.2f}s",
                level=2,
            )
        finally:
            owner.is_fitted_ = previous_fit_state


class GraphGeneratorBatchAndEpochSnapshotCallback(pl.callbacks.Callback):
    """Persist streaming snapshots on both batch cadence and validation epoch end."""

    def __init__(self, owner_graph_generator):
        self._batch_callback = GraphGeneratorBatchSnapshotCallback(owner_graph_generator)
        self._epoch_callback = GraphGeneratorEpochSnapshotCallback(owner_graph_generator)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._batch_callback.on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._epoch_callback.on_validation_epoch_end(trainer, pl_module)
