"""Evaluation and visualization helpers for deliberate conditional overfitting."""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import random
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np

from ...conditional_node_field_generator import GraphConditioningBatch


RECONSTRUCTION_ERROR_METRICS = (
    "node_label_hist_l1",
    "degree_hist_l1",
    "edge_label_hist_l1",
)


@contextmanager
def _seeded_generation(seed: int):
    """Make paired stochastic decodes use the same initial random states."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch = None
    torch_state = None
    try:
        import torch as _torch

        torch = _torch
        torch_state = torch.random.get_rng_state()
    except ImportError:
        pass
    random.seed(int(seed))
    np.random.seed(int(seed))
    if torch is not None:
        torch.manual_seed(int(seed))
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if torch is not None and torch_state is not None:
            torch.random.set_rng_state(torch_state)


def same_labeled_graph(reference: nx.Graph, candidate: nx.Graph) -> bool:
    return nx.is_isomorphic(
        reference,
        candidate,
        node_match=lambda left, right: left.get("label") == right.get("label"),
        edge_match=lambda left, right: left.get("label") == right.get("label"),
    )


def normalized_histogram(values: Sequence[Any]) -> dict[Any, float]:
    counts = Counter(values)
    total = max(1, sum(counts.values()))
    return {key: value / total for key, value in counts.items()}


def histogram_l1(left: Sequence[Any], right: Sequence[Any]) -> float:
    left_hist = normalized_histogram(left)
    right_hist = normalized_histogram(right)
    support = set(left_hist) | set(right_hist)
    return float(sum(abs(left_hist.get(key, 0.0) - right_hist.get(key, 0.0)) for key in support))


def relaxed_graph_errors(reference: nx.Graph, candidate: nx.Graph) -> dict[str, float]:
    return {
        "node_label_hist_l1": histogram_l1(
            [data.get("label") for _, data in reference.nodes(data=True)],
            [data.get("label") for _, data in candidate.nodes(data=True)],
        ),
        "degree_hist_l1": histogram_l1(
            [degree for _, degree in reference.degree()],
            [degree for _, degree in candidate.degree()],
        ),
        "edge_label_hist_l1": histogram_l1(
            [data.get("label") for _, _, data in reference.edges(data=True)],
            [data.get("label") for _, _, data in candidate.edges(data=True)],
        ),
    }


def score_conditioned_reconstruction(
    reference_graphs: Sequence[nx.Graph],
    generated_by_reference: Sequence[Sequence[nx.Graph]],
) -> dict[str, float]:
    per_reference_exact_rates = []
    per_reference_any_hits = []
    node_count_errors = []
    edge_count_errors = []
    relaxed_errors = {metric: [] for metric in RECONSTRUCTION_ERROR_METRICS}

    for reference, candidates in zip(reference_graphs, generated_by_reference):
        candidates = list(candidates)
        exact = [same_labeled_graph(reference, candidate) for candidate in candidates]
        per_reference_exact_rates.append(float(np.mean(exact)) if exact else 0.0)
        per_reference_any_hits.append(float(any(exact)))
        node_count_errors.extend(
            abs(candidate.number_of_nodes() - reference.number_of_nodes()) for candidate in candidates
        )
        edge_count_errors.extend(
            abs(candidate.number_of_edges() - reference.number_of_edges()) for candidate in candidates
        )
        for candidate in candidates:
            for metric, value in relaxed_graph_errors(reference, candidate).items():
                relaxed_errors[metric].append(value)

    return {
        "exact_rate_per_sample": float(np.mean(per_reference_exact_rates)),
        "exact_rate_any_sample": float(np.mean(per_reference_any_hits)),
        "node_count_mae": float(np.mean(node_count_errors)) if node_count_errors else float("nan"),
        "edge_count_mae": float(np.mean(edge_count_errors)) if edge_count_errors else float("nan"),
        **{
            metric: float(np.mean(values)) if values else float("nan")
            for metric, values in relaxed_errors.items()
        },
    }


def shuffled_embedding_conditioning(
    conditioning: GraphConditioningBatch,
    donor_conditioning: GraphConditioningBatch,
) -> GraphConditioningBatch:
    """Pair target counts with a cyclically mismatched donor embedding batch."""
    if len(donor_conditioning) < 1:
        return conditioning
    permutation = np.roll(np.arange(len(donor_conditioning), dtype=np.int64), 1)
    donor = donor_conditioning.take(permutation)
    donor_indices = np.arange(len(conditioning), dtype=np.int64) % len(donor)
    shuffled = donor.take(donor_indices)
    return GraphConditioningBatch(
        graph_embeddings=shuffled.graph_embeddings,
        node_counts=conditioning.node_counts,
        edge_counts=conditioning.edge_counts,
        condition_node_embeddings=shuffled.condition_node_embeddings,
        condition_node_presence_mask=shuffled.condition_node_presence_mask,
    )


def decode_conditioning_repeated(
    graph_generator: Any,
    conditioning: GraphConditioningBatch,
    *,
    repeats: int,
    use_ilp_decoder: bool,
) -> list[list[nx.Graph]]:
    repeated = graph_generator._repeat_graph_conditioning(conditioning, repeats=repeats)
    decoded = graph_generator.decode(
        repeated,
        apply_feasibility_filtering=False,
        use_ilp_decoder=use_ilp_decoder,
    )
    return [
        decoded[index:index + repeats]
        for index in range(0, len(decoded), repeats)
    ]


def composite_reconstruction_error(score: Mapping[str, float]) -> float:
    return float(np.mean([score[metric] for metric in RECONSTRUCTION_ERROR_METRICS]))


def evaluate_overfit_generalization(
    graph_generator: Any,
    fit_graphs: Sequence[nx.Graph],
    heldout_graphs: Sequence[nx.Graph],
    *,
    samples_per_graph: int = 6,
    use_ilp_decoder: bool = True,
    evaluation_seed: int = 2026,
    targets: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate seen, held-out, and within-split mismatched conditioning controls."""
    if len(graph_generator.training_graph_conditioning_) != len(fit_graphs):
        raise RuntimeError(
            "This generator was not trained on the holdout split. "
            "Retrain with fit_graphs before testing generalization."
        )

    targets = dict(targets or {
        "max_seen_error": 0.90,
        "min_heldout_gap": 0.20,
        "min_shuffled_gap": 0.20,
    })
    print("Evaluate conditional reconstruction on seen and held-out graphs.")
    with _seeded_generation(evaluation_seed):
        train_conditioned = graph_generator.conditional_sample(
            fit_graphs,
            n_samples=samples_per_graph,
            feasibility_effort=0,
            feasibility_filter="none",
            use_ilp_decoder=use_ilp_decoder,
        )
    with _seeded_generation(evaluation_seed):
        heldout_conditioned = graph_generator.conditional_sample(
            heldout_graphs,
            n_samples=samples_per_graph,
            feasibility_effort=0,
            feasibility_filter="none",
            use_ilp_decoder=use_ilp_decoder,
        )
    train_score = score_conditioned_reconstruction(fit_graphs, train_conditioned)
    heldout_score = score_conditioned_reconstruction(heldout_graphs, heldout_conditioned)
    print("Seen/train conditions:", train_score)
    print("Held-out conditions:", heldout_score)
    print(
        "Generalization gap (exact any-sample rate):",
        train_score["exact_rate_any_sample"] - heldout_score["exact_rate_any_sample"],
    )

    fit_conditioning = graph_generator.graph_encode(fit_graphs)
    heldout_conditioning = graph_generator.graph_encode(heldout_graphs)
    with _seeded_generation(evaluation_seed):
        shuffled_fit_conditioned = decode_conditioning_repeated(
            graph_generator,
            shuffled_embedding_conditioning(fit_conditioning, fit_conditioning),
            repeats=samples_per_graph,
            use_ilp_decoder=use_ilp_decoder,
        )
    with _seeded_generation(evaluation_seed):
        shuffled_heldout_conditioned = decode_conditioning_repeated(
            graph_generator,
            shuffled_embedding_conditioning(heldout_conditioning, heldout_conditioning),
            repeats=samples_per_graph,
            use_ilp_decoder=use_ilp_decoder,
        )
    shuffled_fit_score = score_conditioned_reconstruction(fit_graphs, shuffled_fit_conditioned)
    shuffled_heldout_score = score_conditioned_reconstruction(
        heldout_graphs, shuffled_heldout_conditioned
    )
    print("Shuffled seen/train conditions:", shuffled_fit_score)
    print("Shuffled held-out conditions:", shuffled_heldout_score)

    seen_error = composite_reconstruction_error(train_score)
    heldout_error = composite_reconstruction_error(heldout_score)
    shuffled_error = composite_reconstruction_error(shuffled_fit_score)
    heldout_gap = heldout_error - seen_error
    shuffled_gap = shuffled_error - seen_error
    seen_quality = max(0.0, targets["max_seen_error"] - seen_error)
    overfit_objective = seen_quality * (heldout_gap + shuffled_gap)
    overfit_success = (
        seen_error <= targets["max_seen_error"]
        and heldout_gap >= targets["min_heldout_gap"]
        and shuffled_gap >= targets["min_shuffled_gap"]
    )
    overfit_metrics = {
        "seen_error": seen_error,
        "heldout_error": heldout_error,
        "shuffled_error": shuffled_error,
        "heldout_gap": heldout_gap,
        "shuffled_gap": shuffled_gap,
        "overfit_objective": overfit_objective,
        "overfit_success": overfit_success,
        "targets": targets,
    }
    print(overfit_metrics)
    return {
        "train_conditioned": train_conditioned,
        "heldout_conditioned": heldout_conditioned,
        "shuffled_fit_conditioned": shuffled_fit_conditioned,
        "shuffled_heldout_conditioned": shuffled_heldout_conditioned,
        "train_score": train_score,
        "heldout_score": heldout_score,
        "shuffled_fit_score": shuffled_fit_score,
        "shuffled_heldout_score": shuffled_heldout_score,
        "overfit_metrics": overfit_metrics,
    }


def plot_overfit_summary(
    evaluation: Mapping[str, Any],
    *,
    figsize: tuple[float, float] = (15.0, 7.0),
) -> None:
    """Display intuitive error, gap, and target comparisons for an evaluation."""
    from IPython.display import HTML, display
    import matplotlib.pyplot as plt

    scores = {
        "Seen/train": evaluation["train_score"],
        "Held-out": evaluation["heldout_score"],
        "Shuffled seen": evaluation["shuffled_fit_score"],
        "Shuffled held-out": evaluation["shuffled_heldout_score"],
    }
    metrics = evaluation["overfit_metrics"]
    targets = metrics["targets"]
    success = bool(metrics["overfit_success"])
    status = "OVERFIT TARGETS MET" if success else "OVERFIT TARGETS NOT MET"
    status_color = "#16803c" if success else "#b42318"
    display(HTML(
        '<hr style="border:0;border-top:3px solid #555;margin:24px 0 12px;">'
        f'<h2>OVERFIT EVALUATION SUMMARY</h2>'
        f'<p style="color:{status_color};font-size:18px;font-weight:bold;">{status}</p>'
    ))

    labels = list(scores)
    composite_values = [composite_reconstruction_error(score) for score in scores.values()]
    component_values = np.asarray([
        [score[metric] for metric in RECONSTRUCTION_ERROR_METRICS]
        for score in scores.values()
    ])
    component_labels = ["Node labels", "Degree", "Edge labels"]

    figure, axes = plt.subplots(
        1,
        3,
        figsize=figsize,
        gridspec_kw={"width_ratios": [1.0, 1.25, 1.0]},
    )

    bars = axes[0].bar(
        labels,
        composite_values,
        color=["#4c78a8", "#f58518", "#54a24b", "#e45756"],
    )
    axes[0].axhline(
        targets["max_seen_error"],
        color="#b42318",
        linestyle="--",
        linewidth=1.5,
        label=f"max seen = {targets['max_seen_error']:.2f}",
    )
    axes[0].set_title("Composite reconstruction error\n(lower is better)")
    axes[0].set_ylabel("mean L1 error")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].legend(fontsize=8)
    axes[0].bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

    image = axes[1].imshow(
        component_values,
        cmap="YlOrRd",
        aspect="auto",
        vmin=0.0,
        vmax=max(1.0, float(component_values.max())),
    )
    axes[1].set_title("Error by reconstruction component\n(lower is better)")
    axes[1].set_xticks(
        range(len(component_labels)),
        component_labels,
        rotation=25,
        ha="right",
    )
    axes[1].set_yticks(range(len(labels)), labels)
    for row in range(component_values.shape[0]):
        for column in range(component_values.shape[1]):
            axes[1].text(
                column,
                row,
                f"{component_values[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )
    figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)

    gap_labels = ["Held-out gap", "Shuffled gap"]
    gap_values = [metrics["heldout_gap"], metrics["shuffled_gap"]]
    gap_targets = [targets["min_heldout_gap"], targets["min_shuffled_gap"]]
    gap_colors = [
        "#16803c" if value >= target else "#b42318"
        for value, target in zip(gap_values, gap_targets)
    ]
    bars = axes[2].bar(gap_labels, gap_values, color=gap_colors)
    axes[2].axhline(0.0, color="black", linewidth=0.8)
    axes[2].plot(range(len(gap_targets)), gap_targets, "k--", label="required gap")
    axes[2].set_title("Generalization gaps\n(higher is better)")
    axes[2].set_ylabel("gap relative to seen error")
    axes[2].tick_params(axis="x", rotation=25)
    axes[2].legend(fontsize=8)
    axes[2].bar_label(bars, fmt="%.2f", padding=3, fontsize=8)

    figure.suptitle(
        f"Seen error={metrics['seen_error']:.2f} | "
        f"Held-out gap={metrics['heldout_gap']:+.2f} | "
        f"Shuffled gap={metrics['shuffled_gap']:+.2f}",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.92])
    display(figure)
    plt.close(figure)


def plot_overfit_reconstructions(
    plot_artificial_graphs: Any,
    evaluation: Mapping[str, Any],
    fit_graphs: Sequence[nx.Graph],
    heldout_graphs: Sequence[nx.Graph],
    *,
    n_candidates: int = 6,
) -> None:
    """Display labeled reference/reconstruction sections in the notebook."""
    from IPython.display import HTML, display
    import matplotlib.pyplot as plt

    def show_section(title: str, references: Sequence[nx.Graph], generated: Any) -> None:
        display(HTML(
            f'<hr style="border:0;border-top:3px solid #555;margin:24px 0 12px;">'
            f'<h2>{title.upper()}</h2>'
        ))
        total_references = len(references)
        for index, (reference, candidates) in enumerate(zip(references, generated)):
            candidates = list(candidates)[:n_candidates]
            comparison = [reference, *candidates]
            titles = [f"REFERENCE {index}"] + [
                f"GENERATED {sample_index}" for sample_index in range(len(candidates))
            ]
            figure = plot_artificial_graphs(
                comparison,
                n_cols=min(7, len(comparison)),
                titles=titles,
                size=3.5,
            )
            figure.suptitle(
                f"{title} {index + 1}/{total_references}",
                fontsize=16,
                fontweight="bold",
                y=1.02,
            )
            figure.tight_layout(rect=[0, 0, 1, 0.95])
            display(figure)
            plt.close(figure)

    show_section("Seen/train conditions", fit_graphs, evaluation["train_conditioned"])
    show_section("Held-out conditions", heldout_graphs, evaluation["heldout_conditioned"])
    show_section(
        "Mismatched/shuffled seen conditions",
        fit_graphs,
        evaluation["shuffled_fit_conditioned"],
    )
    show_section(
        "Mismatched/shuffled held-out conditions",
        heldout_graphs,
        evaluation["shuffled_heldout_conditioned"],
    )


__all__ = [
    "composite_reconstruction_error",
    "decode_conditioning_repeated",
    "evaluate_overfit_generalization",
    "histogram_l1",
    "normalized_histogram",
    "plot_overfit_summary",
    "plot_overfit_reconstructions",
    "relaxed_graph_errors",
    "same_labeled_graph",
    "score_conditioned_reconstruction",
    "shuffled_embedding_conditioning",
]
