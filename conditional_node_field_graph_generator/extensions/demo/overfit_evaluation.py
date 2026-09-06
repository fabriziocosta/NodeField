"""Evaluation and visualization helpers for deliberate conditional overfitting."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import networkx as nx
import numpy as np

from ...conditional_node_field_generator import GraphConditioningBatch


RECONSTRUCTION_ERROR_METRICS = (
    "node_label_hist_l1",
    "degree_hist_l1",
    "edge_label_hist_l1",
)


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
    """Pair target counts with graph embeddings from the opposite split."""
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
    targets: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate seen, held-out, and opposite-split conditioning controls."""
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
    train_conditioned = graph_generator.conditional_sample(
        fit_graphs,
        n_samples=samples_per_graph,
        feasibility_effort=0,
        feasibility_filter="none",
        use_ilp_decoder=use_ilp_decoder,
    )
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
    shuffled_fit_conditioned = decode_conditioning_repeated(
        graph_generator,
        shuffled_embedding_conditioning(fit_conditioning, heldout_conditioning),
        repeats=samples_per_graph,
        use_ilp_decoder=use_ilp_decoder,
    )
    shuffled_heldout_conditioned = decode_conditioning_repeated(
        graph_generator,
        shuffled_embedding_conditioning(heldout_conditioning, fit_conditioning),
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
    "plot_overfit_reconstructions",
    "relaxed_graph_errors",
    "same_labeled_graph",
    "score_conditioned_reconstruction",
    "shuffled_embedding_conditioning",
]
