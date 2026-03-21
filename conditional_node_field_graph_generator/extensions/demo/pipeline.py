"""Demo-oriented dataset and model-construction helpers."""

from __future__ import annotations

import os
from pathlib import Path
import math
from typing import Any, Callable, Optional
import warnings

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from abstractgraph.operators import compose, cycle, neighborhood, unlabel, combination
except ModuleNotFoundError:
    compose = None
    cycle = None
    neighborhood = None
    unlabel = None
    combination = None

try:
    from abstractgraph_ml.feasibility import (
        FeasibilityEstimator,
        FeasibilityEstimatorFeatureCannotExist,
        WithinRangeFeasibilityEstimatorFromNumericalFunction,
    )
except ModuleNotFoundError:
    FeasibilityEstimator = None
    FeasibilityEstimatorFeatureCannotExist = None
    WithinRangeFeasibilityEstimatorFromNumericalFunction = None

try:
    from NSPPK.nsppk import NSPPK, NodeNSPPK
except ModuleNotFoundError:
    from nsppk import NSPPK, NodeNSPPK

from ...conditional_node_field_generator import ConditionalNodeFieldGenerator
from ...conditional_node_field_graph_generator import (
    ConditionalNodeFieldGraphDecoder,
    ConditionalNodeFieldGraphGenerator,
    GeneratedGuidanceBatch,
)
from ...persistence import save_graph_generator
from ..molecular import (
    PubChemLoader,
    SupervisedDataSetLoader,
    build_zinc_graph_corpus,
    download_zinc_dataset,
    draw_molecules,
    load_zinc_graph_dataset,
)
from ..synthetic import ArtificialGraphDatasetConstructor
from .storage import describe_resume_checkpoint, find_latest_checkpoint
from .visualization import offset_neg_graphs, plot_networkx_graphs, select_pos_neg


def _has_demo_feasibility_support():
    return not any(
        dependency is None
        for dependency in (
            compose,
            cycle,
            neighborhood,
            unlabel,
            combination,
            FeasibilityEstimator,
            FeasibilityEstimatorFeatureCannotExist,
            WithinRangeFeasibilityEstimatorFromNumericalFunction,
        )
    )


def _resolve_pubchem_dir() -> Path:
    env_path = os.environ.get("PUBCHEM_DATA_DIR")
    if env_path:
        return Path(env_path).expanduser().resolve()
    return Path(__file__).resolve().parents[3] / "notebooks" / "datasets" / "PUBCHEM"


def build_dataset(dataset_type, dataset_size=50, size=5, assay_id="651610"):
    if dataset_type == "ARTIFICIAL":
        alphabet_size = 3
        graphs, targets = ArtificialGraphDatasetConstructor(
            graph_generator_target_type_pos="cycle",
            graph_generator_context_type_pos="cycle",
            graph_generator_target_type_neg="tree",
            graph_generator_context_type_neg="tree",
            target_size_pos=size,
            context_size_pos=size,
            n_link_edges_pos=1,
            alphabet_size_pos=alphabet_size,
            target_size_neg=size,
            context_size_neg=size,
            n_link_edges_neg=1,
            alphabet_size_neg=alphabet_size,
        ).sample(dataset_size // 2)
        graphs, targets = offset_neg_graphs(graphs, targets, offset=alphabet_size)
        n_graphs_per_line = 8
        pos_graphs, neg_graphs = select_pos_neg(graphs, targets, n_lines=1, n_graphs_per_line=n_graphs_per_line)
        plot_networkx_graphs(pos_graphs, n_cols=n_graphs_per_line)
        plot_networkx_graphs(neg_graphs, n_cols=n_graphs_per_line)
        return graphs, targets

    if dataset_type == "MOLECULAR":
        pubchem_dir = _resolve_pubchem_dir()

        def pubchem_loader():
            loader = PubChemLoader()
            loader.pubchem_dir = str(pubchem_dir)
            return loader.load(assay_id, dirname=str(pubchem_dir))

        original_graphs, original_targets = SupervisedDataSetLoader(
            pubchem_loader,
            size=dataset_size,
            use_equalized=False,
        ).load()
        original_graphs = np.array(original_graphs, dtype=object)
        original_targets = np.array(original_targets)
        idxs = [idx for idx, graph in enumerate(original_graphs) if nx.number_of_nodes(graph) <= size]
        graphs = original_graphs[idxs].tolist()
        targets = original_targets[idxs]
        draw_molecules(graphs[:14])
        return graphs, targets

    raise ValueError(f"Unsupported dataset_type={dataset_type!r}")


def build_zinc_dataset(
    dataset_dir=None,
    num_examples=10_000,
    min_size=10,
    max_size=15,
    random_state=0,
):
    """Load a random ZINC subset after filtering the full cached corpus by node-count range."""
    if int(num_examples) < 1:
        raise ValueError("num_examples must be >= 1")
    if int(min_size) < 1:
        raise ValueError("min_size must be >= 1")
    if int(max_size) < int(min_size):
        raise ValueError("max_size must be >= min_size")

    if dataset_dir is None:
        dataset_dir = Path(__file__).resolve().parents[3] / "notebooks" / "datasets" / "zinc"
    dataset_dir = Path(dataset_dir).expanduser().resolve()

    csv_path = download_zinc_dataset(dataset_dir)
    manifest = build_zinc_graph_corpus(dataset_dir, csv_path=csv_path)
    max_molecules = int(manifest.get("total_graphs", int(num_examples)))
    graphs, metadata = load_zinc_graph_dataset(
        dataset_dir,
        max_molecules=max_molecules,
        min_node_count=int(min_size),
        max_node_count=int(max_size),
    )
    if len(graphs) > int(num_examples):
        rng = np.random.default_rng(random_state)
        selected_indices = np.sort(rng.choice(len(graphs), size=int(num_examples), replace=False))
        graphs = [graphs[idx] for idx in selected_indices.tolist()]
        metadata = metadata.iloc[selected_indices].reset_index(drop=True)
    return graphs, metadata, manifest


def prepare_experiment(build_dataset_fn: Callable, dataset_size=200, test_size=10, random_state=42, **build_kwargs):
    graphs, targets = build_dataset_fn(dataset_size=dataset_size, **build_kwargs)
    train_graphs, test_graphs, train_targets, test_targets = train_test_split(
        graphs,
        targets,
        test_size=test_size,
        random_state=random_state,
    )
    print(f"train_graphs:{len(train_graphs)}   test_graphs:{len(test_graphs)}")
    return graphs, targets, train_graphs, test_graphs, train_targets, test_targets


def score_graph_generator_feasible_rate(
    graph_generator,
    n_samples=32,
    max_feasibility_attempts=None,
    feasibility_candidates_per_attempt=None,
    feasibility_oracle_candidates_per_attempt=None,
    interpolate_between_n_samples=None,
    desired_target=None,
    guidance_scale=1.0,
    verbose=False,
):
    """Estimate generation quality from the fraction of feasible decoded candidates."""
    return graph_generator.score_feasible_rate(
        n_samples=n_samples,
        max_feasibility_attempts=max_feasibility_attempts,
        feasibility_candidates_per_attempt=feasibility_candidates_per_attempt,
        feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
        interpolate_between_n_samples=interpolate_between_n_samples,
        desired_target=desired_target,
        guidance_scale=guidance_scale,
        verbose=verbose,
    )


def _resolve_violation_counts(feasibility_estimator, decoded_graphs):
    try:
        raw_violation_counts = feasibility_estimator.number_of_violations(decoded_graphs)
    except AttributeError as exc:
        if "has no attribute 'get'" not in str(exc):
            raise
        raw_violation_counts = feasibility_estimator.number_of_violations(
            [graph.graph if hasattr(graph, "graph") else graph for graph in decoded_graphs]
        )
    return np.asarray(raw_violation_counts, dtype=np.int64).reshape(-1)


def _evaluate_guidance_mode(
    graph_generator,
    graph_conditioning,
    *,
    sampling_mode,
    desired_target,
    guidance_scale,
    predictor_scale,
):
    effective_desired_target = None if sampling_mode == "unguided" else desired_target
    generated_nodes = graph_generator._predict_generated_nodes(
        graph_conditioning,
        sampling_mode=sampling_mode,
        desired_target=effective_desired_target,
        guidance_scale=guidance_scale,
        predictor_scale=predictor_scale,
    )
    decoded_graphs = graph_generator._decode_generated_nodes(generated_nodes)
    violation_counts = _resolve_violation_counts(graph_generator.feasibility_estimator, decoded_graphs)
    if violation_counts.shape[0] != len(decoded_graphs):
        raise RuntimeError(
            "Feasibility estimator returned an unexpected number of violation counts "
            f"({violation_counts.shape[0]} for {len(decoded_graphs)} graphs)."
        )
    return GeneratedGuidanceBatch(
        node_embeddings_list=[
            np.asarray(embedding, dtype=float)
            for embedding in generated_nodes.node_embeddings_list or []
        ],
        graph_conditioning=graph_conditioning,
        decoded_graphs=decoded_graphs,
        violation_counts=violation_counts,
        guidance_targets=graph_generator._compute_guidance_targets(violation_counts),
        feasible_mask=np.asarray(violation_counts == 0, dtype=bool),
        sampling_mode=str(sampling_mode),
    )


def _wilson_interval(successes, total, confidence_level=0.95):
    if int(total) <= 0:
        return 0.0, 0.0
    z = 1.959963984540054 if float(confidence_level) == 0.95 else 1.959963984540054
    n = float(total)
    p = float(successes) / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / n) + (z * z) / (4.0 * n * n))
    return max(0.0, center - margin), min(1.0, center + margin)


def _bootstrap_interval(values, rng, confidence_level=0.95, n_resamples=1000):
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        return float(arr[0]), float(arr[0])
    resampled_means = np.empty(int(n_resamples), dtype=float)
    for idx in range(int(n_resamples)):
        sample = rng.choice(arr, size=arr.size, replace=True)
        resampled_means[idx] = float(np.mean(sample))
    alpha = 1.0 - float(confidence_level)
    lower = float(np.quantile(resampled_means, alpha / 2.0))
    upper = float(np.quantile(resampled_means, 1.0 - alpha / 2.0))
    return lower, upper


def benchmark_regression_guidance(
    graph_generator,
    n_samples=200,
    interpolate_between_n_samples=None,
    desired_target=1.0,
    guidance_scale=1.0,
    predictor_scale=1.0,
    bootstrap_samples=1000,
    confidence_level=0.95,
    random_state=0,
):
    """Compare unguided and regression-guided generation on the same sampled conditioning."""
    if graph_generator.feasibility_estimator is None:
        raise RuntimeError("benchmark_regression_guidance() requires graph_generator.feasibility_estimator.")
    graph_conditioning = graph_generator._sample_conditions(
        int(n_samples),
        interpolate_between_n_samples=interpolate_between_n_samples,
    )
    unguided_batch = _evaluate_guidance_mode(
        graph_generator,
        graph_conditioning,
        sampling_mode="unguided",
        desired_target=desired_target,
        guidance_scale=guidance_scale,
        predictor_scale=predictor_scale,
    )
    guided_batch = _evaluate_guidance_mode(
        graph_generator,
        graph_conditioning,
        sampling_mode="regression_guided",
        desired_target=desired_target,
        guidance_scale=guidance_scale,
        predictor_scale=predictor_scale,
    )

    rng = np.random.default_rng(random_state)
    rows = []
    for batch in (unguided_batch, guided_batch):
        feasible_count = int(np.sum(batch.feasible_mask))
        count = len(batch)
        feasible_ci_low, feasible_ci_high = _wilson_interval(
            feasible_count,
            count,
            confidence_level=confidence_level,
        )
        rows.append(
            {
                "label": batch.sampling_mode,
                "count": count,
                "feasible_count": feasible_count,
                "feasible_rate": float(np.mean(batch.feasible_mask)) if count else 0.0,
                "feasible_rate_ci_low": feasible_ci_low,
                "feasible_rate_ci_high": feasible_ci_high,
                "mean_violations": float(np.mean(batch.violation_counts)) if count else 0.0,
                "median_violations": float(np.median(batch.violation_counts)) if count else 0.0,
                "mean_target": float(np.mean(batch.guidance_targets)) if count else 0.0,
                "median_target": float(np.median(batch.guidance_targets)) if count else 0.0,
            }
        )

    feasible_delta = np.asarray(guided_batch.feasible_mask, dtype=float) - np.asarray(unguided_batch.feasible_mask, dtype=float)
    violation_delta = np.asarray(guided_batch.violation_counts, dtype=float) - np.asarray(unguided_batch.violation_counts, dtype=float)
    target_delta = np.asarray(guided_batch.guidance_targets, dtype=float) - np.asarray(unguided_batch.guidance_targets, dtype=float)
    feasible_delta_ci_low, feasible_delta_ci_high = _bootstrap_interval(
        feasible_delta,
        rng,
        confidence_level=confidence_level,
        n_resamples=bootstrap_samples,
    )
    violation_delta_ci_low, violation_delta_ci_high = _bootstrap_interval(
        violation_delta,
        rng,
        confidence_level=confidence_level,
        n_resamples=bootstrap_samples,
    )
    target_delta_ci_low, target_delta_ci_high = _bootstrap_interval(
        target_delta,
        rng,
        confidence_level=confidence_level,
        n_resamples=bootstrap_samples,
    )
    paired_summary = pd.DataFrame(
        [
            {
                "count": len(unguided_batch),
                "guided_only_feasible": int(np.sum(guided_batch.feasible_mask & ~unguided_batch.feasible_mask)),
                "unguided_only_feasible": int(np.sum(unguided_batch.feasible_mask & ~guided_batch.feasible_mask)),
                "both_feasible": int(np.sum(guided_batch.feasible_mask & unguided_batch.feasible_mask)),
                "neither_feasible": int(np.sum(~guided_batch.feasible_mask & ~unguided_batch.feasible_mask)),
                "mean_feasible_rate_delta": float(np.mean(feasible_delta)) if feasible_delta.size else 0.0,
                "feasible_rate_delta_ci_low": feasible_delta_ci_low,
                "feasible_rate_delta_ci_high": feasible_delta_ci_high,
                "mean_violation_delta": float(np.mean(violation_delta)) if violation_delta.size else 0.0,
                "violation_delta_ci_low": violation_delta_ci_low,
                "violation_delta_ci_high": violation_delta_ci_high,
                "mean_target_delta": float(np.mean(target_delta)) if target_delta.size else 0.0,
                "target_delta_ci_low": target_delta_ci_low,
                "target_delta_ci_high": target_delta_ci_high,
            }
        ]
    )
    per_sample = pd.DataFrame(
        {
            "conditioning_index": np.arange(len(unguided_batch), dtype=int),
            "unguided_feasible": np.asarray(unguided_batch.feasible_mask, dtype=bool),
            "guided_feasible": np.asarray(guided_batch.feasible_mask, dtype=bool),
            "unguided_violations": np.asarray(unguided_batch.violation_counts, dtype=int),
            "guided_violations": np.asarray(guided_batch.violation_counts, dtype=int),
            "violation_delta": violation_delta,
            "unguided_target": np.asarray(unguided_batch.guidance_targets, dtype=float),
            "guided_target": np.asarray(guided_batch.guidance_targets, dtype=float),
            "target_delta": target_delta,
        }
    )
    return {
        "summary": pd.DataFrame(rows),
        "paired_summary": paired_summary,
        "per_sample": per_sample,
        "unguided_batch": unguided_batch,
        "guided_batch": guided_batch,
    }


def sample_hyperparameter_configuration(
    search_space: dict[str, dict[str, Any]],
    random_state: Optional[int] = None,
):
    """Sample one hyperparameter configuration from typed ranges."""
    rng = np.random.default_rng(random_state)
    sampled = {}
    for name, spec in search_space.items():
        param_type = spec["type"]
        low = spec["low"]
        high = spec["high"]
        if param_type == "int":
            sampled[name] = int(rng.integers(int(low), int(high) + 1))
        elif param_type == "real":
            sampled[name] = float(rng.uniform(float(low), float(high)))
        else:
            raise ValueError(f"Unsupported hyperparameter type for {name!r}: {param_type!r}")
    return sampled


def build_graph_generator(
    verbose=2,
    nbits=None,
    node_vectorizer_radius=2,
    node_vectorizer_distance=4,
    node_vectorizer_connector=1,
    node_vectorizer_nbits=None,
    node_vectorizer_dense=True,
    node_vectorizer_parallel=True,
    node_vectorizer_use_edges_as_features=True,
    graph_vectorizer_radius=2,
    graph_vectorizer_distance=4,
    graph_vectorizer_connector=1,
    graph_vectorizer_nbits=None,
    graph_vectorizer_dense=True,
    graph_vectorizer_parallel=True,
    graph_vectorizer_use_edges_as_features=True,
    feasibility_size_quantile=None,
    feasibility_unlabeled_radius=2,
    feasibility_valence_radius=1,
    feasibility_unlabeled_nbits=19,
    feasibility_valence_nbits=19,
    feasibility_cycle_nbits=19,
    feasibility_parallel=True,
    feasibility_n_jobs=None,
    feasibility_backend="dill",
    latent_embedding_dimension=128,
    number_of_transformer_layers=4,
    transformer_attention_head_count=4,
    transformer_dropout=0.2,
    learning_rate=1e-4,
    maximum_epochs=250,
    batch_size=16,
    total_steps=100,
    verbose_epoch_interval=10,
    enable_early_stopping=True,
    early_stopping_monitor="val_total",
    early_stopping_mode="min",
    early_stopping_patience=20,
    early_stopping_min_delta=0.1,
    early_stopping_ema_alpha=0.3,
    restore_best_checkpoint=True,
    important_feature_index=1,
    lambda_degree_importance=2.0,
    default_exist_pos_weight=1.0,
    lambda_node_exist_importance=2.0,
    lambda_node_count_importance=0.5,
    lambda_node_label_importance=2.0,
    lambda_edge_label_importance=2.0,
    lambda_direct_edge_importance=2.0,
    lambda_edge_count_importance=0.5,
    lambda_degree_edge_consistency_importance=0.5,
    lambda_auxiliary_edge_importance=1.0,
    degree_temperature=1,
    pool_condition_tokens=False,
    node_field_sigma=0.2,
    sampling_step_size=0.05,
    sampling_steps=None,
    langevin_noise_scale=0.0,
    cfg_condition_dropout_prob=0.1,
    cfg_null_target_strategy="zero",
    locality_horizon=1,
    locality_sample_fraction=0.5,
    negative_sample_factor=1,
    locality_sampling_strategy="stratified_preserve",
    locality_target_positive_ratio=0.5,
    feasibility_oracle_candidates_per_attempt=2,
    max_oracle_iterations=8,
    use_feasibility_filtering=True,
    max_feasibility_attempts=20,
    feasibility_candidates_per_attempt=8,
    feasibility_failure_mode="return_partial",
    decoder_existence_threshold=0.5,
    decoder_enforce_connectivity=True,
    decoder_degree_slack_penalty=1e6,
    decoder_warm_start_mst=True,
    decoder_n_jobs=1,
    artifact_root=None,
    checkpoint_root=None,
    model_name=None,
    model_dir=None,
):
    if nbits is not None:
        if node_vectorizer_nbits is None:
            node_vectorizer_nbits = nbits
        if graph_vectorizer_nbits is None:
            graph_vectorizer_nbits = nbits
    if node_vectorizer_nbits is None:
        node_vectorizer_nbits = 11
    if graph_vectorizer_nbits is None:
        graph_vectorizer_nbits = 11

    node_graph_vectorizer = NodeNSPPK(
        radius=node_vectorizer_radius,
        distance=node_vectorizer_distance,
        connector=node_vectorizer_connector,
        nbits=node_vectorizer_nbits,
        dense=node_vectorizer_dense,
        parallel=node_vectorizer_parallel,
        use_edges_as_features=node_vectorizer_use_edges_as_features,
    )
    graph_vectorizer = NSPPK(
        radius=graph_vectorizer_radius,
        distance=graph_vectorizer_distance,
        connector=graph_vectorizer_connector,
        nbits=graph_vectorizer_nbits,
        dense=graph_vectorizer_dense,
        parallel=graph_vectorizer_parallel,
        use_edges_as_features=graph_vectorizer_use_edges_as_features,
    )

    feasibility_estimator = None
    if _has_demo_feasibility_support():
        feasibility_size = WithinRangeFeasibilityEstimatorFromNumericalFunction(
            numerical_function=lambda graph: len(graph),
            quantile=feasibility_size_quantile,
        )
        feasibility_unlabeled_structure = FeasibilityEstimatorFeatureCannotExist(
            decomposition_function=compose(neighborhood(radius=feasibility_unlabeled_radius), unlabel()),
            nbits=feasibility_unlabeled_nbits,
            parallel=feasibility_parallel,
            backend=feasibility_backend,
            n_jobs=feasibility_n_jobs,
        )
        feasibility_valence = FeasibilityEstimatorFeatureCannotExist(
            decomposition_function=neighborhood(radius=feasibility_valence_radius),
            nbits=feasibility_valence_nbits,
            parallel=feasibility_parallel,
            backend=feasibility_backend,
            n_jobs=feasibility_n_jobs,
        )
        feasibility_cycle = FeasibilityEstimatorFeatureCannotExist(
            decomposition_function=cycle(),
            nbits=feasibility_cycle_nbits,
            parallel=feasibility_parallel,
            backend=feasibility_backend,
            n_jobs=feasibility_n_jobs,
        )
        feasibility_cycle_composition = FeasibilityEstimatorFeatureCannotExist(
            decomposition_function=compose(combination(number_of_elements=(2,3), distance=0), cycle(), unlabel()),
            nbits=feasibility_cycle_nbits,
            parallel=feasibility_parallel,
            backend=feasibility_backend,
            n_jobs=feasibility_n_jobs,
        )
        feasibility_estimator = FeasibilityEstimator(
            [
                feasibility_size,
                feasibility_valence,
                feasibility_cycle,
                feasibility_unlabeled_structure,
                feasibility_cycle_composition,
            ]
        )
    else:
        if feasibility_oracle_candidates_per_attempt or use_feasibility_filtering:
            warnings.warn(
                "Optional dependencies 'abstractgraph' and 'abstractgraph_ml' are not installed; "
                "disabling feasibility oracle and feasibility filtering.",
                RuntimeWarning,
                stacklevel=2,
            )
        feasibility_oracle_candidates_per_attempt = 0
        use_feasibility_filtering = False

    conditional_node_generator_model = ConditionalNodeFieldGenerator(
        latent_embedding_dimension=latent_embedding_dimension,
        number_of_transformer_layers=number_of_transformer_layers,
        transformer_attention_head_count=transformer_attention_head_count,
        transformer_dropout=transformer_dropout,
        learning_rate=learning_rate,
        maximum_epochs=maximum_epochs,
        batch_size=batch_size,
        total_steps=total_steps,
        lambda_degree_importance=lambda_degree_importance,
        lambda_node_exist_importance=lambda_node_exist_importance,
        lambda_node_count_importance=lambda_node_count_importance,
        lambda_node_label_importance=lambda_node_label_importance,
        lambda_edge_label_importance=lambda_edge_label_importance,
        lambda_direct_edge_importance=lambda_direct_edge_importance,
        lambda_edge_count_importance=lambda_edge_count_importance,
        lambda_degree_edge_consistency_importance=lambda_degree_edge_consistency_importance,
        lambda_auxiliary_edge_importance=lambda_auxiliary_edge_importance,
        degree_temperature=degree_temperature,
        node_field_sigma=node_field_sigma,
        sampling_step_size=sampling_step_size,
        langevin_noise_scale=langevin_noise_scale,
        verbose=verbose,
        verbose_epoch_interval=verbose_epoch_interval,
        enable_early_stopping=enable_early_stopping,
        early_stopping_monitor=early_stopping_monitor,
        early_stopping_mode=early_stopping_mode,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=early_stopping_min_delta,
        early_stopping_ema_alpha=early_stopping_ema_alpha,
        restore_best_checkpoint=restore_best_checkpoint,
        important_feature_index=important_feature_index,
        cfg_condition_dropout_prob=cfg_condition_dropout_prob,
        cfg_null_target_strategy=cfg_null_target_strategy,
        default_exist_pos_weight=default_exist_pos_weight,
        artifact_root_dir=str(artifact_root) if artifact_root is not None else None,
        checkpoint_root_dir=str(checkpoint_root) if checkpoint_root is not None else None,
        pool_condition_tokens=pool_condition_tokens,
        sampling_steps=sampling_steps,
        model_name=model_name,
        model_dir=str(model_dir) if model_dir is not None else None,
    )
    graph_decoder = ConditionalNodeFieldGraphDecoder(
        verbose=verbose,
        existence_threshold=decoder_existence_threshold,
        enforce_connectivity=decoder_enforce_connectivity,
        degree_slack_penalty=decoder_degree_slack_penalty,
        warm_start_mst=decoder_warm_start_mst,
        n_jobs=decoder_n_jobs,
    )
    return ConditionalNodeFieldGraphGenerator(
        graph_vectorizer=graph_vectorizer,
        node_graph_vectorizer=node_graph_vectorizer,
        conditional_node_generator_model=conditional_node_generator_model,
        graph_decoder=graph_decoder,
        feasibility_estimator=feasibility_estimator,
        feasibility_oracle_candidates_per_attempt=feasibility_oracle_candidates_per_attempt,
        max_oracle_iterations=max_oracle_iterations,
        locality_sample_fraction=locality_sample_fraction,
        locality_horizon=locality_horizon,
        negative_sample_factor=negative_sample_factor,
        locality_sampling_strategy=locality_sampling_strategy,
        locality_target_positive_ratio=locality_target_positive_ratio,
        use_feasibility_filtering=use_feasibility_filtering,
        max_feasibility_attempts=max_feasibility_attempts,
        feasibility_candidates_per_attempt=feasibility_candidates_per_attempt,
        feasibility_failure_mode=feasibility_failure_mode,
        model_name=model_name,
        model_dir=str(model_dir) if model_dir is not None else None,
        verbose=verbose,
    )


def fit_graph_generator(
    graph_generator,
    train_graphs,
    targets=None,
    ckpt_path=None,
    resume_latest_checkpoint=False,
    checkpoint_root=None,
):
    if ckpt_path is not None and resume_latest_checkpoint:
        raise ValueError("Provide either ckpt_path or resume_latest_checkpoint, not both.")
    resolved_ckpt_path = ckpt_path
    if resume_latest_checkpoint:
        resolved_ckpt_path = find_latest_checkpoint(checkpoint_root=checkpoint_root)
    describe_resume_checkpoint(resolved_ckpt_path)
    try:
        graph_generator.fit(train_graphs, targets=targets, ckpt_path=resolved_ckpt_path)
    except RuntimeError as exc:
        if not (resume_latest_checkpoint and resolved_ckpt_path is not None and _is_incompatible_resume_error(exc)):
            raise
        print(
            "Latest checkpoint is incompatible with the current generator configuration; "
            "retrying from scratch."
        )
        print(Path(resolved_ckpt_path).expanduser().resolve())
        graph_generator.fit(train_graphs, targets=targets, ckpt_path=None)
    if getattr(graph_generator, "model_name", None) is not None:
        save_graph_generator(graph_generator)
    return graph_generator


def _is_incompatible_resume_error(exc: RuntimeError) -> bool:
    message = str(exc)
    return (
        "Error(s) in loading state_dict" in message
        or "Missing key(s) in state_dict" in message
        or "Unexpected key(s) in state_dict" in message
        or "size mismatch for" in message
    )
