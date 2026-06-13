"""Active-node MILP projection for graph adjacency decoding."""

from __future__ import annotations

import heapq
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Optional, Sequence

import networkx as nx
import numpy as np
import pulp

from .graph_decode_utils import _normalize_violating_edge_sets

Edge = tuple[int, int]
_PROBABILITY_EPS = 1e-6
_BINARY_TOLERANCE = 1e-5


@dataclass(frozen=True)
class AdjacencySolveReport:
    solver_status: str
    solver_status_code: int
    solution_status: str
    solution_status_code: int
    optimal: bool
    used_incumbent: bool
    elapsed_seconds: float
    active_node_count: int
    solve_count: int
    horizon_iterations: int
    solver_termination_reason: str = "unknown"
    horizon_termination_reason: str = "not_enabled"
    objective_value: Optional[float] = None
    degree_slack_total: float = 0.0
    edge_count_slack: float = 0.0
    unresolved_horizon_pair_count: int = 0
    horizon_path_search_truncated: bool = False
    horizon_path_expansion_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def edge_key(i: int, j: int) -> Edge:
    return (int(i), int(j)) if int(i) < int(j) else (int(j), int(i))


def path_edges(path: Sequence[int]) -> list[Edge]:
    return [edge_key(path[idx], path[idx + 1]) for idx in range(len(path) - 1)]


def edge_logit(probability: float) -> float:
    probability = float(
        np.clip(probability, _PROBABILITY_EPS, 1.0 - _PROBABILITY_EPS)
    )
    return float(np.log(probability) - np.log1p(-probability))


def resolve_target_edge_count(
    n_nodes: int,
    desired_edge_count: Optional[int],
    *,
    connectivity: bool,
) -> Optional[int]:
    if desired_edge_count is None:
        return None
    n_nodes = max(0, int(n_nodes))
    max_edges = n_nodes * (n_nodes - 1) // 2
    min_edges = n_nodes - 1 if connectivity and n_nodes >= 2 else 0
    return int(np.clip(int(np.rint(desired_edge_count)), min_edges, max_edges))


def select_horizon_pairs(
    horizon_probability_matrix: np.ndarray,
    *,
    positive_threshold: float,
    negative_threshold: float,
    pair_budget: int,
) -> tuple[list[tuple[int, int, float, float]], list[tuple[int, int, float, float]]]:
    horizon_probs = np.asarray(horizon_probability_matrix, dtype=float)
    positive_pairs = []
    negative_pairs = []
    for i in range(horizon_probs.shape[0]):
        for j in range(i + 1, horizon_probs.shape[0]):
            probability = float(
                np.clip((horizon_probs[i, j] + horizon_probs[j, i]) / 2.0, 0.0, 1.0)
            )
            confidence = abs(probability - 0.5) * 2.0
            item = (i, j, probability, confidence)
            if probability >= positive_threshold:
                positive_pairs.append(item)
            elif probability <= negative_threshold:
                negative_pairs.append(item)

    positive_pairs.sort(key=lambda item: (-item[3], item[0], item[1]))
    negative_pairs.sort(key=lambda item: (-item[3], item[0], item[1]))
    budget = max(0, int(pair_budget))
    if budget == 0:
        return [], []
    if positive_pairs and negative_pairs:
        positive_budget = max(1, budget // 2)
        negative_budget = budget - positive_budget
    elif positive_pairs:
        positive_budget, negative_budget = budget, 0
    else:
        positive_budget, negative_budget = 0, budget
    return positive_pairs[:positive_budget], negative_pairs[:negative_budget]


def enumerate_horizon_paths(
    source: int,
    target: int,
    *,
    horizon: int,
    edge_logit_matrix: np.ndarray,
    paths_per_pair: int,
    expansion_budget: int = 4096,
    deadline_monotonic: Optional[float] = None,
    return_stats: bool = False,
):
    max_paths = max(1, int(paths_per_pair))
    n_nodes = int(edge_logit_matrix.shape[0])
    source, target = int(source), int(target)
    if source == target or not (0 <= source < n_nodes and 0 <= target < n_nodes):
        return ([], 0, False) if return_stats else []

    def edge_cost(i: int, j: int) -> float:
        logit = float(edge_logit_matrix[i, j])
        return float(np.logaddexp(0.0, -logit))

    queue: list[tuple[float, tuple[int, ...]]] = [(0.0, (source,))]
    paths: list[tuple[list[int], float]] = []
    expansion_count = 0
    truncated = False
    while queue and len(paths) < max_paths:
        if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
            truncated = True
            break
        if expansion_count >= int(expansion_budget):
            truncated = True
            break
        cost, path = heapq.heappop(queue)
        expansion_count += 1
        current = path[-1]
        if current == target:
            score = sum(
                float(edge_logit_matrix[path[idx], path[idx + 1]])
                for idx in range(len(path) - 1)
            )
            paths.append((list(path), score))
            continue
        if len(path) - 1 >= int(horizon):
            continue
        for neighbor in range(n_nodes):
            if neighbor == current or neighbor in path:
                continue
            next_path = path + (neighbor,)
            heapq.heappush(
                queue,
                (cost + edge_cost(current, neighbor), next_path),
            )
    if return_stats:
        return paths, expansion_count, truncated
    return paths


def _validate_probability_matrix(matrix: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values.")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"{name} values must be within [0, 1].")
    return values


def _validated_slack_total(values, label: str) -> float:
    if any(
        value is None or not np.isfinite(value) or float(value) < -_BINARY_TOLERANCE
        for value in values
    ):
        raise RuntimeError(
            f"Adjacency incumbent contains invalid {label} slack values."
        )
    return float(sum(float(value) for value in values))


def find_negative_horizon_cuts(
    adjacency: np.ndarray,
    negative_pairs: Sequence[tuple[int, int, float, float]],
    *,
    horizon: int,
) -> list[tuple[list[Edge], float]]:
    graph = nx.from_numpy_array(np.asarray(adjacency, dtype=int))
    cuts = []
    seen = set()
    for i, j, _probability, confidence in negative_pairs:
        try:
            path = nx.shortest_path(graph, int(i), int(j))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        if len(path) - 1 > int(horizon):
            continue
        edges = path_edges(path)
        frozen = frozenset(edges)
        if frozen and frozen not in seen:
            seen.add(frozen)
            cuts.append((edges, float(confidence)))
    return cuts


def count_negative_horizon_violations(
    adjacency: np.ndarray,
    negative_pairs: Sequence[tuple[int, int, float, float]],
    *,
    horizon: int,
) -> int:
    graph = nx.from_numpy_array(np.asarray(adjacency, dtype=int))
    violation_count = 0
    for i, j, _probability, _confidence in negative_pairs:
        try:
            if nx.shortest_path_length(graph, int(i), int(j)) <= int(horizon):
                violation_count += 1
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
    return violation_count


def _project_problem_to_active_nodes(
    prob_matrix: np.ndarray,
    target_degrees: Sequence[int],
    active_node_mask: Optional[np.ndarray],
    forbidden_edge_sets: Optional[Iterable[Iterable[Sequence[Any]]]],
    horizon_probability_matrix: Optional[np.ndarray],
):
    full_prob_matrix = _validate_probability_matrix(prob_matrix, "prob_matrix")
    n_slots = full_prob_matrix.shape[0]
    if full_prob_matrix.shape != (n_slots, n_slots):
        raise ValueError("prob_matrix must be square.")
    if len(target_degrees) != n_slots:
        raise ValueError("target_degrees must align with prob_matrix.")
    active_mask = (
        np.ones(n_slots, dtype=bool)
        if active_node_mask is None
        else np.asarray(active_node_mask, dtype=bool)
    )
    if active_mask.shape != (n_slots,):
        raise ValueError("active_node_mask must align with prob_matrix.")
    active_indices = np.flatnonzero(active_mask)
    original_to_local = {int(original): local for local, original in enumerate(active_indices)}
    local_prob_matrix = full_prob_matrix[np.ix_(active_indices, active_indices)]
    local_target_degrees = [int(target_degrees[idx]) for idx in active_indices]

    normalized_cuts = _normalize_violating_edge_sets(
        [] if forbidden_edge_sets is None else forbidden_edge_sets,
        n_nodes=n_slots,
    )
    local_cuts = []
    for edge_set in normalized_cuts:
        if not all(u in original_to_local and v in original_to_local for u, v in edge_set):
            continue
        local_cuts.append(
            frozenset(
                edge_key(original_to_local[u], original_to_local[v])
                for u, v in edge_set
            )
        )

    local_horizon = None
    if horizon_probability_matrix is not None:
        horizon_matrix = _validate_probability_matrix(
            horizon_probability_matrix,
            "horizon_probability_matrix",
        )
        if horizon_matrix.shape != (n_slots, n_slots):
            raise ValueError(
                "horizon_probability_matrix must align with prob_matrix; "
                f"received {horizon_matrix.shape} for n={n_slots}."
            )
        local_horizon = horizon_matrix[np.ix_(active_indices, active_indices)]
    return (
        active_indices,
        local_prob_matrix,
        local_target_degrees,
        local_cuts,
        local_horizon,
    )


def _validate_adjacency(
    adjacency: np.ndarray,
    *,
    connectivity: bool,
    exact_edge_count: Optional[int],
    forbidden_edge_sets,
) -> None:
    adjacency = np.asarray(adjacency, dtype=float)
    if not np.allclose(adjacency, adjacency.T, atol=_BINARY_TOLERANCE):
        raise RuntimeError("Adjacency incumbent is not symmetric.")
    if not np.allclose(np.diag(adjacency), 0.0, atol=_BINARY_TOLERANCE):
        raise RuntimeError("Adjacency incumbent contains self-loops.")
    rounded = np.rint(adjacency)
    if not np.allclose(adjacency, rounded, atol=_BINARY_TOLERANCE):
        raise RuntimeError("Adjacency incumbent contains fractional edge values.")
    if not np.all(np.isfinite(adjacency)):
        raise RuntimeError("Adjacency incumbent contains non-finite edge values.")
    if np.any(rounded < 0.0) or np.any(rounded > 1.0):
        raise RuntimeError("Adjacency incumbent contains edge values outside {0, 1}.")
    adjacency = rounded.astype(int)
    n_nodes = adjacency.shape[0]
    if connectivity and n_nodes >= 2 and not nx.is_connected(nx.from_numpy_array(adjacency)):
        raise RuntimeError("Adjacency incumbent violates the connectivity constraint.")
    if exact_edge_count is not None and int(adjacency.sum() // 2) != int(exact_edge_count):
        raise RuntimeError("Adjacency incumbent violates the exact edge-count constraint.")
    for edge_set in forbidden_edge_sets:
        if sum(int(adjacency[u, v]) for u, v in edge_set) >= len(edge_set):
            raise RuntimeError("Adjacency incumbent violates a forbidden edge-set constraint.")


def solve_adjacency(
    prob_matrix: np.ndarray,
    target_degrees: Sequence[int],
    *,
    target_edge_count: Optional[int] = None,
    time_limit_seconds: Optional[float] = None,
    verbose: bool = False,
    alpha: float = 0.7,
    connectivity: bool = True,
    forbidden_edge_sets: Optional[Iterable[Iterable[Sequence[Any]]]] = None,
    active_node_mask: Optional[np.ndarray] = None,
    degree_slack_penalty: float = 1e6,
    edge_count_slack_penalty: Optional[float] = 2.0,
    warm_start_mst: bool = True,
    horizon_probability_matrix: Optional[np.ndarray] = None,
    horizon: Optional[int] = None,
    use_horizon_constraints: bool = True,
    horizon_constraint_weight: float = 2.0,
    horizon_positive_threshold: float = 0.8,
    horizon_negative_threshold: float = 0.2,
    horizon_pair_budget: int = 24,
    horizon_paths_per_pair: int = 8,
    horizon_path_expansion_budget: int = 4096,
    horizon_max_iterations: int = 1,
    solver_threads: Optional[int] = None,
    deadline_monotonic: Optional[float] = None,
) -> tuple[np.ndarray, AdjacencySolveReport]:
    started_at = time.monotonic()
    if deadline_monotonic is None and time_limit_seconds is not None:
        deadline_monotonic = started_at + float(time_limit_seconds)
    elif deadline_monotonic is not None and time_limit_seconds is not None:
        deadline_monotonic = min(
            float(deadline_monotonic),
            started_at + float(time_limit_seconds),
        )

    def check_deadline(stage: str) -> Optional[float]:
        if deadline_monotonic is None:
            return None
        remaining = float(deadline_monotonic) - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError(
                f"Adjacency solve exhausted its shared time budget during {stage}."
            )
        return remaining
    if not np.isfinite(alpha) or float(alpha) <= 0.0:
        raise ValueError("alpha must be finite and > 0.")
    if not np.isfinite(degree_slack_penalty) or float(degree_slack_penalty) <= 0.0:
        raise ValueError("degree_slack_penalty must be finite and > 0.")
    if edge_count_slack_penalty is not None and (
        not np.isfinite(edge_count_slack_penalty)
        or float(edge_count_slack_penalty) <= 0.0
    ):
        raise ValueError("edge_count_slack_penalty must be finite and > 0 when provided.")
    if time_limit_seconds is not None and (
        not np.isfinite(time_limit_seconds) or float(time_limit_seconds) <= 0.0
    ):
        raise ValueError("time_limit_seconds must be finite and > 0 when provided.")
    target_degree_values = np.asarray(target_degrees, dtype=float)
    if not np.all(np.isfinite(target_degree_values)) or np.any(target_degree_values < 0.0):
        raise ValueError("target_degrees must contain finite nonnegative values.")
    if target_edge_count is not None and int(target_edge_count) < 0:
        raise ValueError("target_edge_count must be >= 0 when provided.")
    if horizon is not None and int(horizon) < 1:
        raise ValueError("horizon must be >= 1 when provided.")
    if (
        not np.isfinite(horizon_constraint_weight)
        or float(horizon_constraint_weight) < 0.0
    ):
        raise ValueError("horizon_constraint_weight must be finite and >= 0.")
    for value, name in (
        (horizon_positive_threshold, "horizon_positive_threshold"),
        (horizon_negative_threshold, "horizon_negative_threshold"),
    ):
        if not np.isfinite(value) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be within [0, 1].")
    if float(horizon_negative_threshold) > float(horizon_positive_threshold):
        raise ValueError(
            "horizon_negative_threshold must be <= horizon_positive_threshold."
        )
    if int(horizon_pair_budget) < 0:
        raise ValueError("horizon_pair_budget must be >= 0.")
    if int(horizon_paths_per_pair) < 1:
        raise ValueError("horizon_paths_per_pair must be >= 1.")
    if int(horizon_path_expansion_budget) < 1:
        raise ValueError("horizon_path_expansion_budget must be >= 1.")
    if int(horizon_max_iterations) < 0:
        raise ValueError("horizon_max_iterations must be >= 0.")
    if solver_threads is not None and int(solver_threads) < 1:
        raise ValueError("solver_threads must be >= 1 when provided.")
    check_deadline("active-node projection")
    (
        active_indices,
        local_prob_matrix,
        local_target_degrees,
        local_forbidden_sets,
        local_horizon_matrix,
    ) = _project_problem_to_active_nodes(
        prob_matrix,
        target_degrees,
        active_node_mask,
        forbidden_edge_sets,
        horizon_probability_matrix,
    )
    n_slots = np.asarray(prob_matrix).shape[0]
    n_active = len(active_indices)
    resolved_edge_count = resolve_target_edge_count(
        n_active,
        target_edge_count,
        connectivity=connectivity,
    )

    if n_active <= 1:
        adjacency = np.zeros((n_slots, n_slots), dtype=int)
        report = AdjacencySolveReport(
            solver_status="Trivial",
            solver_status_code=pulp.LpStatusOptimal,
            solution_status="Optimal Solution Found",
            solution_status_code=pulp.LpSolutionOptimal,
            optimal=True,
            used_incumbent=False,
            elapsed_seconds=time.monotonic() - started_at,
            active_node_count=n_active,
            solve_count=0,
            horizon_iterations=0,
            solver_termination_reason="trivial",
            horizon_termination_reason="not_enabled",
        )
        return adjacency, report

    if alpha != 1.0:
        local_prob_matrix = np.power(local_prob_matrix, alpha)
    check_deadline("probability preprocessing")
    edge_logit_matrix = np.zeros((n_active, n_active), dtype=float)
    for i in range(n_active):
        for j in range(i + 1, n_active):
            logit = edge_logit(float(local_prob_matrix[i, j]))
            edge_logit_matrix[i, j] = edge_logit_matrix[j, i] = logit

    horizon_enabled = (
        use_horizon_constraints
        and local_horizon_matrix is not None
        and horizon is not None
        and int(horizon) > 1
        and float(horizon_constraint_weight) > 0.0
    )
    if horizon_enabled:
        positive_pairs, negative_pairs = select_horizon_pairs(
            local_horizon_matrix,
            positive_threshold=horizon_positive_threshold,
            negative_threshold=horizon_negative_threshold,
            pair_budget=horizon_pair_budget,
        )
    else:
        positive_pairs, negative_pairs = [], []

    positive_path_sets = []
    horizon_path_expansion_count = 0
    horizon_path_search_truncated = False
    for pair in positive_pairs:
        check_deadline("horizon path enumeration")
        paths, expansion_count, truncated = enumerate_horizon_paths(
            pair[0],
            pair[1],
            horizon=int(horizon),
            edge_logit_matrix=edge_logit_matrix,
            paths_per_pair=horizon_paths_per_pair,
            expansion_budget=horizon_path_expansion_budget,
            deadline_monotonic=deadline_monotonic,
            return_stats=True,
        )
        horizon_path_expansion_count += expansion_count
        horizon_path_search_truncated = horizon_path_search_truncated or truncated
        positive_path_sets.append((pair, paths))

    solve_count = 0
    last_status_code = pulp.LpStatusNotSolved
    last_solution_status_code = pulp.LpSolutionNoSolutionFound
    last_objective_value: Optional[float] = None
    last_degree_slack_total = 0.0
    last_edge_count_slack = 0.0

    def solve_once(negative_cuts, solve_name: str) -> np.ndarray:
        nonlocal solve_count, last_status_code, last_solution_status_code
        nonlocal last_objective_value, last_degree_slack_total, last_edge_count_slack
        check_deadline("MILP construction")

        problem = pulp.LpProblem(solve_name, pulp.LpMaximize)
        x = {
            (i, j): pulp.LpVariable(f"x_{i}_{j}", cat="Binary")
            for i in range(n_active)
            for j in range(i + 1, n_active)
        }
        degree_under = {
            i: pulp.LpVariable(f"degree_under_{i}", lowBound=0, cat="Integer")
            for i in range(n_active)
        }
        degree_over = {
            i: pulp.LpVariable(f"degree_over_{i}", lowBound=0, cat="Integer")
            for i in range(n_active)
        }
        objective_terms = [
            edge_logit_matrix[i, j] * x[(i, j)] for i, j in x
        ]
        objective_terms.extend(
            -float(degree_slack_penalty) * (degree_under[i] + degree_over[i])
            for i in range(n_active)
        )
        for i in range(n_active):
            incident = [x[edge] for edge in x if i in edge]
            problem += (
                pulp.lpSum(incident) + degree_under[i] - degree_over[i]
                == local_target_degrees[i]
            ), f"Degree_{i}"

        count_under = None
        count_over = None
        if resolved_edge_count is not None:
            if edge_count_slack_penalty is None:
                problem += pulp.lpSum(x.values()) == resolved_edge_count, "EdgeCount"
            else:
                count_under = pulp.LpVariable("edge_count_under", lowBound=0, cat="Integer")
                count_over = pulp.LpVariable("edge_count_over", lowBound=0, cat="Integer")
                problem += (
                    pulp.lpSum(x.values()) + count_under - count_over
                    == resolved_edge_count
                ), "EdgeCountSoft"
                objective_terms.append(
                    -float(edge_count_slack_penalty) * (count_under + count_over)
                )

        if connectivity:
            directed_edges = list(x) + [(j, i) for i, j in x]
            flow = {
                edge: pulp.LpVariable(
                    f"flow_{edge[0]}_{edge[1]}",
                    lowBound=0,
                    cat="Continuous",
                )
                for edge in directed_edges
            }
            capacity = n_active - 1
            for node in range(n_active):
                inflow = pulp.lpSum(value for (u, v), value in flow.items() if v == node)
                outflow = pulp.lpSum(value for (u, v), value in flow.items() if u == node)
                problem += (
                    outflow - inflow == capacity
                    if node == 0
                    else inflow - outflow == 1
                ), f"Flow_{node}"
            for (u, v), value in flow.items():
                problem += value <= capacity * x[edge_key(u, v)], f"FlowCouple_{u}_{v}"

        for cut_idx, edge_set in enumerate(local_forbidden_sets):
            problem += (
                pulp.lpSum(x[edge] for edge in edge_set) <= len(edge_set) - 1
            ), f"ForbiddenMotif_{cut_idx}"

        for pair_idx, ((i, j, _probability, confidence), paths) in enumerate(
            positive_path_sets
        ):
            if not paths:
                continue
            slack = pulp.LpVariable(
                f"hpos_slack_{pair_idx}_{i}_{j}",
                lowBound=0,
                upBound=1,
                cat="Continuous",
            )
            objective_terms.append(
                -float(horizon_constraint_weight) * float(confidence) * slack
            )
            path_vars = []
            for path_idx, (path, _score) in enumerate(paths):
                edges = path_edges(path)
                path_var = pulp.LpVariable(
                    f"hpos_path_{pair_idx}_{path_idx}_{i}_{j}",
                    cat="Binary",
                )
                path_vars.append(path_var)
                for edge in edges:
                    problem += path_var <= x[edge]
                problem += path_var >= pulp.lpSum(x[edge] for edge in edges) - len(edges) + 1
            problem += pulp.lpSum(path_vars) + slack >= 1

        for cut_idx, (edge_set, confidence) in enumerate(negative_cuts):
            slack = pulp.LpVariable(
                f"hneg_slack_{cut_idx}",
                lowBound=0,
                upBound=1,
                cat="Continuous",
            )
            objective_terms.append(
                -float(horizon_constraint_weight) * float(confidence) * slack
            )
            problem += (
                pulp.lpSum(x[edge] for edge in edge_set)
                <= len(edge_set) - 1 + slack
            )
        problem += pulp.lpSum(objective_terms)

        if warm_start_mst:
            graph = nx.Graph()
            graph.add_nodes_from(range(n_active))
            for i, j in x:
                graph.add_edge(i, j, weight=local_prob_matrix[i, j])
            tree = nx.maximum_spanning_tree(graph)
            for (i, j), variable in x.items():
                variable.start = 1 if tree.has_edge(i, j) else 0

        solver_kwargs: dict[str, Any] = {"msg": verbose}
        remaining = check_deadline("CBC solve")
        if remaining is not None:
            solver_kwargs["timeLimit"] = remaining
        if solver_threads is not None:
            solver_kwargs["threads"] = max(1, int(solver_threads))
        solve_count += 1
        problem.solve(pulp.PULP_CBC_CMD(**solver_kwargs))
        last_status_code = int(getattr(problem, "status", pulp.LpStatusUndefined))
        last_solution_status_code = int(
            getattr(problem, "sol_status", pulp.LpSolutionNoSolutionFound)
        )
        if (
            last_status_code == pulp.LpStatusOptimal
            and last_solution_status_code == pulp.LpSolutionNoSolutionFound
        ):
            # Some solver adapters and test doubles only populate ``status``.
            last_solution_status_code = pulp.LpSolutionOptimal
        has_acceptable_solution = last_solution_status_code in {
            pulp.LpSolutionOptimal,
            pulp.LpSolutionIntegerFeasible,
        }
        if not has_acceptable_solution:
            status_label = pulp.LpStatus.get(last_status_code, str(last_status_code))
            solution_label = pulp.LpSolution.get(
                last_solution_status_code,
                str(last_solution_status_code),
            )
            raise RuntimeError(
                "Adjacency ILP did not produce a feasible solution "
                f"(status={status_label}, solution_status={solution_label}, "
                f"n_active={n_active}, connectivity={connectivity})."
            )

        adjacency = np.zeros((n_active, n_active), dtype=float)
        for (i, j), variable in x.items():
            value = pulp.value(variable)
            if value is None:
                raise RuntimeError(
                    "Adjacency ILP finished without assigning all decision variables "
                    f"(missing_edge=({i}, {j}))."
                )
            adjacency[i, j] = adjacency[j, i] = float(value)
        _validate_adjacency(
            adjacency,
            connectivity=connectivity,
            exact_edge_count=(
                resolved_edge_count if edge_count_slack_penalty is None else None
            ),
            forbidden_edge_sets=local_forbidden_sets,
        )
        objective_value = pulp.value(problem.objective)
        last_objective_value = (
            None
            if objective_value is None or not np.isfinite(objective_value)
            else float(objective_value)
        )
        degree_slack_values = [
            pulp.value(variable)
            for i in range(n_active)
            for variable in (degree_under[i], degree_over[i])
        ]
        if any(
            value is None or not np.isfinite(value) or float(value) < -_BINARY_TOLERANCE
            for value in degree_slack_values
        ):
            raise RuntimeError("Adjacency incumbent contains invalid degree slack values.")
        last_degree_slack_total = float(sum(float(value) for value in degree_slack_values))
        last_edge_count_slack = (
            0.0
            if count_under is None or count_over is None
            else _validated_slack_total(
                (pulp.value(count_under), pulp.value(count_over)),
                "edge-count",
            )
        )
        return np.rint(adjacency).astype(int)

    negative_cuts: list[tuple[list[Edge], float]] = []
    adjacency = solve_once(negative_cuts, "AdjacencyMatrixOptimization")
    horizon_iterations = 0
    seen_negative_cuts: set[frozenset[Edge]] = set()
    if negative_pairs:
        horizon_termination_reason = "iteration_limit"
    elif positive_pairs:
        horizon_termination_reason = "positive_only"
    else:
        horizon_termination_reason = "not_enabled"
    for iteration in range(max(0, int(horizon_max_iterations))):
        discovered = find_negative_horizon_cuts(
            adjacency,
            negative_pairs,
            horizon=int(horizon) if horizon is not None else 0,
        )
        new_cuts = []
        for edges, confidence in discovered:
            frozen = frozenset(edges)
            if frozen not in seen_negative_cuts:
                seen_negative_cuts.add(frozen)
                new_cuts.append((edges, confidence))
        if not new_cuts:
            horizon_termination_reason = (
                "satisfied" if not discovered else "no_new_cuts"
            )
            break
        negative_cuts.extend(new_cuts)
        horizon_iterations = iteration + 1
        try:
            adjacency = solve_once(
                negative_cuts,
                f"AdjacencyMatrixOptimizationHorizonRepair{horizon_iterations}",
            )
        except TimeoutError:
            horizon_termination_reason = "time_budget_exhausted"
            break

    unresolved_horizon_pair_count = count_negative_horizon_violations(
        adjacency,
        negative_pairs,
        horizon=int(horizon) if horizon is not None else 0,
    )
    if negative_pairs and unresolved_horizon_pair_count == 0:
        horizon_termination_reason = "satisfied"
    elif negative_pairs and horizon_termination_reason == "satisfied":
        horizon_termination_reason = "no_new_cuts"

    full_adjacency = np.zeros((n_slots, n_slots), dtype=int)
    full_adjacency[np.ix_(active_indices, active_indices)] = adjacency
    optimal = last_solution_status_code == pulp.LpSolutionOptimal
    report = AdjacencySolveReport(
        solver_status=pulp.LpStatus.get(last_status_code, str(last_status_code)),
        solver_status_code=last_status_code,
        solution_status=pulp.LpSolution.get(
            last_solution_status_code,
            str(last_solution_status_code),
        ),
        solution_status_code=last_solution_status_code,
        optimal=optimal,
        used_incumbent=not optimal,
        elapsed_seconds=time.monotonic() - started_at,
        active_node_count=n_active,
        solve_count=solve_count,
        horizon_iterations=horizon_iterations,
        solver_termination_reason=(
            "optimal" if optimal else "integer_feasible"
        ),
        horizon_termination_reason=horizon_termination_reason,
        objective_value=last_objective_value,
        degree_slack_total=last_degree_slack_total,
        edge_count_slack=last_edge_count_slack,
        unresolved_horizon_pair_count=unresolved_horizon_pair_count,
        horizon_path_search_truncated=horizon_path_search_truncated,
        horizon_path_expansion_count=horizon_path_expansion_count,
    )
    return full_adjacency, report
