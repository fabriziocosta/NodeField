"""Active-node MILP projection for graph adjacency decoding."""

from __future__ import annotations

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
) -> list[tuple[list[int], float]]:
    max_paths = max(1, int(paths_per_pair))
    max_candidates = max_paths * 16
    nodes = set(range(edge_logit_matrix.shape[0]))
    ordered_neighbors = {
        node: sorted(
            (candidate for candidate in nodes if candidate != node),
            key=lambda candidate: (-float(edge_logit_matrix[node, candidate]), candidate),
        )
        for node in nodes
    }
    candidates: list[tuple[list[int], float]] = []

    def walk(path: list[int], visited: set[int]) -> None:
        if len(candidates) >= max_candidates:
            return
        current = path[-1]
        if current == target:
            score = sum(
                float(edge_logit_matrix[path[idx], path[idx + 1]])
                for idx in range(len(path) - 1)
            )
            candidates.append((list(path), score))
            return
        if len(path) - 1 >= int(horizon):
            return
        for neighbor in ordered_neighbors[current]:
            if neighbor in visited:
                continue
            walk(path + [neighbor], visited | {neighbor})
            if len(candidates) >= max_candidates:
                return

    walk([int(source)], {int(source)})
    candidates.sort(key=lambda item: (-item[1], len(item[0]), item[0]))
    return candidates[:max_paths]


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


def _project_problem_to_active_nodes(
    prob_matrix: np.ndarray,
    target_degrees: Sequence[int],
    active_node_mask: Optional[np.ndarray],
    forbidden_edge_sets: Optional[Iterable[Iterable[Sequence[Any]]]],
    horizon_probability_matrix: Optional[np.ndarray],
):
    full_prob_matrix = np.asarray(prob_matrix, dtype=float)
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
        horizon_matrix = np.asarray(horizon_probability_matrix, dtype=float)
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
    horizon_max_iterations: int = 1,
    solver_threads: Optional[int] = None,
) -> tuple[np.ndarray, AdjacencySolveReport]:
    started_at = time.monotonic()
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
        )
        return adjacency, report

    if alpha != 1.0:
        local_prob_matrix = np.power(local_prob_matrix, alpha)
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

    solve_count = 0
    last_status_code = pulp.LpStatusNotSolved
    last_solution_status_code = pulp.LpSolutionNoSolutionFound

    def solve_once(negative_cuts, solve_name: str) -> np.ndarray:
        nonlocal solve_count, last_status_code, last_solution_status_code
        remaining = None
        if time_limit_seconds is not None:
            remaining = float(time_limit_seconds) - (time.monotonic() - started_at)
            if remaining <= 0.0:
                raise TimeoutError("Adjacency solve exhausted its shared time budget.")

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

        for pair_idx, (i, j, _probability, confidence) in enumerate(positive_pairs):
            paths = enumerate_horizon_paths(
                i,
                j,
                horizon=int(horizon),
                edge_logit_matrix=edge_logit_matrix,
                paths_per_pair=horizon_paths_per_pair,
            )
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
        if remaining is not None:
            solver_kwargs["timeLimit"] = max(0.01, remaining)
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
        return np.rint(adjacency).astype(int)

    negative_cuts: list[tuple[list[Edge], float]] = []
    adjacency = solve_once(negative_cuts, "AdjacencyMatrixOptimization")
    horizon_iterations = 0
    seen_negative_cuts: set[frozenset[Edge]] = set()
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
            break
        negative_cuts.extend(new_cuts)
        horizon_iterations = iteration + 1
        adjacency = solve_once(
            negative_cuts,
            f"AdjacencyMatrixOptimizationHorizonRepair{horizon_iterations}",
        )

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
    )
    return full_adjacency, report
