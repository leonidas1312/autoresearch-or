from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

import prepare
from prepare import TSPInstance

EXACT_NEAREST_NEIGHBOR_LIMIT = 12_000
FULL_TWO_OPT_LIMIT = 400
WINDOWED_TWO_OPT_LIMIT = 100_000
TIME_BOXED_MULTI_START_LIMIT = 128
RELOCATE_RESERVE_FRACTION = 0.10
PER_CANDIDATE_RELOCATE_LIMIT_S = 0.01
ITERATED_LOCAL_SEARCH_MIN_DIMENSION = 40
ITERATED_LOCAL_SEARCH_MAX_DIMENSION = 90
ITERATED_LOCAL_SEARCH_BLOCK_SHIFT_WIDTH = 6
PILOT_START_RANKING_LIMIT = 128
PILOT_START_RANKING_MAX_S = 0.015
GRID_NEAREST_SIDE = 184
CANDIDATE_MATRIX_LIMIT = 1_200
VALIDATION_RESERVE_BASE_S = 0.05
VALIDATION_RESERVE_EUC_S_PER_NODE = 0.0000002
VALIDATION_RESERVE_LARGE_EUC_S_PER_NODE = 0.0000007
VALIDATION_RESERVE_GEO_S_PER_NODE = 0.0000015
VALIDATION_RESERVE_MAX_FRACTION = 0.60


@dataclass(frozen=True, slots=True)
class SolverSpec:
    solver_name: str
    solver_kind: str = "multistart"
    start_order: str = "time_boxed"
    max_starts: int | None = None
    grid_side: int | None = None
    grid_polish: bool = False
    restart_reserve_fraction: float = RELOCATE_RESERVE_FRACTION
    candidate_relocate_limit_s: float = PER_CANDIDATE_RELOCATE_LIMIT_S
    ils_enabled: bool = False
    ils_block_width: int = ITERATED_LOCAL_SEARCH_BLOCK_SHIFT_WIDTH
    candidate_neighbors: int = 48
    candidate_constructions: tuple[str, ...] = ()
    candidate_relocate_near: int = 24
    candidate_relocate_blocks: tuple[int, ...] = (1,)


# The scheduler receives the total selected-suite budget directly from the V3 harness.
# Weights are normalized over the active benchmark instances before allocation.
SCHEDULER_BUDGET_WEIGHTS = {
    "att48": 0.09,
    "eil51": 0.08,
    "berlin52": 0.06,
    "pr76": 0.20,
    "rd100": 0.30,
    "lin318": 2.00,
    "pcb442": 2.10,
    "rat783": 0.15,
    "pr1002": 4.90,
    "nrw1379": 0.30,
    "pcb3038": 0.45,
    "qa194": 0.25,
    "uy734": 0.15,
    "lu980": 0.60,
    "gr9882": 0.90,
    "ch71009": 1.50,
    "world": 1.40,
}

SCHEDULER_RUN_ORDER = {
    "lin318": 0,
    "pr1002": 1,
    "pcb442": 2,
}


BENCHMARK_SOLVERS: dict[str, SolverSpec] = {
    "att48": SolverSpec(
        solver_name="att48_candidate_q1",
        solver_kind="candidate_local",
        candidate_neighbors=48,
        candidate_constructions=("nn_q1", "insertion_diameter_far"),
        candidate_relocate_near=24,
        candidate_relocate_blocks=(1, 2),
    ),
    "eil51": SolverSpec(
        solver_name="eil51_candidate_last",
        solver_kind="candidate_local",
        candidate_neighbors=48,
        candidate_constructions=("nn_last", "nn_q1"),
        candidate_relocate_near=24,
        candidate_relocate_blocks=(1, 2),
    ),
    "berlin52": SolverSpec(
        solver_name="berlin52_candidate_diameter",
        solver_kind="candidate_local",
        candidate_neighbors=52,
        candidate_constructions=("insertion_diameter_far",),
        candidate_relocate_near=24,
        candidate_relocate_blocks=(1, 2, 3),
    ),
    "pr76": SolverSpec(
        solver_name="pr76_candidate_nn0",
        solver_kind="candidate_local",
        candidate_neighbors=64,
        candidate_constructions=("nn_0", "insertion_other_far"),
        candidate_relocate_near=32,
        candidate_relocate_blocks=(1, 2, 3),
    ),
    "rd100": SolverSpec(
        solver_name="rd100_candidate_extreme",
        solver_kind="candidate_local",
        candidate_neighbors=72,
        candidate_constructions=("insertion_extreme_far", "nn_0", "nn_q1"),
        candidate_relocate_near=32,
        candidate_relocate_blocks=(1, 2, 3),
    ),
    "lin318": SolverSpec(
        solver_name="lin318_center_anchor",
        start_order="center_anchor",
        max_starts=1,
        restart_reserve_fraction=0.15,
        candidate_relocate_limit_s=0.0,
    ),
    "rat783": SolverSpec(
        solver_name="rat783_grid_polished",
        solver_kind="grid_nearest",
        max_starts=2,
        grid_side=48,
        grid_polish=True,
    ),
    "nrw1379": SolverSpec(
        solver_name="nrw1379_grid_polished",
        solver_kind="grid_nearest",
        max_starts=2,
        grid_side=128,
        grid_polish=True,
    ),
    "pcb3038": SolverSpec(
        solver_name="pcb3038_grid_polished",
        solver_kind="grid_nearest",
        max_starts=2,
        grid_side=128,
        grid_polish=True,
    ),
    "qa194": SolverSpec(
        solver_name="qa194_grid_polished",
        solver_kind="grid_nearest",
        max_starts=2,
        grid_side=16,
        grid_polish=True,
    ),
    "uy734": SolverSpec(
        solver_name="uy734_grid_polished",
        solver_kind="grid_nearest",
        max_starts=2,
        grid_side=48,
        grid_polish=True,
    ),
    "pcb442": SolverSpec(
        solver_name="pcb442_candidate_midpoint",
        solver_kind="candidate_local",
        candidate_neighbors=80,
        candidate_constructions=("nn_mid", "nn_0", "insertion_diameter_near"),
        candidate_relocate_near=40,
        candidate_relocate_blocks=(1, 2, 3),
    ),
    "gr9882": SolverSpec(
        solver_name="gr9882_grid_nearest",
        solver_kind="grid_nearest",
        max_starts=2,
        grid_side=280,
    ),
    "ch71009": SolverSpec(
        solver_name="ch71009_grid_nearest",
        solver_kind="grid_nearest",
        max_starts=1,
        grid_side=320,
    ),
    "pr1002": SolverSpec(
        solver_name="pr1002_diameter_insertion",
        solver_kind="candidate_local",
        candidate_neighbors=80,
        candidate_constructions=("insertion_diameter_far",),
        candidate_relocate_near=36,
        candidate_relocate_blocks=(1, 2, 3),
    ),
}

DEFAULT_SOLVER_SPEC = SolverSpec(
    solver_name="generic_multistart",
    solver_kind="multistart",
    start_order="time_boxed",
    candidate_relocate_limit_s=0.0,
    ils_enabled=False,
)

_DISTANCE_MATRIX_CACHE: dict[tuple[str, int], list[list[int]]] = {}


def compute_tour_length(instance: TSPInstance, tour: list[int]) -> float:
    if not tour:
        raise ValueError(f"Empty tour for {instance.name}")
    total = 0
    size = len(tour)
    for index in range(size):
        total += _distance(instance, tour[index], tour[(index + 1) % size])
    return float(total)


def _distance_matrix(instance: TSPInstance) -> list[list[int]] | None:
    if instance.dimension > FULL_TWO_OPT_LIMIT:
        return None

    cache_key = (instance.name, instance.dimension)
    matrix = _DISTANCE_MATRIX_CACHE.get(cache_key)
    if matrix is not None:
        return matrix

    n = instance.dimension
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            distance = prepare.edge_distance(instance, i, j)
            matrix[i][j] = distance
            matrix[j][i] = distance
    _DISTANCE_MATRIX_CACHE[cache_key] = matrix
    return matrix


def _distance(instance: TSPInstance, a: int, b: int) -> int:
    matrix = _distance_matrix(instance)
    if matrix is not None:
        return matrix[a][b]
    return prepare.edge_distance(instance, a, b)


def schedule_budgets(
    instances: Sequence[TSPInstance],
    total_budget_s: float,
) -> dict[str, tuple[float, float]]:
    if not instances:
        return {}
    raw_weights = {
        instance.name: SCHEDULER_BUDGET_WEIGHTS.get(instance.name, 1.0)
        for instance in instances
    }
    total_weight = sum(raw_weights.values())
    if total_weight <= 0:
        equal = 1.0 / len(instances)
        return {
            instance.name: (total_budget_s * equal, equal)
            for instance in instances
        }
    return {
        instance.name: (total_budget_s * (weight / total_weight), weight / total_weight)
        for instance in instances
        for weight in (raw_weights[instance.name],)
    }


def estimate_validation_reserve_s(
    instances: Sequence[TSPInstance],
    total_budget_s: float,
) -> float:
    estimate = VALIDATION_RESERVE_BASE_S
    for instance in instances:
        if instance.edge_weight_type in {"GEO", "GEOM"}:
            estimate += instance.dimension * VALIDATION_RESERVE_GEO_S_PER_NODE
        elif instance.dimension > EXACT_NEAREST_NEIGHBOR_LIMIT:
            estimate += instance.dimension * VALIDATION_RESERVE_LARGE_EUC_S_PER_NODE
        else:
            estimate += instance.dimension * VALIDATION_RESERVE_EUC_S_PER_NODE
    return min(total_budget_s * VALIDATION_RESERVE_MAX_FRACTION, estimate)


def identity_tour(instance: TSPInstance) -> list[int]:
    return list(range(instance.dimension))


def choose_start_nodes(
    instance: TSPInstance,
    seed: int,
    start_order: str = "anchor_nodes",
) -> list[int]:
    n = instance.dimension
    if start_order == "center_anchor":
        anchor_nodes = [n // 2, n // 4, (3 * n) // 4, n - 1, 0]
    else:
        anchor_nodes = [n // 4, n // 2, (3 * n) // 4, n - 1, 0]
    rng = random.Random(seed)
    candidates = [node for node in anchor_nodes if 0 <= node < n]

    if n <= 100:
        target = min(8, n)
    elif n <= 1_000:
        target = min(4, n)
    elif n <= EXACT_NEAREST_NEIGHBOR_LIMIT:
        target = min(2, n)
    else:
        target = 1

    while len(candidates) < target:
        candidates.append(rng.randrange(n))

    unique: list[int] = []
    seen: set[int] = set()
    for node in candidates:
        if node not in seen:
            unique.append(node)
            seen.add(node)
        if len(unique) >= target:
            break
    return unique


def order_time_boxed_starts(instance: TSPInstance, seed: int) -> list[int]:
    starts = list(range(instance.dimension))
    if instance.dimension < 64:
        random.Random(seed).shuffle(starts)
        return starts

    centroid_x = sum(x for x, _ in instance.coords) / instance.dimension
    centroid_y = sum(y for _, y in instance.coords) / instance.dimension
    starts.sort(
        key=lambda node: (
            math.atan2(
                instance.coords[node][1] - centroid_y,
                instance.coords[node][0] - centroid_x,
            ),
            node,
        )
    )

    quantiles = max(5, min(8, round(instance.dimension / 16)))
    preferred = [starts[0], starts[-1]]
    bucket = instance.dimension / quantiles
    preferred.extend(
        starts[min(instance.dimension - 1, int((index + 0.5) * bucket))]
        for index in range(quantiles)
    )
    seen: set[int] = set()
    ordered: list[int] = []
    for node in preferred + starts:
        if node not in seen:
            ordered.append(node)
            seen.add(node)
    return ordered


def nearest_neighbor_tour(instance: TSPInstance, start: int, deadline: float) -> list[int]:
    n = instance.dimension
    visited = bytearray(n)
    visited[start] = 1
    tour = [start]
    current = start

    while len(tour) < n:
        if time.perf_counter() >= deadline:
            break
        best_node = -1
        best_distance = math.inf
        for candidate in range(n):
            if visited[candidate]:
                continue
            distance = _distance(instance, current, candidate)
            if distance < best_distance or (
                distance == best_distance and candidate < best_node
            ):
                best_distance = distance
                best_node = candidate
        if best_node < 0:
            break
        visited[best_node] = 1
        tour.append(best_node)
        current = best_node

    if len(tour) < n:
        for node in range(n):
            if not visited[node]:
                tour.append(node)
    return tour


def build_grid_index(
    instance: TSPInstance,
    grid_side: int,
) -> tuple[list[list[int]], list[tuple[int, int]]]:
    xs = [x for x, _ in instance.coords]
    ys = [y for _, y in instance.coords]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    span_x = max(max_x - min_x, 1e-9)
    span_y = max(max_y - min_y, 1e-9)

    cells: list[list[int]] = [[] for _ in range(grid_side * grid_side)]
    node_cells: list[tuple[int, int]] = []
    for node, (x, y) in enumerate(instance.coords):
        cell_x = min(grid_side - 1, max(0, int(((x - min_x) / span_x) * grid_side)))
        cell_y = min(grid_side - 1, max(0, int(((y - min_y) / span_y) * grid_side)))
        node_cells.append((cell_x, cell_y))
        cells[(cell_y * grid_side) + cell_x].append(node)
    return cells, node_cells


def grid_nearest_neighbor_tour(
    instance: TSPInstance,
    start: int,
    grid_side: int,
    cells: list[list[int]],
    node_cells: list[tuple[int, int]],
    deadline: float,
) -> tuple[list[int], bool]:
    n = instance.dimension
    visited = bytearray(n)
    visited[start] = 1
    tour = [start]
    current = start
    completed = True

    while len(tour) < n:
        if time.perf_counter() >= deadline:
            completed = False
            break

        cell_x, cell_y = node_cells[current]
        best_node = -1
        best_distance = math.inf
        for radius in range(grid_side):
            found_unvisited = False
            min_cell_x = max(0, cell_x - radius)
            max_cell_x = min(grid_side - 1, cell_x + radius)
            min_cell_y = max(0, cell_y - radius)
            max_cell_y = min(grid_side - 1, cell_y + radius)
            for candidate_cell_y in range(min_cell_y, max_cell_y + 1):
                row_offset = candidate_cell_y * grid_side
                for candidate_cell_x in range(min_cell_x, max_cell_x + 1):
                    for candidate in cells[row_offset + candidate_cell_x]:
                        if visited[candidate]:
                            continue
                        found_unvisited = True
                        distance = _distance(instance, current, candidate)
                        if distance < best_distance or (
                            distance == best_distance
                            and (best_node < 0 or candidate < best_node)
                        ):
                            best_distance = distance
                            best_node = candidate
            if found_unvisited:
                break

        if best_node < 0:
            completed = False
            break
        visited[best_node] = 1
        tour.append(best_node)
        current = best_node

    if len(tour) < n:
        for node in range(n):
            if not visited[node]:
                tour.append(node)
    return tour, completed


def build_start_order(
    instance: TSPInstance,
    spec: SolverSpec,
    seed: int,
    deadline: float,
) -> tuple[list[int], str]:
    base_order = order_time_boxed_starts(instance, seed)
    if spec.start_order != "pilot_ranked" or instance.dimension > PILOT_START_RANKING_LIMIT:
        return base_order, spec.start_order

    pilot_deadline = min(deadline, time.perf_counter() + PILOT_START_RANKING_MAX_S)
    scored: list[tuple[float, int]] = []
    seen: set[int] = set()
    for start in base_order:
        if time.perf_counter() >= pilot_deadline:
            break
        candidate_tour = nearest_neighbor_tour(instance, start, pilot_deadline)
        scored.append((compute_tour_length(instance, candidate_tour), start))
        seen.add(start)

    if not scored:
        return base_order, "pilot_ranked_fallback"

    scored.sort()
    ranked = [start for _, start in scored]
    ranked.extend(start for start in base_order if start not in seen)
    return ranked, "pilot_ranked"


def block_shift_kick(tour: list[int], rng: random.Random, width: int) -> list[int]:
    n = len(tour)
    if n < 4:
        return tour[:]
    width = min(width, n - 1)
    start = rng.randrange(0, n - width)
    block = tour[start : start + width]
    remainder = tour[:start] + tour[start + width :]
    insert_at = rng.randrange(0, len(remainder) + 1)
    return remainder[:insert_at] + block + remainder[insert_at:]


def sweep_tour(instance: TSPInstance, deadline: float) -> tuple[list[int], dict[str, Any]]:
    if time.perf_counter() >= deadline:
        return identity_tour(instance), {"sweep_mode": "identity_timeout"}

    n = instance.dimension
    if n > 100_000:
        if instance.edge_weight_type != "GEOM":
            return identity_tour(instance), {"sweep_mode": "identity_large"}
        bucket_size = 3584
        tour: list[int] = []
        previous_node: int | None = None
        completed_blocks = 0
        for start in range(0, n, bucket_size):
            if time.perf_counter() >= deadline:
                tour.extend(range(start, n))
                return tour, {
                    "sweep_mode": "input_order_greedy_partial_timeout",
                    "sweep_bucket_size": bucket_size,
                    "sweep_completed_blocks": completed_blocks,
                }
            end = min(n, start + bucket_size)
            block = sorted(
                range(start, end),
                key=lambda node: (instance.coords[node][1], instance.coords[node][0], node),
            )
            if previous_node is not None:
                forward_distance = _distance(instance, previous_node, block[0])
                reverse_distance = _distance(instance, previous_node, block[-1])
                if reverse_distance < forward_distance:
                    block.reverse()
            tour.extend(block)
            previous_node = tour[-1]
            completed_blocks += 1
        return tour, {
            "sweep_mode": "input_order_greedy_sweep_large",
            "sweep_bucket_size": bucket_size,
            "sweep_completed_blocks": completed_blocks,
        }

    order = list(range(n))
    order.sort(key=lambda node: (instance.coords[node][0], instance.coords[node][1], node))
    if time.perf_counter() >= deadline:
        return order, {"sweep_mode": "x_sorted_timeout"}

    bucket_size = max(32, int(math.sqrt(n)))
    reverse = False
    completed_blocks = 0
    for start in range(0, n, bucket_size):
        if time.perf_counter() >= deadline:
            return order, {
                "sweep_mode": "partial_timeout",
                "sweep_bucket_size": bucket_size,
                "sweep_completed_blocks": completed_blocks,
            }
        end = min(n, start + bucket_size)
        order[start:end] = sorted(
            order[start:end],
            key=lambda node: (instance.coords[node][1], instance.coords[node][0], node),
            reverse=reverse,
        )
        reverse = not reverse
        completed_blocks += 1
    return order, {
        "sweep_mode": "sweep",
        "sweep_bucket_size": bucket_size,
        "sweep_completed_blocks": completed_blocks,
    }


def _two_opt_delta(instance: TSPInstance, a: int, b: int, c: int, d: int) -> int:
    return (
        _distance(instance, a, c)
        + _distance(instance, b, d)
        - _distance(instance, a, b)
        - _distance(instance, c, d)
    )


def two_opt(instance: TSPInstance, tour: list[int], deadline: float) -> tuple[list[int], dict[str, Any]]:
    n = len(tour)
    if n < 4 or time.perf_counter() >= deadline:
        return tour, {"two_opt_mode": "skipped", "passes": 0, "improvements": 0}

    if n <= FULL_TWO_OPT_LIMIT:
        window = n - 1
        max_passes = 50
        mode = "full"
    elif n <= WINDOWED_TWO_OPT_LIMIT:
        window = 30 if n <= 10_000 else 16
        max_passes = 2 if n <= 10_000 else 1
        mode = "windowed"
    else:
        return tour, {"two_opt_mode": "skipped", "passes": 0, "improvements": 0}

    improvements = 0
    passes = 0
    restart_after_improvement = mode == "full" and n <= TIME_BOXED_MULTI_START_LIMIT
    rotate_between_passes = mode == "windowed" and max_passes > 1 and n > window + 3

    while passes < max_passes and time.perf_counter() < deadline:
        improved = False
        passes += 1
        for i in range(n - 3):
            if time.perf_counter() >= deadline:
                return tour, {"two_opt_mode": mode, "passes": passes, "improvements": improvements}
            a = tour[i]
            b = tour[i + 1]
            upper_exclusive = n if mode == "full" else min(n, i + window + 1)
            for j in range(i + 2, upper_exclusive):
                if i == 0 and j == n - 1:
                    continue
                c = tour[j]
                d = tour[(j + 1) % n]
                if _two_opt_delta(instance, a, b, c, d) < 0:
                    tour[i + 1 : j + 1] = reversed(tour[i + 1 : j + 1])
                    improvements += 1
                    improved = True
                    break
            if improved and restart_after_improvement:
                break
        if not improved:
            break
        if rotate_between_passes and passes == 1 and time.perf_counter() < deadline:
            split = n // 2
            tour[:] = tour[split:] + tour[:split]

    return tour, {"two_opt_mode": mode, "passes": passes, "improvements": improvements}


def relocate(instance: TSPInstance, tour: list[int], deadline: float) -> tuple[list[int], dict[str, Any]]:
    n = len(tour)
    if n < 4 or time.perf_counter() >= deadline:
        return tour, {"relocate_mode": "skipped", "relocate_moves": 0}

    moves = 0

    while time.perf_counter() < deadline:
        best_delta = 0
        best_move: tuple[int, int] | None = None
        for i in range(n):
            prev_i = tour[i - 1]
            node = tour[i]
            next_i = tour[(i + 1) % n]
            remove_delta = (
                _distance(instance, prev_i, next_i)
                - _distance(instance, prev_i, node)
                - _distance(instance, node, next_i)
            )
            for j in range(n):
                if j == i or j == (i - 1) % n:
                    continue
                a = tour[j]
                b = tour[(j + 1) % n]
                insert_delta = (
                    _distance(instance, a, node)
                    + _distance(instance, node, b)
                    - _distance(instance, a, b)
                )
                delta = remove_delta + insert_delta
                if delta < best_delta:
                    best_delta = delta
                    best_move = (i, j)
            if time.perf_counter() >= deadline:
                return tour, {"relocate_mode": "best_improvement", "relocate_moves": moves}
        if best_move is None:
            break
        i, j = best_move
        node = tour[i]
        reduced = tour[:i] + tour[i + 1 :]
        insert_at = j + 1 if j < i else j
        tour = reduced[:insert_at] + [node] + reduced[insert_at:]
        moves += 1

    return tour, {"relocate_mode": "best_improvement", "relocate_moves": moves}


def prefix_meta(meta: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{key}": value for key, value in meta.items()}


def solve_with_multistart(
    instance: TSPInstance,
    spec: SolverSpec,
    budget_s: float,
    seed: int,
    deadline: float,
) -> tuple[list[int], dict[str, Any]]:
    if instance.dimension > EXACT_NEAREST_NEIGHBOR_LIMIT:
        tour, sweep_meta = sweep_tour(instance, deadline)
        return tour, {
            "construction": "sweep",
            "solver_name": spec.solver_name,
            "solver_kind": spec.solver_kind,
            "start_order_mode": "sweep",
            "candidate_starts": 1,
            "starts_tried": 1,
            "best_start": None,
            **sweep_meta,
        }

    if instance.dimension <= TIME_BOXED_MULTI_START_LIMIT:
        starts, start_order_mode = build_start_order(instance, spec, seed, deadline)
    else:
        start_order_mode = (
            spec.start_order if spec.start_order != "time_boxed" else "anchor_nodes"
        )
        starts = choose_start_nodes(instance, seed, start_order_mode)
    if spec.max_starts is not None:
        starts = starts[: spec.max_starts]

    restart_deadline = deadline - (budget_s * spec.restart_reserve_fraction)
    best_tour: list[int] | None = None
    best_objective = math.inf
    best_start: int | str | None = None
    starts_tried = 0
    best_local_search_meta = {
        "restart_two_opt_mode": "skipped",
        "restart_passes": 0,
        "restart_improvements": 0,
        "restart_relocate_mode": "skipped",
        "restart_relocate_moves": 0,
    }
    incumbent_meta: dict[str, Any] = {}

    if instance.dimension > TIME_BOXED_MULTI_START_LIMIT and time.perf_counter() < restart_deadline:
        sweep_candidate, sweep_meta = sweep_tour(instance, restart_deadline)
        best_tour = sweep_candidate[:]
        best_objective = compute_tour_length(instance, sweep_candidate)
        best_start = "sweep"
        incumbent_meta = prefix_meta(sweep_meta, "incumbent_")

    for start in starts:
        if time.perf_counter() >= restart_deadline:
            break
        candidate_tour = nearest_neighbor_tour(instance, start, restart_deadline)
        candidate_tour, candidate_two_opt_meta = two_opt(instance, candidate_tour, restart_deadline)
        candidate_relocate_meta = {"relocate_mode": "skipped", "relocate_moves": 0}
        if time.perf_counter() < restart_deadline:
            relocate_slice = min(
                spec.candidate_relocate_limit_s,
                max(0.0, restart_deadline - time.perf_counter()),
            )
            if relocate_slice > 0.0:
                candidate_tour, candidate_relocate_meta = relocate(
                    instance,
                    candidate_tour,
                    min(restart_deadline, time.perf_counter() + relocate_slice),
                )
        candidate_objective = compute_tour_length(instance, candidate_tour)
        starts_tried += 1
        if candidate_objective < best_objective:
            best_tour = candidate_tour[:]
            best_objective = candidate_objective
            best_start = start
            best_local_search_meta = {
                **prefix_meta(candidate_two_opt_meta, "restart_"),
                **prefix_meta(candidate_relocate_meta, "restart_"),
            }

    if best_tour is None:
        best_tour = identity_tour(instance)

    final_two_opt_meta = {
        "final_two_opt_mode": "skipped",
        "final_two_opt_passes": 0,
        "final_two_opt_improvements": 0,
    }
    final_relocate_meta = {"final_relocate_mode": "skipped", "final_relocate_moves": 0}

    if time.perf_counter() < deadline:
        best_tour, polish_two_opt_meta = two_opt(instance, best_tour, deadline)
        final_two_opt_meta = {
            "final_two_opt_mode": polish_two_opt_meta["two_opt_mode"],
            "final_two_opt_passes": polish_two_opt_meta["passes"],
            "final_two_opt_improvements": polish_two_opt_meta["improvements"],
        }
    if time.perf_counter() < deadline:
        best_tour, polish_relocate_meta = relocate(instance, best_tour, deadline)
        final_relocate_meta = {
            "final_relocate_mode": polish_relocate_meta["relocate_mode"],
            "final_relocate_moves": polish_relocate_meta["relocate_moves"],
        }

    return best_tour, {
        "construction": "multi_start_nearest_neighbor",
        "solver_name": spec.solver_name,
        "solver_kind": spec.solver_kind,
        "start_order_mode": start_order_mode,
        "candidate_starts": len(starts),
        "starts_tried": starts_tried,
        "best_start": best_start,
        **best_local_search_meta,
        **final_two_opt_meta,
        **final_relocate_meta,
        **incumbent_meta,
    }


def solve_with_grid_nearest(
    instance: TSPInstance,
    spec: SolverSpec,
    budget_s: float,
    seed: int,
    deadline: float,
) -> tuple[list[int], dict[str, Any]]:
    del seed

    grid_side = spec.grid_side or GRID_NEAREST_SIDE
    cells, node_cells = build_grid_index(instance, grid_side)
    n = instance.dimension
    starts = [n - 1, 0, n // 4, n // 2, (3 * n) // 4]
    if spec.max_starts is not None:
        starts = starts[: spec.max_starts]
    construction_deadline = deadline
    if spec.grid_polish:
        construction_slice_s = max(0.02, budget_s * 0.45)
        construction_deadline = min(deadline, time.perf_counter() + construction_slice_s)

    best_tour: list[int] | None = None
    best_objective = math.inf
    best_start: int | None = None
    starts_tried = 0
    completed_starts = 0

    for start in starts:
        if time.perf_counter() >= construction_deadline:
            break
        candidate_tour, completed = grid_nearest_neighbor_tour(
            instance,
            start,
            grid_side,
            cells,
            node_cells,
            construction_deadline,
        )
        starts_tried += 1
        if completed:
            completed_starts += 1
        if time.perf_counter() >= construction_deadline and best_tour is not None:
            break
        candidate_objective = compute_tour_length(instance, candidate_tour)
        if candidate_objective < best_objective:
            best_tour = candidate_tour
            best_objective = candidate_objective
            best_start = start

    final_two_opt_meta = {
        "final_two_opt_mode": "skipped",
        "final_two_opt_passes": 0,
        "final_two_opt_improvements": 0,
    }

    if best_tour is None:
        best_tour, sweep_meta = sweep_tour(instance, deadline)
        return best_tour, {
            "construction": "sweep",
            "solver_name": spec.solver_name,
            "solver_kind": spec.solver_kind,
            "start_order_mode": "grid_nearest_fallback",
            "candidate_starts": 0,
            "starts_tried": starts_tried,
            "best_start": None,
            "grid_side": grid_side,
            "grid_polish": spec.grid_polish,
            "grid_completed_starts": completed_starts,
            **final_two_opt_meta,
            **sweep_meta,
        }

    if spec.grid_polish and time.perf_counter() < deadline:
        best_tour, polish_two_opt_meta = two_opt(instance, best_tour, deadline)
        final_two_opt_meta = {
            "final_two_opt_mode": polish_two_opt_meta["two_opt_mode"],
            "final_two_opt_passes": polish_two_opt_meta["passes"],
            "final_two_opt_improvements": polish_two_opt_meta["improvements"],
        }

    return best_tour, {
        "construction": "grid_nearest_neighbor",
        "solver_name": spec.solver_name,
        "solver_kind": spec.solver_kind,
        "start_order_mode": "grid_anchor_nodes",
        "candidate_starts": len(starts),
        "starts_tried": starts_tried,
        "best_start": best_start,
        "grid_side": grid_side,
        "grid_polish": spec.grid_polish,
        "grid_completed_starts": completed_starts,
        **final_two_opt_meta,
    }


def build_candidate_distance_matrix(
    instance: TSPInstance,
    deadline: float,
) -> list[list[int]] | None:
    n = instance.dimension
    if n > CANDIDATE_MATRIX_LIMIT:
        return None

    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        if (i & 15) == 0 and time.perf_counter() >= deadline:
            return None
        row_i = matrix[i]
        for j in range(i + 1, n):
            distance = prepare.edge_distance(instance, i, j)
            row_i[j] = distance
            matrix[j][i] = distance
    return matrix


def matrix_tour_length(matrix: list[list[int]], tour: list[int]) -> float:
    n = len(tour)
    return float(sum(matrix[tour[i]][tour[(i + 1) % n]] for i in range(n)))


def matrix_nearest_neighbor_tour(
    matrix: list[list[int]],
    start: int,
    deadline: float,
) -> list[int]:
    n = len(matrix)
    visited = bytearray(n)
    visited[start] = 1
    tour = [start]
    current = start

    while len(tour) < n and time.perf_counter() < deadline:
        row = matrix[current]
        best_node = -1
        best_distance = math.inf
        for candidate, distance in enumerate(row):
            if not visited[candidate] and distance < best_distance:
                best_node = candidate
                best_distance = distance
        if best_node < 0:
            break
        visited[best_node] = 1
        tour.append(best_node)
        current = best_node

    if len(tour) < n:
        tour.extend(node for node in range(n) if not visited[node])
    return tour


def candidate_lists_from_matrix(
    matrix: list[list[int]],
    candidate_count: int,
    deadline: float,
) -> list[list[int]]:
    if time.perf_counter() >= deadline:
        return [[] for _ in matrix]
    n = len(matrix)
    count = min(candidate_count, max(0, n - 1))
    return [
        sorted(range(n), key=row.__getitem__)[1 : count + 1]
        for row in matrix
    ]


def tour_positions(tour: list[int]) -> list[int]:
    positions = [0] * len(tour)
    for index, node in enumerate(tour):
        positions[node] = index
    return positions


def refresh_positions(
    tour: list[int],
    positions: list[int],
    start: int = 0,
    end: int | None = None,
) -> None:
    if end is None:
        end = len(tour) - 1
    for index in range(start, end + 1):
        positions[tour[index]] = index


def candidate_two_opt(
    matrix: list[list[int]],
    tour: list[int],
    candidates: list[list[int]],
    deadline: float,
) -> tuple[list[int], dict[str, Any]]:
    n = len(tour)
    if n < 4 or time.perf_counter() >= deadline:
        return tour, {"candidate_two_opt_moves": 0, "candidate_two_opt_passes": 0}

    positions = tour_positions(tour)
    moves = 0
    passes = 0

    while time.perf_counter() < deadline:
        improved = False
        passes += 1
        for i in range(n):
            if time.perf_counter() >= deadline:
                return tour, {
                    "candidate_two_opt_moves": moves,
                    "candidate_two_opt_passes": passes,
                }
            a = tour[i]
            b = tour[(i + 1) % n]
            old_ab = matrix[a][b]
            for c in candidates[a]:
                j = positions[c]
                if i < j:
                    if j == i + 1 or (i == 0 and j == n - 1):
                        continue
                    d = tour[(j + 1) % n]
                    delta = matrix[a][c] + matrix[b][d] - old_ab - matrix[c][d]
                    if delta < 0:
                        tour[i + 1 : j + 1] = reversed(tour[i + 1 : j + 1])
                        refresh_positions(tour, positions, i + 1, j)
                        moves += 1
                        improved = True
                        break
                elif j < i - 1:
                    if j == 0 and i == n - 1:
                        continue
                    d = tour[(j + 1) % n]
                    delta = matrix[c][a] + matrix[d][b] - matrix[c][d] - old_ab
                    if delta < 0:
                        tour[j + 1 : i + 1] = reversed(tour[j + 1 : i + 1])
                        refresh_positions(tour, positions, j + 1, i)
                        moves += 1
                        improved = True
                        break
            if improved:
                break
        if not improved:
            break

    return tour, {
        "candidate_two_opt_moves": moves,
        "candidate_two_opt_passes": passes,
    }


def candidate_relocate(
    matrix: list[list[int]],
    tour: list[int],
    candidates: list[list[int]],
    near_limit: int,
    block_widths: tuple[int, ...],
    deadline: float,
) -> tuple[list[int], dict[str, Any]]:
    n = len(tour)
    if n < 4 or time.perf_counter() >= deadline:
        return tour, {"candidate_relocate_moves": 0}

    positions = tour_positions(tour)
    moves = 0

    while time.perf_counter() < deadline:
        moved = False
        for width in block_widths:
            if moved or width >= n:
                break
            for i in range(0, n - width + 1):
                if time.perf_counter() >= deadline:
                    return tour, {"candidate_relocate_moves": moves}
                first = tour[i]
                last = tour[i + width - 1]
                previous_node = tour[i - 1]
                next_node = tour[(i + width) % n]
                remove_delta = (
                    matrix[previous_node][next_node]
                    - matrix[previous_node][first]
                    - matrix[last][next_node]
                )

                seen_neighbors: set[int] = set()
                neighbor_iter = (
                    candidates[first][:near_limit] + candidates[last][:near_limit]
                )
                for neighbor in neighbor_iter:
                    if neighbor in seen_neighbors:
                        continue
                    seen_neighbors.add(neighbor)
                    j = positions[neighbor]
                    if i - 1 <= j <= i + width - 1:
                        continue
                    insert_next_index = (j + 1) % n
                    if i <= insert_next_index <= i + width - 1:
                        continue
                    insert_next = tour[insert_next_index]
                    delta = (
                        remove_delta
                        + matrix[neighbor][first]
                        + matrix[last][insert_next]
                        - matrix[neighbor][insert_next]
                    )
                    if delta < 0:
                        block = tour[i : i + width]
                        del tour[i : i + width]
                        if j > i:
                            j -= width
                        insert_at = j + 1
                        tour[insert_at:insert_at] = block
                        refresh_positions(tour, positions)
                        moves += 1
                        moved = True
                        break
                if moved:
                    break
            if moved:
                break
        if not moved:
            break

    return tour, {"candidate_relocate_moves": moves}


def convex_hull_nodes(instance: TSPInstance) -> list[int]:
    points = sorted((x, y, node) for node, (x, y) in enumerate(instance.coords))

    def cross(
        origin: tuple[float, float, int],
        left: tuple[float, float, int],
        right: tuple[float, float, int],
    ) -> float:
        return (
            (left[0] - origin[0]) * (right[1] - origin[1])
            - (left[1] - origin[1]) * (right[0] - origin[0])
        )

    lower: list[tuple[float, float, int]] = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper: list[tuple[float, float, int]] = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    hull = lower[:-1] + upper[:-1]
    return [point[2] for point in hull]


def insertion_seed_tour(
    instance: TSPInstance,
    matrix: list[list[int]],
    seed_kind: str,
) -> list[int]:
    n = instance.dimension
    xs = [x for x, _ in instance.coords]
    ys = [y for _, y in instance.coords]

    if seed_kind == "hull":
        seeds = convex_hull_nodes(instance)
    elif seed_kind == "diameter":
        first = 0
        second = max(range(n), key=lambda node: matrix[first][node])
        first = max(range(n), key=lambda node: matrix[node][second])
        seeds = [first, second]
    elif seed_kind == "extreme":
        seeds = []
        for key in (
            lambda node: xs[node],
            lambda node: -xs[node],
            lambda node: ys[node],
            lambda node: -ys[node],
        ):
            node = min(range(n), key=key)
            if node not in seeds:
                seeds.append(node)
        if seeds:
            ordered = [seeds[0]]
            remaining = seeds[1:]
            while remaining:
                current = ordered[-1]
                next_node = min(remaining, key=lambda node: matrix[current][node])
                remaining.remove(next_node)
                ordered.append(next_node)
            seeds = ordered
    else:
        seeds = [0, n // 3, (2 * n) // 3]

    unique: list[int] = []
    seen: set[int] = set()
    for node in seeds:
        if 0 <= node < n and node not in seen:
            unique.append(node)
            seen.add(node)
    if len(unique) < 2:
        unique.extend(node for node in range(n) if node not in seen)
    return unique[: max(2, min(len(unique), n))]


def insertion_tour_from_matrix(
    instance: TSPInstance,
    matrix: list[list[int]],
    construction: str,
    deadline: float,
) -> list[int]:
    parts = construction.split("_")
    seed_kind = parts[1] if len(parts) >= 3 else "diameter"
    pick_mode = parts[2] if len(parts) >= 3 else "far"
    n = instance.dimension
    tour = insertion_seed_tour(instance, matrix, seed_kind)
    unvisited = [True] * n
    for node in tour:
        unvisited[node] = False
    unvisited_nodes = [node for node in range(n) if unvisited[node]]

    nearest = [math.inf] * n
    for node in unvisited_nodes:
        nearest[node] = min(matrix[node][seed] for seed in tour)

    while unvisited_nodes and time.perf_counter() < deadline:
        selected = -1
        selected_index = -1
        if pick_mode == "near":
            selected_value = math.inf
            for index, node in enumerate(unvisited_nodes):
                if nearest[node] < selected_value:
                    selected = node
                    selected_index = index
                    selected_value = nearest[node]
        else:
            selected_value = -1.0
            for index, node in enumerate(unvisited_nodes):
                if nearest[node] > selected_value:
                    selected = node
                    selected_index = index
                    selected_value = nearest[node]
        if selected < 0:
            break

        selected_row = matrix[selected]
        best_position = 0
        best_delta = math.inf
        tour_size = len(tour)
        for position, left in enumerate(tour):
            right = tour[(position + 1) % tour_size]
            delta = selected_row[left] + selected_row[right] - matrix[left][right]
            if delta < best_delta:
                best_delta = delta
                best_position = position + 1

        tour.insert(best_position, selected)
        unvisited[selected] = False
        unvisited_nodes.pop(selected_index)

        for node in unvisited_nodes:
            if selected_row[node] < nearest[node]:
                nearest[node] = selected_row[node]

    while unvisited_nodes and time.perf_counter() < deadline:
        current = tour[-1]
        current_row = matrix[current]
        selected_index = min(
            range(len(unvisited_nodes)),
            key=lambda index: current_row[unvisited_nodes[index]],
        )
        selected = unvisited_nodes.pop(selected_index)
        unvisited[selected] = False
        tour.append(selected)

    if unvisited_nodes:
        tour.extend(unvisited_nodes)
    return tour


def candidate_construction_tour(
    instance: TSPInstance,
    matrix: list[list[int]],
    construction: str,
    seed: int,
    deadline: float,
) -> list[int]:
    if construction.startswith("baseline_"):
        construction = construction.removeprefix("baseline_")

    n = instance.dimension
    start_nodes = {
        "nn_0": 0,
        "nn_q1": n // 4,
        "nn_mid": n // 2,
        "nn_q3": (3 * n) // 4,
        "nn_last": n - 1,
    }
    if construction in start_nodes:
        return matrix_nearest_neighbor_tour(matrix, start_nodes[construction], deadline)
    if construction == "nn_random":
        rng = random.Random(seed)
        return matrix_nearest_neighbor_tour(matrix, rng.randrange(n), deadline)
    if construction.startswith("insertion_"):
        return insertion_tour_from_matrix(instance, matrix, construction, deadline)
    return matrix_nearest_neighbor_tour(matrix, n // 2, deadline)


def solve_with_candidate_local(
    instance: TSPInstance,
    spec: SolverSpec,
    budget_s: float,
    seed: int,
    deadline: float,
) -> tuple[list[int], dict[str, Any]]:
    matrix = build_candidate_distance_matrix(instance, deadline)
    if matrix is None:
        return solve_with_multistart(instance, spec, budget_s, seed, deadline)

    constructions = spec.candidate_constructions or ("nn_mid", "insertion_diameter_far")
    candidates: list[list[int]] | None = None
    best_tour: list[int] | None = None
    best_objective = math.inf
    best_meta: dict[str, Any] = {}
    starts_tried = 0

    for construction in constructions:
        if time.perf_counter() >= deadline:
            break
        starts_tried += 1
        tour = candidate_construction_tour(instance, matrix, construction, seed, deadline)
        objective = matrix_tour_length(matrix, tour)
        if objective < best_objective:
            best_tour = tour[:]
            best_objective = objective
            best_meta = {
                "candidate_best_stage": "construction",
                "candidate_best_construction": construction,
            }

        if construction.startswith("baseline_"):
            continue

        if time.perf_counter() >= deadline:
            break
        if candidates is None:
            candidates = candidate_lists_from_matrix(
                matrix,
                spec.candidate_neighbors,
                deadline,
            )

        tour, two_opt_meta = candidate_two_opt(matrix, tour, candidates, deadline)
        objective = matrix_tour_length(matrix, tour)
        if objective < best_objective:
            best_tour = tour[:]
            best_objective = objective
            best_meta = {
                "candidate_best_stage": "two_opt",
                "candidate_best_construction": construction,
                **prefix_meta(two_opt_meta, "best_"),
            }

        if time.perf_counter() < deadline:
            tour, relocate_meta = candidate_relocate(
                matrix,
                tour,
                candidates,
                spec.candidate_relocate_near,
                spec.candidate_relocate_blocks,
                deadline,
            )
            objective = matrix_tour_length(matrix, tour)
            if objective < best_objective:
                best_tour = tour[:]
                best_objective = objective
                best_meta = {
                    "candidate_best_stage": "relocate",
                    "candidate_best_construction": construction,
                    **prefix_meta(relocate_meta, "best_"),
                }

        if time.perf_counter() < deadline:
            tour, second_two_opt_meta = candidate_two_opt(
                matrix,
                tour,
                candidates,
                deadline,
            )
            objective = matrix_tour_length(matrix, tour)
            if objective < best_objective:
                best_tour = tour[:]
                best_objective = objective
                best_meta = {
                    "candidate_best_stage": "second_two_opt",
                    "candidate_best_construction": construction,
                    **prefix_meta(second_two_opt_meta, "best_second_"),
                }

    if best_tour is None:
        best_tour = identity_tour(instance)

    return best_tour, {
        "construction": "candidate_local_search",
        "solver_name": spec.solver_name,
        "solver_kind": spec.solver_kind,
        "candidate_starts": len(constructions),
        "starts_tried": starts_tried,
        "candidate_neighbors": spec.candidate_neighbors,
        "candidate_relocate_near": spec.candidate_relocate_near,
        "candidate_relocate_blocks": spec.candidate_relocate_blocks,
        "candidate_best_objective": best_objective,
        **best_meta,
    }


SolverFn = Callable[[TSPInstance, SolverSpec, float, int, float], tuple[list[int], dict[str, Any]]]

SOLVER_REGISTRY: dict[str, SolverFn] = {
    "candidate_local": solve_with_candidate_local,
    "grid_nearest": solve_with_grid_nearest,
    "multistart": solve_with_multistart,
}


def run_ils(
    instance: TSPInstance,
    spec: SolverSpec,
    incumbent_tour: list[int],
    incumbent_objective: float,
    seed: int,
    deadline: float,
) -> tuple[list[int], float, dict[str, Any]]:
    if not spec.ils_enabled or time.perf_counter() >= deadline:
        return incumbent_tour, incumbent_objective, {
            "ils_mode": "skipped",
            "ils_iterations": 0,
            "ils_improvements": 0,
        }

    if instance.dimension < ITERATED_LOCAL_SEARCH_MIN_DIMENSION:
        return incumbent_tour, incumbent_objective, {
            "ils_mode": "skipped_small_dimension",
            "ils_iterations": 0,
            "ils_improvements": 0,
        }
    if instance.dimension > ITERATED_LOCAL_SEARCH_MAX_DIMENSION:
        return incumbent_tour, incumbent_objective, {
            "ils_mode": "skipped_large_dimension",
            "ils_iterations": 0,
            "ils_improvements": 0,
        }

    rng = random.Random(seed)
    best_tour = incumbent_tour[:]
    best_objective = incumbent_objective
    iterations = 0
    improvements = 0

    while time.perf_counter() < deadline:
        candidate_tour = block_shift_kick(best_tour, rng, spec.ils_block_width)
        candidate_tour, _ = two_opt(instance, candidate_tour, deadline)
        if time.perf_counter() < deadline:
            candidate_tour, _ = relocate(instance, candidate_tour, deadline)
        if time.perf_counter() < deadline:
            candidate_tour, _ = two_opt(instance, candidate_tour, deadline)
        candidate_objective = compute_tour_length(instance, candidate_tour)
        iterations += 1
        if candidate_objective < best_objective:
            best_tour = candidate_tour
            best_objective = candidate_objective
            improvements += 1

    return best_tour, best_objective, {
        "ils_mode": "block_shift",
        "ils_iterations": iterations,
        "ils_improvements": improvements,
    }


def solver_spec_for(instance: TSPInstance) -> SolverSpec:
    return BENCHMARK_SOLVERS.get(instance.name, DEFAULT_SOLVER_SPEC)


def make_result(
    solution: list[int],
    objective: float | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    result = {
        "solution": solution,
        "metadata": metadata,
    }
    if objective is not None:
        result["objective"] = objective
    return result


def solve_one_instance(
    instance: TSPInstance,
    assigned_budget_s: float,
    scheduler_weight: float,
    seed: int,
    global_deadline: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    spec = solver_spec_for(instance)
    instance_deadline = min(global_deadline, started + max(0.0, assigned_budget_s))

    if assigned_budget_s <= 0.0 or time.perf_counter() >= global_deadline:
        tour = identity_tour(instance)
        objective = compute_tour_length(instance, tour)
        return make_result(
            tour,
            objective,
            {
                "solver_name": spec.solver_name,
                "solver_kind": spec.solver_kind,
                "solver_params": asdict(spec),
                "assigned_budget_s": assigned_budget_s,
                "scheduler_weight": scheduler_weight,
                "runtime_s": time.perf_counter() - started,
                "seed": seed,
                "deadline_hit": True,
                "stop_reason": "no_time",
                "construction": "identity",
            },
        )

    try:
        solver_fn = SOLVER_REGISTRY[spec.solver_kind]
    except KeyError as exc:
        raise KeyError(f"{instance.name}: unknown solver_kind {spec.solver_kind!r}") from exc

    incumbent_tour, solver_meta = solver_fn(
        instance,
        spec,
        assigned_budget_s,
        seed,
        instance_deadline,
    )
    incumbent_objective: float | None = None
    ils_meta = {
        "ils_mode": "skipped_unscored",
        "ils_iterations": 0,
        "ils_improvements": 0,
    }
    should_score_for_ils = (
        spec.ils_enabled
        and ITERATED_LOCAL_SEARCH_MIN_DIMENSION
        <= instance.dimension
        <= ITERATED_LOCAL_SEARCH_MAX_DIMENSION
    )
    if should_score_for_ils:
        incumbent_objective = compute_tour_length(instance, incumbent_tour)
        incumbent_tour, incumbent_objective, ils_meta = run_ils(
            instance,
            spec,
            incumbent_tour,
            incumbent_objective,
            seed,
            instance_deadline,
        )

    now = time.perf_counter()
    deadline_hit = now >= instance_deadline or now >= global_deadline
    return make_result(
        incumbent_tour,
        incumbent_objective,
        {
            **solver_meta,
            **ils_meta,
            "solver_params": asdict(spec),
            "assigned_budget_s": assigned_budget_s,
            "scheduler_weight": scheduler_weight,
            "runtime_s": now - started,
            "seed": seed,
            "deadline_hit": deadline_hit,
            "stop_reason": "deadline" if deadline_hit else "complete",
        },
    )


def solve_benchmark(
    instances: Sequence[TSPInstance],
    total_budget_s: float,
    seed: int,
    deadline: float,
) -> Mapping[str, Any]:
    validation_reserve_s = estimate_validation_reserve_s(instances, total_budget_s)
    solver_budget_s = max(0.0, total_budget_s - validation_reserve_s)
    solver_deadline = min(deadline, max(time.perf_counter(), deadline - validation_reserve_s))
    allocations = schedule_budgets(instances, solver_budget_s)
    results: dict[str, dict[str, Any]] = {}

    if any(instance.name == "lin318" for instance in instances):
        ordered_instances = sorted(
            enumerate(instances),
            key=lambda item: (SCHEDULER_RUN_ORDER.get(item[1].name, 100), item[0]),
        )
    else:
        ordered_instances = list(enumerate(instances))
    for index, instance in ordered_instances:
        assigned_budget_s, scheduler_weight = allocations[instance.name]
        result = solve_one_instance(
            instance,
            assigned_budget_s,
            scheduler_weight,
            seed + index,
            solver_deadline,
        )
        result["metadata"]["benchmark_total_budget_s"] = total_budget_s
        result["metadata"]["benchmark_solver_budget_s"] = solver_budget_s
        result["metadata"]["validation_reserve_s"] = validation_reserve_s
        results[instance.name] = result

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one local TSP heuristic benchmark.")
    parser.add_argument(
        "--size",
        choices=prepare.BENCHMARK_SIZE_CHOICES,
        help="Filter by benchmark size. Defaults to small for --suite all, otherwise all.",
    )
    parser.add_argument(
        "--suite",
        choices=prepare.REFERENCE_SUITE_CHOICES,
        default="all",
        help="Filter by reference suite: all, opt_tour/optimal, or baseline_ref/baseline.",
    )
    parser.add_argument("--budget", type=float, default=30.0, help="Total wall-clock budget in seconds.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--status", choices=("baseline", "keep", "discard", "crash"), default="baseline")
    parser.add_argument("--description", default="baseline")
    parser.add_argument("--allow-missing", action="store_true", help="Allow missing tier instances.")
    args = parser.parse_args()
    if args.size is None:
        args.size = "small" if args.suite == "all" else "all"
    return args


def print_run_summary(artifact: dict[str, Any], artifact_path: str, status: str) -> None:
    print(f"[run] status={status}")
    print(f"[run] score_schema={artifact['score_schema']}")
    print(f"[run] size={artifact['size']}")
    print(f"[run] suite={artifact['suite']}")
    print(f"[run] score={artifact['aggregate_metrics']['score']:.6f}")
    print(f"[run] median_score={artifact['aggregate_metrics']['median_score']:.6f}")
    for reference_kind, metrics in artifact["aggregate_metrics"].get(
        "reference_kind_metrics", {}
    ).items():
        print(
            f"[run] {reference_kind}_score={metrics['score']:.6f} "
            f"n={metrics['num_instances']}"
        )
    print(f"[run] runtime_s={artifact['aggregate_metrics']['total_runtime_s']:.3f}")
    for metric in artifact["per_instance_metrics"]:
        print(
            f"[instance] {metric['name']} n={metric['dimension']} "
            f"objective={metric['objective']:.2f} "
            f"relative_gap_pct={metric['score']:.6f} "
            f"runtime_s={metric['solver_runtime_s']}"
        )
    print(f"[run] artifact={artifact_path}")


def main() -> int:
    args = parse_args()
    cases = prepare.load_benchmark_instances(
        args.size,
        suite=args.suite,
        verbose=True,
        allow_missing=args.allow_missing,
    )
    benchmark_names = [case.name for case in cases]
    run_started = time.perf_counter()

    try:
        artifact = prepare.run_benchmark(
            cases=cases,
            solve_benchmark=solve_benchmark,
            size=args.size,
            suite=args.suite,
            budget_s=args.budget,
            seed=args.seed,
            allow_missing=args.allow_missing,
        )
        artifact_path = prepare.record_run(
            artifact,
            status=args.status,
            description=args.description,
        )
    except Exception as exc:
        crash_artifact = prepare.build_crash_artifact(
            size=args.size,
            suite=args.suite,
            seed=args.seed,
            budget_s=args.budget,
            benchmark_instance_names=benchmark_names,
            total_runtime_s=time.perf_counter() - run_started,
            error=str(exc),
            error_type=type(exc).__name__,
            allow_missing=args.allow_missing,
        )
        artifact_path = prepare.record_run(
            crash_artifact,
            status="crash",
            description=args.description,
        )
        print("[run] status=crash")
        print(f"[run] error={exc}")
        print(f"[run] artifact={artifact_path}")
        raise

    print_run_summary(artifact, str(artifact_path), args.status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
