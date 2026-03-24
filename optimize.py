from __future__ import annotations

import argparse
import math
import random
import time
from dataclasses import dataclass
from typing import Any

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
ITERATED_LOCAL_SEARCH_TRIGGER_GAP_PCT = 0.5
ITERATED_LOCAL_SEARCH_BLOCK_SHIFT_WIDTH = 6
PILOT_START_RANKING_LIMIT = 64
PILOT_START_RANKING_MAX_S = 0.015


@dataclass(frozen=True, slots=True)
class SolverSpec:
    solver_name: str
    start_order: str = "time_boxed"
    max_starts: int | None = None
    restart_reserve_fraction: float = RELOCATE_RESERVE_FRACTION
    candidate_relocate_limit_s: float = PER_CANDIDATE_RELOCATE_LIMIT_S
    ils_enabled: bool = False
    ils_trigger_gap_pct: float = ITERATED_LOCAL_SEARCH_TRIGGER_GAP_PCT
    ils_block_width: int = ITERATED_LOCAL_SEARCH_BLOCK_SHIFT_WIDTH


# The scheduler maps the fixed harness budget into per-benchmark solver budgets.
SCHEDULER_BUDGET_WEIGHTS = {
    "att48": 0.40,
    "eil51": 0.16,
    "berlin52": 0.05,
    "pr76": 0.25,
    "rd100": 0.14,
}


# Small-tier benchmark solvers. Future experiments can change these solver specs,
# replace them with other heuristics, or retune the scheduler above.
BENCHMARK_SOLVERS: dict[str, SolverSpec] = {
    "att48": SolverSpec(
        solver_name="att48_multistart_ils",
        start_order="pilot_ranked",
        ils_enabled=True,
    ),
    "eil51": SolverSpec(
        solver_name="eil51_ranked_multistart",
        start_order="pilot_ranked",
        restart_reserve_fraction=0.15,
        candidate_relocate_limit_s=0.0,
        ils_enabled=True,
        ils_trigger_gap_pct=0.0,
        ils_block_width=5,
    ),
    "berlin52": SolverSpec(
        solver_name="berlin52_ranked_multistart",
        start_order="pilot_ranked",
        ils_enabled=False,
    ),
    "pr76": SolverSpec(
        solver_name="pr76_multistart_ils",
        start_order="time_boxed",
        max_starts=6,
        restart_reserve_fraction=0.15,
        ils_enabled=True,
    ),
    "rd100": SolverSpec(
        solver_name="rd100_multistart",
        start_order="time_boxed",
        max_starts=4,
        ils_enabled=False,
    ),
}

DEFAULT_SOLVER_SPEC = SolverSpec(
    solver_name="generic_multistart",
    start_order="time_boxed",
    ils_enabled=False,
)


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
    matrix = instance.metadata.get("_distance_matrix")
    if matrix is not None:
        return matrix

    n = instance.dimension
    matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            distance = prepare.edge_distance(instance, i, j)
            matrix[i][j] = distance
            matrix[j][i] = distance
    instance.metadata["_distance_matrix"] = matrix
    return matrix


def _distance(instance: TSPInstance, a: int, b: int) -> int:
    matrix = _distance_matrix(instance)
    if matrix is not None:
        return matrix[a][b]
    return prepare.edge_distance(instance, a, b)


def allocate_instance_budget(instance: TSPInstance, budget_s: float) -> float:
    weight = SCHEDULER_BUDGET_WEIGHTS.get(instance.name)
    if weight is None:
        return budget_s
    total_budget_s = budget_s * len(SCHEDULER_BUDGET_WEIGHTS)
    return total_budget_s * weight


def choose_start_nodes(instance: TSPInstance, seed: int) -> list[int]:
    n = instance.dimension
    anchor_nodes = [0, n // 4, n // 2, (3 * n) // 4, n - 1]
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


def sweep_tour(instance: TSPInstance) -> list[int]:
    n = instance.dimension
    order = list(range(n))
    order.sort(key=lambda node: (instance.coords[node][0], instance.coords[node][1], node))

    bucket_size = max(32, int(math.sqrt(n)))
    tour: list[int] = []
    reverse = False
    for start in range(0, n, bucket_size):
        block = order[start : start + bucket_size]
        block.sort(
            key=lambda node: (instance.coords[node][1], instance.coords[node][0], node),
            reverse=reverse,
        )
        tour.extend(block)
        reverse = not reverse
    return tour


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
        window = 80 if n <= 10_000 else 16
        max_passes = 2 if n <= 10_000 else 1
        mode = "windowed"
    else:
        return tour, {"two_opt_mode": "skipped", "passes": 0, "improvements": 0}

    improvements = 0
    passes = 0

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
            if improved:
                break
        if not improved:
            break

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
        return sweep_tour(instance), {
            "construction": "sweep",
            "solver_name": spec.solver_name,
            "start_order_mode": "sweep",
            "candidate_starts": 1,
            "starts_tried": 1,
            "best_start": None,
        }

    if instance.dimension <= TIME_BOXED_MULTI_START_LIMIT:
        starts, start_order_mode = build_start_order(instance, spec, seed, deadline)
    else:
        starts = choose_start_nodes(instance, seed)
        start_order_mode = "anchor_nodes"
    if spec.max_starts is not None:
        starts = starts[: spec.max_starts]

    restart_deadline = deadline - (budget_s * spec.restart_reserve_fraction)
    best_tour: list[int] | None = None
    best_objective = math.inf
    best_start: int | None = None
    starts_tried = 0
    best_local_search_meta = {
        "restart_two_opt_mode": "skipped",
        "restart_passes": 0,
        "restart_improvements": 0,
        "restart_relocate_mode": "skipped",
        "restart_relocate_moves": 0,
    }

    for start_index, start in enumerate(starts):
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
                candidate_relocate_meta = {"relocate_mode": "best_improvement", "relocate_moves": 0}
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
        best_tour = list(range(instance.dimension))

    final_two_opt_meta = {"final_two_opt_mode": "skipped", "final_two_opt_passes": 0, "final_two_opt_improvements": 0}
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
        "start_order_mode": start_order_mode,
        "candidate_starts": len(starts),
        "starts_tried": starts_tried,
        "best_start": best_start,
        "allocated_budget_s": budget_s,
        **best_local_search_meta,
        **final_two_opt_meta,
        **final_relocate_meta,
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

    if instance.best_known is not None:
        trigger_objective = instance.best_known * (1.0 + (spec.ils_trigger_gap_pct / 100.0))
        if incumbent_objective <= trigger_objective:
            return incumbent_tour, incumbent_objective, {
                "ils_mode": "skipped_below_trigger",
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


def solve_instance(instance: TSPInstance, budget_s: float, seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    effective_budget = max(0.01, allocate_instance_budget(instance, budget_s))
    deadline = started + effective_budget
    spec = solver_spec_for(instance)

    incumbent_tour, solver_meta = solve_with_multistart(
        instance,
        spec,
        effective_budget,
        seed,
        deadline,
    )
    incumbent_objective = compute_tour_length(instance, incumbent_tour)
    incumbent_tour, incumbent_objective, ils_meta = run_ils(
        instance,
        spec,
        incumbent_tour,
        incumbent_objective,
        seed,
        deadline,
    )

    return {
        "solution": incumbent_tour,
        "objective": incumbent_objective,
        "metadata": {
            **solver_meta,
            **ils_meta,
            "elapsed_s": time.perf_counter() - started,
            "scheduler_budget_s": effective_budget,
            "scheduler_base_budget_s": budget_s,
            "seed": seed,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one local TSP heuristic benchmark.")
    parser.add_argument("--size", choices=tuple(prepare.BENCHMARK_TIERS), default="small")
    parser.add_argument("--budget", type=float, default=30.0, help="Total wall-clock budget in seconds.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--status", choices=("baseline", "keep", "discard", "crash"), default="baseline")
    parser.add_argument("--description", default="baseline")
    return parser.parse_args()


def print_run_summary(artifact: dict[str, Any], artifact_path: str, status: str) -> None:
    print(f"[run] status={status}")
    print(f"[run] score={artifact['aggregate_metrics']['score']:.6f}")
    print(f"[run] median_score={artifact['aggregate_metrics']['median_score']:.6f}")
    print(f"[run] runtime_s={artifact['aggregate_metrics']['total_runtime_s']:.3f}")
    for metric in artifact["per_instance_metrics"]:
        if metric["score_kind"] == "gap_pct":
            score_text = f"gap_pct={metric['score']:.6f}"
        else:
            score_text = f"raw_objective={metric['score']:.2f}"
        print(
            f"[instance] {metric['name']} n={metric['dimension']} "
            f"objective={metric['objective']:.2f} {score_text} runtime_s={metric['runtime_s']:.3f}"
        )
    print(f"[run] artifact={artifact_path}")


def main() -> int:
    args = parse_args()
    instances = prepare.load_benchmark_instances(args.size, verbose=True)
    if not instances:
        return 0

    benchmark_names = [instance.name for instance in instances]
    run_started = time.perf_counter()

    try:
        artifact = prepare.run_benchmark(
            instances=instances,
            solve_instance=solve_instance,
            size=args.size,
            budget_s=args.budget,
            seed=args.seed,
        )
        artifact_path = prepare.record_run(
            artifact,
            status=args.status,
            description=args.description,
        )
    except Exception as exc:
        crash_artifact = prepare.build_crash_artifact(
            size=args.size,
            seed=args.seed,
            budget_s=args.budget,
            benchmark_instance_names=benchmark_names,
            total_runtime_s=time.perf_counter() - run_started,
            error=str(exc),
        )
        artifact_path = prepare.record_run(
            crash_artifact,
            status="crash",
            description=args.description,
        )
        print(f"[run] status=crash")
        print(f"[run] error={exc}")
        print(f"[run] artifact={artifact_path}")
        raise

    print_run_summary(artifact, str(artifact_path), args.status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
