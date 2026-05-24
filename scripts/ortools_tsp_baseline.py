#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from ortools.constraint_solver import pywrapcp, routing_enums_pb2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import prepare

OUTPUT_DIR = ROOT / "ortools_baselines"
OUTPUT_TSV = ROOT / "ortools_baselines.tsv"


def solve_with_ortools(instance: prepare.TSPInstance, time_limit_s: int) -> dict[str, Any]:
    started = time.perf_counter()
    manager = pywrapcp.RoutingIndexManager(instance.dimension, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    distance_cache: dict[tuple[int, int], int] = {}

    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        key = (from_node, to_node)
        value = distance_cache.get(key)
        if value is None:
            value = prepare.edge_distance(instance, from_node, to_node)
            distance_cache[key] = value
        return value

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_parameters.time_limit.FromSeconds(time_limit_s)
    search_parameters.log_search = False

    solution = routing.SolveWithParameters(search_parameters)
    elapsed_s = time.perf_counter() - started
    if solution is None:
        return {
            "instance": instance.name,
            "dimension": instance.dimension,
            "status": "no_solution",
            "time_limit_s": time_limit_s,
            "elapsed_s": elapsed_s,
            "objective": None,
            "gap_to_known_pct": None,
            "tour": None,
        }

    index = routing.Start(0)
    tour: list[int] = []
    while not routing.IsEnd(index):
        tour.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))

    objective = prepare.compute_tour_length(instance, tour)
    gap = None
    if instance.best_known is not None:
        gap = ((objective - instance.best_known) / instance.best_known) * 100.0

    return {
        "instance": instance.name,
        "dimension": instance.dimension,
        "status": "ok",
        "time_limit_s": time_limit_s,
        "elapsed_s": elapsed_s,
        "objective": objective,
        "gap_to_known_pct": gap,
        "tour": tour,
    }


def append_tsv(result: dict[str, Any]) -> None:
    if not OUTPUT_TSV.exists():
        OUTPUT_TSV.write_text(
            "instance\tdimension\tstatus\ttime_limit_s\telapsed_s\tobjective\tgap_to_known_pct\n",
            encoding="utf-8",
        )
    with OUTPUT_TSV.open("a", encoding="utf-8") as handle:
        objective_text = "" if result["objective"] is None else f"{result['objective']:.6f}"
        gap_text = "" if result["gap_to_known_pct"] is None else f"{result['gap_to_known_pct']:.6f}"
        handle.write(
            f"{result['instance']}\t{result['dimension']}\t{result['status']}\t"
            f"{result['time_limit_s']}\t{result['elapsed_s']:.3f}\t"
            f"{objective_text}\t{gap_text}\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run OR-Tools time-limited TSP baselines.")
    parser.add_argument("--size", choices=tuple(prepare.BENCHMARK_TIERS), default="medium")
    parser.add_argument("--instances", nargs="*", help="Optional instance names. Defaults to unknown-optimum instances in size.")
    parser.add_argument("--time-limit", type=int, required=True, help="Per-instance OR-Tools time limit in seconds.")
    parser.add_argument("--max-nodes", type=int, default=20_000, help="Skip instances larger than this many nodes.")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    instances = prepare.load_benchmark_instances(args.size, verbose=True)
    if args.instances:
        wanted = set(args.instances)
        instances = [instance for instance in instances if instance.name in wanted]
    else:
        instances = [instance for instance in instances if instance.best_known is None]

    for instance in instances:
        if instance.dimension > args.max_nodes:
            result = {
                "instance": instance.name,
                "dimension": instance.dimension,
                "status": f"skipped_over_max_nodes_{args.max_nodes}",
                "time_limit_s": args.time_limit,
                "elapsed_s": 0.0,
                "objective": None,
                "gap_to_known_pct": None,
                "tour": None,
            }
        else:
            print(f"[ortools] solving {instance.name} n={instance.dimension} limit={args.time_limit}s", flush=True)
            result = solve_with_ortools(instance, args.time_limit)
        artifact = OUTPUT_DIR / f"{instance.name}_ortools_{args.time_limit}s.json"
        artifact.write_text(json.dumps(result, indent=2), encoding="utf-8")
        append_tsv(result)
        print(
            f"[ortools] {instance.name} status={result['status']} "
            f"objective={result['objective']} elapsed={result['elapsed_s']:.3f}s artifact={artifact}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
