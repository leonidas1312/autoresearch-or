from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "tsp"
RESULTS_DIR = ROOT / "results"
RESULTS_TSV = ROOT / "results.tsv"

BENCHMARK_TIERS: dict[str, tuple[str, ...]] = {
    "small": ("att48", "eil51", "berlin52", "pr76", "rd100"),
    "medium": ("lin318", "pcb442", "rat783", "pr1002", "nrw1379", "pcb3038"),
    "large": ("qa194", "uy734", "lu980", "gr9882", "ch71009", "world"),
}

SUPPORTED_EDGE_WEIGHT_TYPES = {"EUC_2D", "CEIL_2D", "ATT", "GEO", "GEOM"}


@dataclass(slots=True)
class TSPInstance:
    name: str
    coords: list[tuple[float, float]]
    dimension: int
    edge_weight_type: str
    best_known: float | None = None
    reference_tour: list[int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _parse_header_line(line: str) -> tuple[str, str] | None:
    if not line or line.upper() in {"NODE_COORD_SECTION", "TOUR_SECTION", "EOF"}:
        return None
    if ":" in line:
        key, value = line.split(":", 1)
        return key.strip().upper(), value.strip()
    parts = line.split(None, 1)
    if len(parts) == 2:
        return parts[0].strip().upper(), parts[1].strip()
    return None


def inspect_instance_file(path: Path) -> dict[str, Any]:
    header: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.upper() == "NODE_COORD_SECTION":
                break
            parsed = _parse_header_line(line)
            if parsed is not None:
                key, value = parsed
                header[key] = value
    return {
        "name": header.get("NAME", path.stem),
        "dimension": int(header["DIMENSION"]),
        "edge_weight_type": header.get("EDGE_WEIGHT_TYPE", "EUC_2D").upper(),
    }


def _geo_to_radians(value: float) -> float:
    degrees = int(value)
    minutes = value - degrees
    return math.pi * (degrees + (5.0 * minutes / 3.0)) / 180.0


def _geom_distance(a: tuple[float, float], b: tuple[float, float]) -> int:
    radius = 6378.388
    lat1 = math.radians(a[0])
    lon1 = math.radians(a[1])
    lat2 = math.radians(b[0])
    lon2 = math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    hav = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    )
    central_angle = 2.0 * math.asin(min(1.0, math.sqrt(hav)))
    return int(radius * central_angle + 0.5)


def edge_distance(instance: TSPInstance, i: int, j: int) -> int:
    x1, y1 = instance.coords[i]
    x2, y2 = instance.coords[j]
    dx = x1 - x2
    dy = y1 - y2
    edge_type = instance.edge_weight_type

    if edge_type == "EUC_2D":
        return int(math.hypot(dx, dy) + 0.5)
    if edge_type == "CEIL_2D":
        return math.ceil(math.hypot(dx, dy))
    if edge_type == "ATT":
        value = math.sqrt((dx * dx + dy * dy) / 10.0)
        rounded = int(value + 0.5)
        return rounded if rounded >= value else rounded + 1
    if edge_type == "GEO":
        radius = 6378.388
        lat_i = _geo_to_radians(x1)
        lon_i = _geo_to_radians(y1)
        lat_j = _geo_to_radians(x2)
        lon_j = _geo_to_radians(y2)
        q1 = math.cos(lon_i - lon_j)
        q2 = math.cos(lat_i - lat_j)
        q3 = math.cos(lat_i + lat_j)
        return int(radius * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)
    if edge_type == "GEOM":
        return _geom_distance((x1, y1), (x2, y2))
    raise ValueError(f"Unsupported edge weight type for {instance.name}: {edge_type}")


def compute_tour_length(instance: TSPInstance, tour: Sequence[int]) -> float:
    if not tour:
        raise ValueError(f"Empty tour for {instance.name}")
    total = 0
    size = len(tour)
    for index in range(size):
        total += edge_distance(instance, tour[index], tour[(index + 1) % size])
    return float(total)


def validate_tour(instance: TSPInstance, tour: Sequence[int]) -> tuple[bool, str | None]:
    if len(tour) != instance.dimension:
        return False, f"{instance.name}: expected {instance.dimension} nodes, got {len(tour)}"
    seen = bytearray(instance.dimension)
    for node in tour:
        if not isinstance(node, int):
            return False, f"{instance.name}: non-integer node id {node!r}"
        if node < 0 or node >= instance.dimension:
            return False, f"{instance.name}: node id {node} out of range"
        if seen[node]:
            return False, f"{instance.name}: duplicate node id {node}"
        seen[node] = 1
    return True, None


def _normalize_tour(nodes: list[int], dimension: int) -> list[int]:
    if not nodes:
        raise ValueError("Reference tour is empty")
    if min(nodes) >= 1 and max(nodes) <= dimension:
        return [node - 1 for node in nodes]
    if min(nodes) >= 0 and max(nodes) < dimension:
        return nodes
    raise ValueError(f"Reference tour indices do not match dimension={dimension}")


def load_reference_tour(path: Path, dimension: int) -> list[int]:
    nodes: list[int] = []
    in_section = False
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            upper = line.upper()
            if upper == "TOUR_SECTION":
                in_section = True
                continue
            if upper == "EOF":
                break
            if not in_section:
                continue
            for token in line.split():
                value = int(token)
                if value == -1:
                    return _normalize_tour(nodes, dimension)
                nodes.append(value)
    return _normalize_tour(nodes, dimension)


def load_tsp_instance(path: Path) -> TSPInstance:
    header: dict[str, str] = {}
    coords: list[tuple[float, float]] = []
    in_coord_section = False

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            upper = line.upper()
            if upper == "EOF":
                break
            if upper == "NODE_COORD_SECTION":
                in_coord_section = True
                continue
            if not in_coord_section:
                parsed = _parse_header_line(line)
                if parsed is not None:
                    key, value = parsed
                    header[key] = value
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Malformed node line in {path}: {line}")
            coords.append((float(parts[1]), float(parts[2])))

    name = header.get("NAME", path.stem)
    dimension = int(header["DIMENSION"])
    edge_weight_type = header.get("EDGE_WEIGHT_TYPE", "EUC_2D").upper()
    if edge_weight_type not in SUPPORTED_EDGE_WEIGHT_TYPES:
        raise ValueError(f"Unsupported edge type for {name}: {edge_weight_type}")
    if len(coords) != dimension:
        raise ValueError(f"{name}: expected {dimension} coordinates, found {len(coords)}")

    reference_path = None
    for suffix in (".opt.tour", ".tour"):
        candidate = path.with_name(f"{path.stem}{suffix}")
        if candidate.exists():
            reference_path = candidate
            break

    reference_tour = None
    best_known = None
    if reference_path is not None:
        reference_tour = load_reference_tour(reference_path, dimension)
        feasible, error = validate_tour(
            TSPInstance(
                name=name,
                coords=coords,
                dimension=dimension,
                edge_weight_type=edge_weight_type,
            ),
            reference_tour,
        )
        if not feasible:
            raise ValueError(f"{name}: invalid reference tour: {error}")
        best_known = compute_tour_length(
            TSPInstance(
                name=name,
                coords=coords,
                dimension=dimension,
                edge_weight_type=edge_weight_type,
            ),
            reference_tour,
        )

    return TSPInstance(
        name=name,
        coords=coords,
        dimension=dimension,
        edge_weight_type=edge_weight_type,
        best_known=best_known,
        reference_tour=reference_tour,
        metadata={
            "source_path": str(path),
            "reference_path": str(reference_path) if reference_path is not None else None,
        },
    )


def discover_instance_paths() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in sorted(DATA_DIR.rglob("*.tsp")):
        stem = path.stem.lower()
        if stem in paths:
            raise ValueError(f"Duplicate instance stem discovered: {stem}")
        paths[stem] = path
    return paths


def describe_tier(size: str) -> dict[str, list[dict[str, Any]]]:
    available = discover_instance_paths()
    found: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []

    for stem in BENCHMARK_TIERS[size]:
        path = available.get(stem)
        if path is None:
            missing.append({"name": stem})
            continue
        inspected = inspect_instance_file(path)
        found.append(
            {
                "name": inspected["name"],
                "dimension": inspected["dimension"],
                "edge_weight_type": inspected["edge_weight_type"],
                "path": str(path),
                "reference_tour": any(
                    path.with_name(f"{path.stem}{suffix}").exists()
                    for suffix in (".opt.tour", ".tour")
                ),
            }
        )

    return {"found": found, "missing": missing}


def load_benchmark_instances(size: str, verbose: bool = True) -> list[TSPInstance]:
    if size not in BENCHMARK_TIERS:
        raise ValueError(f"Unknown benchmark size: {size}")

    available = discover_instance_paths()
    instances: list[TSPInstance] = []
    found_names: list[str] = []
    missing_names: list[str] = []

    for stem in BENCHMARK_TIERS[size]:
        path = available.get(stem)
        if path is None:
            missing_names.append(stem)
            continue
        instances.append(load_tsp_instance(path))
        found_names.append(stem)

    if verbose:
        print(f"[prepare] size={size}")
        if found_names:
            print(f"[prepare] found={', '.join(found_names)}")
        if missing_names:
            print(f"[prepare] missing={', '.join(missing_names)}")
        if not found_names:
            print(f"[prepare] no {size} benchmark instances found under {DATA_DIR}")

    if size == "large" and not instances:
        print("[prepare] no large instances are available locally, exiting cleanly.")

    return instances


def score_objective(instance: TSPInstance, objective: float) -> dict[str, Any]:
    if instance.best_known is not None:
        score = ((objective - instance.best_known) / instance.best_known) * 100.0
        return {
            "score": score,
            "score_kind": "gap_pct",
            "reference_objective": instance.best_known,
        }
    return {
        "score": objective,
        "score_kind": "raw_objective",
        "reference_objective": None,
    }


def summarize_scores(per_instance_metrics: Sequence[dict[str, Any]], total_runtime_s: float) -> dict[str, Any]:
    if per_instance_metrics:
        scores = [item["score"] for item in per_instance_metrics]
        mean_score = statistics.fmean(scores)
        median_score = statistics.median(scores)
    else:
        mean_score = math.inf
        median_score = math.inf
    return {
        "score": mean_score,
        "mean_score": mean_score,
        "median_score": median_score,
        "total_runtime_s": total_runtime_s,
        "num_instances": len(per_instance_metrics),
    }


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "nogit"
    return result.stdout.strip() or "nogit"


def build_run_stub(
    *,
    size: str,
    seed: int,
    budget_s: float,
    benchmark_instance_names: Sequence[str],
) -> dict[str, Any]:
    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    return {
        "run_id": f"{run_id}_{size}_seed{seed}",
        "timestamp": timestamp.isoformat(),
        "size": size,
        "seed": seed,
        "budget_s": budget_s,
        "commit": get_git_commit(),
        "benchmark_instance_names": list(benchmark_instance_names),
    }


def build_artifact(
    *,
    size: str,
    seed: int,
    budget_s: float,
    benchmark_instance_names: Sequence[str],
    per_instance_metrics: Sequence[dict[str, Any]],
    total_runtime_s: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact = build_run_stub(
        size=size,
        seed=seed,
        budget_s=budget_s,
        benchmark_instance_names=benchmark_instance_names,
    )
    artifact["per_instance_metrics"] = list(per_instance_metrics)
    artifact["aggregate_metrics"] = summarize_scores(per_instance_metrics, total_runtime_s)
    if extra:
        artifact.update(extra)
    return artifact


def build_crash_artifact(
    *,
    size: str,
    seed: int,
    budget_s: float,
    benchmark_instance_names: Sequence[str],
    total_runtime_s: float,
    error: str,
) -> dict[str, Any]:
    return build_artifact(
        size=size,
        seed=seed,
        budget_s=budget_s,
        benchmark_instance_names=benchmark_instance_names,
        per_instance_metrics=[],
        total_runtime_s=total_runtime_s,
        extra={"error": error},
    )


def _allocate_instance_budgets(instances: Sequence[TSPInstance], total_budget_s: float) -> list[float]:
    if not instances:
        return []
    per_instance = total_budget_s / len(instances)
    return [per_instance for _ in instances]


def run_benchmark(
    *,
    instances: Sequence[TSPInstance],
    solve_instance: Callable[[TSPInstance, float, int], dict[str, Any]],
    size: str,
    budget_s: float,
    seed: int,
) -> dict[str, Any]:
    per_instance_metrics: list[dict[str, Any]] = []
    budgets = _allocate_instance_budgets(instances, budget_s)
    benchmark_start = time.perf_counter()

    for index, (instance, instance_budget) in enumerate(zip(instances, budgets, strict=False)):
        instance_seed = seed + index
        solve_start = time.perf_counter()
        result = solve_instance(instance, instance_budget, instance_seed)
        runtime_s = time.perf_counter() - solve_start

        solution = result["solution"]
        feasible, error = validate_tour(instance, solution)
        if not feasible:
            raise ValueError(error)

        objective = compute_tour_length(instance, solution)
        scored = score_objective(instance, objective)
        per_instance_metrics.append(
            {
                "name": instance.name,
                "dimension": instance.dimension,
                "edge_weight_type": instance.edge_weight_type,
                "budget_s": instance_budget,
                "seed": instance_seed,
                "objective": objective,
                "reported_objective": result.get("objective"),
                "score": scored["score"],
                "score_kind": scored["score_kind"],
                "reference_objective": scored["reference_objective"],
                "runtime_s": runtime_s,
                "metadata": result.get("metadata", {}),
            }
        )

    total_runtime_s = time.perf_counter() - benchmark_start
    return build_artifact(
        size=size,
        seed=seed,
        budget_s=budget_s,
        benchmark_instance_names=[instance.name for instance in instances],
        per_instance_metrics=per_instance_metrics,
        total_runtime_s=total_runtime_s,
    )


def initialize_results_tsv() -> None:
    if RESULTS_TSV.exists():
        return
    RESULTS_TSV.write_text(
        "commit\tscore\truntime_s\tstatus\tdescription\n",
        encoding="utf-8",
    )


def record_run(artifact: dict[str, Any], *, status: str, description: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    initialize_results_tsv()

    artifact_path = RESULTS_DIR / f"{artifact['run_id']}.json"
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    score = artifact["aggregate_metrics"]["score"]
    runtime_s = artifact["aggregate_metrics"]["total_runtime_s"]
    clean_description = description.replace("\t", " ").strip()
    with RESULTS_TSV.open("a", encoding="utf-8") as handle:
        handle.write(
            f"{artifact['commit']}\t{score:.6f}\t{runtime_s:.3f}\t{status}\t{clean_description}\n"
        )
    return artifact_path


def _print_tier_summary(size: str) -> None:
    description = describe_tier(size)
    print(f"{size}:")
    if description["found"]:
        for item in description["found"]:
            reference = "yes" if item["reference_tour"] else "no"
            print(
                f"  - {item['name']} (n={item['dimension']}, edge={item['edge_weight_type']}, "
                f"reference_tour={reference})"
            )
    else:
        print("  - none found")
    if description["missing"]:
        print("  missing:", ", ".join(item["name"] for item in description["missing"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect local TSP benchmark availability.")
    parser.add_argument("--list", action="store_true", help="List all benchmark tiers.")
    parser.add_argument("--size", choices=tuple(BENCHMARK_TIERS), help="Inspect one benchmark tier.")
    args = parser.parse_args()

    if args.list or args.size is None:
        for size in BENCHMARK_TIERS:
            _print_tier_summary(size)
        return 0

    _print_tier_summary(args.size)
    if args.size == "large" and not describe_tier("large")["found"]:
        print("[prepare] no large instances are available locally, exiting cleanly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
