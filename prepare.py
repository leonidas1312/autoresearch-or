from __future__ import annotations

import argparse
import json
import math
import signal
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "tsp"
RESULTS_DIR = ROOT / "results_v3"
RESULTS_TSV = ROOT / "results_v3.tsv"

SCORE_SCHEMA = "relative_gap_pct_v3"
TIMER_GRACE_S = 0.25
OVER_BUDGET_TOLERANCE_S = 0.05

BENCHMARK_TIERS: dict[str, tuple[str, ...]] = {
    "small": ("att48", "eil51", "berlin52", "pr76", "rd100"),
    "medium": ("lin318", "pcb442", "rat783", "pr1002", "nrw1379", "pcb3038"),
    "large": ("qa194", "uy734", "lu980", "gr9882", "ch71009", "world"),
}
BENCHMARK_SIZE_CHOICES = ("all", *BENCHMARK_TIERS)

REFERENCE_SUITE_ALIASES = {
    "all": "all",
    "opt_tour": "opt_tour",
    "optimal": "opt_tour",
    "baseline_ref": "baseline_ref",
    "baseline": "baseline_ref",
}
REFERENCE_SUITE_CHOICES = tuple(REFERENCE_SUITE_ALIASES)

SUPPORTED_EDGE_WEIGHT_TYPES = {"EUC_2D", "CEIL_2D", "ATT", "GEO", "GEOM"}

# V3 uses one normalized score for every selected instance. Known TSPLIB instances use their
# local .opt.tour objective; all other instances use this deterministic harness
# baseline. The suite filter keeps known-optimum and baseline-reference results
# separable while preserving size filters for quick experiments.
REFERENCE_OBJECTIVES: dict[str, tuple[float, str]] = {
    "att48": (10628.0, "opt_tour"),
    "eil51": (426.0, "opt_tour"),
    "berlin52": (7542.0, "opt_tour"),
    "pr76": (108159.0, "opt_tour"),
    "rd100": (7910.0, "opt_tour"),
    "lin318": (68360.0, "baseline_sweep_v1"),
    "pcb442": (50778.0, "opt_tour"),
    "rat783": (15299.0, "baseline_sweep_v1"),
    "pr1002": (259045.0, "opt_tour"),
    "nrw1379": (79914.0, "baseline_sweep_v1"),
    "pcb3038": (231633.0, "baseline_sweep_v1"),
    "qa194": (16342.0, "baseline_sweep_v1"),
    "uy734": (122837.0, "baseline_sweep_v1"),
    "lu980": (19338.0, "baseline_sweep_v1"),
    "gr9882": (603290.0, "baseline_sweep_v1"),
    "ch71009": (11757850.0, "baseline_sweep_v1"),
    "world": (35917135.0, "baseline_sweep_v1"),
}


class BenchmarkTimeout(RuntimeError):
    pass


@dataclass(slots=True)
class TSPInstance:
    name: str
    coords: list[tuple[float, float]]
    dimension: int
    edge_weight_type: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    name: str
    instance: TSPInstance
    reference_objective: float
    reference_kind: str


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

    return TSPInstance(
        name=name,
        coords=coords,
        dimension=dimension,
        edge_weight_type=edge_weight_type,
        metadata={"source_path": str(path)},
    )


def baseline_sweep_tour(instance: TSPInstance) -> list[int]:
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


def discover_instance_paths() -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in sorted(DATA_DIR.rglob("*.tsp")):
        stem = path.stem.lower()
        if stem in paths:
            raise ValueError(f"Duplicate instance stem discovered: {stem}")
        paths[stem] = path
    return paths


def _all_benchmark_stems() -> tuple[str, ...]:
    return tuple(stem for tier in BENCHMARK_TIERS.values() for stem in tier)


def _canonical_suite(suite: str) -> str:
    try:
        return REFERENCE_SUITE_ALIASES[suite]
    except KeyError as exc:
        choices = ", ".join(REFERENCE_SUITE_CHOICES)
        raise ValueError(f"Unknown reference suite {suite!r}; expected one of: {choices}") from exc


def _benchmark_stems_for_size(size: str) -> tuple[str, ...]:
    if size == "all":
        return _all_benchmark_stems()
    try:
        return BENCHMARK_TIERS[size]
    except KeyError as exc:
        choices = ", ".join(BENCHMARK_SIZE_CHOICES)
        raise ValueError(f"Unknown benchmark size {size!r}; expected one of: {choices}") from exc


def _reference_kind_matches_suite(reference_kind: str, suite: str) -> bool:
    canonical_suite = _canonical_suite(suite)
    if canonical_suite == "all":
        return True
    if canonical_suite == "opt_tour":
        return reference_kind == "opt_tour"
    if canonical_suite == "baseline_ref":
        return reference_kind != "opt_tour"
    raise AssertionError(f"unhandled suite: {canonical_suite}")


def _selected_benchmark_stems(size: str, suite: str) -> tuple[str, ...]:
    selected: list[str] = []
    for stem in _benchmark_stems_for_size(size):
        _, reference_kind = _reference_entry_for(stem)
        if _reference_kind_matches_suite(reference_kind, suite):
            selected.append(stem)
    return tuple(selected)


def _reference_path_for(path: Path) -> Path | None:
    for suffix in (".opt.tour", ".tour"):
        candidate = path.with_name(f"{path.stem}{suffix}")
        if candidate.exists():
            return candidate
    return None


def _reference_entry_for(instance_name: str) -> tuple[float, str]:
    try:
        return REFERENCE_OBJECTIVES[instance_name]
    except KeyError as exc:
        raise ValueError(f"{instance_name}: missing V3 reference objective") from exc


def _validate_opt_tour_reference(path: Path, instance: TSPInstance, expected: float) -> None:
    reference_path = _reference_path_for(path)
    if reference_path is None:
        raise ValueError(f"{instance.name}: reference objective expects an opt tour, but none exists")
    reference_tour = load_reference_tour(reference_path, instance.dimension)
    feasible, error = validate_tour(instance, reference_tour)
    if not feasible:
        raise ValueError(f"{instance.name}: invalid reference tour: {error}")
    actual = compute_tour_length(instance, reference_tour)
    if actual != expected:
        raise ValueError(
            f"{instance.name}: reference objective mismatch, expected {expected}, got {actual}"
        )


def load_benchmark_case(path: Path) -> BenchmarkCase:
    instance = load_tsp_instance(path)
    reference_objective, reference_kind = _reference_entry_for(instance.name)
    if reference_kind == "opt_tour":
        _validate_opt_tour_reference(path, instance, reference_objective)
    return BenchmarkCase(
        name=instance.name,
        instance=instance,
        reference_objective=reference_objective,
        reference_kind=reference_kind,
    )


def describe_selection(size: str, suite: str = "all") -> dict[str, list[dict[str, Any]]]:
    available = discover_instance_paths()
    found: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    stems = _selected_benchmark_stems(size, suite)

    for stem in stems:
        path = available.get(stem)
        if path is None:
            missing.append({"name": stem})
            continue
        inspected = inspect_instance_file(path)
        reference_objective, reference_kind = REFERENCE_OBJECTIVES.get(
            inspected["name"], (None, "missing")
        )
        found.append(
            {
                "name": inspected["name"],
                "dimension": inspected["dimension"],
                "edge_weight_type": inspected["edge_weight_type"],
                "path": str(path),
                "reference_objective": reference_objective,
                "reference_kind": reference_kind,
            }
        )

    return {"found": found, "missing": missing}


def describe_tier(size: str) -> dict[str, list[dict[str, Any]]]:
    return describe_selection(size=size, suite="all")


def load_benchmark_instances(
    size: str,
    suite: str = "all",
    verbose: bool = True,
    allow_missing: bool = False,
) -> list[BenchmarkCase]:
    available = discover_instance_paths()
    cases: list[BenchmarkCase] = []
    found_names: list[str] = []
    missing_names: list[str] = []
    canonical_suite = _canonical_suite(suite)

    for stem in _selected_benchmark_stems(size, canonical_suite):
        path = available.get(stem)
        if path is None:
            missing_names.append(stem)
            continue
        cases.append(load_benchmark_case(path))
        found_names.append(stem)

    if missing_names and not allow_missing:
        raise FileNotFoundError(
            f"{size}: missing required benchmark instances: {', '.join(missing_names)}"
        )

    if verbose:
        print(f"[prepare] size={size}")
        print(f"[prepare] suite={canonical_suite}")
        if found_names:
            print(f"[prepare] found={', '.join(found_names)}")
        if missing_names:
            mode = "allowed" if allow_missing else "not allowed"
            print(f"[prepare] missing ({mode})={', '.join(missing_names)}")

    if not cases:
        raise FileNotFoundError(
            f"size={size} suite={canonical_suite}: no benchmark instances found under {DATA_DIR}"
        )

    return cases


def score_objective(case: BenchmarkCase, objective: float) -> dict[str, Any]:
    score = ((objective - case.reference_objective) / case.reference_objective) * 100.0
    return {
        "score": score,
        "score_kind": "relative_gap_pct",
        "reference_objective": case.reference_objective,
        "reference_kind": case.reference_kind,
    }


def summarize_scores(per_instance_metrics: Sequence[dict[str, Any]], total_runtime_s: float) -> dict[str, Any]:
    if per_instance_metrics:
        scores = [item["score"] for item in per_instance_metrics]
        mean_score = statistics.fmean(scores)
        median_score = statistics.median(scores)
    else:
        mean_score = math.inf
        median_score = math.inf
    reference_kind_metrics: dict[str, dict[str, Any]] = {}
    for reference_kind in sorted({item["reference_kind"] for item in per_instance_metrics}):
        kind_scores = [
            item["score"]
            for item in per_instance_metrics
            if item["reference_kind"] == reference_kind
        ]
        reference_kind_metrics[reference_kind] = {
            "score": statistics.fmean(kind_scores),
            "mean_score": statistics.fmean(kind_scores),
            "median_score": statistics.median(kind_scores),
            "num_instances": len(kind_scores),
        }
    return {
        "score": mean_score,
        "mean_score": mean_score,
        "median_score": median_score,
        "total_runtime_s": total_runtime_s,
        "num_instances": len(per_instance_metrics),
        "reference_kind_metrics": reference_kind_metrics,
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


def get_git_dirty() -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return bool(result.stdout.strip())


def build_run_stub(
    *,
    size: str,
    suite: str,
    seed: int,
    budget_s: float,
    benchmark_instance_names: Sequence[str],
    allow_missing: bool = False,
) -> dict[str, Any]:
    canonical_suite = _canonical_suite(suite)
    timestamp = datetime.now(timezone.utc)
    run_id = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    return {
        "run_id": f"{run_id}_{size}_{canonical_suite}_seed{seed}",
        "timestamp": timestamp.isoformat(),
        "size": size,
        "suite": canonical_suite,
        "seed": seed,
        "budget_s": budget_s,
        "commit": get_git_commit(),
        "git_dirty": get_git_dirty(),
        "benchmark_instance_names": list(benchmark_instance_names),
        "allow_missing": allow_missing,
        "score_schema": SCORE_SCHEMA,
    }


def build_artifact(
    *,
    size: str,
    suite: str,
    seed: int,
    budget_s: float,
    benchmark_instance_names: Sequence[str],
    per_instance_metrics: Sequence[dict[str, Any]],
    total_runtime_s: float,
    allow_missing: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact = build_run_stub(
        size=size,
        suite=suite,
        seed=seed,
        budget_s=budget_s,
        benchmark_instance_names=benchmark_instance_names,
        allow_missing=allow_missing,
    )
    artifact["per_instance_metrics"] = list(per_instance_metrics)
    artifact["aggregate_metrics"] = summarize_scores(per_instance_metrics, total_runtime_s)
    artifact["over_budget"] = total_runtime_s > budget_s + OVER_BUDGET_TOLERANCE_S
    if extra:
        artifact.update(extra)
    return artifact


def build_crash_artifact(
    *,
    size: str,
    suite: str,
    seed: int,
    budget_s: float,
    benchmark_instance_names: Sequence[str],
    total_runtime_s: float,
    error: str,
    error_type: str = "Exception",
    allow_missing: bool = False,
) -> dict[str, Any]:
    return build_artifact(
        size=size,
        suite=suite,
        seed=seed,
        budget_s=budget_s,
        benchmark_instance_names=benchmark_instance_names,
        per_instance_metrics=[],
        total_runtime_s=total_runtime_s,
        allow_missing=allow_missing,
        extra={"error": error, "error_type": error_type},
    )


def clone_solver_instance(instance: TSPInstance) -> TSPInstance:
    return TSPInstance(
        name=instance.name,
        coords=list(instance.coords),
        dimension=instance.dimension,
        edge_weight_type=instance.edge_weight_type,
        metadata=dict(instance.metadata),
    )


def _timeout_handler(signum: int, frame: Any) -> None:
    raise BenchmarkTimeout("benchmark solver exceeded hard wall-clock timeout")


def _arm_timeout(seconds: float) -> Any:
    if not hasattr(signal, "SIGALRM") or seconds <= 0:
        return None
    previous = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    return previous


def _clear_timeout(previous: Any) -> None:
    if previous is None or not hasattr(signal, "SIGALRM"):
        return
    signal.setitimer(signal.ITIMER_REAL, 0)
    signal.signal(signal.SIGALRM, previous)


def _normalize_result(result: Any, instance_name: str) -> Mapping[str, Any]:
    if not isinstance(result, Mapping):
        raise TypeError(f"{instance_name}: solver result must be a mapping")
    if "solution" not in result:
        raise KeyError(f"{instance_name}: solver result missing 'solution'")
    return result


def run_benchmark(
    *,
    cases: Sequence[BenchmarkCase],
    solve_benchmark: Callable[[Sequence[TSPInstance], float, int, float], Mapping[str, Any]],
    size: str,
    suite: str,
    budget_s: float,
    seed: int,
    allow_missing: bool = False,
) -> dict[str, Any]:
    if budget_s <= 0:
        raise ValueError("budget_s must be positive")

    expected_names = [case.name for case in cases]
    solver_instances = [clone_solver_instance(case.instance) for case in cases]
    benchmark_start = time.perf_counter()
    deadline = benchmark_start + budget_s
    previous_handler = _arm_timeout(budget_s + TIMER_GRACE_S)

    try:
        raw_results = solve_benchmark(solver_instances, budget_s, seed, deadline)

        total_runtime_s = time.perf_counter() - benchmark_start
        if total_runtime_s > budget_s + OVER_BUDGET_TOLERANCE_S:
            raise BenchmarkTimeout(
                f"benchmark exceeded budget: runtime={total_runtime_s:.3f}s budget={budget_s:.3f}s"
            )
        if not isinstance(raw_results, Mapping):
            raise TypeError("solve_benchmark must return a mapping keyed by instance name")

        actual_names = set(raw_results)
        expected_set = set(expected_names)
        missing = sorted(expected_set - actual_names)
        extra = sorted(actual_names - expected_set)
        if missing:
            raise KeyError(f"solver did not return results for: {', '.join(missing)}")
        if extra:
            raise KeyError(f"solver returned unexpected results for: {', '.join(extra)}")

        per_instance_metrics: list[dict[str, Any]] = []
        for case in cases:
            result = _normalize_result(raw_results[case.name], case.name)
            solution = result["solution"]
            feasible, error = validate_tour(case.instance, solution)
            if not feasible:
                raise ValueError(error)

            objective = compute_tour_length(case.instance, solution)
            scored = score_objective(case, objective)
            metadata = dict(result.get("metadata", {}))
            per_instance_metrics.append(
                {
                    "name": case.name,
                    "dimension": case.instance.dimension,
                    "edge_weight_type": case.instance.edge_weight_type,
                    "objective": objective,
                    "reported_objective": result.get("objective"),
                    "score": scored["score"],
                    "score_kind": scored["score_kind"],
                    "reference_objective": scored["reference_objective"],
                    "reference_kind": scored["reference_kind"],
                    "assigned_budget_s": metadata.get("assigned_budget_s"),
                    "solver_runtime_s": metadata.get("runtime_s", metadata.get("elapsed_s")),
                    "deadline_hit": metadata.get("deadline_hit"),
                    "stop_reason": metadata.get("stop_reason"),
                    "metadata": metadata,
                }
            )

        total_runtime_s = time.perf_counter() - benchmark_start
        if total_runtime_s > budget_s + OVER_BUDGET_TOLERANCE_S:
            raise BenchmarkTimeout(
                f"benchmark exceeded budget: runtime={total_runtime_s:.3f}s budget={budget_s:.3f}s"
            )
        return build_artifact(
            size=size,
            suite=suite,
            seed=seed,
            budget_s=budget_s,
            benchmark_instance_names=expected_names,
            per_instance_metrics=per_instance_metrics,
            total_runtime_s=total_runtime_s,
            allow_missing=allow_missing,
        )
    finally:
        _clear_timeout(previous_handler)


def initialize_results_tsv() -> None:
    if RESULTS_TSV.exists():
        return
    RESULTS_TSV.write_text(
        "\t".join(
            [
                "run_id",
                "commit",
                "score",
                "runtime_s",
                "size",
                "suite",
                "seed",
                "budget_s",
                "num_instances",
                "opt_tour_score",
                "baseline_ref_score",
                "over_budget",
                "score_schema",
                "status",
                "artifact_path",
                "description",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def record_run(artifact: dict[str, Any], *, status: str, description: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    initialize_results_tsv()

    artifact_path = RESULTS_DIR / f"{artifact['run_id']}.json"
    relative_artifact_path = artifact_path.relative_to(ROOT)
    clean_description = description.replace("\t", " ").strip()
    artifact["status"] = status
    artifact["description"] = clean_description
    artifact["artifact_path"] = str(relative_artifact_path)
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    aggregate = artifact["aggregate_metrics"]
    reference_metrics = aggregate.get("reference_kind_metrics", {})

    def score_field(reference_kind: str) -> str:
        metric = reference_metrics.get(reference_kind)
        if metric is None:
            return ""
        return f"{metric['score']:.6f}"

    fields = [
        artifact["run_id"],
        artifact["commit"],
        f"{aggregate['score']:.6f}",
        f"{aggregate['total_runtime_s']:.3f}",
        artifact["size"],
        artifact["suite"],
        str(artifact["seed"]),
        f"{artifact['budget_s']:.6f}",
        str(aggregate["num_instances"]),
        score_field("opt_tour"),
        score_field("baseline_sweep_v1"),
        str(bool(artifact.get("over_budget", False))).lower(),
        artifact["score_schema"],
        status,
        str(relative_artifact_path),
        clean_description,
    ]
    with RESULTS_TSV.open("a", encoding="utf-8") as handle:
        handle.write("\t".join(fields) + "\n")
    return artifact_path


def _print_selection_summary(size: str, suite: str = "all") -> None:
    canonical_suite = _canonical_suite(suite)
    description = describe_selection(size=size, suite=canonical_suite)
    print(f"{size} [{canonical_suite}]:")
    if description["found"]:
        for item in description["found"]:
            print(
                f"  - {item['name']} (n={item['dimension']}, edge={item['edge_weight_type']}, "
                f"reference={item['reference_objective']}:{item['reference_kind']})"
            )
    else:
        print("  - none found")
    if description["missing"]:
        print("  missing:", ", ".join(item["name"] for item in description["missing"]))


def _print_tier_summary(size: str) -> None:
    _print_selection_summary(size=size, suite="all")


def check_reference_objectives() -> None:
    cases = load_benchmark_instances("all", suite="all", verbose=False)
    for case in cases:
        if case.reference_kind == "baseline_sweep_v1":
            tour = baseline_sweep_tour(case.instance)
            actual = compute_tour_length(case.instance, tour)
            if actual != case.reference_objective:
                raise ValueError(
                    f"{case.name}: baseline reference mismatch, "
                    f"expected {case.reference_objective}, got {actual}"
                )
        print(
            f"{case.name}\t{case.reference_objective:.0f}\t{case.reference_kind}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect local TSP benchmark availability.")
    parser.add_argument("--list", action="store_true", help="List benchmark instances.")
    parser.add_argument(
        "--size",
        choices=BENCHMARK_SIZE_CHOICES,
        help="Filter by benchmark size; use all for the cross-size suite.",
    )
    parser.add_argument(
        "--suite",
        choices=REFERENCE_SUITE_CHOICES,
        default="all",
        help="Filter by reference suite: all, opt_tour/optimal, or baseline_ref/baseline.",
    )
    parser.add_argument("--allow-missing", action="store_true", help="Allow missing tier instances.")
    parser.add_argument(
        "--check-references",
        action="store_true",
        help="Recompute and validate all V3 reference objectives.",
    )
    args = parser.parse_args()

    if args.check_references:
        check_reference_objectives()
        return 0

    if args.size is not None:
        _print_selection_summary(args.size, args.suite)
        load_benchmark_instances(
            args.size,
            suite=args.suite,
            verbose=False,
            allow_missing=args.allow_missing,
        )
        return 0

    if args.list:
        _print_selection_summary("all", args.suite)
    else:
        for size in BENCHMARK_TIERS:
            _print_selection_summary(size, args.suite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
