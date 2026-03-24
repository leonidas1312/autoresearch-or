# autoresearch-or

Minimal autoresearch scaffold for heuristic algorithms in Operations Research, starting with Euclidean TSP on real local benchmark data only.

`prepare.py` is the fixed harness for benchmark discovery, TSPLIB parsing, scoring, validation, and run logging. `optimize.py` is the only editable experiment file. `program_TSP.md` is the instruction file for future agent runs.

Benchmarks are grouped into `small`, `medium`, and `large` tiers and are discovered from the local files under `data/tsp/`. Reference tours are used when present; otherwise scoring falls back to raw objective. Lower is better.

The baseline in `optimize.py` is deterministic nearest-neighbor construction with budget-aware 2-opt, plus a simple sweep fallback for extreme optional large instances where exact nearest-neighbor is not practical.

Run one benchmark with:

```bash
python optimize.py
```

Other common runs:

```bash
python optimize.py --size medium --budget 300 --seed 0 --description "medium baseline"
python optimize.py --size large --budget 300 --seed 0 --description "large baseline"
python prepare.py --list
```

The intended loop is simple: run the baseline, make one focused change in `optimize.py`, rerun the benchmark, and keep or discard the commit based on the logged score in `results.tsv` and the JSON artifact in `results/`.
