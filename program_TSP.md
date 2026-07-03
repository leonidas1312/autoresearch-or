# program_TSP.md

This is an experiment to have the LLM do its own research.


## Setup

1. Propose a run tag based on today's date.
2. Create a fresh branch named `autoresearch/<tag>`.
3. Read `README.md`, `prepare.py`, and `optimize.py`.
4. Confirm that the repo is ready.
5. Ensure `run.log`, `results_v3/`, and `results_v3.tsv` are ignored locally and are never committed.
6. Initialize `results_v3.tsv` by running one baseline experiment if it does not exist.
7. Run the current solver/scheduler baseline first before making changes.

## Experimentation

What you CAN do:
- modify `optimize.py`
- change the scheduler inside `optimize.py`
- change the solver assigned to any benchmark inside `optimize.py`
- add or remove benchmark-specific solver specs inside `optimize.py`
- refactor `optimize.py` so the scheduler and solver registry are clearer
- FULLY modify the logic of `optimize.py`

What you CANNOT do:
- modify `prepare.py`
- modify the fixed evaluation metric
- read or use hidden reference objectives from `prepare.py` inside solver logic
- add new dependencies
- add new files unless explicitly required by the human
- turn this into a bigger framework outside `optimize.py`

Goal:
- minimize the aggregate `relative_gap_pct_v3` score produced by `prepare.py` under the fixed wall-clock budget
- lower is better
- track `--suite opt_tour` and `--suite baseline_ref` separately when the question is known optimality versus baseline improvement

Primary design principle:
- `optimize.py` should hold two things:
- `solve_benchmark`, a scheduler that allocates the fixed harness budget across the active benchmark instances
- a solver registry and heuristic solver entries for benchmarks currently under study

Research principle:
- benchmark-specific solvers are allowed
- scheduler changes and solver changes are both valid experiments
- the harness and total time budget stay fixed
- size filters and suite filters are independent; use `--suite opt_tour` for all known-optimum instances, `--suite baseline_ref` for non-optimal baseline references, and combine with `--size` when needed
- `prepare.py` passes sanitized public `TSPInstance` objects to the solver; reference objectives are scoring-only data

Simplicity criterion:
- all else equal, simpler is better
- tiny gains are not worth ugly complexity
- a cleaner scheduler or clearer solver registry with equal or better score is a win

## Output Format

`optimize.py` prints:
- aggregate score
- score schema
- suite
- median score
- per-reference-kind aggregate scores when present
- total runtime
- per-instance objective, score, and runtime
- JSON artifact path in `results_v3/`

Inspect prior results with `results_v3.tsv` and the per-run JSON files in `results_v3/`.

When reading results, separate:
- scheduler effects
- per-benchmark solver effects
- accidental runtime noise

## Logging Results

Append one row to `results_v3.tsv` after each experiment. The harness writes:
- `run_id`
- `commit`
- `score`
- `runtime_s`
- `size`
- `suite`
- `seed`
- `budget_s`
- `num_instances`
- `opt_tour_score`
- `baseline_ref_score`
- `over_budget`
- `score_schema`
- `status`
- `artifact_path`
- `description`

Keep failed, discarded, and crashed experiments in `results_v3.tsv`.
Use the appropriate `status` and the commit that produced the run even if you later revert that solver or scheduler change.

In the description, say whether the change primarily touched:
- scheduler
- solver
- both

## Experiment Loop

LOOP FOREVER:
1. Check git state.
2. Inspect the current scheduler and benchmark solver assignments in `optimize.py`.
3. Make one focused change to either the scheduler, one benchmark solver, or one shared heuristic component.
4. `git commit`
5. Run the benchmark and redirect output to `run.log`.
6. Read the final metric from the log.
7. If the run crashed, inspect `run.log` and either fix once or discard.
8. Confirm the result was recorded in `results_v3.tsv`, including failed or discarded runs.
9. If the score improved, keep the commit.
10. If the score is equal or worse, revert to the previous good commit unless the human explicitly wants a new architectural baseline.

Guidance:
- prefer one small change at a time
- keep `schedule_budgets` explicit and readable
- keep each benchmark solver easy to identify
- do not hide benchmark-specific logic in scattered conditionals if a solver spec or registry entry would be clearer
- respect the benchmark-level `deadline` passed into `solve_benchmark`
- if a change crashes twice, discard it and move on
- do not mutate the harness to rescue a weak solver or scheduler idea

## Candidate Ideas

- retune scheduler weights across the active benchmarks
- change start-order policy for a single benchmark solver
- swap one benchmark from pure multi-start local search to perturb-and-restart
- add candidate lists for a specific benchmark solver
- improve 2-opt move ordering for one solver or for all solvers
- use different perturbation strengths by benchmark
- move a benchmark from generic defaults to a dedicated solver entry
- simplify duplicate solver logic by extracting shared components
- test when a benchmark should skip ILS entirely
