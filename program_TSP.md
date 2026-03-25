# program_TSP.md

This is an experiment to have the LLM do its own research.


## Setup

1. Propose a run tag based on today's date.
2. Create a fresh branch named `autoresearch/<tag>`.
3. Read `README.md`, `prepare.py`, and `optimize.py`.
4. Confirm that the repo is ready.
5. Ensure `run.log` and `results.tsv` are ignored locally and are never committed.
6. Initialize `results.tsv` if it does not exist.
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
- add new dependencies
- add new files unless explicitly required by the human
- turn this into a bigger framework outside `optimize.py`

Goal:
- minimize the aggregate benchmark score produced by `prepare.py` under the fixed wall-clock budget
- lower is better

Primary design principle:
- `optimize.py` should hold two things:
- a scheduler that allocates the fixed harness budget across the benchmarks being optimized
- a heuristic solver entry for each benchmark currently under study

Research principle:
- benchmark-specific solvers are allowed
- scheduler changes and solver changes are both valid experiments
- the harness and total time budget stay fixed

Simplicity criterion:
- all else equal, simpler is better
- tiny gains are not worth ugly complexity
- a cleaner scheduler or clearer solver registry with equal or better score is a win

## Output Format

`optimize.py` prints:
- aggregate score
- median score
- total runtime
- per-instance objective, score, and runtime
- JSON artifact path in `results/`

Inspect prior results with `results.tsv` and the per-run JSON files in `results/`.

When reading results, separate:
- scheduler effects
- per-benchmark solver effects
- accidental runtime noise

## Logging Results

Append one row to `results.tsv` after each experiment. Keep logging simple:
- `commit`
- `score`
- `runtime_s`
- `status`
- `description`

Keep failed, discarded, and crashed experiments in `results.tsv`.
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
8. Record the result in `results.tsv`, including failed or discarded runs.
9. If the score improved, keep the commit.
10. If the score is equal or worse, revert to the previous good commit unless the human explicitly wants a new architectural baseline.

Guidance:
- prefer one small change at a time
- keep the scheduler explicit and readable
- keep each benchmark solver easy to identify
- do not hide benchmark-specific logic in scattered conditionals if a solver spec or registry entry would be clearer
- use explicit wall-clock timeouts for longer runs
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
