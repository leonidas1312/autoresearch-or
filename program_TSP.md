# program_TSP.md

## Setup

1. Propose a run tag based on today's date.
2. Create a fresh branch named `autoresearch/<tag>`.
3. Read `README.md`, `prepare.py`, and `optimize.py`.
4. Confirm that the repo is ready.
5. Ensure `run.log` and `results.tsv` are ignored locally and are never committed.
6. Initialize `results.tsv` if it does not exist.
7. Run the baseline first before making changes.

## Experimentation

What you CAN do:
- modify `optimize.py` only

What you CANNOT do:
- modify `prepare.py`
- modify the fixed evaluation metric
- add new dependencies
- add new files unless explicitly required by the human
- turn this into a bigger framework

Goal:
- minimize the aggregate benchmark score produced by `prepare.py` under the fixed wall-clock budget
- lower is better

Simplicity criterion:
- all else equal, simpler is better
- tiny gains are not worth ugly complexity
- simplification with equal or better score is a win

The first run must always establish the baseline.

## Output Format

`optimize.py` prints:
- aggregate score
- median score
- total runtime
- per-instance objective, score, and runtime
- JSON artifact path in `results/`

Inspect prior results with `results.tsv` and the per-run JSON files in `results/`.

## Logging Results

Append one row to `results.tsv` after each experiment. Keep logging simple:
- `commit`
- `score`
- `runtime_s`
- `status`
- `description`

Keep failed, discarded, and crashed experiments in `results.tsv`.
Use the appropriate `status` and the commit that produced the run even if you later revert that solver change.

## Experiment Loop

LOOP FOREVER:
1. Check git state.
2. Make one focused change to `optimize.py`.
3. `git commit`
4. Run the benchmark and redirect output to `run.log`.
5. Read the final metric from the log.
6. If the run crashed, inspect `run.log` and either fix once or discard.
7. Record the result in `results.tsv`, including failed or discarded runs.
8. If the score improved, keep the commit.
9. If the score is equal or worse, revert to the previous good commit.

Guidance:
- prefer one small change at a time
- use explicit wall-clock timeouts for longer runs
- if a change crashes twice, discard it and move on
- do not mutate the harness to rescue a weak solver idea

## Candidate Ideas

- multi-start construction
- better start node selection
- candidate lists
- improved 2-opt move ordering
- perturbation and restart
- iterated local search
- simulated annealing
- tabu memory
- VNS-style neighborhoods
