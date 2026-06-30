# softrobot

Physics-based robot simulation built on [PyBullet](https://pybullet.org/).
The current focus is an **evolutionary algorithm** that learns wheel-control
parameters to drive an R2D2 model toward a target, plus an interactive
keyboard-driven simulation of the same robot.

R2D2 is treated as a **differential-drive base**: only the left- and right-side
wheel motors are actuated, so each candidate solution is just two
values — one target velocity per side (equal values drive straight, a
difference steers). None of the other joints (legs, gripper, head) are driven.

> This started as an unfinished college project (originally prototyped with
> vPython, now built on PyBullet) and is being brought to completion. The
> legacy vPython experiments are kept under [`vPython/`](vPython/) for
> reference and are not part of the maintained code.

## Repository layout

```
pybullet/
  r2d2/
    sim-key-control.py            # interactive, keyboard-controlled R2D2 sim
    evolutionary_algorithm/       # the EA: training loop + objective function
    analysis/                     # plots of fitness and parameter evolution
    specs/                        # robot joint specifications (CSV)
  requirements.txt
vPython/                          # legacy vPython prototypes (unmaintained)
tests/                            # unit tests for the EA operators
```

## Local development

Requires **Python 3.9+**.

```bash
git clone https://github.com/james-worx/softrobot.git
cd softrobot

python3 -m venv .venv
source .venv/bin/activate

# Either install the package (recommended) ...
pip install -e .
# ... or just the runtime dependencies:
pip install -r pybullet/requirements.txt
```

## Running

### Using `make` (recommended)

A [`Makefile`](Makefile) wraps the common commands. Each target invokes the
repo-local `.venv` interpreter and sets `PYTHONPATH` for you, so you don't
have to keep the virtualenv activated or remember to export `PYTHONPATH`.
Run these from the **repository root**:

```bash
make sim      # interactive, keyboard-controlled R2D2 simulation (GUI window)
make train    # evolutionary-algorithm training (headless), then GUI replay + plots
make test     # run the unit-test suite
make lint     # run the ruff linter
```

Training runs **headless and in parallel** (one worker process per CPU core),
printing live per-generation stats (best/avg/worst fitness, time per
generation, and ETA). Tune it with make variables:

```bash
make train WORKERS=4         # cap fitness evaluation at 4 worker processes
make train WORKERS=1         # evaluate sequentially (no multiprocessing)
make train GENERATIONS=20    # evolve for more generations
make train POPULATION=40     # larger population (fresh runs only)
```

`make sim` opens a GUI window for the whole session, and `make train` opens
one for the final best-solution replay plus the analysis plots — so run them
on a local desktop session rather than over a plain SSH connection.

### Running the scripts directly

If you'd rather not use `make`, run the scripts yourself from the
`pybullet/` directory so the `r2d2` package resolves. Make sure the `.venv`
is active first (`source .venv/bin/activate`):

```bash
cd pybullet
```

#### Interactive simulation

```bash
python3 r2d2/sim-key-control.py
```

Opens a GUI window. Drive R2D2 with the arrow keys (up/down for velocity,
left/right to turn) and toggle the gripper with the space bar.

#### Evolutionary-algorithm training

```bash
export PYTHONPATH=$(pwd)
python3 r2d2/evolutionary_algorithm/main.py [--workers N] [--generations N] [--population N]
```

The algorithm evolves a population of two-gene candidates (left- and
right-side wheel velocities) to minimise the distance between R2D2 and a
target placed directly ahead of it. Fitness
evaluation runs headless (PyBullet `DIRECT`) and is parallelised across worker
processes — pass `--workers` (default: all CPU cores), `--generations`
(default: 10) and `--population` (default: 20); see `--help`. Training
artifacts (final population, fitness history, best generation) are written to
`r2d2/evolutionary_algorithm/trained_models/` and are git-ignored; a later run
automatically resumes from the most recent saved population if one is present.

After training, the run plots fitness evolution and per-parameter evolution
(via the modules in `r2d2/analysis/`) and replays the best solution.

#### Live telemetry HUD

The best-solution replay hides PyBullet's default preview panels (the RGB,
depth and segmentation tiles, which are not useful here) and overlays live
locomotion telemetry that tracks the robot:

* **path travelled** – ground distance covered,
* **displacement** – straight-line distance from the start,
* **speed** – current (and peak) horizontal speed,
* **to target** – distance remaining to the goal, and
* **fitness** – the score the evolutionary algorithm optimises.

The path taken is traced on the ground and the camera follows the robot. The
per-parameter charts are drawn as a single gridded figure with one labelled
subplot per drive side (left/right wheels), instead of one pop-up window per
parameter.

## Development workflow

```bash
pip install -e ".[dev]"   # installs ruff + pytest

ruff check .              # lint  (or: make lint)
pytest -q                 # run the test suite  (or: make test)
```

The same two checks run in CI on every pull request and push to `master`
(`.github/workflows/ci.yml`).

## Releases

Releases are automated with
[python-semantic-release](https://python-semantic-release.readthedocs.io/).
On every push to `master`, commits are analysed: the version in
`pyproject.toml` is bumped, `CHANGELOG.md` is updated, and a tagged GitHub
Release is published. This only works if commits follow the conventions in
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

Released under the [MIT License](LICENSE).
