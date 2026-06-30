# Convenience targets so you never have to think about which Python
# interpreter is active. Everything runs through the repo-local .venv and
# sets PYTHONPATH so the `r2d2` package resolves.

VENV    := $(CURDIR)/.venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
PYBDIR  := $(CURDIR)/pybullet

# Training knobs, overridable on the command line, e.g.
#   make train WORKERS=4 GENERATIONS=20 POPULATION=10
# Empty by default so main.py's own defaults apply (all CPU cores, 10 gens).
WORKERS     ?=
GENERATIONS ?=
POPULATION  ?=
TRAIN_FLAGS := $(if $(WORKERS),--workers $(WORKERS)) \
               $(if $(GENERATIONS),--generations $(GENERATIONS)) \
               $(if $(POPULATION),--population $(POPULATION))

.PHONY: sim train test lint venv

## sim   - interactive, keyboard-controlled R2D2 simulation (opens a GUI window)
sim:
	cd $(PYBDIR) && PYTHONPATH=$(PYBDIR) $(PY) r2d2/sim-key-control.py

## train - run the EA (headless + parallel training; GUI replay + plots at the end)
##         tune with WORKERS=/GENERATIONS=/POPULATION=, e.g. make train WORKERS=4
train:
	cd $(PYBDIR) && PYTHONPATH=$(PYBDIR) $(PY) r2d2/evolutionary_algorithm/main.py $(TRAIN_FLAGS)

## test  - run the unit-test suite
test:
	$(PY) -m pytest -q

## lint  - run the ruff linter
lint:
	$(VENV)/bin/ruff check .
