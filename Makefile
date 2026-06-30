# Convenience targets so you never have to think about which Python
# interpreter is active. Everything runs through the repo-local .venv and
# sets PYTHONPATH so the `r2d2` package resolves.

VENV    := $(CURDIR)/.venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
PYBDIR  := $(CURDIR)/pybullet

.PHONY: sim train test lint venv

## sim   - interactive, keyboard-controlled R2D2 simulation (opens a GUI window)
sim:
	cd $(PYBDIR) && PYTHONPATH=$(PYBDIR) $(PY) r2d2/sim-key-control.py

## train - run the evolutionary-algorithm training (opens a GUI window + plots)
train:
	cd $(PYBDIR) && PYTHONPATH=$(PYBDIR) $(PY) r2d2/evolutionary_algorithm/main.py

## test  - run the unit-test suite
test:
	$(PY) -m pytest -q

## lint  - run the ruff linter
lint:
	$(VENV)/bin/ruff check .
