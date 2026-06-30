"""Unit tests for the evolutionary-algorithm operators.

These cover the pure functions in ``r2d2.evolutionary_algorithm.training``
that do not need a PyBullet physics server, so they run headless in CI.
"""
import numpy as np

from r2d2.evolutionary_algorithm.training import (
    crossover,
    initialize_population,
    mutate,
    reproduce,
    select_candidates,
)
from r2d2.evolutionary_algorithm.main import load_latest_population


def test_initialize_population_shape_and_range():
    pop = initialize_population(pop_size=6, num_parameters=4)
    assert pop.shape == (6, 4)
    assert pop.min() >= -10
    assert pop.max() <= 10


def test_select_candidates_returns_highest_fitness():
    pop = np.arange(20, dtype=float).reshape(5, 4)
    fitnesses = [1.0, 5.0, 2.0, 4.0, 3.0]  # best two are indices 1 and 3

    selected = select_candidates(pop, fitnesses, num_selected=2)

    assert selected.shape == (2, 4)
    chosen = {tuple(row) for row in selected}
    assert chosen == {tuple(pop[1]), tuple(pop[3])}


def test_crossover_preserves_length():
    parent1 = np.zeros(8)
    parent2 = np.ones(8)

    child = crossover(parent1, parent2)

    assert len(child) == 8
    # Every gene comes from one of the two parents.
    assert set(np.unique(child)).issubset({0.0, 1.0})


def test_mutate_with_zero_rate_is_noop():
    candidate = np.arange(10, dtype=float)
    out = mutate(candidate.copy(), mutation_rate=0.0)
    assert np.array_equal(out, candidate)


def test_reproduce_doubles_population():
    selected = np.random.uniform(-1, 1, (4, 5))
    offspring = reproduce(selected, mutation_rate=0.1)
    assert len(offspring) == len(selected) * 2
    assert all(len(child) == 5 for child in offspring)


def test_load_latest_population_uses_newest_final_population(tmp_path):
    older = np.array([[1.0, 2.0], [3.0, 4.0]])
    newer = np.array([[5.0, 6.0], [7.0, 8.0]])
    np.save(tmp_path / "final_population_20260630010101.npy", older)
    np.save(tmp_path / "final_population_20260630020202.npy", newer)

    loaded = load_latest_population(tmp_path, num_parameters=2)

    assert np.array_equal(loaded, newer)


def test_load_latest_population_ignores_best_generation_files(tmp_path):
    best_generation = np.array(
        [{"parameters": [1.0, 2.0], "fitness": -3.0}],
        dtype=object,
    )
    np.save(tmp_path / "best_generation_info_20260630020202.npy", best_generation)

    loaded = load_latest_population(tmp_path, num_parameters=2)

    assert loaded is None
