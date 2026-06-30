"""Evolutionary algorithm that optimises R2D2's wheel-control parameters.

Steps:
1. Size the parameter vector to the two differential-drive controls (the only
   motors that matter): left- and right-side wheel velocities.
2. Load or initialise a population of candidate solutions.
3. Evolve the population, evaluating fitness headless and in parallel.
4. Plot fitness/parameter evolution and replay the best solution in the GUI.

Training runs in PyBullet's DIRECT (no-GUI) mode and spreads fitness
evaluation across worker processes, so it is dramatically faster than the
original GUI, one-candidate-at-a-time loop. The GUI is only used for the final
best-solution replay.

NOTE: the executable code lives under ``if __name__ == "__main__"`` because
parallel evaluation uses multiprocessing; on macOS workers are spawned by
re-importing this module, and an unguarded body would re-run training in every
worker.
"""

import argparse
import os
import re

import numpy as np

from r2d2.evolutionary_algorithm.training import initialize_population, evolve_population
from r2d2.evolutionary_algorithm.objective import objective_function, GENOME_SIZE

from r2d2.analysis.fitness_plot import plot_fitness_evolution
from r2d2.analysis.parameter_analysis import analyze_parameters
from r2d2.analysis.final_performance import visualize_best_solution


def main(population_size=6, num_generations=10, num_workers=None):
    """Run the evolutionary algorithm.

    Args:
        population_size: candidates per generation (used only for a fresh run;
            a resumed run keeps the saved population's size).
        num_generations: number of generations to evolve.
        num_workers: parallel worker processes for fitness evaluation;
            ``None`` uses all available CPU cores.
    """
    # The robot is driven as a differential-drive base, so each candidate only
    # needs two genes (left- and right-side wheel velocities) no matter how many
    # joints the URDF exposes. Those are the only motors actuated.
    num_parameters = GENOME_SIZE

    # Resume from the most recent saved population if one exists.
    directory = 'r2d2/evolutionary_algorithm/trained_models'
    os.makedirs(directory, exist_ok=True)
    date_pattern = r'best_generation_info_\d{14}.npy'
    matching_files = sorted(f for f in os.listdir(directory) if re.match(date_pattern, f))

    population = None
    if matching_files:
        selected_file = matching_files[-1]  # most recent by timestamp
        population_file_path = os.path.join(directory, selected_file)
        saved = np.load(population_file_path, allow_pickle=True)
        candidate = np.array([item['parameters'] for item in saved])
        # Populations saved under the older, wider (per-joint) genome are
        # incompatible with the current two-gene model; start fresh instead of
        # crashing on the dimension mismatch.
        if candidate.ndim == 2 and candidate.shape[1] == num_parameters:
            population = candidate
            print("Found matching file:", selected_file, "\nLoading population from file.")
        else:
            width = candidate.shape[1] if candidate.ndim == 2 else "?"
            print(f"Ignoring {selected_file}: it has {width} genes per candidate "
                  f"but the current model uses {num_parameters}. Starting fresh.")

    if population is None:
        print("Initializing a new population.")
        population = initialize_population(population_size, num_parameters)

    # Evolve the population (headless + parallel; prints live per-gen stats).
    final_population, best_fitnesses, avg_fitnesses, worst_fitnesses, population_history = evolve_population(
        population, num_generations, objective_function, workers=num_workers)

    # Post-training analysis (these open plot windows / a GUI replay).
    plot_fitness_evolution(best_fitnesses, avg_fitnesses, worst_fitnesses)
    analyze_parameters(population_history)
    best_solution = max(final_population, key=lambda candidate: objective_function(candidate))
    visualize_best_solution(best_solution)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evolve R2D2's wheel-control parameters (headless + parallel).")
    parser.add_argument(
        "-w", "--workers", type=int, default=None,
        help="parallel worker processes for fitness evaluation "
             "(default: all CPU cores; use 1 for sequential)")
    parser.add_argument(
        "-g", "--generations", type=int, default=10,
        help="number of generations to evolve (default: 10)")
    parser.add_argument(
        "-p", "--population", type=int, default=6,
        help="candidates per generation for a fresh run (default: 6)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(population_size=args.population,
         num_generations=args.generations,
         num_workers=args.workers)
