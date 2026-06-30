"""Evolutionary algorithm that optimises R2D2's wheel-control parameters.

Steps:
1. Count the robot's joints (headless) to size the parameter vector.
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
import pybullet as p
import pybullet_data  # type: ignore

from r2d2.evolutionary_algorithm.training import initialize_population, evolve_population
from r2d2.evolutionary_algorithm.objective import objective_function

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
    # Count the robot's joints with a headless connection (no GUI needed),
    # then dynamically size the parameter vector to match.
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot_id = p.loadURDF("r2d2.urdf")
    num_parameters = p.getNumJoints(robot_id)
    p.resetSimulation()
    p.disconnect()

    # Resume from the most recent saved population if one exists.
    directory = 'r2d2/evolutionary_algorithm/trained_models'
    os.makedirs(directory, exist_ok=True)
    date_pattern = r'best_generation_info_\d{14}.npy'
    matching_files = sorted(f for f in os.listdir(directory) if re.match(date_pattern, f))

    if matching_files:
        selected_file = matching_files[-1]  # most recent by timestamp
        population_file_path = os.path.join(directory, selected_file)
        population = np.load(population_file_path, allow_pickle=True)
        parameters_list = [item['parameters'] for item in population]
        population = np.array(parameters_list)
        print("Found matching file:", selected_file, "\nLoading population from file.")
    else:
        print("No matching file found. Initializing a new population.")
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
