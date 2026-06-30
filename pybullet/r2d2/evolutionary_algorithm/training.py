import numpy as np
import random
import os
import datetime
import time
import multiprocessing as mp

def initialize_population(pop_size, num_parameters):
    return np.random.uniform(-10, 10, (pop_size, num_parameters))  # Increased range of initial parameters


def _init_worker():
    """Give each parallel worker process its own headless PyBullet connection.

    PyBullet keeps a single physics client per process, so fitness evaluation
    is parallelised with processes (not threads). Each worker connects once in
    DIRECT mode and reuses that connection for every candidate it evaluates.
    """
    import pybullet as p
    if p.getConnectionInfo()['isConnected'] == 0:
        p.connect(p.DIRECT)


def evolve_population(population, num_generations, objective_function, mutation_rate=0.1, workers=None):
    """
    Evolves a population over a specified number of generations using an evolutionary algorithm.

    Args:
        population (numpy.ndarray): The initial population of candidate solutions.
        num_generations (int): The number of generations to evolve the population.
        objective_function (function): The function used to evaluate the fitness of each candidate solution.
        mutation_rate (float, optional): The rate at which mutations occur during reproduction. Defaults to 0.1.

    Returns:
        tuple: A tuple containing the final population, best fitnesses, average fitnesses, worst fitnesses, and population history.

    Raises:
        None

    """

    best_fitnesses = []
    avg_fitnesses = []
    worst_fitnesses = []
    population_history = []

    # Fitness evaluation is the bottleneck: each candidate runs a full headless
    # simulation. Spread the population across worker processes (each with its
    # own PyBullet connection) so the candidates in a generation evaluate in
    # parallel. workers=1 falls back to sequential evaluation.
    if workers is None:
        workers = os.cpu_count() or 1
    workers = min(workers, len(population))
    pool = mp.Pool(processes=workers, initializer=_init_worker) if workers > 1 else None
    print(f"Evaluating {len(population)} candidates/generation across "
          f"{workers} worker(s) over {num_generations} generations.")

    start_time = time.time()
    try:
        for generation in range(num_generations):
            gen_start = time.time()

            # Evaluate each candidate (in parallel when workers > 1).
            candidates = list(population)
            if pool is not None:
                fitnesses = pool.map(objective_function, candidates)
            else:
                fitnesses = [objective_function(candidate) for candidate in candidates]

            # Log fitness values
            best_fitnesses.append(max(fitnesses))
            avg_fitnesses.append(np.mean(fitnesses))
            worst_fitnesses.append(min(fitnesses))

            # Store population history
            generation_info = [{'parameters': candidate.tolist(), 'fitness': fitness} for candidate, fitness in zip(population, fitnesses)]
            population_history.append(generation_info)

            # Select the best candidates
            selected = select_candidates(population, fitnesses)

            # Reproduce
            offspring = reproduce(selected, mutation_rate)

            # Create the new population
            population = np.array(offspring)

            # Live per-generation stats: fitness spread, time, and ETA.
            done = generation + 1
            gen_time = time.time() - gen_start
            eta = (time.time() - start_time) / done * (num_generations - done)
            print(f"Generation {done}/{num_generations}: "
                  f"best={best_fitnesses[-1]:.3f} "
                  f"avg={avg_fitnesses[-1]:.3f} "
                  f"worst={worst_fitnesses[-1]:.3f} | "
                  f"{gen_time:.1f}s/gen | ETA {eta:.0f}s")
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    # Save training artifacts for analysis and future training sessions.
    # Use a single timestamp so all files from this run share one suffix.
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    model_dir = 'r2d2/evolutionary_algorithm/trained_models'
    os.makedirs(model_dir, exist_ok=True)

    # Save the full population history (every generation), not just the last one.
    np.save(os.path.join(model_dir, f'final_population_history_{timestamp}.npy'),
            np.array(population_history, dtype=object))

    # Save the final population so a future run can resume from it.
    final_population_path = os.path.join(model_dir, f'final_population_{timestamp}.npy')
    np.save(final_population_path, population)

    # Save the best generation's info and the fitness history.
    best_fitness_index = np.argmax(best_fitnesses)
    best_generation_info = population_history[best_fitness_index]
    np.save(os.path.join(model_dir, f'best_generation_info_{timestamp}.npy'),
            np.array(best_generation_info))
    np.save(os.path.join(model_dir, f'avg_fitnesses_{timestamp}.npy'), np.array(avg_fitnesses))
    np.save(os.path.join(model_dir, f'worst_fitnesses_{timestamp}.npy'), np.array(worst_fitnesses))
    print('Final population saved to:', final_population_path)

    return population, best_fitnesses, avg_fitnesses, worst_fitnesses, population_history

def select_candidates(population, fitnesses, num_selected=4):
    """
    Selects the top candidates from the population based on their fitness scores.

    Args:
        population (numpy.ndarray): The population of candidates.
        fitnesses (numpy.ndarray): The fitness scores of the candidates.
        num_selected (int, optional): The number of candidates to select. Defaults to 10.

    Returns:
        numpy.ndarray: The selected candidates from the population.
    """
    selected_indices = np.argsort(fitnesses)[-num_selected:]
    return population[selected_indices]

def reproduce(selected, mutation_rate):
    """
    Reproduces the selected individuals by performing crossover and mutation.

    Args:
        selected (list): A list of selected individuals.
        mutation_rate (float): The probability of mutation for each gene.

    Returns:
        list: A list of offspring generated through reproduction.
    """
    offspring = []
    for _ in range(len(selected) * 2):
        parent1, parent2 = random.sample(list(selected), 2)
        child = crossover(parent1, parent2)
        child = mutate(child, mutation_rate)
        offspring.append(child)
    return offspring

def crossover(parent1, parent2):
    """
    Perform crossover between two parents to create a child.

    Args:
        parent1 (numpy.ndarray): The first parent.
        parent2 (numpy.ndarray): The second parent.

    Returns:
        numpy.ndarray: The child created through crossover.
    """
    crossover_point = random.randint(0, len(parent1) - 1)
    child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
    return child

def mutate(candidate, mutation_rate):
    """
    Mutates the given candidate by adding random noise to each element.

    Args:
        candidate (list): The candidate solution to be mutated.
        mutation_rate (float): The probability of mutation for each element.

    Returns:
        list: The mutated candidate solution.
    """
    for i in range(len(candidate)):
        if random.random() < mutation_rate:
            candidate[i] += np.random.normal()
    return candidate