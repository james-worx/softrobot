import numpy as np
import matplotlib.pyplot as plt

from r2d2.evolutionary_algorithm.objective import WHEEL_JOINT_INDICES


def analyze_parameters(population_history):
    """Plot how each parameter evolved, as a single gridded figure.

    Previously this opened one window per parameter; for a robot with a dozen
    joints that meant a dozen pop-up windows. Everything is now drawn as a grid
    of subplots in one figure. The driven wheel joints (the only parameters the
    objective actually uses) are highlighted so the meaningful ones stand out.

    Args:
        population_history (list): a list of generations, where each generation
            is a list of candidate dicts with ``parameters`` and ``fitness``.

    Returns:
        None
    """
    num_generations = len(population_history)
    num_parameters = len(population_history[0][0]['parameters'])

    avg_parameters = np.zeros((num_generations, num_parameters))
    best_parameters = np.zeros((num_generations, num_parameters))

    for gen_idx, generation in enumerate(population_history):
        parameters_list = [candidate['parameters'] for candidate in generation]

        best_candidate = max(generation, key=lambda candidate: candidate['fitness'])
        avg_parameters[gen_idx, :] = np.mean(parameters_list, axis=0)
        best_parameters[gen_idx, :] = best_candidate['parameters']

    # Lay the per-parameter plots out on a roughly-square grid.
    cols = int(np.ceil(np.sqrt(num_parameters)))
    rows = int(np.ceil(num_parameters / cols))
    generations = range(num_generations)

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for param_idx in range(num_parameters):
        ax = axes_flat[param_idx]
        is_wheel = param_idx in WHEEL_JOINT_INDICES
        ax.plot(generations, avg_parameters[:, param_idx], label='Average')
        ax.plot(generations, best_parameters[:, param_idx], label='Best')
        title = f'Parameter {param_idx}'
        if is_wheel:
            title += ' (wheel)'
            ax.set_facecolor('#f3f8ff')
        ax.set_title(title)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Value')
        ax.grid(True)
        ax.legend(fontsize='small')

    # Hide any unused grid cells.
    for spare_idx in range(num_parameters, len(axes_flat)):
        axes_flat[spare_idx].axis('off')

    fig.suptitle('Parameter Evolution (wheel joints highlighted)')
    fig.tight_layout()
    plt.show()
