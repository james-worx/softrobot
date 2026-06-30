import numpy as np
import matplotlib.pyplot as plt

# Human-readable labels for the two differential-drive genes.
PARAMETER_LABELS = {0: "left wheels", 1: "right wheels"}


def analyze_parameters(population_history):
    """Plot how each parameter evolved, as a single gridded figure.

    With the differential-drive model the genome is just two values (left- and
    right-side wheel velocities), so this draws one labelled subplot per side in
    a single figure instead of a window per joint.

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
        ax.plot(generations, avg_parameters[:, param_idx], label='Average')
        ax.plot(generations, best_parameters[:, param_idx], label='Best')
        label = PARAMETER_LABELS.get(param_idx)
        title = f'Parameter {param_idx}'
        if label:
            title += f' ({label})'
        ax.set_title(title)
        ax.set_xlabel('Generation')
        ax.set_ylabel('Target velocity')
        ax.grid(True)
        ax.legend(fontsize='small')

    # Hide any unused grid cells.
    for spare_idx in range(num_parameters, len(axes_flat)):
        axes_flat[spare_idx].axis('off')

    fig.suptitle('Parameter Evolution (differential-drive wheel velocities)')
    fig.tight_layout()
    plt.show()
