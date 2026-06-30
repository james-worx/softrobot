import time

import pybullet as p
import pybullet_data

from r2d2.evolutionary_algorithm.objective import (
    TARGET_POSITION,
    apply_wheel_velocities,
)
from r2d2.analysis.telemetry import TelemetryHUD, configure_visualizer


def visualize_best_solution(best_parameters, steps=6500, realtime=True):
    """Replay the best evolved solution with a live telemetry overlay.

    The robot is driven exactly as it was during training (the genome sets the
    wheel velocities), the default preview panels are hidden, and a HUD tracks
    distance travelled, speed, distance-to-target and fitness while the path is
    traced on the ground.

    Args:
        best_parameters (list): the evolved per-joint parameter vector.
        steps (int): number of simulation steps to run. Defaults to the same
            horizon used to evaluate fitness during training.
        realtime (bool): if True, sleep between steps so the replay plays at
            roughly wall-clock speed instead of as fast as possible.
    """
    # Training leaves the main process connected to DIRECT after selecting the
    # best candidate. A DIRECT connection cannot show the final replay, so
    # switch to GUI unless an existing GUI connection is already active.
    connection_info = p.getConnectionInfo()
    if (
        connection_info['isConnected']
        and connection_info.get('connectionMethod') != p.GUI
    ):
        p.disconnect()
        connection_info = p.getConnectionInfo()

    if connection_info['isConnected'] == 0:
        p.connect(p.GUI)

    configure_visualizer()

    # Reset the simulation
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    # Load the plane
    p.loadURDF("plane.urdf")

    # Set the robot's initial position above the plane
    start_position = [0, 0, 1]  # Adjust z value as needed
    start_orientation = p.getQuaternionFromEuler([0, 0, 0])

    # Load the robot
    robot_id = p.loadURDF("r2d2.urdf", start_position, start_orientation)

    # Drive the robot the same way the objective function does, so the replay
    # faithfully reproduces the behaviour that earned this genome its fitness.
    apply_wheel_velocities(robot_id, best_parameters)

    hud = TelemetryHUD(robot_id, TARGET_POSITION, start_position=start_position)

    # Decouple physics from rendering. Stepping the GUI one physics step at a
    # time means a render + HUD round-trip for every one of `steps` iterations,
    # which dominates the runtime (physics itself is a fraction of a second).
    # Instead, advance the physics in small batches with rendering disabled,
    # then re-enable rendering and refresh the HUD once per batch. This cuts the
    # replay from minutes back down to roughly real time while looking the same.
    steps_per_frame = 8
    hud.update()
    remaining = steps
    while remaining > 0:
        batch = min(steps_per_frame, remaining)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        for _ in range(batch):
            p.stepSimulation()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        hud.update()
        remaining -= batch
        if realtime:
            # Pace the whole batch at once so playback stays at wall-clock speed.
            time.sleep(batch / 240.)

    final = hud.update()
    print(
        "Replay finished | path travelled: {path:.2f} m | "
        "distance to target: {dist:.2f} m | fitness: {fit:.2f}".format(
            path=final["path_distance"],
            dist=final["distance_to_target"],
            fit=final["fitness"],
        )
    )

    # Disconnect from the physics server
    p.disconnect()
    return final

# Example usage:
# best_parameters = np.array([...])
# visualize_best_solution(best_parameters)
