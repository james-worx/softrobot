"""Live in-simulation telemetry HUD for the R2D2 evolutionary-algorithm sim.

PyBullet's GUI shows three preview-buffer panels by default (the RGB, depth
and segmentation-mask tiles). They carry no useful information for a
locomotion task, so this module hides them and overlays metrics that actually
matter for the EA instead:

* horizontal distance travelled along the ground (path length),
* straight-line displacement from the start,
* current horizontal speed,
* distance remaining to the target, and
* the resulting fitness (negative distance to target).

The numbers are drawn as text anchored above the robot so they follow it
around the arena, and the path taken is traced on the ground.
"""

import numpy as np
import pybullet as p


def configure_visualizer():
    """Hide PyBullet's default preview panels so the arena is uncluttered.

    These are the three image tiles (RGB / depth / segmentation) shown by the
    GUI client out of the box. They are replaced by the telemetry overlay.
    """
    p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)


def locomotion_metrics(position, velocity, start_position, target_position,
                       path_distance):
    """Compute the locomotion metrics shown on the HUD.

    Pure function (no PyBullet calls) so it can be unit-tested directly.

    Args:
        position: current robot base position ``(x, y, z)``.
        velocity: current robot base linear velocity ``(vx, vy, vz)``.
        start_position: where the robot started ``(x, y, z)``.
        target_position: the goal the robot is driving toward ``(x, y, z)``.
        path_distance: accumulated ground-path length travelled so far (m).

    Returns:
        dict with ``displacement``, ``path_distance``, ``speed``,
        ``distance_to_target``, ``progress`` and ``fitness`` (all floats).
        ``progress`` is the signed advance toward the target measured along
        the horizontal start->target axis.
    """
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    start = np.asarray(start_position, dtype=float)
    target = np.asarray(target_position, dtype=float)

    displacement = float(np.linalg.norm(position[:2] - start[:2]))
    speed = float(np.linalg.norm(velocity[:2]))
    distance_to_target = float(np.linalg.norm(target - position))

    # Progress: how far the robot has advanced along the line from its start
    # toward the target, projected onto the horizontal plane.
    axis = target[:2] - start[:2]
    axis_length = np.linalg.norm(axis)
    if axis_length > 0:
        progress = float(np.dot(position[:2] - start[:2], axis) / axis_length)
    else:
        progress = 0.0

    return {
        "displacement": displacement,
        "path_distance": float(path_distance),
        "speed": speed,
        "distance_to_target": distance_to_target,
        "progress": progress,
        # Mirror the objective function: fitness is the negative distance to
        # the target (closer is better, so less-negative is better).
        "fitness": -distance_to_target,
    }


class TelemetryHUD:
    """Tracks locomotion metrics and renders them live during a GUI replay.

    The HUD keeps the debug-item ids it creates and updates them in place via
    ``replaceItemUniqueId`` so the text refreshes instead of spawning a new
    overlay every frame.
    """

    # HUD text colour and the local-frame heights of each stacked line.
    _TEXT_COLOR = (0.1, 0.1, 0.1)
    _TRAIL_COLOR = (0.0, 0.4, 1.0)

    def __init__(self, robot_id, target_position, start_position=None,
                 follow_camera=True):
        self.robot_id = robot_id
        self.target_position = np.asarray(target_position, dtype=float)
        self.follow_camera = follow_camera

        if start_position is None:
            start_position = p.getBasePositionAndOrientation(robot_id)[0]
        self.start_position = np.asarray(start_position, dtype=float)

        self.path_distance = 0.0
        self.max_speed = 0.0
        self._prev_xy = self.start_position[:2].copy()
        self._trail_anchor = self.start_position.copy()
        self._text_ids = []  # one debug-text id per HUD line

        self._draw_target_marker()

    def _draw_target_marker(self):
        """Drop a green sphere at the target and a guide line from the start."""
        visual = p.createVisualShape(p.GEOM_SPHERE, radius=0.25,
                                     rgbaColor=[0, 1, 0, 1])
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual,
                          basePosition=self.target_position.tolist())
        p.addUserDebugLine(self.start_position.tolist(),
                           self.target_position.tolist(),
                           lineColorRGB=[0, 0.6, 0], lineWidth=1.0)

    def update(self):
        """Sample the simulation, refresh the overlay, and return the metrics."""
        position = np.asarray(
            p.getBasePositionAndOrientation(self.robot_id)[0], dtype=float)
        velocity = np.asarray(p.getBaseVelocity(self.robot_id)[0], dtype=float)

        # Accumulate ground-path length from the previous sample.
        self.path_distance += float(np.linalg.norm(position[:2] - self._prev_xy))
        self._prev_xy = position[:2].copy()

        metrics = locomotion_metrics(position, velocity, self.start_position,
                                     self.target_position, self.path_distance)
        self.max_speed = max(self.max_speed, metrics["speed"])

        self._draw_trail(position)
        self._draw_text(metrics)

        if self.follow_camera:
            p.resetDebugVisualizerCamera(
                cameraDistance=5.0, cameraYaw=50, cameraPitch=-35,
                cameraTargetPosition=position.tolist())

        return metrics

    def _draw_trail(self, position):
        """Extend the ground trail from the last anchor to the current spot."""
        anchor = self._trail_anchor.copy()
        anchor[2] = 0.05
        here = position.copy()
        here[2] = 0.05
        # Only draw once the robot has actually moved, to avoid zero-length
        # segments that PyBullet rejects.
        if np.linalg.norm(here[:2] - anchor[:2]) > 1e-3:
            p.addUserDebugLine(anchor.tolist(), here.tolist(),
                               lineColorRGB=list(self._TRAIL_COLOR),
                               lineWidth=2.0)
            self._trail_anchor = position.copy()

    def _draw_text(self, metrics):
        """Render (or refresh) the stacked HUD lines above the robot."""
        lines = [
            f"path travelled : {metrics['path_distance']:6.2f} m",
            f"displacement   : {metrics['displacement']:6.2f} m",
            f"speed          : {metrics['speed']:6.2f} m/s  (max {self.max_speed:5.2f})",
            f"to target      : {metrics['distance_to_target']:6.2f} m",
            f"fitness        : {metrics['fitness']:6.2f}",
        ]
        creating = not self._text_ids
        for i, line in enumerate(lines):
            local_pos = [0, 0, 1.5 - i * 0.22]
            kwargs = dict(
                textColorRGB=list(self._TEXT_COLOR),
                textSize=1.2,
                parentObjectUniqueId=self.robot_id,
            )
            if creating:
                self._text_ids.append(
                    p.addUserDebugText(line, local_pos, **kwargs))
            else:
                p.addUserDebugText(line, local_pos,
                                   replaceItemUniqueId=self._text_ids[i],
                                   **kwargs)
