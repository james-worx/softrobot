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

    # Trail tuning. Each visible segment is a PyBullet debug line; the GUI
    # re-syncs every live debug item on each add, so an unbounded trail makes
    # the work per update grow without limit (O(n^2) over a replay) and is what
    # turned a ~27 s episode into a 10+ minute animation. We therefore (a) only
    # start a new segment after the robot has travelled a minimum distance, and
    # (b) cap the number of live segments, reusing the oldest segment's id in a
    # ring once the cap is hit. This keeps per-update cost flat.
    _TRAIL_MIN_SEGMENT = 0.1   # metres of travel before drawing a new segment
    _TRAIL_MAX_SEGMENTS = 256  # hard cap on simultaneously live trail lines

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
        self._trail_ids = []  # live trail-segment debug-item ids (ring buffer)
        self._trail_cursor = 0  # next ring slot to overwrite once at capacity

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
        """Extend the ground trail, keeping the live segment count bounded."""
        anchor = self._trail_anchor.copy()
        anchor[2] = 0.05
        here = position.copy()
        here[2] = 0.05
        # Only draw once the robot has travelled a meaningful distance. This
        # both skips zero-length segments (which PyBullet rejects) and keeps
        # the total segment count low for a normal-length path.
        if np.linalg.norm(here[:2] - anchor[:2]) < self._TRAIL_MIN_SEGMENT:
            return

        if len(self._trail_ids) < self._TRAIL_MAX_SEGMENTS:
            self._trail_ids.append(
                p.addUserDebugLine(anchor.tolist(), here.tolist(),
                                   lineColorRGB=list(self._TRAIL_COLOR),
                                   lineWidth=2.0))
        else:
            # At capacity: reuse the oldest segment's id (replaceItemUniqueId)
            # so the number of live debug lines never grows. The visible trail
            # becomes the most recent _TRAIL_MAX_SEGMENTS segments.
            reuse_id = self._trail_ids[self._trail_cursor]
            self._trail_ids[self._trail_cursor] = p.addUserDebugLine(
                anchor.tolist(), here.tolist(),
                lineColorRGB=list(self._TRAIL_COLOR), lineWidth=2.0,
                replaceItemUniqueId=reuse_id)
            self._trail_cursor = (self._trail_cursor + 1) % self._TRAIL_MAX_SEGMENTS

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
