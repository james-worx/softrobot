"""Unit tests for the telemetry metric maths.

``locomotion_metrics`` is a pure function (no PyBullet calls), so the figures
shown on the in-sim HUD can be verified headless in CI.
"""
import numpy as np

from r2d2.analysis.telemetry import locomotion_metrics


def test_metrics_at_start_are_zeroed():
    m = locomotion_metrics(
        position=[0, 0, 1],
        velocity=[0, 0, 0],
        start_position=[0, 0, 1],
        target_position=[0, 10, 1],
        path_distance=0.0,
    )
    assert m["displacement"] == 0.0
    assert m["speed"] == 0.0
    assert m["progress"] == 0.0
    assert m["distance_to_target"] == 10.0
    # Fitness mirrors the objective: negative distance to target.
    assert m["fitness"] == -10.0


def test_displacement_and_progress_are_horizontal():
    # Moved 3m forward and 4m sideways -> 5m horizontal displacement, but only
    # 3m of progress along the start->target (forward) axis. Vertical change is
    # ignored.
    m = locomotion_metrics(
        position=[4, 3, 5],
        velocity=[0, 0, 0],
        start_position=[0, 0, 1],
        target_position=[0, 10, 1],
        path_distance=7.0,
    )
    assert np.isclose(m["displacement"], 5.0)
    assert np.isclose(m["progress"], 3.0)
    assert m["path_distance"] == 7.0


def test_speed_uses_horizontal_velocity_only():
    m = locomotion_metrics(
        position=[0, 0, 1],
        velocity=[3, 4, 99],  # vertical component must not count
        start_position=[0, 0, 1],
        target_position=[0, 10, 1],
        path_distance=0.0,
    )
    assert np.isclose(m["speed"], 5.0)


def test_degenerate_target_gives_zero_progress():
    # Target coincident with start (no axis) must not divide by zero.
    m = locomotion_metrics(
        position=[1, 1, 1],
        velocity=[0, 0, 0],
        start_position=[0, 0, 0],
        target_position=[0, 0, 0],
        path_distance=0.0,
    )
    assert m["progress"] == 0.0
