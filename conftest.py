"""Pytest configuration.

Make the ``r2d2`` package importable when the project has not been
installed (e.g. ``pytest`` run straight from a clone) by adding the
``pybullet`` source directory to ``sys.path``.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pybullet"))
