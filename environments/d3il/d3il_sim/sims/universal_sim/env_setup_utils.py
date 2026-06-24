
import os

import numpy as np
import yaml


SHRINK_FACTOR = 2


def _ensure_planar_array(points_xy):
    points = np.asarray(points_xy, dtype=float)
    if points.ndim == 1:
        if points.shape[0] not in (2, 4):
            raise ValueError("Expected a point with shape (2,) or (4,)")
        points = points.reshape(1, -1)
    if points.ndim != 2 or points.shape[-1] not in (2, 4):
        raise ValueError("Expected points with shape (N, 2) or (N, 4)")
    return points


def _rotate_heading_sincos(points, delta_theta):
    if points.shape[1] != 4:
        return points

    heading = np.arctan2(points[:, 2], points[:, 3]) + delta_theta
    rotated = points.copy()
    rotated[:, 2] = np.sin(heading)
    rotated[:, 3] = np.cos(heading)
    return rotated


def shrink_size(values, shrink_factor=SHRINK_FACTOR):
    scale = 1.0 / float(shrink_factor)
    array = np.asarray(values, dtype=float)

    if array.ndim == 0:
        return array * scale

    if array.ndim == 1:
        if array.shape[0] in (2, 4):
            shrunk = array.copy()
            shrunk[:2] *= scale
            return shrunk
        return array * scale

    if array.ndim == 2 and array.shape[1] in (2, 4):
        shrunk = array.copy()
        shrunk[:, :2] *= scale
        return shrunk

    return array * scale


class EnvSetup:
    @staticmethod
    def planner_to_sim_xy(points_xy, rotate_90on_z=False):
        points = _ensure_planar_array(points_xy)
        if rotate_90on_z:
            # Rotate 90 degrees counterclockwise around Z-axis: (x, y) -> (-y, x)
            rotated = points.copy()
            rotated[:, 0] = -points[:, 1]
            rotated[:, 1] = points[:, 0]
            return _rotate_heading_sincos(rotated, np.pi / 2.0)
        return points

    @staticmethod
    def sim_to_planner_xy(points_xy, rotate_90on_z=False):
        points = _ensure_planar_array(points_xy)
        if rotate_90on_z:
            # Inverse of 90-degree counterclockwise rotation: (x, y) -> (y, -x)
            rotated = points.copy()
            rotated[:, 0] = points[:, 1]
            rotated[:, 1] = -points[:, 0]
            return _rotate_heading_sincos(rotated, -np.pi / 2.0)
        return points

    @staticmethod
    def translate_in_x(points_xy, offset_x):
        """Translate points in x direction by offset_x amount."""
        points = _ensure_planar_array(points_xy)
        translated = points.copy()
        translated[:, 0] = translated[:, 0] + offset_x
        return translated

    @staticmethod
    def shrink_size(values, shrink_factor=SHRINK_FACTOR):
        return shrink_size(values, shrink_factor=shrink_factor)

    @staticmethod
    def load_obs_centers(file_path, rotate_90on_z=False):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Obstacle layout YAML not found: {file_path}")

        with open(file_path, "r") as f:
            params_main = yaml.safe_load(f) or {}

        if "obstacles" not in params_main:
            raise KeyError(
                f"Missing 'obstacles' key in layout YAML: {file_path}. "
                "Expected format: obstacles: [[x, y, ...], ...]"
            )

        obstacles = np.array(params_main["obstacles"], dtype=float)
        print(f"Loaded obstacles from YAML: {obstacles.shape[0]} obstacles")

        obstacles = obstacles[:, :2]

        if rotate_90on_z:
            obstacles = EnvSetup.planner_to_sim_xy(obstacles, rotate_90on_z=True)
            print("Applied 90-degree Z-axis rotation to obstacles")

        return obstacles


def load_obs_centers(file_path, rotate_90on_z=False):
    return EnvSetup.load_obs_centers(file_path=file_path, rotate_90on_z=rotate_90on_z)


def planner_to_sim_xy(points_xy, rotate_90on_z=False):
    return EnvSetup.planner_to_sim_xy(points_xy=points_xy, rotate_90on_z=rotate_90on_z)


def sim_to_planner_xy(points_xy, rotate_90on_z=False):
    return EnvSetup.sim_to_planner_xy(points_xy=points_xy, rotate_90on_z=rotate_90on_z)


def translate_in_x(points_xy, offset_x):
    return EnvSetup.translate_in_x(points_xy=points_xy, offset_x=offset_x)


def shrink_size(values, shrink_factor=SHRINK_FACTOR):
    scale = 1.0 / float(shrink_factor)
    array = np.asarray(values, dtype=float)

    if array.ndim == 0:
        return array * scale

    if array.ndim == 1:
        if array.shape[0] in (2, 4):
            shrunk = array.copy()
            shrunk[:2] *= scale
            return shrunk
        return array * scale

    if array.ndim == 2 and array.shape[1] in (2, 4):
        shrunk = array.copy()
        shrunk[:, :2] *= scale
        return shrunk

    return array * scale