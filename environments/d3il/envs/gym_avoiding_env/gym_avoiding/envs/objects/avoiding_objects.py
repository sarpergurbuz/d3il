import numpy as np

from environments.d3il.d3il_sim.sims.universal_sim.PrimitiveObjects import Box, Cylinder
from environments.d3il.d3il_sim.sims.universal_sim.CompoundObjects import CylinderCompound
from environments.d3il.d3il_sim.sims.universal_sim.env_setup_utils import EnvSetup, shrink_size, SHRINK_FACTOR
from agents.utils.sim_path import sim_framework_path
init_end_eff_pos = [0.47565762, -0.29119676, 0.12]


_MID_POS = 0.5
_OFFSET = 0.075
_FIRST_LEVEL_Y = -0.1
_LEVEL_DISTANCE = 0.18

_OBSTACLE_NAMES = [
    "l1_obs",
    "l2_top_obs",
    "l2_bottom_obs",
    "l3_top_obs",
    "l3_mid_obs",
    "l3_bottom_obs",
]

_OBSTACLE_CENTERS = [
    [_MID_POS, _FIRST_LEVEL_Y],
    [_MID_POS - _OFFSET, _FIRST_LEVEL_Y + _LEVEL_DISTANCE],
    [_MID_POS + _OFFSET, _FIRST_LEVEL_Y + _LEVEL_DISTANCE],
    [_MID_POS - 2 * _OFFSET, _FIRST_LEVEL_Y + 2 * _LEVEL_DISTANCE],
    [_MID_POS, _FIRST_LEVEL_Y + 2 * _LEVEL_DISTANCE],
    [_MID_POS + 2 * _OFFSET, _FIRST_LEVEL_Y + 2 * _LEVEL_DISTANCE],
]

_OBSTACLE_RADII = [0.03, 0.025, 0.025, 0.025, 0.025, 0.025]
_OBSTACLE_HEIGHTS = [0.07, 0.1, 0.1, 0.1, 0.1, 0.1]
_OBSTACLE_BASE_NAME = "maze_obs"

GOAL_SAFE_RADIUS = 0.03 / SHRINK_FACTOR
DEFAULT_GOAL_XY = [0.4, _FIRST_LEVEL_Y + 2.5 * _LEVEL_DISTANCE]

obs_centers_maze = EnvSetup.load_obs_centers(
    file_path=sim_framework_path(
        "environments/d3il/d3il_sim/sims/universal_sim/test_envs/layout_6.yaml"
    ),
    rotate_90on_z=True,
)

# Translate maze in x direction by 0.2 meters
obs_centers_maze = shrink_size(obs_centers_maze, shrink_factor=SHRINK_FACTOR)
obs_centers_maze = EnvSetup.translate_in_x(obs_centers_maze, offset_x=0.3)


def get_finish_point(goal_xy=None, safe_radius=GOAL_SAFE_RADIUS):
    if goal_xy is None:
        goal_xy = DEFAULT_GOAL_XY
    goal_xy = np.asarray(goal_xy, dtype=float)

    return Cylinder(
        name='finish_point',
        init_pos=[goal_xy[0], goal_xy[1], 0],
        init_quat=[1, 0, 0, 0],
        size=[safe_radius, 0.005],
        rgba=[0., 1., 0., 0.3],
        visual_only=True,
        static=True,
    )


def get_obj_list():
    obstacle_names = get_obstacle_names()
    obstacle_compound = CylinderCompound(
        centers=obs_centers_maze,
        names=obstacle_names,
        radius=shrink_size(0.03, shrink_factor=SHRINK_FACTOR),
        height=0.1,
        rgba=[1, 0, 0, 1],
        static=True,
    )

    return obstacle_compound.to_objects()


def get_obj_xy_list():
    return obs_centers_maze


def get_obstacle_names():
    return [f"{_OBSTACLE_BASE_NAME}_{i}" for i in range(len(obs_centers_maze))]
