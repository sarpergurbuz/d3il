import numpy as np
import copy
from environments.d3il.d3il_sim.sims.mj_beta.mj_utils.mj_helper import has_collision
from environments.d3il.d3il_sim.utils.sim_path import d3il_path

from environments.d3il.d3il_sim.core import Scene
from environments.d3il.d3il_sim.gyms.gym_env_wrapper import GymEnvWrapper
from environments.d3il.d3il_sim.core.logger import ObjectLogger, CamLogger
from environments.d3il.d3il_sim.sims.mj_beta.MjRobot import MjRobot
from environments.d3il.d3il_sim.sims.mj_beta.MjFactory import MjFactory
from environments.d3il.d3il_sim.sims import MjCamera
from environments.d3il.d3il_sim.sims.universal_sim.PrimitiveObjects import Box

from .objects.avoiding_objects import get_obj_list, \
    init_end_eff_pos, \
    get_obj_xy_list, \
    get_obstacle_names, \
    get_finish_point, \
    GOAL_SAFE_RADIUS

obj_list = get_obj_list()


def create_trajectory_markers(trajectory_xy, fixed_z=0.12, marker_radius=0.005):
    """
    Create small sphere markers for trajectory visualization.
    
    Args:
        trajectory_xy: (N, 2) array of [x, y] waypoints
        fixed_z: z-position for all markers
        marker_radius: radius of small spheres
    
    Returns:
        List of Box objects (used as sphere markers, visual_only)
    """
    markers = []
    trajectory_xy = np.asarray(trajectory_xy)
    if trajectory_xy.ndim != 2 or trajectory_xy.shape[1] < 2:
        raise ValueError("trajectory_xy must have shape (N, 2) or (N, 4)")

    trajectory_xy = trajectory_xy[:, :2]
    for i, (x, y) in enumerate(trajectory_xy):
        marker = Box(
            name=f'traj_marker_{i}',
            init_pos=[x, y, fixed_z],
            init_quat=[1, 0, 0, 0],
            size=[marker_radius, marker_radius, marker_radius],
            rgba=[0, 1, 1, 0.6],  # Cyan color
            visual_only=True,
            static=True
        )
        markers.append(marker)
    return markers


class BPCageCam(MjCamera):
    """
    Cage camera. Extends the camera base class.
    """

    def __init__(self, width: int = 1024, height: int = 1024, *args, **kwargs):
        super().__init__(
            "bp_cam",
            width,
            height,
            init_pos=[1.05, 0, 1.2],
            init_quat=[
                0.6830127,
                0.1830127,
                0.1830127,
                0.683012,
            ],  # Looking with 30 deg to the robot
            *args,
            **kwargs,
        )



class ObstacleAvoidanceManager:
    def __init__(self):
        self.index = 0
        pass

    def start(self):
        pass


class ObstacleAvoidanceEnv(GymEnvWrapper):
    def __init__(
            self,
            n_substeps: int = 35,
            max_steps_per_episode: int = 250,
            debug: bool = False,
            render: bool = False,
            trajectory_xy: np.ndarray = None,
    initial_cart_position: np.ndarray = None,
    ):

        sim_factory = MjFactory()
        render_mode = Scene.RenderMode.HUMAN if render else Scene.RenderMode.BLIND
        scene = sim_factory.create_scene(
            object_list=obj_list, render=render_mode, dt=0.001
        )
        robot = MjRobot(
            scene,
            xml_path=d3il_path("./models/mj/robot/panda_rod_invisible.xml")
        )
        controller = robot.cartesianPosQuatTrackingController

        super().__init__(
            scene=scene,
            controller=controller,
            max_steps_per_episode=max_steps_per_episode,
            n_substeps=n_substeps,
            debug=debug,
        )

        self.manager = ObstacleAvoidanceManager()

        if initial_cart_position is None:
            self.initial_cart_position = copy.deepcopy(init_end_eff_pos)
        else:
            self.initial_cart_position = np.asarray(initial_cart_position, dtype=float).copy()

        self.bp_cam = BPCageCam()

        self.scene.add_object(self.bp_cam)

        self.log_dict = {}
        self.cam_dict = {"bp-cam": CamLogger(scene, self.bp_cam)}

        for _, v in self.cam_dict.items():
            scene.add_logger(v)

        self.obj_xy_list = get_obj_xy_list()
        self.obstacle_names = get_obstacle_names()

        if trajectory_xy is not None:
            trajectory_xy = np.asarray(trajectory_xy, dtype=float)
            self.goal_xy = trajectory_xy[-1, :2].copy()
        else:
            self.goal_xy = np.array([0.4, -0.1 + 2.5 * 0.18], dtype=float)

        self.finish_point = get_finish_point(self.goal_xy, safe_radius=GOAL_SAFE_RADIUS)
        self.scene.add_object(self.finish_point)

        # Add trajectory markers if provided
        if trajectory_xy is not None:
            trajectory_xy = np.asarray(trajectory_xy)
            init_z = self.initial_cart_position[2]  # Use same Z as initial end effector
            trajectory_markers = create_trajectory_markers(trajectory_xy, fixed_z=init_z)
            #for marker in trajectory_markers:
                #self.scene.add_object(marker)
            #print(f"Added {len(trajectory_markers)} trajectory markers to scene")

        self.target_min_dist = GOAL_SAFE_RADIUS

        level_distance = 0.18
        obstacle_offset = 0.075
        self.l1_ypos = -0.1
        self.l2_ypos = -0.1 + level_distance
        self.l3_ypos = -0.1 + 2 * level_distance
        self.goal_ypos = -0.1 + 2.5 * level_distance
        self.l1_xpos = 0.5
        self.l2_top_xpos = 0.5 - obstacle_offset
        self.l2_bottom_xpos = 0.5 + obstacle_offset
        self.l3_top_xpos = 0.5 - 2 * obstacle_offset
        self.l3_mid_xpos = 0.5
        self.l3_bottom_xpos = 0.5 + 2 * obstacle_offset

        self.l1_passed = False
        self.l2_passed = False
        self.l3_passed = False

        self.mode_encoding = np.zeros(2 + 3 + 4)

        self.success = False

    def get_observation(self) -> np.ndarray:
        robot_c_pos = self.robot_state()[:2]
        return robot_c_pos.astype(np.float32)

    def start(self):
        self.scene.start()

        # reset view of the camera
        try:
            self.scene.viewer.cam.elevation = -55
            self.scene.viewer.cam.distance = 1.8
            self.scene.viewer.cam.lookat[0] += -0.1
            self.scene.viewer.cam.lookat[2] -= 0.2

            # self.scene.viewer.cam.elevation = -55
            # self.scene.viewer.cam.distance = 2.0
            # self.scene.viewer.cam.lookat[0] += 0
            # self.scene.viewer.cam.lookat[2] -= 0.2
            # self.scene.viewer.cam.elevation = -60
            # self.scene.viewer.cam.distance = 1.6
            # self.scene.viewer.cam.lookat[0] += 0.1
            # self.scene.viewer.cam.lookat[2] -= 0.1
        except:
            pass

        # reset the initial state of the robot
        initial_cart_position = copy.deepcopy(self.initial_cart_position)
        # initial_cart_position[2] = 0.12
        self.robot.gotoCartPosQuatController.setDesiredPos(
            [
                initial_cart_position[0],
                initial_cart_position[1],
                initial_cart_position[2],
                0,
                1,
                0,
                0,
            ]
        )
        self.robot.gotoCartPosQuatController.initController(self.robot, 1)

        self.robot.init_qpos = self.robot.gotoCartPosQuatController.trajectory[-1].copy()
        self.robot.init_tcp_pos = initial_cart_position
        self.robot.init_tcp_quat = [0, 1, 0, 0]

        self.robot.beam_to_joint_pos(self.robot.gotoCartPosQuatController.trajectory[-1])

        self.robot.gotoCartPositionAndQuat(
            desiredPos=initial_cart_position, desiredQuat=[0, 1, 0, 0], duration=0.5
        )

    def step(self, action, gripper_width=None):
        observation, reward, done, _ = super().step(action, gripper_width)
        self.check_mode()
        return observation, reward, done, (self.mode_encoding, self.success)

    def check_mode(self):
        r_x_pos = self.robot.current_c_pos[0]
        r_y_pos = self.robot.current_c_pos[1]
        if r_y_pos - 0.03 <= self.l1_ypos <= r_y_pos + 0.03 and (not self.l1_passed):
            if r_x_pos < self.l1_xpos:
                self.mode_encoding[0] = 1
            elif r_x_pos > self.l1_xpos:
                self.mode_encoding[1] = 1
            self.l1_passed = True

        if r_y_pos - 0.03 <= self.l2_ypos <= r_y_pos + 0.03 and (not self.l2_passed):
            if r_x_pos < self.l2_top_xpos:
                self.mode_encoding[2] = 1
            elif self.l2_top_xpos < r_x_pos < self.l2_bottom_xpos:
                self.mode_encoding[3] = 1
            elif r_x_pos > self.l2_bottom_xpos:
                self.mode_encoding[4] = 1
            self.l2_passed = True

        # if r_y_pos - 0.015 <= self.l3_ypos and (not self.l3_passed):
        if r_y_pos >= self.l3_ypos and (not self.l3_passed):
            if r_x_pos < self.l3_top_xpos:
                self.mode_encoding[5] = 1
            if self.l3_top_xpos < r_x_pos < self.l3_mid_xpos:
                self.mode_encoding[6] = 1
            elif self.l3_mid_xpos < r_x_pos < self.l3_bottom_xpos:
                self.mode_encoding[7] = 1
            elif r_x_pos > self.l3_top_xpos:
                self.mode_encoding[8] = 1
            self.l3_passed = True

    def check_failure(self):
        for obstacle_name in self.obstacle_names:
            if has_collision(obstacle_name, 'rod', self.scene.model, self.scene.data):
                return True
        return False

    def check_success(self):
        goal_pos = np.asarray(self.goal_xy, dtype=float)
        current_pos = np.asarray(self.robot.current_c_pos[:2], dtype=float)
        return np.linalg.norm(current_pos - goal_pos) <= self.target_min_dist

    def reset_mode_encoding(self):
        self.l1_passed = False
        self.l2_passed = False
        self.l3_passed = False
        assert np.sum(self.mode_encoding) <= 3
        self.mode_encoding = np.zeros(2 + 3 + 4)

    def get_reward(self):
        ...

    def _check_early_termination(self) -> bool:

        # print(self.check_failure())

        if self.check_success() or self.check_failure():
            if self.check_success():
                self.success = True
            self.terminated = True
            return True

        return False

    def reset(self, random=True, context=None):
        self.terminated = False
        self.env_step_counter = 0
        self.episode += 1
        self.reset_mode_encoding()
        self.success = False
        obs = self._reset_env(random=random, context=context)
        return obs

    def _reset_env(self, random=True, context=None):
        self.scene.reset()
        self.robot.beam_to_joint_pos(self.robot.init_qpos)
        self.scene.next_step()
        observation = self.get_observation()
        return observation

    def reward(self, x):
        def squared_exp_kernel(x, mean, scale, bandwidth):
            return scale * np.exp(np.square(np.linalg.norm(x - mean, axis=1)) / bandwidth)

        rewards = np.zeros(x.shape[0])
        for obs in self.obj_xy_list:
            rewards -= squared_exp_kernel(x, np.array(obs), 1, 1)
        # rewards += np.abs(x[:, 1]- 0.4)
        rewards -= np.abs(x[:, 0] - 0.4)
        return rewards

    def mode_decoding(self, data):
        data_decimal = data.dot(1 << np.arange(data.shape[-1]))
        _, counts = np.unique(data_decimal, return_counts=True)
        mode_dist = counts / np.sum(counts)
        entropy = - np.sum(mode_dist * (np.log(mode_dist) / np.log(24)))
        return counts, entropy


    def action_space(self):
        ...
