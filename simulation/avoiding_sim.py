import logging
import multiprocessing as mp
import os
import random
from envs.gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv

import numpy as np
import torch
import wandb
from environments.d3il.d3il_sim.sims.universal_sim.env_setup_utils import EnvSetup

from simulation.base_sim import BaseSim

log = logging.getLogger(__name__)

# Example smoothed trajectory for direct trajectory following
EXAMPLE_TRAJECTORY_XY = np.array([
    [-0.29119676, -0.27565762],
    [-0.29114598, -0.27563619],
    [-0.29080054, -0.27549205],
    [-0.28991558, -0.27513126],
    [-0.28830393, -0.27449233],
    [-0.2858328, -0.27354314],
    [-0.28241975, -0.27227735],
    [-0.27802773, -0.27071028],
    [-0.27265944, -0.26887465],
    [-0.26635183, -0.26681646],
    [-0.25917104, -0.2645914],
    [-0.25120672, -0.2622612],
    [-0.24256643, -0.25989014],
    [-0.2333701, -0.25754198],
    [-0.22374479, -0.25527714],
    [-0.21381966, -0.2531503],
    [-0.2037213, -0.25120826],
    [-0.19356943, -0.24948815],
    [-0.18347313, -0.24801593],
    [-0.17352763, -0.24680534],
    [-0.16381164, -0.2458575],
    [-0.15438517, -0.2451611],
    [-0.1452882, -0.2446932],
    [-0.13654001, -0.24442117],
    [-0.12813925, -0.24430587],
    [-0.12006498, -0.24430645],
    [-0.11227881, -0.24438676],
    [-0.10472873, -0.24452311],
    [-0.09735479, -0.2447126],
    [-0.09009668, -0.24498116],
    [-0.08290179, -0.2453891],
    [-0.07573256, -0.24603238],
    [-0.0685715, -0.24703752],
    [-0.06142298, -0.24855217],
    [-0.05431214, -0.25073389],
    [-0.04728136, -0.25373723],
    [-0.04038523, -0.25769866],
    [-0.03368466, -0.26272032],
    [-0.02724072, -0.26885392],
    [-0.02110891, -0.2760865],
    [-0.01533418, -0.28433077],
    [-0.00994662, -0.29342255],
    [-0.00495761, -0.30312771],
    [-0.00035726, -0.31315837],
    [0.00388558, -0.32319587],
    [0.00782161, -0.33291857],
    [0.01152, -0.34203216],
    [0.01506823, -0.35029815],
    [0.01857208, -0.3575556],
    [0.02215292, -0.36373172],
    [0.02594098, -0.36883966],
    [0.03006551, -0.37296705],
    [0.03464282, -0.37626098],
    [0.03976324, -0.37891147],
    [0.04547875, -0.38113481],
    [0.05179476, -0.3831593],
    [0.058665, -0.38521268],
    [0.06599006, -0.38750977],
    [0.07362056, -0.3902416],
    [0.08136625, -0.39356648],
    [0.08901113, -0.39760319],
    [0.09633298, -0.40242711],
    [0.10312517, -0.40806998],
    [0.10921754, -0.41452432],
    [0.11449214, -0.42175109],
    [0.11888959, -0.42968711],
    [0.12240598, -0.43825067],
    [0.12508494, -0.4473466],
    [0.12700808, -0.45687206],
    [0.12828534, -0.46672339],
    [0.12904532, -0.47680198],
    [0.12942677, -0.48701913],
    [0.12957166, -0.49729899],
    [0.12961981, -0.50757896],
    [0.12970475, -0.51780744],
    [0.12994996, -0.52794027],
    [0.13046313, -0.53793668],
    [0.13132988, -0.54775656],
    [0.13260987, -0.55735951],
    [0.13433604, -0.56670624],
    [0.13651644, -0.57576183],
    [0.1391377, -0.58449964],
    [0.1421693, -0.59290465],
    [0.14556773, -0.60097495],
    [0.14927995, -0.6087207],
    [0.15324601, -0.61616064],
    [0.15740097, -0.62331704],
    [0.161676, -0.63021018],
    [0.16599946, -0.63685327],
    [0.17029829, -0.64324851],
    [0.17449977, -0.64938476],
    [0.17853351, -0.6552368],
    [0.18233345, -0.66076602],
    [0.18583982, -0.66592249],
    [0.18900109, -0.67064839],
    [0.1917758, -0.67488273],
    [0.19413433, -0.67856703],
    [0.19606076, -0.68165222],
    [0.1975549, -0.68410663],
    [0.19863445, -0.68592505],
    [0.19933707, -0.68713747],
    [0.19972219, -0.68781714],
    [0.19987246, -0.68808764],
    [0.19989483, -0.68812848],
])


def assign_process_to_cpu(pid, cpus):
    os.sched_setaffinity(pid, cpus)


def interpolate_xy_trajectory(trajectory_xy, total_duration, dt):
    """Interpolate 2D knot points onto a uniform time grid."""
    knot_times = np.linspace(0.0, total_duration, trajectory_xy.shape[0])
    query_times = np.arange(0.0, total_duration, dt)

    x = np.interp(query_times, knot_times, trajectory_xy[:, 0])
    y = np.interp(query_times, knot_times, trajectory_xy[:, 1])

    return np.column_stack((x, y))


class Avoiding_Sim(BaseSim):
    # Slowdown factor: execution dt = physics dt * slowdown_factor
    # physics dt = 0.001s, so slowdown_factor=20 means trajectory dt = 0.020s
    slowdown_factor = 80

    def __init__(
            self,
            seed: int,
            device: str,
            render: bool,
            n_cores: int = 1,
            n_trajectories: int = 30,
            rotate_planner_frame_90on_z: bool = True,
    ):
        super().__init__(seed, device, render, n_cores)

        self.n_trajectories = n_trajectories
        self.rotate_planner_frame_90on_z = rotate_planner_frame_90on_z

    def eval_agent(self, agent, n_trajectories, mode_encoding, successes, robot_c_pos, pid, cpu_set):

        print(os.getpid(), cpu_set)
        assign_process_to_cpu(os.getpid(), cpu_set)

        # Prepare trajectory once (for both visualization and execution)
        trajectory_xy = EXAMPLE_TRAJECTORY_XY.copy()
        
        # Apply planner-to-sim frame rotation
        if self.rotate_planner_frame_90on_z:
            trajectory_xy = EnvSetup.planner_to_sim_xy(
                trajectory_xy,
                rotate_90on_z=True,
            )
        
        # Translate trajectory in x by 0.2 meters (after rotation, in sim frame)
        trajectory_xy = EnvSetup.translate_in_x(trajectory_xy, offset_x=0.1)
        
        initial_cart_position = np.array([
            trajectory_xy[0, 0],
            trajectory_xy[0, 1],
            0.12,
        ])

        env = ObstacleAvoidanceEnv(
            render=self.render,
            trajectory_xy=trajectory_xy,
            initial_cart_position=initial_cart_position,
        )
        env.start()

        random.seed(pid)
        torch.manual_seed(pid)
        np.random.seed(pid)

        for i in range(n_trajectories):

            agent.reset()

            print(f'core {pid}, Rollout {i}')

            obs = env.reset()

            fixed_quat = np.array([0, 1, 0, 0])

            # Treat the 104 points as knots of a time-parameterized trajectory.
            # The controller still runs at 1 ms, but we query the interpolated target at each step.
            total_duration = trajectory_xy.shape[0] * self.slowdown_factor * env.robot.dt
            desired_xy = interpolate_xy_trajectory(
                trajectory_xy,
                total_duration=total_duration,
                dt=env.robot.dt,
            )
            desired_pos = np.column_stack(
                (
                    desired_xy[:, 0],
                    desired_xy[:, 1],
                    np.full(desired_xy.shape[0], 0.12),
                )
            )
            desired_quat = np.repeat(fixed_quat[None, :], desired_pos.shape[0], axis=0)

            c_pos = [env.robot.current_c_pos]

            # Use the robot's full trajectory tracker, but run non-blocking so we can
            # keep per-step collision/success checks during execution.
            env.robot.follow_CartPositionAndQuatTraj(
                desiredPos=desired_pos,
                desiredQuat=desired_quat,
                goto_start=False,
                global_coord=True,
                block=False,
            )

            done = False
            tracker = env.robot.cartPosQuatTrajectoryTracker
            while not tracker.isFinished(env.robot):
                env.robot.nextStep()
                c_pos.append(env.robot.current_c_pos)

                env.check_mode()
                done = env._check_early_termination()
                if done:
                    break

            if not done:
                env.check_mode()
                env._check_early_termination()

            info = (env.mode_encoding, env.success)

            c_pos = torch.tensor(np.array(c_pos))[:, :2]
            robot_c_pos[pid * n_trajectories + i, :c_pos.shape[0], :] = c_pos

            mode_encoding[pid * n_trajectories + i, :] = torch.tensor(info[0])
            successes[pid * n_trajectories + i] = torch.tensor(info[1])

    ################################
    # we use multi-process for the simulation
    # n_trajectories: rollout policy for n times
    # n_cores: the number of cores used for simulation
    ###############################
    def test_agent(self, agent):

        log.info('Starting trained model evaluation')

        robot_c_pos = torch.zeros([self.n_trajectories, 150, 2]).share_memory_()

        mode_encoding = torch.zeros([self.n_trajectories, 9]).share_memory_()
        successes = torch.zeros(self.n_trajectories).share_memory_()

        num_cpu = mp.cpu_count()
        cpu_set = list(range(num_cpu))

        # start = self.seed * 20
        # end = start + 20
        #
        # cpu_set = cpu_set[start:end]
        print("there are cpus: ", num_cpu)

        ctx = mp.get_context('spawn')

        p_list = []
        if self.n_cores > 1:
            for i in range(self.n_cores):
                p = ctx.Process(
                    target=self.eval_agent,
                    kwargs={
                        "agent": agent,
                        "n_trajectories": self.n_trajectories // self.n_cores,
                        "mode_encoding": mode_encoding,
                        "successes": successes,
                        "robot_c_pos": robot_c_pos,
                        "pid": i,
                        "cpu_set": set(cpu_set[i:i + 1])
                    },
                )
                print("Start {}".format(i))
                p.start()
                p_list.append(p)
            [p.join() for p in p_list]

        else:
            self.eval_agent(agent, self.n_trajectories, mode_encoding, successes, robot_c_pos, 0, set([0]))
            
        # TODO: save robot_c_pos

        success_rate = torch.mean(successes).item()

        # calculate entropy
        data = mode_encoding[successes == 1].numpy()
        data_decimal = data.dot(1 << np.arange(data.shape[-1]))
        _, counts = np.unique(data_decimal, return_counts=True)
        mode_dist = counts / np.sum(counts)
        entropy = - np.sum(mode_dist * (np.log(mode_dist) / np.log(24)))

        wandb.log({'score': (success_rate * 0.8 + entropy * 0.2)})
        wandb.log({'Metrics/successes': success_rate})
        wandb.log({'Metrics/entropy': entropy})

        print(f'Successrate {success_rate}')
        print(f'entropy {entropy}')

        return successes, entropy