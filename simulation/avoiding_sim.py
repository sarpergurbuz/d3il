import logging
import multiprocessing as mp
import os
import random
from envs.gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv

import numpy as np
import torch
import wandb
from environments.d3il.d3il_sim.sims.universal_sim.env_setup_utils import EnvSetup, shrink_size, SHRINK_FACTOR

from simulation.base_sim import BaseSim
import sys

# Ensure repo root is on sys.path so package imports like `flow_matcher` work
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

log = logging.getLogger(__name__)

# fixed trajectory speed (m/s) used to compute total duration from curve length
TRAJ_SPEED = 0.05

# Example smoothed trajectory for direct trajectory following


EXAMPLE_TRAJECTORY = np.fromstring(
    """
example smoothed traj: [[-0.3237884  -0.18898983 -0.3291086   0.94429207]
 [-0.32342687 -0.18840377 -0.3283459   0.94455755]
 [-0.32274202 -0.18754888 -0.32859483  0.944471  ]
 [-0.3216591  -0.18757771 -0.32667285  0.9451375 ]
 [-0.31938934 -0.1870756  -0.3224817   0.9465757 ]
 [-0.31556502 -0.18704012 -0.31905416  0.94773644]
 [-0.30918068 -0.1874013  -0.31111562  0.95037204]
 [-0.30006325 -0.18821377 -0.30424944  0.95259243]
 [-0.29209605 -0.18992569 -0.29729727  0.954785  ]
 [-0.28329057 -0.19159633 -0.2914994   0.95657104]
 [-0.27415946 -0.19340481 -0.28321305  0.95905703]
 [-0.2645545  -0.19527277 -0.27255523  0.96214014]
 [-0.25493842 -0.19715962 -0.26143175  0.965222  ]
 [-0.2450205  -0.1988978  -0.2470463   0.9690037 ]
 [-0.23513794 -0.2004779  -0.23198359  0.97271967]
 [-0.225187   -0.20189168 -0.21496154  0.9766225 ]
 [-0.21555883 -0.20316336 -0.1969306   0.98041743]
 [-0.20588253 -0.20414981 -0.17860453  0.98392093]
 [-0.19641098 -0.20507112 -0.1612344   0.9869161 ]
 [-0.18684423 -0.20579258 -0.14526184  0.9893933 ]
 [-0.17764091 -0.20660305 -0.13183789  0.9912713 ]
 [-0.1682598  -0.20712493 -0.12042125  0.99272287]
 [-0.15911889 -0.2077092  -0.1133325   0.9935571 ]
 [-0.14994162 -0.2081968  -0.10938013  0.994     ]
 [-0.14115807 -0.20873591 -0.10986698  0.99394625]
 [-0.13227186 -0.20908105 -0.11358398  0.9935284 ]
 [-0.12376487 -0.209326   -0.12313365  0.99239004]
 [-0.11539119 -0.20932351 -0.13722499  0.9905399 ]
 [-0.10755917 -0.2091143  -0.15749086  0.98752046]
 [-0.09971026 -0.20861872 -0.18281452  0.98314744]
 [-0.09224927 -0.2078062  -0.21535571  0.9765357 ]
 [-0.08485839 -0.20705564 -0.2543673   0.9671077 ]
 [-0.07773719 -0.20635243 -0.3014028   0.9534969 ]
 [-0.07086544 -0.20608059 -0.35660928  0.93425363]
 [-0.06436993 -0.20601875 -0.41877788  0.9080887 ]
 [-0.05821211 -0.20681992 -0.4886441   0.87248325]
 [-0.05208004 -0.20835403 -0.5619085   0.82719946]
 [-0.04660774 -0.21059397 -0.6372954   0.7706196 ]
 [-0.04158976 -0.21402323 -0.70856273  0.7056478 ]
 [-0.03710108 -0.21789715 -0.7725218   0.6349882 ]
 [-0.03270566 -0.22292964 -0.8259754   0.5637061 ]
 [-0.02838298 -0.22869542 -0.8676861   0.4971125 ]
 [-0.02404585 -0.235805   -0.8976457   0.4407178 ]
 [-0.01992842 -0.24415557 -0.9170522   0.39876714]
 [-0.01592617 -0.2536471  -0.9271317   0.37473568]
 [-0.01184369 -0.2644285  -0.92897356  0.37014613]
 [-0.00694788 -0.27653605 -0.9223459   0.38636518]
 [-0.00161097 -0.28962278 -0.906694    0.42178908]
 [ 0.00459203 -0.30306584 -0.8797506   0.47543544]
 [ 0.01118989 -0.31599665 -0.8404508   0.5418879 ]
 [ 0.01832524 -0.3291182  -0.78697526  0.61698455]
 [ 0.02510706 -0.34050027 -0.7207605   0.69318414]
 [ 0.03238833 -0.3508953  -0.6446298   0.7644949 ]
 [ 0.03967786 -0.35996446 -0.5650123   0.8250824 ]
 [ 0.04693503 -0.36769068 -0.4924411   0.87034583]
 [ 0.05356245 -0.37447414 -0.43120065  0.9022561 ]
 [ 0.05980375 -0.3801871  -0.3855363   0.92269266]
 [ 0.06592795 -0.38496777 -0.3619681   0.9321905 ]
 [ 0.07222652 -0.3884756  -0.35797042  0.9337329 ]
 [ 0.07833117 -0.39192283 -0.37877116  0.9254903 ]
 [ 0.08460736 -0.39486533 -0.4165016   0.909135  ]
 [ 0.09063429 -0.39818    -0.47585207  0.8795253 ]
 [ 0.09664412 -0.4019366  -0.5445146   0.8387514 ]
 [ 0.10219003 -0.40692094 -0.6227919   0.7823875 ]
 [ 0.10770468 -0.4133719  -0.70169413  0.7124783 ]
 [ 0.11316991 -0.42022094 -0.7767842   0.6297668 ]
 [ 0.1184254  -0.4278365  -0.8425454   0.53862536]
 [ 0.12343522 -0.43573272 -0.8955821   0.44489634]
 [ 0.12796052 -0.44475552 -0.93536025  0.35369667]
 [ 0.13126597 -0.4537139  -0.96252966  0.27117637]
 [ 0.13355821 -0.4627886  -0.9796338   0.20079257]
 [ 0.13468724 -0.47210234 -0.98951775  0.14441128]
 [ 0.13494395 -0.48125386 -0.9947891   0.10195357]
 [ 0.13480787 -0.49048185 -0.99741066  0.07191696]
 [ 0.13449074 -0.49937835 -0.9985262   0.05427134]
 [ 0.13410261 -0.5082202  -0.9989209   0.04644383]
 [ 0.1337023  -0.51713336 -0.99888587  0.04719159]
 [ 0.1333802  -0.5261568  -0.9985107   0.05455521]
 [ 0.13323683 -0.53556955 -0.9977301   0.0673401 ]
 [ 0.13330896 -0.54527473 -0.9965059   0.08352217]
 [ 0.13386226 -0.5553041  -0.9947773   0.10206965]
 [ 0.1347805  -0.5654286  -0.99255705  0.12178017]
 [ 0.13617788 -0.5756819  -0.98980695  0.14241552]
 [ 0.13767743 -0.58593714 -0.9866574   0.16281044]
 [ 0.13937077 -0.5960475  -0.98315984  0.18274803]
 [ 0.1412728  -0.6059336  -0.97950196  0.20143463]
 [ 0.14365652 -0.61553085 -0.97570497  0.21908855]
 [ 0.14636663 -0.6247239  -0.9718316   0.23567624]
 [ 0.14961967 -0.63355803 -0.96790123  0.25133085]
 [ 0.15299092 -0.6421174  -0.96410054  0.2655374 ]
 [ 0.1566087  -0.6504487  -0.9603748   0.2787119 ]
 [ 0.16030338 -0.6585136  -0.95671713  0.29101932]
 [ 0.16412069 -0.6663969  -0.95322406  0.30226472]
 [ 0.1679835  -0.67418337 -0.94999266  0.3122722 ]
 [ 0.17175879 -0.6817311  -0.94698626  0.32127413]
 [ 0.17550465 -0.68883777 -0.9444452   0.32866892]
 [ 0.17886911 -0.69540435 -0.94225997  0.33488235]
 [ 0.18309537 -0.7027162  -0.939225    0.34330225]
 [ 0.18643937 -0.7079307  -0.9369141   0.34955972]
 [ 0.18844187 -0.71077645 -0.9350923   0.35440442]
 [ 0.18995002 -0.7120835  -0.93379337  0.35781273]
 [ 0.19054969 -0.71250165 -0.9329072   0.36011693]
 [ 0.19062933 -0.7126737  -0.9321453   0.36208445]
 [ 0.19030446 -0.7123393  -0.9317409   0.36312386]]

""".replace("example smoothed traj:", " ").replace("[", " ").replace("]", " "),
    sep=" ",
).reshape(-1, 4)


def assign_process_to_cpu(pid, cpus):
    os.sched_setaffinity(pid, cpus)


def interpolate_trajectory(trajectory, total_duration, dt):
    """Interpolate 2D or 4D knot points onto a uniform time grid."""
    trajectory = np.asarray(trajectory, dtype=float)
    knot_times = np.linspace(0.0, total_duration, trajectory.shape[0])
    query_times = np.arange(0.0, total_duration, dt)

    interpolated = np.column_stack(
        [np.interp(query_times, knot_times, trajectory[:, i]) for i in range(trajectory.shape[1])]
    )

    if interpolated.shape[1] == 4:
        heading_norm = np.hypot(interpolated[:, 2], interpolated[:, 3])
        heading_norm[heading_norm == 0.0] = 1.0
        interpolated[:, 2] = interpolated[:, 2] / heading_norm
        interpolated[:, 3] = interpolated[:, 3] / heading_norm

    return interpolated


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def heading_sincos_to_quat(heading_sin, heading_cos, base_quat):
    theta = np.arctan2(heading_sin, heading_cos)
    half_theta = 0.5 * theta
    yaw_quat = np.column_stack(
        (
            np.cos(half_theta),
            np.zeros_like(half_theta),
            np.zeros_like(half_theta),
            np.sin(half_theta),
        )
    )
    return np.array([quat_multiply(yaw_q, base_quat) for yaw_q in yaw_quat])


class Avoiding_Sim(BaseSim):
    # Slowdown factor: execution dt = physics dt * slowdown_factor
    # physics dt = 0.001s, so slowdown_factor=20 means trajectory dt = 0.020s
    slowdown_factor = 110

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

        # We'll sample a fresh trajectory for each rollout below (so each run is different)

        random.seed(pid)
        torch.manual_seed(pid)
        np.random.seed(pid)

        # Sample initial trajectory and create/start env once so only one window opens
        try:
            from flow_matcher.generatesamplesfrommodel_Unet_generalized import sample_a_collision_free_trajectory
            sampled_traj = sample_a_collision_free_trajectory()
        except Exception as exc:
            log.warning(f"Falling back to EXAMPLE_TRAJECTORY due to sampling error: {exc}")
            sampled_traj = EXAMPLE_TRAJECTORY

        # Coerce sampled trajectory to (N,2) or (N,4)
        sampled_arr = np.asarray(sampled_traj, dtype=float)
        if sampled_arr.ndim == 3 and sampled_arr.shape[0] == 1:
            sampled_arr = sampled_arr.squeeze(0)
        elif sampled_arr.ndim == 1:
            if sampled_arr.size % 4 == 0:
                sampled_arr = sampled_arr.reshape(-1, 4)
        elif sampled_arr.ndim == 2 and sampled_arr.shape[1] not in (2, 4):
            if sampled_arr.shape[0] in (2, 4) and sampled_arr.shape[1] > 4:
                sampled_arr = sampled_arr.T

        if sampled_arr.ndim != 2 or sampled_arr.shape[1] not in (2, 4):
            raise ValueError(f"Sampled trajectory has unsupported shape after coercion: {sampled_arr.shape}")

        trajectory = shrink_size(sampled_arr, shrink_factor=SHRINK_FACTOR)
        if self.rotate_planner_frame_90on_z:
            trajectory = EnvSetup.planner_to_sim_xy(trajectory, rotate_90on_z=True)
        trajectory = EnvSetup.translate_in_x(trajectory, offset_x=0.3)

        initial_cart_position = np.array([trajectory[0, 0], trajectory[0, 1], 0.12])

        env = ObstacleAvoidanceEnv(render=self.render, trajectory_xy=trajectory, initial_cart_position=initial_cart_position)
        env.start()

        # run rollouts, updating the trajectory between runs
        for i in range(n_trajectories):
            if i == 0:
                # already have trajectory used to construct env
                pass
            else:
                try:
                    sampled_traj = sample_a_collision_free_trajectory()
                except Exception as exc:
                    log.warning(f"Falling back to EXAMPLE_TRAJECTORY due to sampling error: {exc}")
                    sampled_traj = EXAMPLE_TRAJECTORY

                sampled_arr = np.asarray(sampled_traj, dtype=float)
                if sampled_arr.ndim == 3 and sampled_arr.shape[0] == 1:
                    sampled_arr = sampled_arr.squeeze(0)
                elif sampled_arr.ndim == 1:
                    if sampled_arr.size % 4 == 0:
                        sampled_arr = sampled_arr.reshape(-1, 4)
                elif sampled_arr.ndim == 2 and sampled_arr.shape[1] not in (2, 4):
                    if sampled_arr.shape[0] in (2, 4) and sampled_arr.shape[1] > 4:
                        sampled_arr = sampled_arr.T

                if sampled_arr.ndim != 2 or sampled_arr.shape[1] not in (2, 4):
                    raise ValueError(f"Sampled trajectory has unsupported shape after coercion: {sampled_arr.shape}")

                trajectory = shrink_size(sampled_arr, shrink_factor=SHRINK_FACTOR)
                if self.rotate_planner_frame_90on_z:
                    trajectory = EnvSetup.planner_to_sim_xy(trajectory, rotate_90on_z=True)
                trajectory = EnvSetup.translate_in_x(trajectory, offset_x=0.3)

                # update env visualization for new trajectory without restarting scene
                try:
                    env.set_trajectory(trajectory)
                    
                except Exception:
                    pass

            agent.reset()

            print(f'core {pid}, Rollout {i}')

            obs = env.reset()

            fixed_quat = np.array([0, 1, 0, 0])

            # Treat the knot points as a time-parameterized trajectory.
            # Compute total duration from curve length (XY) and fixed speed, then
            # derive a per-trajectory slowdown factor (used implicitly below).
            xy = trajectory[:, :2]
            if xy.shape[0] >= 2:
                seg_dists = np.linalg.norm(np.diff(xy, axis=0), axis=1)
                curve_length = float(np.sum(seg_dists))
            else:
                curve_length = 0.0

            total_duration = curve_length / TRAJ_SPEED if TRAJ_SPEED > 0 else float(trajectory.shape[0] * self.slowdown_factor * env.robot.dt)
            # local slowdown factor (float) representing how many controller dt steps per knot
            slowdown_local = total_duration / (trajectory.shape[0] * env.robot.dt) if trajectory.shape[0] * env.robot.dt > 0 else float(self.slowdown_factor)
            desired_traj = interpolate_trajectory(
                trajectory,
                total_duration=total_duration,
                dt=env.robot.dt,
            )
            desired_pos = np.column_stack(
                (
                    desired_traj[:, 0],
                    desired_traj[:, 1],
                    np.full(desired_traj.shape[0], 0.12),
                )
            )
            if desired_traj.shape[1] == 4:
                desired_quat = heading_sincos_to_quat(
                    desired_traj[:, 2],
                    desired_traj[:, 3],
                    base_quat=fixed_quat,
                )
            else:
                desired_quat = np.repeat(fixed_quat[None, :], desired_pos.shape[0], axis=0)

            c_pos = [env.robot.current_c_pos]

            # Use the robot's full trajectory tracker, but run non-blocking so we can
            # keep per-step collision/success checks during execution.
            env.robot.follow_CartPositionAndQuatTraj(
                desiredPos=desired_pos,
                desiredQuat=desired_quat,
                goto_start=True,
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

        # Estimate maximum number of per-rollout position steps so we can
        # allocate a sufficiently large shared buffer. Each knot in
        # EXAMPLE_TRAJECTORY will be expanded by `slowdown_factor` steps
        # at controller rate, so expected steps ~= len(EXAMPLE_TRAJECTORY) * slowdown_factor.
        try:
            est_steps = int(EXAMPLE_TRAJECTORY.shape[0] * float(self.slowdown_factor)) + 10
        except Exception:
            est_steps = 150
        max_steps = max(150, est_steps) * 10
        robot_c_pos = torch.zeros([self.n_trajectories, max_steps, 2]).share_memory_()

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

        if getattr(wandb, "run", None) is not None:
            wandb.log({'score': (success_rate * 0.8 + entropy * 0.2)})
            wandb.log({'Metrics/successes': success_rate})
            wandb.log({'Metrics/entropy': entropy})

        print(f'Successrate {success_rate}')
        print(f'entropy {entropy}')

        return successes, entropy