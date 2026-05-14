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

        # Prepare trajectory once (for both visualization and execution)
        trajectory = shrink_size(EXAMPLE_TRAJECTORY, shrink_factor=SHRINK_FACTOR)
        
        # Apply planner-to-sim frame rotation
        if self.rotate_planner_frame_90on_z:
            trajectory = EnvSetup.planner_to_sim_xy(
                trajectory,
                rotate_90on_z=True,
            )
        
        # Translate trajectory in x by 0.2 meters (after rotation, in sim frame)
        trajectory = EnvSetup.translate_in_x(trajectory, offset_x=0.2)
        
        initial_cart_position = np.array([
            trajectory[0, 0],
            trajectory[0, 1],
            0.12,
        ])

        env = ObstacleAvoidanceEnv(
            render=self.render,
            trajectory_xy=trajectory,
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

            # Treat the knot points as a time-parameterized trajectory.
            # The controller still runs at 1 ms, but we query the interpolated target at each step.
            total_duration = trajectory.shape[0] * self.slowdown_factor * env.robot.dt
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
        max_steps = max(150, est_steps)
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