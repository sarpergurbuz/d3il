import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
import time
import os
import yaml
from potential_field_with_walls import generate_potential_field_trajectories, sample_jerky_potential_field_path
import importlib.util
import sys

# Robustly load train_flow_matching_Unet_generalized from the same package folder
tfm_mod_name = "flow_matcher.train_flow_matching_Unet_generalized"
try:
    # Prefer normal import if available
    import train_flow_matching_Unet_generalized as _tfm
except Exception:
    # Fallback: load by file path and register under package module name
    _this_dir = os.path.dirname(__file__)
    _train_path = os.path.join(_this_dir, "train_flow_matching_Unet_generalized.py")
    spec = importlib.util.spec_from_file_location(tfm_mod_name, _train_path)
    _tfm = importlib.util.module_from_spec(spec)
    sys.modules[tfm_mod_name] = _tfm
    spec.loader.exec_module(_tfm)

# Export required symbols from loaded module
build_soft_obstacle_field = _tfm.build_soft_obstacle_field
COLLISION_DILATION_KERNEL = getattr(_tfm, "COLLISION_DILATION_KERNEL", None)
COLLISION_BLUR_KERNEL = getattr(_tfm, "COLLISION_BLUR_KERNEL", None)
COORD_BOUNDS = getattr(_tfm, "COORD_BOUNDS", None)
RECT_CENTER_OFFSET = getattr(_tfm, "RECT_CENTER_OFFSET", 0.0)
device = "cuda" if torch.cuda.is_available() else "cpu"


#---------------------------------------------------------------------------------------------------------------
CKPT_FILENAME = "models/checkpoint_epoch_200_offset.pt"

# cfg/state_dict will be loaded at runtime inside the sampling function if available
cfg = None
state_dict = None

# ---------------------------------------------------
# Build model from saved config
# ---------------------------------------------------

RECT_LENGTH=0.12
RECT_WIDTH=0.015
OBSTACLE_REGION_LIMITS = (-0.25, 0.25, -0.65, -0.3)

# ---------------------------------------------------
# Default config dictionary
# ---------------------------------------------------
DEFAULT_CONFIG = {
    'POTENTIAL_PATHS_AS_INITIAL_DIST': True,
    'USE_RK4_INSTEAD_OF_EULER': False,
    'SAMPLING_INTEGRATION_STEPS': 10,
    'SMOOTHING': True,
    'NUM_SAMPLES': 150,
    'PLOT_INITIAL_DIST_FOR_TRAJS': True,
    'ONLY_PLOT_COMPLEX': True,
    'POTENTIAL_PATHS_INITIAL_DIST_NOISE_STD': 0.001,
    'ONLY_PLOT_COLLISION_FREE': False,
    'COLLISION_FREE_SAMPLES': 12,
    'PLOT_RANDOM_NUMBER_OF_SAMPLES_OUT_OF_NUM_SAMPLES': 12,
    'OBSTACLE_RADIUS': 0.03,
    'SAFETY_RADIUS': 0.006,
    'USE_JERKY_PATHS_FOR_INITIALIZATION': False,
    'POTENTIAL_PATHS_ON_GPU': True,
    'PLOT_REFERENCE_TRAJ_POTENTIAL_FIELD': False,
    'GAUSSIAN_NOISE_INITIALIZATION_MEAN': [0.0, -0.4],
    'GAUSSIAN_NOISE_INITIALIZATION_STD': 0.2,
}


def sync_device():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def plot_rectangle(ax, x, y, theta, length, width, color='blue', alpha=0.25, linewidth=0.8):
    """Plot a rotated rectangle centered at (x, y) with heading theta."""
    corners = np.array([
        [-length / 2, -width / 2],
        [ length / 2, -width / 2],
        [ length / 2,  width / 2],
        [-length / 2,  width / 2],
        [-length / 2, -width / 2],
    ])

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rot = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    corners_world = corners @ rot.T + np.array([x, y])
    ax.plot(corners_world[:, 0], corners_world[:, 1], color=color, alpha=alpha, linewidth=linewidth)


def normalize_sincos_trajectory(traj):
    """Project the sin/cos orientation channels back onto the unit circle."""
    traj = np.asarray(traj, dtype=np.float32)
    norm = np.linalg.norm(traj[:, 2:4], axis=-1, keepdims=True)
    traj[:, 2:4] = traj[:, 2:4] / np.clip(norm, 1e-8, None)
    return traj



# ---------------------------------------------------
# Checkpoint path (do not load at import time)
# ---------------------------------------------------


# ---------------------------------------------------
# Sampling classes
# ---------------------------------------------------
class Sampler:
    def __init__(self, model, config=None, potential_field_paths=None):
        self.model = model
        self.potential_field_paths = potential_field_paths
        self.config = config or DEFAULT_CONFIG.copy()

    def _xy_trajs_to_4d(self, tau_xy, noise_std):
        """
        Convert (B, T, 2) position-only trajectories into (B, T, 4)
        using a path-derived heading representation.
        """
        waypoint_diff = tau_xy[:, 1:, :] - tau_xy[:, :-1, :]          # (B, T-1, 2)
        theta = torch.atan2(waypoint_diff[:, :, 1], waypoint_diff[:, :, 0])
        theta_full = torch.cat([theta, theta[:, -1:]], dim=1)         # (B, T)

        theta_noisy = theta_full + torch.randn_like(theta_full) * noise_std
        sin_theta = torch.sin(theta_noisy).unsqueeze(-1)              # (B, T, 1)
        cos_theta = torch.cos(theta_noisy).unsqueeze(-1)              # (B, T, 1)

        xy_noisy = tau_xy + torch.randn_like(tau_xy) * noise_std

        return torch.cat([xy_noisy, sin_theta, cos_theta], dim=-1)    # (B, T, 4)

    def apply_cfg_rescale(self, v_cond, v_uncond, w, phi=1, eps=1e-7):
        """
        v_cond, v_uncond: (B, T, C)
        Returns CFG-combined velocity with optional variance/std rescaling.
        """
        v_cfg = (1.0 - w) * v_uncond + w * v_cond

        # Start by testing rescaling only for amplified CFG
        if w <= 1.0:
            return v_cfg

        # Per-sample std over trajectory/time and channel dims
        std_cond = v_cond.std(dim=(1, 2), keepdim=True)
        std_cfg = v_cfg.std(dim=(1, 2), keepdim=True)

        v_rescaled = v_cfg * (std_cond / (std_cfg + eps))

        # Blend raw CFG and rescaled CFG
        return phi * v_rescaled + (1.0 - phi) * v_cfg

    def sample_tau0_based_on_potential_field(self, potential_field_paths, batch_size, noise_std=0.01):
        N, _, _ = potential_field_paths.shape
        indices = torch.randint(0, N, (batch_size,), device=potential_field_paths.device)
        tau0_reference_xy = potential_field_paths[indices]
        return self._xy_trajs_to_4d(tau0_reference_xy, noise_std)

    def prepare_initial_dist(self, batch_size, T, D):
        if self.config['POTENTIAL_PATHS_AS_INITIAL_DIST']:
            if self.potential_field_paths is None:
                raise ValueError("potential_field_paths is not set in Sampler")
            tau = self.sample_tau0_based_on_potential_field(
                self.potential_field_paths,
                batch_size,
                noise_std=self.config['POTENTIAL_PATHS_INITIAL_DIST_NOISE_STD'],
            )
        else:
            mean_init = torch.tensor(self.config['GAUSSIAN_NOISE_INITIALIZATION_MEAN'], dtype=torch.float32, device=device)
            std_init = self.config['GAUSSIAN_NOISE_INITIALIZATION_STD']
            tau0_xy = torch.randn(batch_size, T, 2, device=device) * std_init + mean_init

            # Compute heading from local trajectory direction
            diff = tau0_xy[:, 1:, :] - tau0_xy[:, :-1, :]          # (B, T-1, 2)
            theta_init = torch.atan2(diff[..., 1], diff[..., 0])   # (B, T-1)

            # Repeat last heading to get T orientations
            theta_init = torch.cat([theta_init, theta_init[:, -1:]], dim=1)  # (B, T)

            sin_theta_init = torch.sin(theta_init).unsqueeze(-1)
            cos_theta_init = torch.cos(theta_init).unsqueeze(-1)

            tau = torch.cat([
                tau0_xy,
                sin_theta_init,
                cos_theta_init,
            ], dim=-1)

        return tau

    def prepare_conditioning(self, cond, batch_size, w=1.0):
        if isinstance(cond, dict):
            cond_batch = {}
            for k, v in cond.items():
                if v.ndim == 2:
                    cond_batch[k] = v.repeat(batch_size, 1)
                elif v.ndim == 3:
                    cond_batch[k] = v.repeat(batch_size, 1, 1)
                else:
                    cond_batch[k] = v.expand(batch_size, *v.shape[1:])

            cond_null = {k: v.clone() for k, v in cond_batch.items()}
            cond_null['obstacle_mask'] = torch.zeros_like(cond_batch['obstacle_mask'])
            #cond_null['presence_flag'] = torch.zeros_like(cond_batch['presence_flag'])

            with torch.no_grad():
                cond_feat = self.model.encode_global_condition(cond_batch)
                cond_null_feat = self.model.encode_global_condition(cond_null) if w != 1.0 else None
        elif torch.is_tensor(cond):
            cond_feat = cond
            if cond_feat.ndim != 2:
                raise ValueError(f"Cached conditioning must have shape (B,C), got {cond_feat.shape}")
            if cond_feat.shape[0] == 1 and batch_size > 1:
                cond_feat = cond_feat.expand(batch_size, -1)
            elif cond_feat.shape[0] != batch_size:
                raise ValueError(
                    f"Cached conditioning batch mismatch: cond {cond_feat.shape[0]} vs requested {batch_size}"
                )
            cond_null_feat = torch.zeros_like(cond_feat) if w != 1.0 else None
        else:
            raise ValueError("cond must be a dict or a cached tensor from model.encode_global_condition")

        return cond_feat, cond_null_feat

    def integrate_field(self, tau, t_vals, dt, batch_size, cond_feat, cond_null_feat, w=1.0):
        with torch.no_grad():
            for i, t in enumerate(t_vals[:-1]):
                t_in = t.unsqueeze(0).expand(batch_size)

                if self.config['USE_RK4_INSTEAD_OF_EULER']:
                    v_cond_k1 = self.model(tau, t_in, global_cond=cond_feat)
                    if w == 1.0:
                        k1 = v_cond_k1
                    else:
                        v_uncond_k1 = self.model(tau, t_in, global_cond=cond_null_feat)
                        k1 = self.apply_cfg_rescale(v_cond_k1, v_uncond_k1, w)

                    t_mid = (t + t_vals[i+1]) / 2.0
                    t_mid_in = t_mid.unsqueeze(0).expand(batch_size)

                    tau_k2 = tau + dt * k1 / 2.0
                    v_cond_k2 = self.model(tau_k2, t_mid_in, global_cond=cond_feat)
                    if w == 1.0:
                        k2 = v_cond_k2
                    else:
                        v_uncond_k2 = self.model(tau_k2, t_mid_in, global_cond=cond_null_feat)
                        k2 = self.apply_cfg_rescale(v_cond_k2, v_uncond_k2, w)

                    tau_k3 = tau + dt * k2 / 2.0
                    v_cond_k3 = self.model(tau_k3, t_mid_in, global_cond=cond_feat)
                    if w == 1.0:
                        k3 = v_cond_k3
                    else:
                        v_uncond_k3 = self.model(tau_k3, t_mid_in, global_cond=cond_null_feat)
                        k3 = self.apply_cfg_rescale(v_cond_k3, v_uncond_k3, w)

                    t_next = t_vals[i+1]
                    t_next_in = t_next.unsqueeze(0).expand(batch_size)
                    tau_k4 = tau + dt * k3
                    v_cond_k4 = self.model(tau_k4, t_next_in, global_cond=cond_feat)
                    if w == 1.0:
                        k4 = v_cond_k4
                    else:
                        v_uncond_k4 = self.model(tau_k4, t_next_in, global_cond=cond_null_feat)
                        k4 = self.apply_cfg_rescale(v_cond_k4, v_uncond_k4, w)

                    tau = tau + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
                else:
                    v_cond_k1 = self.model(tau, t_in, global_cond=cond_feat)
                    if w == 1.0:
                        v = v_cond_k1
                    else:
                        v_uncond_k1 = self.model(tau, t_in, global_cond=cond_null_feat)
                        v = self.apply_cfg_rescale(v_cond_k1, v_uncond_k1, w)

                    tau = tau + dt * v

                # Keep the orientation state on the unit circle after every update.
                tau[:, :, 2:4] = tau[:, :, 2:4] / (torch.norm(tau[:, :, 2:4], dim=-1, keepdim=True) + 1e-8)

        return tau

    def load_obs_and_SE(self, file_path, file_is_npz=False):
        if file_is_npz:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"NPZ file not found: {file_path}")

            npz_data = np.load(file_path)
            if "obstacle_mask" not in npz_data or "start" not in npz_data or "end" not in npz_data:
                raise KeyError("NPZ must contain keys: 'obstacle_mask', 'start', 'end'")

            obstacles = np.asarray(npz_data["obstacle_mask"], dtype=np.float32)
            start_xy = np.asarray(npz_data["start"], dtype=np.float32).reshape(-1)[:2]
            end_xy = np.asarray(npz_data["end"], dtype=np.float32).reshape(-1)[:2]

            print(f"Loaded obstacle mask from NPZ: {obstacles.shape}")
        else:
            params_main = {}
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    params_main = yaml.safe_load(f) or {}

            obstacles = np.array(params_main['obstacles'], dtype=float)
            print(f"Loaded obstacles from YAML: {obstacles.shape[0]} obstacles")

            sx, sy = params_main.get('start_point', [[0, 0]])[0]
            ex, ey = params_main.get('goal_point', [[0, 1]])[0]

            start_xy = np.array([sx, sy], dtype=np.float32)
            end_xy = np.array([ex, ey], dtype=np.float32)
            obstacles = obstacles[:, :2]

        start = torch.from_numpy(start_xy).unsqueeze(0).to(device=device, dtype=torch.float32)
        end = torch.from_numpy(end_xy).unsqueeze(0).to(device=device, dtype=torch.float32)

        return start, end, obstacles

    def sample_trajectories_batch(self, T, D, cond, num_samples, steps=200, w=1.0):
        """Sample multiple trajectories in one GPU batch"""
        batch_size = num_samples
        tau = self.prepare_initial_dist(batch_size, T, D)

        tau_initial = tau.clone().detach().cpu().numpy()

        t_vals = torch.linspace(0, 1, steps, device=device)
        dt = 1.0 / steps

        cond_feat, cond_null_feat = self.prepare_conditioning(cond, batch_size, w=w)
        tau = self.integrate_field(tau, t_vals, dt, batch_size, cond_feat, cond_null_feat, w=w)

        return tau.detach().cpu().numpy(), tau_initial

    def prepare_initial_potential_field_distribution(self, start, end, obstacles):
        start_xy = start.squeeze(0).detach().cpu().numpy().astype(np.float32)
        end_xy = end.squeeze(0).detach().cpu().numpy().astype(np.float32)

        traj_mat, phi, X, Y, seeds, r_min, r_max, start_well_radius, obstacle_mask = generate_potential_field_trajectories(
            start_xy, obstacles, end_xy, n_traj=100, N_waypoints=104, n_output_path=50
        )

        multimodal_jerky_paths, sampled_points_xy = sample_jerky_potential_field_path(
            phi, X, Y, start_xy, end_xy, 20
        )

        potential_field_paths_np = traj_mat.astype(np.float32)
        potential_field_paths = torch.tensor(
            potential_field_paths_np,
            dtype=torch.float32,
            device=device if self.config['POTENTIAL_PATHS_ON_GPU'] else "cpu",
        )

        jerky_paths_np = np.asarray(multimodal_jerky_paths, dtype=np.float32)
        if jerky_paths_np.size == 0:
            jerky_paths_np = potential_field_paths_np.copy()
        jerky_paths = torch.tensor(
            jerky_paths_np,
            dtype=torch.float32,
            device=device if self.config['POTENTIAL_PATHS_ON_GPU'] else "cpu",
        )
        if self.config['USE_JERKY_PATHS_FOR_INITIALIZATION']:
            potential_field_paths = jerky_paths
            print(f"Using jerky paths from potential field as initial distribution for sampling (shape: {potential_field_paths.shape})")

        return potential_field_paths, phi, obstacle_mask


class PostSampler:
    def __init__(self, config=None):
        self.config = config or DEFAULT_CONFIG.copy()

    def is_trajectory_entering_narrow_spaces(
        self,
        traj,
        obstacle_region_limits=OBSTACLE_REGION_LIMITS,
        min_waypoint_fraction=0.65,
    ):
        """
        Return True if at least `min_waypoint_fraction` of trajectory waypoints lie
        inside the rectangular obstacle region limits.

        traj: (T, 2+) array. Only x,y channels are used.
        obstacle_region_limits: (x_min, x_max, y_min, y_max)
        """
        traj = np.asarray(traj, dtype=np.float32)
        if traj.ndim != 2 or traj.shape[1] < 2:
            raise ValueError(
                f"Expected trajectory shape (T, 2+) for narrow-space check, got {traj.shape}"
            )

        if not (0.0 <= float(min_waypoint_fraction) <= 1.0):
            raise ValueError(
                f"min_waypoint_fraction must be in [0, 1], got {min_waypoint_fraction}"
            )

        x_min, x_max, y_min, y_max = obstacle_region_limits
        xy = traj[:, :2]

        in_region = (
            (xy[:, 0] >= x_min)
            & (xy[:, 0] <= x_max)
            & (xy[:, 1] >= y_min)
            & (xy[:, 1] <= y_max)
        )

        in_region_count = int(np.count_nonzero(in_region))
        required_count = int(np.ceil(traj.shape[0] * float(min_waypoint_fraction)))

        return in_region_count >= required_count
    
    def smooth_ema(self, traj: np.ndarray, alpha: float = 0.2) -> np.ndarray:
        """
        Fast EMA low-pass smoothing for trajectory.

        traj: (T, D) array, D=2 typically.
        alpha: in (0,1). Smaller -> more smoothing.
        """
        traj = np.asarray(traj, dtype=np.float32)
        out = np.empty_like(traj)
        out[0] = traj[0]
        for t in range(1, traj.shape[0]):
            out[t] = alpha * traj[t] + (1.0 - alpha) * out[t - 1]
        return out

    def smooth_ema_zero_phase_anchor(
        self,
        traj: np.ndarray,
        alpha: float = 0.5,
        k: int = 8
    ) -> np.ndarray:
        """
        Zero-phase EMA smoothing with endpoint anchoring.

        k: number of points near each endpoint to anchor/blend (e.g., 5..15 for T=200)
        start/end: if provided, enforce these endpoints (use your conditioning).
        """
        traj = np.asarray(traj, dtype=np.float32)
        T = traj.shape[0]
        k = int(np.clip(k, 1, max(1, T // 2)))

        fwd = self.smooth_ema(traj, alpha=alpha)
        bwd = self.smooth_ema(fwd[::-1], alpha=alpha)[::-1]

        s0 = traj[0]
        sT = traj[-1]

        out = bwd
        out[0] = s0
        out[-1] = sT

        w = np.linspace(1.0, 0.0, k, dtype=np.float32)[:, None]
        out[:k] = w * s0 + (1.0 - w) * out[:k]

        w2 = np.linspace(0.0, 1.0, k, dtype=np.float32)[:, None]
        out[-k:] = (1.0 - w2) * out[-k:] + w2 * sT

        return out

    def calculate_average_bending_energy(self, samples):
        """
        Calculates the average Bending Energy across multiple trajectory samples.
        Lower = Smoother (closer to a straight line).
        Higher = More jitter/sharp turns.
        """
        total_energies = []

        for path in samples:
            path = np.array(path)

            ddx = np.diff(path[:, 0], n=2)
            ddy = np.diff(path[:, 1], n=2)
            point_energies = ddx**2 + ddy**2
            sample_energy = np.sum(point_energies)

            dx = np.diff(path[:, 0])
            dy = np.diff(path[:, 1])
            length = np.sum(np.sqrt(dx**2 + dy**2))
            sample_energy = sample_energy / (length + 1e-6)

            total_energies.append(sample_energy)

        return np.mean(total_energies)

    def _make_rectangle_edge_points_np(self, rect_length, rect_width, n_len=13, n_width=3):
        """Generate rectangle edge points in local (body) frame."""
        xs = np.linspace(-rect_length / 2, rect_length / 2, n_len)
        ys = np.linspace(-rect_width / 2, rect_width / 2, n_width)

        top = np.stack([xs, np.full_like(xs, rect_width / 2)], axis=-1)
        bottom = np.stack([xs, np.full_like(xs, -rect_width / 2)], axis=-1)
        left = np.stack([np.full_like(ys, -rect_length / 2), ys], axis=-1)
        right = np.stack([np.full_like(ys, rect_length / 2), ys], axis=-1)

        pts = np.concatenate([top, bottom, left, right], axis=0)
        pts = np.unique(pts, axis=0)  # Remove duplicates
        return pts

    def is_trajectory_colliding(
        self,
        traj,
        obstacles,
        radius=0.03,
        obstacles_are_mask=False,
        obstacle_mask_limits=COORD_BOUNDS,
    ):
        """
        Check trajectory collision using rectangle boundaries at each timestep.
        traj: (T, 4) or (T, 2) array -> x, y, sin(theta), cos(theta) or just x, y
        obstacles: (K, 2) array or (H, W) obstacle mask
        radius: collision radius (obstacle_radius + safety_radius)
        returns: True if any rectangle boundary point collides with any obstacle
        """
        if obstacles_are_mask:
            mask = np.asarray(obstacles)
            if mask.ndim != 2:
                raise ValueError(
                    f"obstacles_are_mask=True requires 2D mask input, got shape {mask.shape}"
                )

            x_min, x_max, y_min, y_max = obstacle_mask_limits
            h, w = mask.shape
            x = np.asarray(traj[:, 0], dtype=np.float32)
            y = np.asarray(traj[:, 1], dtype=np.float32)

            ix = np.clip(
                np.round((x - x_min) / (x_max - x_min + 1e-12) * (w - 1)).astype(np.int32),
                0,
                w - 1,
            )
            iy = np.clip(
                np.round((y - y_min) / (y_max - y_min + 1e-12) * (h - 1)).astype(np.int32),
                0,
                h - 1,
            )
            return bool(np.any(mask[iy, ix]))

        # For regular obstacle list, use rectangle-based collision detection
        if traj.shape[1] >= 4:
            # 4D trajectory: x, y, sin(theta), cos(theta)
            return self._is_trajectory_colliding_rect(traj, obstacles, radius)
        else:
            # 2D trajectory: fallback to point-distance
            for obs in obstacles:
                traj_xy = traj[:, :2]
                dists = np.linalg.norm(traj_xy - obs, axis=1)
                if np.any(dists < radius):
                    return True
            return False

    def _is_trajectory_colliding_rect(self, traj, obstacles, collision_radius):
        """
        Check collision using rectangle boundaries at each timestep.
        traj: (T, 4) -> x, y, sin(theta), cos(theta)
        obstacles: (N, 2) obstacle centers
        """
        if obstacles is None or len(obstacles) == 0:
            return False

        T = traj.shape[0]
        rect_length = RECT_LENGTH
        rect_width = RECT_WIDTH

        # Extract components
        xy_ref = traj[:, :2]          # (T, 2): reference/control point positions
        sin_theta = traj[:, 2]    # (T,): sin(theta)
        cos_theta = traj[:, 3]    # (T,): cos(theta)

        # Compute rectangle center as reference point + offset along heading direction
        xy = xy_ref + RECT_CENTER_OFFSET * np.stack([cos_theta, sin_theta], axis=-1)

        # Local rectangle edge points (in body frame)
        local_pts = self._make_rectangle_edge_points_np(rect_length, rect_width)
        # local_pts shape: (P, 2) where P is number of edge points

        # Transform rectangle points to world frame at each timestep
        lx = local_pts[:, 0]  # (P,)
        ly = local_pts[:, 1]  # (P,)

        # Compute world positions: (T, P, 2)
        wx = xy[:, 0:1] + cos_theta[:, None] * lx[None, :] - sin_theta[:, None] * ly[None, :]
        wy = xy[:, 1:2] + sin_theta[:, None] * lx[None, :] + cos_theta[:, None] * ly[None, :]

        rect_points_world = np.stack([wx, wy], axis=-1)  # (T, P, 2)

        # Check collision: for each obstacle, compute distance from all rectangle points
        for obs in obstacles:
            # obs shape: (2,)
            # rect_points_world shape: (T, P, 2)
            # dists shape: (T, P)
            dists = np.linalg.norm(rect_points_world - obs[None, None, :], axis=2)
            if np.any(dists < collision_radius):
                return True

        return False

    def summarize_and_plot_samples(
        self,
        samples,
        initial_samples,
        obstacles,
        collision_flags,
        collision_free_indices,
        guidance_weight,
        sx,
        sy,
        ex,
        ey,
        obstacle_radius,
        complexity_flags=None,
        obstacles_are_mask=False,
        obstacle_mask_limits=COORD_BOUNDS,
        show_plot_instead_of_saving_png=True,
        output_dir=None,
        filename=None,
    ):
        smoothness_for_this_batch = self.calculate_average_bending_energy(samples)
        start_points = np.array([traj[0] for traj in samples])
        end_points = np.array([traj[-1] for traj in samples])

        start_mean = start_points.mean(axis=0)
        start_std = start_points.std(axis=0)
        end_mean = end_points.mean(axis=0)
        end_std = end_points.std(axis=0)

        if obstacles_are_mask:
            x_min_lim, x_max_lim, y_min_lim, y_max_lim = obstacle_mask_limits
            all_x = np.concatenate([traj[:, 0] for traj in samples] + [np.array([x_min_lim, x_max_lim], dtype=float)])
            all_y = np.concatenate([traj[:, 1] for traj in samples] + [np.array([y_min_lim, y_max_lim], dtype=float)])
        else:
            all_x = np.concatenate([traj[:, 0] for traj in samples] + [obstacles[:, 0]])
            all_y = np.concatenate([traj[:, 1] for traj in samples] + [obstacles[:, 1]])

        x_min, x_max = all_x.min(), all_x.max()
        y_min, y_max = all_y.min(), all_y.max()

        pad_x = 0.05 * (x_max - x_min)
        pad_y = 0.05 * (y_max - y_min)

        x_min -= pad_x
        x_max += pad_x
        y_min -= pad_y
        y_max += pad_y

        if complexity_flags is None:
            complexity_flags = [False] * len(samples)

        if len(complexity_flags) != len(samples):
            raise ValueError(
                f"Expected complexity_flags length {len(samples)}, got {len(complexity_flags)}"
            )

        base_indices = list(range(len(samples)))
        if self.config.get('ONLY_PLOT_COMPLEX', False):
            base_indices = [i for i in base_indices if complexity_flags[i]]

        if self.config['ONLY_PLOT_COLLISION_FREE']:
            collision_free_set = set(collision_free_indices)
            base_indices = [i for i in base_indices if i in collision_free_set]

        if self.config['ONLY_PLOT_COLLISION_FREE']:
            if len(base_indices) > 0:
                n_select = min(self.config['COLLISION_FREE_SAMPLES'], len(base_indices))
                plot_indices = np.random.choice(base_indices, size=n_select, replace=False).tolist()
            else:
                plot_indices = []
        else:
            random_plot_count = self.config.get('PLOT_RANDOM_NUMBER_OF_SAMPLES_OUT_OF_NUM_SAMPLES')
            if random_plot_count is not None:
                n_select = min(int(random_plot_count), len(base_indices))
                if n_select > 0:
                    plot_indices = np.random.choice(base_indices, size=n_select, replace=False).tolist()
                else:
                    plot_indices = []
            else:
                plot_indices = base_indices

        cols = 4
        n_plot = len(plot_indices)
        rows = int(np.ceil(n_plot / cols)) if n_plot > 0 else 1
        plt.figure(figsize=(4 * cols, 4 * rows))
        title = (f"Generated Trajectories U-net Model (CFG w={guidance_weight})\n"
                 f"smoothness_val={smoothness_for_this_batch:.4f}\n"
                 f"Start points: mu=({start_mean[0]:.3f}, {start_mean[1]:.3f}), sigma=({start_std[0]:.3f}, {start_std[1]:.3f}) | "
                 f"End points: mu=({end_mean[0]:.3f}, {end_mean[1]:.3f}), sigma=({end_std[0]:.3f}, {end_std[1]:.3f})")
        plt.suptitle(title, y=0.98, fontsize=11)

        for plot_idx, sample_idx in enumerate(plot_indices):
            traj = samples[sample_idx]
            ax = plt.subplot(rows, cols, plot_idx + 1)

            if obstacles_are_mask:
                from matplotlib.colors import ListedColormap

                mask = np.asarray(obstacles, dtype=np.float32)
                x_min_lim, x_max_lim, y_min_lim, y_max_lim = obstacle_mask_limits
                mask_cmap = ListedColormap(["white", "dimgray"])
                ax.imshow(
                    mask,
                    origin="lower",
                    cmap=mask_cmap,
                    vmin=0.0,
                    vmax=1.0,
                    extent=[x_min_lim, x_max_lim, y_min_lim, y_max_lim],
                    interpolation="nearest",
                    alpha=1.0,
                    zorder=0,
                )

            if self.config['PLOT_INITIAL_DIST_FOR_TRAJS']:
                initial_traj = initial_samples[sample_idx]
                ax.scatter(initial_traj[:, 0], initial_traj[:, 1], c='gray', s=5, alpha=0.5, label='Initial' if plot_idx == 0 else '')

                # Plot the initial robot footprint as gray rectangles using the initial orientation.
                if initial_traj.shape[1] >= 4:
                    init_theta = np.arctan2(initial_traj[:, 2], initial_traj[:, 3])
                    for step in range(initial_traj.shape[0]):
                        # Apply offset to get rectangle center from reference point
                        rect_center_x = initial_traj[step, 0] + RECT_CENTER_OFFSET * np.cos(init_theta[step])
                        rect_center_y = initial_traj[step, 1] + RECT_CENTER_OFFSET * np.sin(init_theta[step])
                        plot_rectangle(
                            ax,
                            rect_center_x,
                            rect_center_y,
                            init_theta[step],
                            RECT_LENGTH,
                            RECT_WIDTH,
                            color='black',
                            alpha=0.5,
                            linewidth=0.7,
                        )

            is_colliding = collision_flags[sample_idx]
            color = 'red' if is_colliding else 'blue'
            ax.plot(traj[:, 0], traj[:, 1], color=color)

            # Overlay the robot rectangle at every timestep using the trajectory orientation.
            if traj.shape[1] >= 4:
                sin_theta = traj[:, 2]
                cos_theta = traj[:, 3]
                theta = np.arctan2(sin_theta, cos_theta)
                for step in range(traj.shape[0]):
                    # Apply offset to get rectangle center from reference point
                    rect_center_x = traj[step, 0] + RECT_CENTER_OFFSET * cos_theta[step]
                    rect_center_y = traj[step, 1] + RECT_CENTER_OFFSET * sin_theta[step]
                    step_alpha = 0.12 + 0.18 * (step / max(1, traj.shape[0] - 1))
                    plot_rectangle(
                        ax,
                        rect_center_x,
                        rect_center_y,
                        theta[step],
                        RECT_LENGTH,
                        RECT_WIDTH,
                        color=color,
                        alpha=0.6,
                        linewidth=0.8,
                    )

            ax.plot(sx, sy, marker='o', color='limegreen', markersize=7, linestyle='None',
                    label='Start' if plot_idx == 0 else None)
            ax.plot(ex, ey, marker='*', color='red', markersize=10, linestyle='None',
                    label='Goal' if plot_idx == 0 else None)

            if not obstacles_are_mask:
                for obs in obstacles:
                    circle = plt.Circle((obs[0], obs[1]), obstacle_radius,
                                        fill=True, color='dimgray', linewidth=1.5)
                    ax.add_patch(circle)
            ax.set_aspect("equal")

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)

            ax.grid(True)

            if plot_idx == 0:
                ax.legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        if show_plot_instead_of_saving_png:
            plt.show()
        else:
            if output_dir is None:
                output_path = "generated_trajectories_plot.png"
            else:
                if filename is None:
                    output_path = os.path.join(output_dir, "generated_trajectories_plot.png")
                else:
                    output_path = os.path.join(output_dir, filename)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {output_path}")
            plt.close()



# ---------------------------------------------------
# Generate samples
# ---------------------------------------------------
def sample_a_collision_free_trajectory(return_plot_context=False):
    # Create config and instantiate classes (model may be attached later if checkpoint found)
    config = DEFAULT_CONFIG.copy()
    sampler = Sampler(None, config=config)
    post_sampler = PostSampler(config=config)

    # Obstacles YAML lives outside flow_matcher, so resolve from repo root.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    yaml_path = os.path.join(
        repo_root,
        "environments",
        "d3il",
        "d3il_sim",
        "sims",
        "universal_sim",
        "test_envs",
        
        "layout_6.yaml",
    )
    start, end, obstacles = sampler.load_obs_and_SE(yaml_path)
    sx, sy = start[0, 0].item(), start[0, 1].item()
    ex, ey = end[0, 0].item(), end[0, 1].item()
    obstacle_radius = config['OBSTACLE_RADIUS']

    potential_field_paths, phi, obstacle_mask = sampler.prepare_initial_potential_field_distribution(start, end, obstacles)

    obstacle_mask_t = torch.as_tensor(obstacle_mask, dtype=torch.float32, device=device)
    if obstacle_mask_t.ndim == 2:
        obstacle_mask_t = obstacle_mask_t.unsqueeze(0)
    elif obstacle_mask_t.ndim != 3:
        raise ValueError(f"Expected obstacle_mask with shape (H,W) or (B,H,W), got {tuple(obstacle_mask_t.shape)}")

    phi_t = torch.as_tensor(phi, dtype=torch.float32, device=device)
    if phi_t.ndim == 2:
        phi_t = phi_t.unsqueeze(0)
    elif phi_t.ndim != 3:
        raise ValueError(f"Expected phi with shape (H,W) or (B,H,W), got {tuple(phi_t.shape)}")

    softened_obstacle_field_t = build_soft_obstacle_field(
        obstacle_mask_t,
        dilation_kernel=COLLISION_DILATION_KERNEL,
        blur_kernel=COLLISION_BLUR_KERNEL,
    )

    sampler.potential_field_paths = potential_field_paths

    flag = torch.ones(1, 1, device=device)

    cond_dict_inference = {
        'start': start,
        'end': end,
        'obstacle_mask': softened_obstacle_field_t,
        'presence_flag': flag
    }

    guidance_weight = 5

    # Try to locate and load a checkpoint file at runtime. If not found, fall back
    # to selecting a potential-field path as a sampled trajectory.
    candidate_paths = [
        CKPT_FILENAME,
        os.path.join(os.path.dirname(__file__), CKPT_FILENAME),
        os.path.join(os.path.dirname(__file__), "..", CKPT_FILENAME),
    ]
    ckpt_file = None
    for p in candidate_paths:
        if os.path.exists(p):
            ckpt_file = p
            break

    samples = None
    initial_samples = None
    start_time = None

    if ckpt_file is not None:
        try:
            try:
                from .unet import ConditionalUnet1D
            except Exception:
                from unet import ConditionalUnet1D

            ckpt = torch.load(ckpt_file, map_location=device)
            cfg_loaded = ckpt.get("config", {})
            state = ckpt.get("model_state_dict", ckpt.get("state_dict", None))

            model = ConditionalUnet1D(
                input_dim=cfg_loaded.get("D", 4),
                cond_embed_dim=cfg_loaded.get("cond_embed_dim", 128),
                diffusion_step_embed_dim=cfg_loaded.get("diffusion_step_embed_dim", 64),
                down_dims=cfg_loaded.get("down_dims", [16, 32, 64]),
                kernel_size=cfg_loaded.get("kernel_size", 5),
                n_groups=cfg_loaded.get("n_groups", 8),
                attention=cfg_loaded.get("attention", False),
            ).to(device)

            if state is not None:
                model.load_state_dict(state)
            model.eval()

            sampler.model = model

            with torch.no_grad():
                _ = model.encode_global_condition(cond_dict_inference)

            # Time the sampling
            start_time = time.time()
            samples, initial_samples = sampler.sample_trajectories_batch(
                int(cfg_loaded.get("T", config.get('T', 104))), int(cfg_loaded.get("D", config.get('D', 4))), cond_dict_inference,
                config['NUM_SAMPLES'], steps=config['SAMPLING_INTEGRATION_STEPS'], w=guidance_weight
            )
        except Exception as e:
            print(f"Warning: model sampling failed ({e}), falling back to potential-field path")
            samples = None

    # # Fallback: use potential field paths directly
    # if samples is None:
    #     potential_field_paths_np = potential_field_paths.cpu().numpy() if isinstance(potential_field_paths, torch.Tensor) else np.asarray(potential_field_paths)
    #     N = potential_field_paths_np.shape[0]
    #     if N == 0:
    #         raise RuntimeError("No potential-field paths available to fallback to.")

    #     samples = []
    #     initial_samples = []
    #     collision_flags = []

    #     order = list(range(N))
    #     random.shuffle(order)

    #     for idx in order:
    #         path_xy = potential_field_paths_np[idx]
    #         # compute heading and pack into 4D
    #         diffs = np.diff(path_xy, axis=0)
    #         theta = np.arctan2(diffs[:, 1], diffs[:, 0])
    #         theta = np.concatenate([theta, theta[-1:]])
    #         sin_theta = np.sin(theta)[:, None]
    #         cos_theta = np.cos(theta)[:, None]
    #         traj4 = np.concatenate([path_xy, sin_theta, cos_theta], axis=1)
    #         samples.append(normalize_sincos_trajectory(traj4))
    #         initial_samples.append(path_xy)
    #         is_coll = post_sampler.is_trajectory_colliding(traj4, obstacles, radius=config['OBSTACLE_RADIUS'] + config['SAFETY_RADIUS'])
    #         collision_flags.append(is_coll)

    #     collision_free_indices = [i for i, f in enumerate(collision_flags) if not f]
    #     if len(collision_free_indices) == 0:
    #         selected_idx = 0
    #     else:
    #         selected_idx = int(np.random.choice(collision_free_indices))

    #     selected_traj = np.asarray(samples[selected_idx], dtype=np.float32)

    #     if return_plot_context:
    #         plot_context = {
    #             'post_sampler': post_sampler,
    #             'samples': samples,
    #             'initial_samples': initial_samples,
    #             'obstacles': obstacles,
    #             'collision_flags': collision_flags,
    #             'collision_free_indices': collision_free_indices,
    #             'guidance_weight': guidance_weight,
    #             'sx': sx,
    #             'sy': sy,
    #             'ex': ex,
    #             'ey': ey,
    #             'obstacle_radius': obstacle_radius,
    #         }
    #         return selected_traj, plot_context

    #     return selected_traj


    # Model sampling succeeded: process and return the result
    if config['SMOOTHING']:
        samples = [normalize_sincos_trajectory(post_sampler.smooth_ema_zero_phase_anchor(traj, alpha=0.5, k=8)) for traj in samples]
    #print("example smoothed traj:", samples[0] if len(samples) > 0 else "no samples")
    
    if start_time is not None:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Batch sampling completed in {elapsed_time:.2f} seconds")

    collision_flags = [
        post_sampler.is_trajectory_colliding(
            traj,
            obstacles,
            radius=config['OBSTACLE_RADIUS'] + config['SAFETY_RADIUS'],
        )
        for traj in samples
    ]

    
    complexity_flags= [
        post_sampler.is_trajectory_entering_narrow_spaces(traj,obstacle_region_limits=OBSTACLE_REGION_LIMITS, min_waypoint_fraction=0.10)
        for traj in samples
    ]
    collision_free_indices = [i for i, flag in enumerate(collision_flags) if not flag]
    red_count = len(samples) - len(collision_free_indices)
    print(f"Colliding trajectories: {red_count}/{len(samples)}")

    

    complex_indices = [i for i, flag in enumerate(complexity_flags) if flag]
    print(f"Trajectories entering narrow spaces: {len(complex_indices)}/{len(samples)}")

    if len(collision_free_indices) == 0:
        raise RuntimeError("No collision-free trajectory found in sampled batch.")

    collision_free_and_complex_indices = [
        i for i in collision_free_indices if complexity_flags[i]
    ]
    if len(collision_free_and_complex_indices) == 0:
        #raise RuntimeError("No trajectory found that is both collision-free and complex.")
        print("No trajectory found that is both collision-free and complex. Falling back to any collision-free trajectory.")
        collision_free_and_complex_indices = collision_free_indices

    selected_idx = int(np.random.choice(collision_free_and_complex_indices))
    selected_traj = np.asarray(samples[selected_idx], dtype=np.float32)

    

    print(f"Selected random collision-free & complex trajectory index: {selected_idx}")

    if return_plot_context:
        plot_context = {
            'post_sampler': post_sampler,
            'samples': samples,
            'initial_samples': initial_samples,
            'obstacles': obstacles,
            'collision_flags': collision_flags,
            'complexity_flags': complexity_flags,
            'collision_free_indices': collision_free_indices,
            'guidance_weight': guidance_weight,
            'sx': sx,
            'sy': sy,
            'ex': ex,
            'ey': ey,
            'obstacle_radius': obstacle_radius,
        }
        return selected_traj, plot_context

    return selected_traj


if __name__ == "__main__":
    sampled_traj, plot_context = sample_a_collision_free_trajectory(return_plot_context=True)
    plot_context['post_sampler'].summarize_and_plot_samples(
        samples=plot_context['samples'],
        initial_samples=plot_context['initial_samples'],
        obstacles=plot_context['obstacles'],
        collision_flags=plot_context['collision_flags'],
        complexity_flags=plot_context.get('complexity_flags'),
        collision_free_indices=plot_context['collision_free_indices'],
        guidance_weight=plot_context['guidance_weight'],
        sx=plot_context['sx'],
        sy=plot_context['sy'],
        ex=plot_context['ex'],
        ey=plot_context['ey'],
        obstacle_radius=plot_context['obstacle_radius'],
        show_plot_instead_of_saving_png=True,
        output_dir="generated_plots",
        filename="plotted_trajectoriess.png",
    )

