import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
try:
    from .unet import ConditionalUnet1D
except Exception:
    from unet import ConditionalUnet1D
try:
    from .dataset_dataloader_full_dataset import build_diverse_paths_dataloader
except Exception:
    from dataset_dataloader_full_dataset import build_diverse_paths_dataloader
import torch.nn.functional as F


import math
import os
import time


COORD_BOUNDS = (-0.5, 0.5, -0.95, 0.05)  # (x_min, x_max, y_min, y_max) for normalizing coordinates and defining UNet grid
# Normalization parameter
NORMALIZATION = False  # Set to False to disable data and conditioning normalization
D3IL_SETUP = False      # Switch for D3IL specific dataset and fixed conditioning
GENERALIZED_DATASET_SETUP = True # Switch for new generalized dataset with dict conditioning and potential field paths
POTENTIAL_PATHS_AS_INITIAL_DIST = True  
JERKY_PATHS_FOR_INITIALIZATION = False  
DISTORT_DATA_FOR_INITIAL_DIST = False  
POTENTIAL_PATHS_ON_GPU = True  # Store potential_field_paths on GPU (set True if it fits in VRAM)
POTENTIAL_PATHS_INITIAL_DIST_NOISE_STD = 0.15 # Standard deviation of Gaussian noise added to potential field paths when used as initial distribution


GAUSSIAN_NOISE_INITIALIZATION_MEAN= [0.0, -0.4]  # Mean for Gaussian noise initialization when not using potential field paths
GAUSSIAN_NOISE_INITIALIZATION_STD = 0.3  # Standard deviation for Gaussian noise initialization when not using potential field paths (adjust based on how much randomness you want in the initial trajectories)

######### SOFT MATCTHING HYPERPARAMETERS #########
SOFT_MATCHING=True # If True, use augmented matching loss with multiple components. If False, use only velocity MSE.
SOFT_MATCHING_WEIGHT = 0.5
# Matching loss weights (augmented loss combines all types)
MATCH_W_TRAJ = 1.0
MATCH_W_VEL = 0.1
MATCH_W_END = 0.1
MATCH_W_COS = 0.1


# Start point penalty
START_POINT_PENALTY = True # If True, add explicit MSE loss on start point mismatch
START_END_LOCATION_LOSS_WEIGHT = 1.5  # Weight for start location MSE loss (penalize start point mismatch)
MSE_LOSS_WEIGHT = 1.0

# Position loss
POSITION_LOSS = True  # If True, add explicit MSE loss on x/y position mismatch
POSITION_LOSS_WEIGHT = 0.2

# Orientation loss
ORIENT_LOSS = True  # If True, add explicit MSE loss on orientation (sin/cos) mismatch
ORIENT_LOSS_WEIGHT = 1.0  # Weight for orientation MSE loss

# Velocity and acceleration consistency loss
VELO_ACCEL_CONSISTENCY_LOSS = False  # If True, add loss on velocity and acceleration computed from trajectory
VELO_ACCEL_CONSISTENCY_LOSS_WEIGHT = 1e-7  # Weight for velocity/acceleration consistency loss

######### MODEL AVERAGING (EMA) #########
USE_EMA = True 
EMA_DECAY = 0.999
EMA_START_FRACTION = 0.90  # start EMA updates after 60% of epochs (focus more on last steps)

######### DYNAMIC GATING HYPERPARAMETERS (Option 2) #########
GATE_K = -0.85  # Controls suppression threshold: d0 = mean_dist + k*std_dist. Range [-0.5, +1.0]
              # k=-0.5: aggressive (suppress ~70%), k=0.0: balanced (suppress ~50%), k=+0.5: lenient (suppress ~30%)
GATE_SHARPNESS = 1.5  # Controls gate smoothness: temp = std_dist/sharpness. Range [2.0, 10.0]
                      # sharpness=2.0: smooth transition, sharpness=5.0: sharp cutoff, sharpness=10.0: very sharp


# COLLISION LOSS HYPERPARAMETERS
COLLISION_LOSS = True
COLLISION_LOSS_WEIGHT = 0.2
RECTANGLE_LENGTH = 0.118
RECTANGLE_WIDTH = 0.014
RECT_CENTER_OFFSET = 0.03  # Offset of rectangle center from reference point along heading direction

# Keep these modest because corridors are narrow (~13 px wide)
COLLISION_DILATION_KERNEL = 3
COLLISION_BLUR_KERNEL = 5


# Setup device first
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model and training hyperparameters


BATCH_SIZE = 128
CHECKPOINT_DIR = "checkpoints/generalized_Unet/obs_mask_cond/may7/200epochs_no_veloaccel_loss_collos02_distinitial_onlyxy_rect0118x0014_lr0005_footprint_aware_feature_concat_posloss02_blurkernel_5_OFFSETRECT"  # Directory to save checkpoints
FINAL_MODEL_NAME= "final_model.pt"
DATA_ROOT = "data/generalization/dataset_for_offset_rect/dataset_for_robot_sim_rectangle_v2_fixed_SE_generation_paths_with_ilqr"
LEARNING_RATE = 0.0005
SCHEDULER_GAMMA = 0.999
NUM_EPOCHS = 200
ATTENTION = True
UNET_TIME_EMB_DIM = 128           # diffusion_step_embed_dim
UNET_DOWNS = [16, 32, 64]      # channels per UNet level
KERNEL_SIZE = 3
N_GROUPS = 8
COND_DROP_PROB = 0.25   # percentage unconditional
COND_EMBED_DIM = 256    # embedding size for conditioning vector

# Diagnostic: sensitivity of predictions to obstacle-mask changes
ENABLE_OBSTACLE_MASK_SENSITIVITY_DIAGNOSTIC = False
OBSTACLE_DIAG_BATCH_SAMPLES = 8





# Checkpointing
SAVE_CHECKPOINTS = True
CHECKPOINT_INTERVAL = 50  # save every N epochs
SAVE_FINAL = True


# TODO: normalize data and conditioning for stable training




def update_ema_state(model, ema_state, decay):
    """Update EMA parameters from current model state."""
    with torch.no_grad():
        model_state = model.state_dict()
        for k, v in model_state.items():
            if v.is_floating_point():
                ema_state[k].mul_(decay).add_(v.detach(), alpha=(1.0 - decay))
            else:
                # Keep non-floating tensors (e.g., integer buffers) in sync.
                ema_state[k].copy_(v)

# -----------------------------
# Training loop
# -----------------------------
loss_history = []
epoch_losses = []
epoch_matching_losses_track = []
epoch_start_losses_track = []
epoch_position_losses_track = []
epoch_collision_losses_track = []
epoch_orient_losses_track = []
epoch_velo_accel_losses_track = []
lr_history=[]

# Beta distribution for time sampling (TESTING different from uniform)
beta_dist = torch.distributions.Beta(concentration1=3.0, concentration0=1)



def compute_matching_loss(tau0, tau1):
    """
    Compute augmented matching loss as a weighted sum of all types (batch averaged).

    Args:
        tau0: (B, T, D) random initialization trajectories
        tau1: (B, T, D) target trajectories

    Returns:
        matching_loss: scalar loss value
    """
    loss_per_sample = compute_matching_loss_per_sample(tau0, tau1)
    return torch.mean(loss_per_sample)


def compute_matching_loss_per_sample(tau0, tau1):
    """
    Compute augmented matching loss per sample (not batch-averaged).

    Args:
        tau0: (B, T, D) random initialization trajectories
        tau1: (B, T, D) target trajectories

    Returns:
        loss_per_sample: (B,) per-sample loss values
    """
    B = tau0.shape[0]

    # Focus matching on positions only (x,y). Ignore orientation channels.
    pos0 = tau0[..., :2]  # (B, T, 2)
    pos1 = tau1[..., :2]  # (B, T, 2)

    # trajectory L2 over positions
    pos_diff = torch.norm(pos0 - pos1, dim=(1, 2))  # (B,)
    pos_magnitude = torch.norm(pos1, dim=(1, 2))  # (B,)
    loss_traj = pos_diff / (pos_magnitude + 1e-8)  # (B,)

    # velocity magnitude (positions only)
    v = pos1 - pos0  # (B, T, 2)
    loss_vel = torch.norm(v, dim=(1, 2))  # (B,)

    # endpoint L2 on positions only
    start_diff = torch.norm(pos0[:, 0, :] - pos1[:, 0, :], dim=1)  # (B,)
    end_diff = torch.norm(pos0[:, -1, :] - pos1[:, -1, :], dim=1)  # (B,)
    loss_end = start_diff + end_diff  # (B,)

    # cosine similarity on flattened positions
    tau0_flat = pos0.reshape(B, -1)  # (B, T*2)
    tau1_flat = pos1.reshape(B, -1)  # (B, T*2)
    cos_sim = torch.nn.functional.cosine_similarity(tau0_flat, tau1_flat, dim=1)  # (B,)
    loss_cos = 1.0 - cos_sim  # (B,)

    # weighted sum per sample (position-only)
    return (MATCH_W_TRAJ * loss_traj
        + MATCH_W_VEL * loss_vel
        + MATCH_W_END * loss_end
        + MATCH_W_COS * loss_cos)  # (B,)

def sample_tau0_based_on_potential_field(potential_field_paths, batch_size, noise_std=0.01, output_without_noise=False):
    """
    Sample trajectories from potential field paths and add Gaussian noise.
    
    Args:
        potential_field_paths: (N, T, D) tensor of trajectories
        batch_size: number of trajectories to sample
        noise_std: standard deviation of Gaussian noise to add to each waypoint
    
    Returns:
        tau0_samples: (batch_size, T, D) sampled and noised trajectories
    
    Noise suggestions (for trajectories in ~[0, 1] range):
        - 0.001-0.005: Very subtle perturbation, preserves trajectory shape well
        - 0.005-0.01:  Small noise, slight random variation (RECOMMENDED)
        - 0.01-0.03:   Moderate noise, noticeable jitter but still recognizable
        - 0.03-0.05:   Strong noise, significant deviation from original
        - >0.05:       Very strong noise, may corrupt trajectory too much
    """
    N, T, D = potential_field_paths.shape
    indices = torch.randint(0, N, (batch_size,))
    tau0_samples = potential_field_paths[indices]  # shape: (batch_size, T, D)
    
    # Add Gaussian noise to each waypoint
    noise = torch.randn_like(tau0_samples) * noise_std
    tau0_samples = tau0_samples + noise
    
    if output_without_noise:
        return potential_field_paths[indices].to(device)
    return tau0_samples.to(device)

def sample_tau0_based_on_potential_field_family(
    pf_family_paths,
    jerky_paths,
    pf_counts_b,
    jerky_counts_b,
    JERKY_PATHS_FOR_INITIALIZATION,
    noise_std=0.01,
    output_without_noise=False,
):
    """
    Select one trajectory per batch item from unpadded family sets.

    Inputs are padded tensors:
      - pf_family_paths: (B, max_pf, T, D)
      - jerky_paths:     (B, max_jerky, T, D)
      - pf_counts_b:     (B,)
      - jerky_counts_b:  (B,)

    If JERKY_PATHS_FOR_INITIALIZATION=True, prefer jerky paths first;
    otherwise prefer potential-field family paths first.
    """
    B, _, T, D = pf_family_paths.shape
    selected = []

    for i in range(B):
        pf_count = int(pf_counts_b[i].item())
        jerky_count = int(jerky_counts_b[i].item())

        chosen = None

        if JERKY_PATHS_FOR_INITIALIZATION:
            if jerky_count > 0:
                idx = torch.randint(0, jerky_count, (1,), device=jerky_paths.device).item()
                chosen = jerky_paths[i, idx]
            elif pf_count > 0:
                idx = torch.randint(0, pf_count, (1,), device=pf_family_paths.device).item()
                chosen = pf_family_paths[i, idx]
        else:
            if pf_count > 0:
                idx = torch.randint(0, pf_count, (1,), device=pf_family_paths.device).item()
                chosen = pf_family_paths[i, idx]
            elif jerky_count > 0:
                idx = torch.randint(0, jerky_count, (1,), device=jerky_paths.device).item()
                chosen = jerky_paths[i, idx]

        if chosen is None:
            chosen = torch.zeros((T, D), dtype=pf_family_paths.dtype, device=pf_family_paths.device)

        selected.append(chosen)

    tau0_samples = torch.stack(selected, dim=0)  # (B, T, D)

    if output_without_noise:
        return tau0_samples

    noise = torch.randn_like(tau0_samples) * noise_std
    return tau0_samples + noise

def sample_tau0_based_on_potential_field_paired(
    potential_field_paths,
    data_samples_for_current_batch,
    number_of_pot_path_to_be_sampled,
    batch_size,
    noise_std=0.01,
):
    """
    Sample a subset of potential paths, then pair each data sample with its
    closest potential path (L2 distance), and add Gaussian noise.
    
    Args:
        potential_field_paths: (N, T, D) tensor of trajectories
        data_samples_for_current_batch: (B, T, D) batch trajectories
        number_of_pot_path_to_be_sampled: number of candidate potential paths
        batch_size: number of trajectories in the batch
        noise_std: standard deviation of Gaussian noise to add to each waypoint
    
    Returns:
        tau0_samples: (B, T, D) matched and noised trajectories
    """
    N, T, D = potential_field_paths.shape
    B = data_samples_for_current_batch.shape[0]
    m = min(number_of_pot_path_to_be_sampled, N)

    # Sample candidate potential paths
    pot_indices = torch.randint(0, N, (m,), device=potential_field_paths.device)
    pot_paths = potential_field_paths[pot_indices].to(data_samples_for_current_batch.device)  # (m, T, D)

    # Compute pairwise L2 distance between batch and candidate paths
    # cost[b, m] = ||data[b] - pot[m]||_2 over all timesteps and dims
    data_batch = data_samples_for_current_batch[:B].to(data_samples_for_current_batch.device).unsqueeze(1)  # (B, 1, T, D)
    pot_batch = pot_paths.unsqueeze(0)  # (1, m, T, D)
    diff = data_batch - pot_batch
    cost = torch.norm(diff.reshape(B, m, -1), dim=2)  # (B, m)
    cost_np = cost.detach().cpu().numpy()

    if m >= B:
        # Hungarian assignment for one-to-one pairing
        row_ind, col_ind = linear_sum_assignment(cost_np)
        col_ind_t = torch.tensor(col_ind, device=pot_paths.device)
        matched = pot_paths[col_ind_t]
    else:
        # Fallback: not enough candidates, allow reuse via nearest neighbor
        nearest = np.argmin(cost_np, axis=1)
        matched = pot_paths[torch.tensor(nearest, device=pot_paths.device)]

    tau0_samples = matched

    # Add Gaussian noise to each waypoint
    noise = torch.randn_like(tau0_samples) * noise_std
    tau0_samples = tau0_samples + noise

    return tau0_samples.to(device)


def visualize_noise_effect(potential_field_paths, num_samples=4, noise_std=0.01):
    """
    Visualize the effect of Gaussian noise on sampled trajectories.
    Plots original vs. noisy trajectories side-by-side.
    Uses the SAME trajectories for fair comparison.
    """
    N, T, D = potential_field_paths.shape
    indices = torch.randint(0, N, (num_samples,))
    
    # Original trajectories (no noise)
    tau0_original = potential_field_paths[indices].cpu().numpy()
    
    # Noisy trajectories - add noise to the SAME trajectories
    tau0_noisy_tensor = potential_field_paths[indices].clone()
    noise = torch.randn_like(tau0_noisy_tensor) * noise_std
    tau0_noisy = (tau0_noisy_tensor + noise).detach().cpu().numpy()
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original trajectories
    ax = axes[0]
    for i in range(num_samples):
        ax.plot(tau0_original[i, :, 0], tau0_original[i, :, 1], 'b-', alpha=0.6, linewidth=2, label=f'Traj {i}' if i == 0 else '')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Original Trajectories (No Noise)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Noisy trajectories
    ax = axes[1]
    for i in range(num_samples):
        ax.plot(tau0_noisy[i, :, 0], tau0_noisy[i, :, 1], 'r-', alpha=0.6, linewidth=2, label=f'Noisy Traj {i}' if i == 0 else '')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Noisy Trajectories (noise_std={noise_std})")
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.show()
    
    # Also plot overlay for better comparison
    fig, ax = plt.subplots(figsize=(10, 8))
    for i in range(num_samples):
        ax.plot(tau0_original[i, :, 0], tau0_original[i, :, 1], 'b-', alpha=0.5, linewidth=2, label='Original' if i == 0 else '')
        ax.plot(tau0_noisy[i, :, 0], tau0_noisy[i, :, 1], 'r--', alpha=0.5, linewidth=2, label='Noisy' if i == 0 else '')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Comparison: Original (blue) vs Noisy (red dashed) | noise_std={noise_std}")
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend()
    plt.tight_layout()
    plt.show()



# Test noise with 4 sample trajectories and noise_std=0.001
#visualize_noise_effect(potential_field_paths, num_samples=4, noise_std=0.1)


def coords_to_grid(points_btd: torch.Tensor) -> torch.Tensor:
    """
    points_btd: (B, T, 2), physical coordinates in (x, y)
    Returns grid for grid_sample in [-1, 1] using COORD_BOUNDS.
    """
    x_min, x_max, y_min, y_max = COORD_BOUNDS

    x = points_btd[..., 0]
    y = points_btd[..., 1]

    x = 2.0 * (x - x_min) / (x_max - x_min) - 1.0
    y = 2.0 * (y - y_min) / (y_max - y_min) - 1.0

    return torch.stack((x, y), dim=-1).clamp(-1.0, 1.0)


def sample_field_at_points(field_bhw: torch.Tensor, points_btd: torch.Tensor) -> torch.Tensor:
    """
    field_bhw:  (B, H, W)
    points_btd: (B, T, 2), physical coordinates in (x, y)
    returns:    (B, T) sampled values
    """
    field = field_bhw.unsqueeze(1).float()          # (B, 1, H, W)
    grid = coords_to_grid(points_btd).unsqueeze(2)  # (B, T, 1, 2)

    vals = F.grid_sample(
        field,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )  # (B, 1, T, 1)

    return vals.squeeze(1).squeeze(-1)  # (B, T)


def build_soft_obstacle_field(
    obstacle_mask_b: torch.Tensor,
    dilation_kernel: int = 3,
    blur_kernel: int = 5,
) -> torch.Tensor:
    """
    obstacle_mask_b: (B, H, W), binary {0,1}
    returns:         (B, H, W), soft obstacle field in [0,1]

    dilation_kernel=3 is a mild inflation (~1 pixel per side).
    blur_kernel=5 gives smooth gradients near obstacle boundaries.
    """
    x = obstacle_mask_b.unsqueeze(1).float()  # (B, 1, H, W)

    if dilation_kernel > 1:
        x = F.max_pool2d(
            x,
            kernel_size=dilation_kernel,
            stride=1,
            padding=dilation_kernel // 2,
        )

    if blur_kernel > 1:
        x = F.avg_pool2d(
            x,
            kernel_size=blur_kernel,
            stride=1,
            padding=blur_kernel // 2,
        )

    return x.squeeze(1).clamp(0.0, 1.0)

def make_rectangle_edge_points(rect_length, rect_width, n_len=9, n_width=3, device="cuda"):
    xs = torch.linspace(-rect_length / 2, rect_length / 2, n_len, device=device)
    ys = torch.linspace(-rect_width / 2, rect_width / 2, n_width, device=device)

    top = torch.stack([xs, torch.full_like(xs, rect_width / 2)], dim=-1)
    bottom = torch.stack([xs, torch.full_like(xs, -rect_width / 2)], dim=-1)

    left = torch.stack([torch.full_like(ys, -rect_length / 2), ys], dim=-1)
    right = torch.stack([torch.full_like(ys, rect_length / 2), ys], dim=-1)

    edge_pts = torch.cat([top, bottom, left, right], dim=0)

    # remove duplicate corner points
    edge_pts = torch.unique(edge_pts, dim=0)

    return edge_pts



def rectangle_collision_loss_from_mask(
    tau_hat,
    obstacle_field_b,
    rect_length=RECTANGLE_LENGTH,
    rect_width=RECTANGLE_WIDTH,
    n_len=21,
    n_width=7,
):
    """
    tau_hat: (B, T, 4) -> x, y, sin(theta), cos(theta)
    obstacle_field_b: (B, H, W) soft obstacle field
    returns scalar loss
    """
    B, T, _ = tau_hat.shape
    device = tau_hat.device

    xy_ref = tau_hat[..., :2]

    sincos = tau_hat[..., 2:4]
    sincos = sincos / (torch.norm(sincos, dim=-1, keepdim=True) + 1e-8)
    s = sincos[..., 0]
    c = sincos[..., 1]

    # Compute rectangle center as reference point + offset along heading direction
    xy = xy_ref + RECT_CENTER_OFFSET * torch.stack([c, s], dim=-1)

    # edge-only rectangle sample points
    local_pts = make_rectangle_edge_points(
        rect_length=rect_length,
        rect_width=rect_width,
        n_len=n_len,
        n_width=n_width,
        device=device,
    )  # (P, 2)

    P = local_pts.shape[0]

    lx = local_pts[:, 0][None, None, :]
    ly = local_pts[:, 1][None, None, :]

    # rotate local edge points by predicted theta and translate to rectangle center
    wx = xy[..., 0:1] + c[..., None] * lx - s[..., None] * ly
    wy = xy[..., 1:2] + s[..., None] * lx + c[..., None] * ly

    rect_points = torch.stack([wx, wy], dim=-1)      # (B, T, P, 2)
    rect_points = rect_points.reshape(B, T * P, 2)   # (B, T*P, 2)

    collision_vals = sample_field_at_points(
        obstacle_field_b,
        rect_points,
    )  # (B, T*P)

    k = max(1, int(0.15 * collision_vals.shape[1]))
    worst_vals = torch.topk(collision_vals, k=k, dim=1).values

    # return per-sample losses
    loss_collision_per_sample = worst_vals.mean(dim=1)  # (B,)

    return loss_collision_per_sample






def train_flow_matching(epochs=NUM_EPOCHS):
    dist_initial_list = []
    dist_predicted_list = []
    ema_start_epoch = int(epochs * EMA_START_FRACTION)
    ema_initialized = (not USE_EMA) or (ema_start_epoch <= 0)
    train_start_time = time.time()

    # Initialize running statistics for dynamic gating
    dist_mean = None
    dist_std = None
    dist_count = 0
    dist_sum = 0.0
    dist_sum_sq = 0.0

    for epoch in range(epochs):
        batch_losses = []
        batch_matching_losses = []
        batch_start_losses = []
        batch_position_losses = []
        batch_obstacle_diag_scores = []
        batch_collision_losses = []
        batch_orient_losses = []
        batch_velo_accel_losses = []

        for batch in dataloader:
            # ---------------------------------------------------
            # Unpack batch from new multi-file dataset/dataloader
            # ---------------------------------------------------
            tau1_path = batch[0].to(device)                # (B, T, 2)
            start_b = batch[1].to(device)             # (B, 2)
            end_b = batch[2].to(device)               # (B, 2)
            obstacle_mask_b = batch[3].to(device)     # (B, 200, 200)
            family_id_b = batch[4].to(device)         # (B,)
            phi_b = batch[5].to(device)               # (B, 200, 200)
            start_family = batch[6].to(device)        # (B, 2)
            end_family = batch[7].to(device)          # (B, 2)

            pf_family_paths = batch[8].to(device)     # (B, max_pf_in_batch, T, 2), padded
            jerky_paths = batch[9].to(device)         # (B, max_jerky_in_batch, T, 2), padded

            flag_b = batch[10].to(device)             # (B,)
            pf_counts_b = batch[11].to(device)        # (B,)
            jerky_counts_b = batch[12].to(device)     # (B,)

            # Optional debug/meta info
            shard_idx_b = batch[13].to(device)        # (B,)
            path_local_idx_b = batch[14].to(device)   # (B,)
            global_idx_b = batch[15].to(device)       # (B,)
            obstacle_centers_b = batch[16].to(device) # (B, max_obs_in_batch, 2), padded

            
            ilqr_pose_b= batch[18].to(device)         # (B, 104, 3)  ILQR trajectory poses (x, y, theta)

            # Compute tau1: convert ilqr_pose_b to (x, y, sin(theta), cos(theta))
            theta_b = ilqr_pose_b[:, :, 2]             # (B, 104)

            tau1 = torch.cat([
                ilqr_pose_b[:, :, :2],                 # (B, 104, 2) - x, y
                torch.sin(theta_b).unsqueeze(-1),      # (B, 104, 1) - sin(theta)
                torch.cos(theta_b).unsqueeze(-1),      # (B, 104, 1) - cos(theta)
            ], dim=-1)                                  # (B, 104, 4)
            

            ilqr_vel_b = batch[19].to(device)          # (B, 104, 2)
            ilqr_omega_b = batch[20].to(device)        # (B, 104,)
            ilqr_acc_b = batch[21].to(device)          # (B, 104, 2)
            ilqr_alpha_b = batch[22].to(device)        # (B, 104)
            ilqr_dt_b = batch[25].to(device)           # (B,)
            
            start_specific_ilqr_pose_b = batch[26].to(device)  # (B, 3)
            end_specific_ilqr_pose_b = batch[27].to(device)    # (B, 3)

            batch_size = tau1.shape[0]

            # Randomize obstacle order for robustness (per sample, no cross-sample mixing)
            # B, K, D_obs = obstacle_centers_b.shape
            # shuffled_indices = torch.argsort(torch.rand(B, K, device=device), dim=1)
            # obstacle_centers_b = torch.gather(
            #     obstacle_centers_b,
            #     dim=1,
            #     index=shuffled_indices.unsqueeze(-1).expand(-1, -1, D_obs)
            # )  # (B, K, 2) with shuffled obstacle order

            # ---------------------------------------------------
            # Construct Dict Conditioning
            # ---------------------------------------------------

            soft_obstacle_field_b = build_soft_obstacle_field(
                    obstacle_mask_b,
                    dilation_kernel=COLLISION_DILATION_KERNEL,
                    blur_kernel=COLLISION_BLUR_KERNEL,
                )  # (B, 200, 200)
            
            cond_dict = {
                "start": start_specific_ilqr_pose_b[:, :2],
                "end": end_specific_ilqr_pose_b[:, :2],
                "obstacle_mask": soft_obstacle_field_b,
                "presence_flag": flag_b,
                
            }

            # ---------------------------------------------------
            # Classifier-Free Guidance (CFG)
            # ---------------------------------------------------
            drop_mask = torch.rand(batch_size, device=device) < COND_DROP_PROB

            if COND_DROP_PROB > 0:
                obstacle_mask_cfg = soft_obstacle_field_b.clone()
                obstacle_mask_cfg[drop_mask] = 0.0
                cond_dict_cfg = {
                    "start": start_specific_ilqr_pose_b[:, :2],
                    "end": end_specific_ilqr_pose_b[:, :2],
                    "obstacle_mask": obstacle_mask_cfg,
                    "presence_flag": flag_b,
                    
                }
            else:
                cond_dict_cfg = cond_dict

            # ---------------------------------------------------
            # Sample initial trajectories tau0
            # ---------------------------------------------------
            tau0_reference = None

            if DISTORT_DATA_FOR_INITIAL_DIST:
                # Distort training data directly: tau0 = tau1 + noise
                noise = torch.randn_like(tau1) * POTENTIAL_PATHS_INITIAL_DIST_NOISE_STD
                tau0 = tau1 + noise

            elif POTENTIAL_PATHS_AS_INITIAL_DIST:
                tau0_reference_xy = sample_tau0_based_on_potential_field_family(
                    pf_family_paths,
                    jerky_paths,
                    pf_counts_b,
                    jerky_counts_b,
                    JERKY_PATHS_FOR_INITIALIZATION,
                    noise_std=POTENTIAL_PATHS_INITIAL_DIST_NOISE_STD,
                    output_without_noise=True,
                )  # (B, 104, 2)
                
                # Compute orientation (theta) for each waypoint from consecutive waypoint directions
                waypoint_diff = tau0_reference_xy[:, 1:, :] - tau0_reference_xy[:, :-1, :]  # (B, 103, 2)
                theta = torch.atan2(waypoint_diff[:, :, 1], waypoint_diff[:, :, 0])          # (B, 103)
                
                # Extend theta to full trajectory length: repeat last angle for final waypoint
                theta_full = torch.cat([theta, theta[:, -1:]], dim=1)  # (B, 104)
                
                # Add noise to theta BEFORE converting to sin/cos (maintains unit circle constraint)
                theta_noisy = theta_full + torch.randn_like(theta_full) * POTENTIAL_PATHS_INITIAL_DIST_NOISE_STD
                
                # Convert noised theta to sin/cos representation
                sin_theta = torch.sin(theta_noisy).unsqueeze(-1)   # (B, 104, 1)
                cos_theta = torch.cos(theta_noisy).unsqueeze(-1)   # (B, 104, 1)
                
                # Add noise to x,y coordinates
                xy_noisy = tau0_reference_xy + torch.randn_like(tau0_reference_xy) * POTENTIAL_PATHS_INITIAL_DIST_NOISE_STD
                
                # Construct full 4D tau0 with noised position and orientation: (x, y, sin(theta), cos(theta))
                tau0 = torch.cat([
                    xy_noisy,
                    sin_theta,
                    cos_theta,
                ], dim=-1)  # (B, 104, 4)
                
                # tau0_reference is the clean version (for matching loss comparison)
                tau0_reference = torch.cat([
                    tau0_reference_xy,
                    torch.sin(theta_full).unsqueeze(-1),
                    torch.cos(theta_full).unsqueeze(-1),
                ], dim=-1)  # (B, 104, 4)

            else:
                # Legacy Gaussian initialization 
                mean_init = torch.tensor(
                    GAUSSIAN_NOISE_INITIALIZATION_MEAN,
                    dtype=torch.float32,
                    device=device,
                )
                std_init = GAUSSIAN_NOISE_INITIALIZATION_STD
                traj_T = tau1.shape[1]
                # Initialize x, y with Gaussian noise
                tau0_xy = torch.randn(batch_size, traj_T, 2, device=device) * std_init + mean_init  # (B, T, 2)
                
                # Compute heading from local trajectory direction
                diff = tau0_xy[:, 1:, :] - tau0_xy[:, :-1, :]          # (B, T-1, 2)
                theta_init = torch.atan2(diff[..., 1], diff[..., 0])   # (B, T-1)

                # Repeat last heading to get T orientations
                theta_init = torch.cat([theta_init, theta_init[:, -1:]], dim=1)  # (B, T)

                sin_theta_init = torch.sin(theta_init).unsqueeze(-1)
                cos_theta_init = torch.cos(theta_init).unsqueeze(-1)

                tau0 = torch.cat([tau0_xy, sin_theta_init, cos_theta_init], dim=-1)
                
                # Construct full 4D tau0: (x, y, sin(theta), cos(theta))
                tau0 = torch.cat([
                    tau0_xy,
                    sin_theta_init,
                    cos_theta_init,
                ], dim=-1)  # (B, T, 4)
                
                

            # ---------------------------------------------------
            # Sample random time t
            # ---------------------------------------------------
            # sample random time t
            #t = torch.rand(batch_size, device=device)
            t = beta_dist.sample((batch_size,)).to(device)

            # ---------------------------------------------------
            # interpolate
            # ---------------------------------------------------
            tau_t = (1 - t[:, None, None]) * tau0 + t[:, None, None] * tau1

            

            # target velocity (true flow)
            v_star = tau1 - tau0

            # predict velocity from model
            v_pred = model(tau_t, t, global_cond=cond_dict_cfg)

            # Reconstruct the predicted clean trajectory once and normalize the orientation part.
            # This keeps collision checking and orientation supervision consistent.
            t_exp = t[:, None, None]
            tau1_hat = tau_t + (1.0 - t_exp) * v_pred   # (B, T, 4)
            tau1_hat_norm = tau1_hat.clone()
            pred_sincos_raw = tau1_hat[..., 2:4]
            pred_sincos = pred_sincos_raw / (pred_sincos_raw.norm(dim=-1, keepdim=True) + 1e-8)
            tau1_hat_norm[..., 2:4] = pred_sincos

            # Base flow matching loss
            loss = MSE_LOSS_WEIGHT * loss_fn(v_pred, v_star)


            # ---------------------------------------------------
            # Soft matching gate
            # ---------------------------------------------------
            if SOFT_MATCHING:
                # Per-sample MSE
                se = (v_pred - v_star) ** 2
                per_sample_mse = se.reshape(batch_size, -1).mean(dim=1)  # (B,)

                # Per-sample initialization distance
                dist_initial_per_sample = compute_matching_loss_per_sample(tau0_reference, tau1)
                #print("dist_initial_per_sample:", dist_initial_per_sample.detach().cpu().numpy())
                

                # Update running statistics
                batch_dist_values = dist_initial_per_sample.detach().cpu().numpy()
                for val in batch_dist_values:
                    dist_count += 1
                    dist_sum += val
                    dist_sum_sq += val ** 2

                # Running mean/std
                if dist_count > 0:
                    dist_mean = dist_sum / dist_count
                    dist_var = (dist_sum_sq / dist_count) - (dist_mean ** 2)
                    dist_std = np.sqrt(np.maximum(dist_var, 1e-8))

                # Adaptive threshold/temperature
                if dist_std is not None and dist_std > 0:
                    d0 = dist_mean + GATE_K * dist_std
                    temp = dist_std / GATE_SHARPNESS
                else:
                    d0 = 0.5
                    temp = 0.1

                # Thresholded exponential gate
                excess = torch.clamp(dist_initial_per_sample - d0, min=0.0)
                weight_per_sample = torch.exp(-excess / temp)

                weighted_loss = weight_per_sample * per_sample_mse
                loss = MSE_LOSS_WEIGHT *SOFT_MATCHING_WEIGHT * weighted_loss.mean()

                # Track mean gate weight
                batch_matching_losses.append(weight_per_sample.mean().item())

            # ---------------------------------------------------
            # Collision loss on predicted clean path
            # ---------------------------------------------------
            if COLLISION_LOSS:
                # Build smooth obstacle field from binary mask
                soft_obstacle_field = build_soft_obstacle_field(
                    obstacle_mask_b,
                    dilation_kernel=COLLISION_DILATION_KERNEL,
                    blur_kernel=COLLISION_BLUR_KERNEL,
                )  # (B, 200, 200)

                loss_collision_per_sample = rectangle_collision_loss_from_mask(
                    tau_hat=tau1_hat_norm,
                    obstacle_field_b=soft_obstacle_field,
                    rect_length=RECTANGLE_LENGTH,
                    rect_width=RECTANGLE_WIDTH,
                    n_len=27,
                    n_width=7,
                )  # (B,)

                # Weight collision more strongly near flow time t=1
                collision_time_weight = t ** 2                               # (B,)

                loss_collision = (collision_time_weight * loss_collision_per_sample).mean()

                # record per-batch (unweighted) collision loss for reporting
                batch_collision_losses.append(loss_collision.item())

                loss = loss + COLLISION_LOSS_WEIGHT * loss_collision

            # ---------------------------------------------------
            # Start / end point penalty
            # ---------------------------------------------------
            if START_POINT_PENALTY:
                

                loss_start = torch.nn.functional.mse_loss(tau1_hat_norm[:, 0, :], tau1[:, 0, :])
                loss_end = torch.nn.functional.mse_loss(tau1_hat_norm[:, -1, :], tau1[:, -1, :])
                loss_start_end = loss_start + loss_end

                loss = loss + START_END_LOCATION_LOSS_WEIGHT * loss_start_end
                batch_start_losses.append(loss_start_end.item())

            # ---------------------------------------------------
            # Position loss on x/y only
            # ---------------------------------------------------
            if POSITION_LOSS:
                target_pos = tau1[..., :2]
                loss_pos = torch.nn.functional.mse_loss(tau1_hat_norm[..., :2], target_pos)

                loss = loss + POSITION_LOSS_WEIGHT * loss_pos
                batch_position_losses.append(loss_pos.item())

            # ---------------------------------------------------
            # Orientation (sin/cos) loss
            # ---------------------------------------------------
            if ORIENT_LOSS:
                # Extract target sin/cos
                target_sincos = tau1[..., 2:4]
                
                # Compute MSE loss on normalized orientation
                loss_orient = torch.nn.functional.mse_loss(tau1_hat_norm[..., 2:4], target_sincos)
                
                loss = loss + ORIENT_LOSS_WEIGHT * loss_orient
                batch_orient_losses.append(loss_orient.item())

            # ---------------------------------------------------
            # Velocity and acceleration consistency loss
            # ---------------------------------------------------
            if VELO_ACCEL_CONSISTENCY_LOSS:
                # Compute discrete derivatives with centered differences.
                # Boundary values are fixed to zero to preserve rest-to-rest endpoints.
                tau_pred = tau1_hat_norm  # (B, T, 4) with x, y, sin(theta), cos(theta)

                B, T, _ = tau_pred.shape

                # Use each sample's own timestep instead of collapsing to a batch mean.
                DT_xy = ilqr_dt_b.reshape(B, 1, 1)
                DT_theta = ilqr_dt_b.reshape(B, 1)

                pos = tau_pred[:, :, :2]  # (B, T, 2)

                theta = torch.atan2(
                    tau_pred[:, :, 2],  # sin
                    tau_pred[:, :, 3],  # cos
                )  # (B, T)

                # -----------------------------
                # Velocity: central difference
                # -----------------------------
                vel_pred_xy = torch.zeros_like(pos)
                vel_pred_xy[:, 1:-1] = (pos[:, 2:] - pos[:, :-2]) / (2.0 * DT_xy)
                vel_pred_xy[:, 0] = 0.0
                vel_pred_xy[:, -1] = 0.0

                # -----------------------------
                # Angular velocity
                # -----------------------------
                omega_pred = torch.zeros_like(theta)
                dtheta = torch.atan2(
                    torch.sin(theta[:, 2:] - theta[:, :-2]),
                    torch.cos(theta[:, 2:] - theta[:, :-2]),
                )
                omega_pred[:, 1:-1] = dtheta / (2.0 * DT_theta)
                omega_pred[:, 0] = 0.0
                omega_pred[:, -1] = 0.0

                # -----------------------------
                # Acceleration: central second derivative
                # -----------------------------
                acc_pred_xy = torch.zeros_like(pos)
                acc_pred_xy[:, 1:-1] = (
                    pos[:, 2:] - 2.0 * pos[:, 1:-1] + pos[:, :-2]
                ) / (DT_xy ** 2)
                acc_pred_xy[:, 0] = 0.0
                acc_pred_xy[:, -1] = 0.0

                # -----------------------------
                # Angular acceleration
                # -----------------------------
                alpha_pred = torch.zeros_like(theta)
                alpha_pred[:, 1:-1] = (
                    theta[:, 2:] - 2.0 * theta[:, 1:-1] + theta[:, :-2]
                ) / (DT_theta ** 2)
                alpha_pred[:, 0] = 0.0
                alpha_pred[:, -1] = 0.0
                
                # Compute per-sample MSE losses so we can time-weight them like collision loss.
                loss_vel_xy_per_sample = F.mse_loss(vel_pred_xy, ilqr_vel_b, reduction="none").mean(dim=(1, 2))
                loss_omega_per_sample = F.mse_loss(omega_pred, ilqr_omega_b, reduction="none").mean(dim=1)
                loss_acc_xy_per_sample = F.mse_loss(acc_pred_xy, ilqr_acc_b, reduction="none").mean(dim=(1, 2))
                loss_alpha_per_sample = F.mse_loss(alpha_pred, ilqr_alpha_b, reduction="none").mean(dim=1)

                # Emphasize samples where t is close to 1, where the model prediction is closest to the target.
                velo_accel_time_weight = t ** 2  # (B,)

                loss_velo_accel_per_sample = (
                    loss_vel_xy_per_sample
                    + loss_omega_per_sample
                    + loss_acc_xy_per_sample
                    + loss_alpha_per_sample
                )
                loss_velo_accel = (velo_accel_time_weight * loss_velo_accel_per_sample).mean()
                
                loss = loss + VELO_ACCEL_CONSISTENCY_LOSS_WEIGHT * loss_velo_accel
                batch_velo_accel_losses.append(loss_velo_accel.item())

            loss_item = loss.item()
            loss_history.append(loss_item)
            batch_losses.append(loss_item)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if USE_EMA and epoch >= ema_start_epoch:
                if not ema_initialized:
                    # Late-start EMA should begin from the current trained weights,
                    # not from epoch-0 initialization.
                    ema_state_dict.update({k: v.detach().clone() for k, v in model.state_dict().items()})
                    ema_initialized = True
                update_ema_state(model, ema_state_dict, EMA_DECAY)

        # -------------------------------------------------------
        # End of epoch
        # -------------------------------------------------------
        # Compute average loss for this epoch
        epoch_loss = np.mean(batch_losses)
        epoch_losses.append(epoch_loss)

        # Track per-epoch loss components
        if SOFT_MATCHING and batch_matching_losses:
            epoch_matching_losses_track.append(np.mean(batch_matching_losses))
        if START_POINT_PENALTY and batch_start_losses:
            epoch_start_losses_track.append(np.mean(batch_start_losses) * START_END_LOCATION_LOSS_WEIGHT)
        if POSITION_LOSS and batch_position_losses:
            epoch_position_losses_track.append(np.mean(batch_position_losses) * POSITION_LOSS_WEIGHT)
        if COLLISION_LOSS and batch_collision_losses:
            epoch_collision_losses_track.append(np.mean(batch_collision_losses) * COLLISION_LOSS_WEIGHT)
        if ORIENT_LOSS and batch_orient_losses:
            epoch_orient_losses_track.append(np.mean(batch_orient_losses) * ORIENT_LOSS_WEIGHT)
        if VELO_ACCEL_CONSISTENCY_LOSS and batch_velo_accel_losses:
            epoch_velo_accel_losses_track.append(np.mean(batch_velo_accel_losses) * VELO_ACCEL_CONSISTENCY_LOSS_WEIGHT)

        scheduler_for_plateau.step(epoch_loss)
        lr = optimizer.param_groups[0]["lr"]
        lr_history.append(lr)

        elapsed_sec = time.time() - train_start_time
        elapsed_min = elapsed_sec / 60.0
        avg_epoch_sec = elapsed_sec / (epoch + 1)
        remaining_epochs = epochs - (epoch + 1)
        remaining_min = (avg_epoch_sec * remaining_epochs) / 60.0
        time_info = f" | Time: {elapsed_min:.1f}m elapsed, {remaining_min:.1f}m remaining"
        if batch_obstacle_diag_scores:
            obstacle_diag_epoch = float(np.mean(batch_obstacle_diag_scores))
            diag_info = f" | ObstacleSensitivity {obstacle_diag_epoch:.5f}"
        else:
            diag_info = ""

        # compute epoch-level collision loss if any
        if batch_collision_losses:
            epoch_collision_loss = np.mean(batch_collision_losses) * COLLISION_LOSS_WEIGHT
            coll_info = f" | Collision {epoch_collision_loss:.6f}"
        else:
            coll_info = ""

        # Epoch-level orientation and velocity/acceleration consistency info
        if batch_orient_losses:
            epoch_orient_loss = np.mean(batch_orient_losses) * ORIENT_LOSS_WEIGHT
            orient_info = f" | Orient {epoch_orient_loss:.6f}"
        else:
            orient_info = ""

        if batch_position_losses:
            epoch_position_loss = np.mean(batch_position_losses) * POSITION_LOSS_WEIGHT
            position_info = f" | Position {epoch_position_loss:.6f}"
        else:
            position_info = ""

        if batch_velo_accel_losses:
            epoch_velo_accel_loss = np.mean(batch_velo_accel_losses) * VELO_ACCEL_CONSISTENCY_LOSS_WEIGHT
            velo_info = f" | VeloAccel {epoch_velo_accel_loss:.6f}"
        else:
            velo_info = ""

        if SOFT_MATCHING and batch_matching_losses:
            epoch_matching_loss = np.mean(batch_matching_losses)

            if dist_std is not None and dist_std > 0:
                d0_current = dist_mean + GATE_K * dist_std
                temp_current = dist_std / GATE_SHARPNESS
            else:
                d0_current = 0.5
                temp_current = 0.1

            if START_POINT_PENALTY and batch_start_losses:
                epoch_start_loss = np.mean(batch_start_losses) * START_END_LOCATION_LOSS_WEIGHT
                print(
                    f"Epoch {epoch:3d} | Loss {epoch_loss:.6f} | Match {epoch_matching_loss:.6f} "
                    f"| Start-End {epoch_start_loss:.6f} | d0={d0_current:.4f} temp={temp_current:.4f} "
                    f"| Gate: mean={dist_mean:.4f} std={dist_std:.4f} | LR {lr:.2e}{time_info}{diag_info}{coll_info}{position_info}{orient_info}{velo_info}"
                )
            else:
                print(
                    f"Epoch {epoch:3d} | Loss {epoch_loss:.6f} | Match {epoch_matching_loss:.6f} "
                    f"| d0={d0_current:.4f} temp={temp_current:.4f} "
                    f"| Gate: mean={dist_mean:.4f} std={dist_std:.4f} | LR {lr:.2e}{time_info}{diag_info}{coll_info}{position_info}{orient_info}{velo_info}"
                )
        else:
            if START_POINT_PENALTY and batch_start_losses:
                epoch_start_loss = np.mean(batch_start_losses) * START_END_LOCATION_LOSS_WEIGHT
                print(f"Epoch {epoch:3d} | Loss {epoch_loss:.6f} | Start-End {epoch_start_loss:.6f} | LR {lr:.2e}{time_info}{diag_info}{coll_info}{position_info}{orient_info}{velo_info}")
            else:
                print(f"Epoch {epoch:3d} | Loss {epoch_loss:.6f} | LR {lr:.2e}{time_info}{diag_info}{coll_info}{position_info}{orient_info}{velo_info}")

        # -------------------------------------------------------
        # Optional checkpoint saving
        # -------------------------------------------------------
        if SAVE_CHECKPOINTS and ((epoch + 1) % CHECKPOINT_INTERVAL == 0):
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch + 1}.pt")
            model_state_for_ckpt = model.state_dict()
            traj_T = tau1.shape[1]
            traj_D = tau1.shape[2]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model_state_for_ckpt,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler_for_plateau.state_dict(),
                    "loss_history": loss_history,
                    "epoch_losses": epoch_losses,

                    "config": {
                    "T": traj_T,
                    "D": traj_D,
                    "diffusion_step_embed_dim": UNET_TIME_EMB_DIM,
                    "cond_embed_dim": COND_EMBED_DIM,
                    "down_dims": UNET_DOWNS,
                    "kernel_size": KERNEL_SIZE,
                    "n_groups": N_GROUPS,
                    "attention": ATTENTION,
                    "local_map_dim": getattr(model, "local_map_dim", None),
                    "coarse_local_map_dim": getattr(model, "coarse_local_map_dim", None),
                    "coord_range": getattr(model, "coord_range", None),
                    }
                },
                ckpt_path,
            )
            print(f"Saved checkpoint: {ckpt_path}")





if __name__ == "__main__":


    # -----------------------------
    # Load dataset
    # -----------------------------
    if GENERALIZED_DATASET_SETUP:
        print("Loading generalized dataset with dict conditioning and potential field paths...")

        

        dataset, dataloader = build_diverse_paths_dataloader(
            root_dir=DATA_ROOT,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,      # start with 0 first
            pin_memory=True,
            preload_to_ram=True # good if total size is manageable
        )

        print("Total samples:", len(dataset))

        batch = next(iter(dataloader))
        print("tau1:", batch[0].shape)          # (B, 104, 2)
        print("start:", batch[1].shape)         # (B, 2)
        print("end:", batch[2].shape)           # (B, 2)
        print("mask:", batch[3].shape)           # (B, 200, 200)
        print("fid:", batch[4].shape)           # (B,)
        print("phi:", batch[5].shape)           # (B, 200, 200)
        print("start_family:", batch[6].shape)  # (B, 2)
        print("end_family:", batch[7].shape)    # (B, 2)
        print("pf family paths:", batch[8].shape)     # (B, max_pf_in_batch, 104, 2)
        print("jerky family paths:", batch[9].shape)  # (B, max_jerky_in_batch, 104, 2)
        print("flag:", batch[10].shape)         # (B,)
        print("pf_counts:", batch[11].shape)    # (B,)
        print("jerky_counts:", batch[12].shape) # (B,)
        #print("obstacle_centers:", batch[16].shape) # (B, max_obs_in_batch, 2)
        #print("obstacle_counts:", batch[17].shape)  # (B,)
        print("ilqr_pose:", batch[18].shape)    # (B, 104, 3)
        print("ilqr_vel:", batch[19].shape)     # (B, 104, 2)
        print("ilqr_omega:", batch[20].shape)
        print("ilqr_acc:", batch[21].shape)     # (B, 104, 2)
        print("start_specific_ilqr_pose:", batch[25].shape)  # (B, 3)
        print("end_specific_ilqr_pose:", batch[26].shape)


        dataset_file = DATA_ROOT
        N = len(dataset)
        T = int(batch[0].shape[1])
        D = 4
        print("D is set to:", D)



    # compute per-coordinate (x,y) mean and std across dataset for stable normalization
    if NORMALIZATION:
        print("Warning: NORMALIZATION is currently skipped in generalized dataloader mode.")

    #--------------------------------------------------------------------------

    #                                CONDITIONING

    #--------------------------------------------------------------------------




    # ensure checkpoint dir exists
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Optional separate directory for final model (set to None to use CHECKPOINT_DIR)
    FINAL_CHECKPOINT_DIR = "final_model"  # e.g. "final_models"
    

    # -----------------------------
    # Instantiate model
    # -----------------------------


    model = ConditionalUnet1D(
        input_dim=D,                  # D=4 -> (x, y, sin(theta), cos(theta))
        cond_embed_dim=COND_EMBED_DIM,
        diffusion_step_embed_dim=UNET_TIME_EMB_DIM,
        down_dims=UNET_DOWNS,
        kernel_size=KERNEL_SIZE,
        n_groups=N_GROUPS,
        attention=ATTENTION,
        coord_bounds=COORD_BOUNDS,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable params: {trainable_params:,} ({trainable_params/1e6:.2f}M)")



    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = StepLR(optimizer, step_size=200, gamma=SCHEDULER_GAMMA)  # legacy scheduler

    scheduler_for_plateau = ReduceLROnPlateau(
        optimizer,
        mode='min',          # we want to minimize loss
        factor=0.5,          # LR ← LR * factor when plateau
        patience=20,         # epochs with no improvement before reducing LR
        threshold=1e-3,      # minimum change to be considered an improvement
        threshold_mode='rel',
        cooldown=0,          # epochs to wait after LR reduction
        min_lr=1e-6,         # lower bound on LR
    )
    loss_fn = nn.MSELoss()

    # Exponential Moving Average (EMA) state for model averaging
    ema_state_dict = None
    if USE_EMA:
        ema_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}


    # Print config at the start
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print("Dataset file: ", dataset_file)
    print("number of trajs is :",N)
    print(f"Dataset shape - N: {N}, T: {T}, D: {D}")
    print(f"N = number of trajectories, T = timesteps/waypoints, D = dimensions (x,y,sin(theta),cos(theta))")
    print(f"Device: {device}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Normalization for data and conditioning: {NORMALIZATION}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Scheduler Gamma (decay): {SCHEDULER_GAMMA}")
    print(f"Number of Epochs: {NUM_EPOCHS}")
    print(f"UNet time embedding dim: {UNET_TIME_EMB_DIM}")
    print(f"UNet down dims (channels per level): {UNET_DOWNS}")
    print(f"Kernel size: {KERNEL_SIZE}")
    print(f"Number of groups: {N_GROUPS}")
    print(f"Conditioning embed dim: {COND_EMBED_DIM}")
    print(f"Attention enabled: {ATTENTION}")
    print(f"EMA enabled: {USE_EMA}")
    if USE_EMA:
        print(f"EMA decay: {EMA_DECAY}")
        print(f"EMA start fraction: {EMA_START_FRACTION}")
    print("="*60 + "\n")

    train_flow_matching()

    # Save final checkpoint
    if SAVE_FINAL:
        final_path = os.path.join(CHECKPOINT_DIR, FINAL_MODEL_NAME)
        ema_start_epoch = int(NUM_EPOCHS * EMA_START_FRACTION)
        use_ema_for_final_save = USE_EMA and (NUM_EPOCHS > ema_start_epoch)
        final_model_state = ema_state_dict if use_ema_for_final_save else model.state_dict()
        torch.save({
            'epoch': NUM_EPOCHS,
            'model_state_dict': final_model_state,
            'raw_model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler_for_plateau.state_dict(),
            'loss_history': loss_history,
            'epoch_losses': epoch_losses,

            "config": {
                "T": T,
                "D": D,
                "diffusion_step_embed_dim": UNET_TIME_EMB_DIM,
                "cond_embed_dim": COND_EMBED_DIM,
                "down_dims": UNET_DOWNS,
                "kernel_size": KERNEL_SIZE,
                "n_groups": N_GROUPS,
                "attention": ATTENTION,
                "local_map_dim": getattr(model, "local_map_dim", None),
                "coarse_local_map_dim": getattr(model, "coarse_local_map_dim", None),
                "coord_range": getattr(model, "coord_range", None),
            }
        }, final_path)
        print(f"Saved final model checkpoint: {final_path}")

    # Use EMA-averaged weights for post-training sampling/visualization
    if USE_EMA:
        model.load_state_dict(ema_state_dict)

    # Plot per-batch loss (noisy)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history)
    plt.xlabel("Iteration (batch)")
    plt.ylabel("Loss")
    plt.title("Per-Batch Loss (Noisy)")
    plt.grid(True)

    # Plot per-epoch loss (smooth)
    plt.subplot(1, 2, 2)
    plt.plot(epoch_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Per-Epoch Loss (Smooth)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.plot(lr_history)
    plt.xlabel("Iteration")
    plt.ylabel("LR")
    plt.title("learning rate change")
    plt.grid(True)
    plt.show()

    # Plot individual loss components per epoch
    loss_components = []
    if epoch_matching_losses_track:
        loss_components.append(("Matching", epoch_matching_losses_track))
    if epoch_start_losses_track:
        loss_components.append(("Start-End", epoch_start_losses_track))
    if epoch_position_losses_track:
        loss_components.append(("Position", epoch_position_losses_track))
    if epoch_collision_losses_track:
        loss_components.append(("Collision", epoch_collision_losses_track))
    if epoch_orient_losses_track:
        loss_components.append(("Orient", epoch_orient_losses_track))
    if epoch_velo_accel_losses_track:
        loss_components.append(("VeloAccel", epoch_velo_accel_losses_track))

    if loss_components:
        n_components = len(loss_components)
        n_cols = 3
        n_rows = (n_components + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

        for idx, (component_name, values) in enumerate(loss_components):
            ax = axes[idx]
            ax.plot(values, linewidth=2, label=component_name)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title(f"{component_name} Loss per Epoch")
            ax.grid(True, alpha=0.3)
            ax.legend()

        # Hide unused subplots
        for idx in range(n_components, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.show()


 
