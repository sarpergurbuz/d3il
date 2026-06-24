import numpy as np
import matplotlib.pyplot as plt
import os
import time
import yaml
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import KDTree
import networkx as nx

try:
    import cupy as cp
    import cupyx.scipy.sparse as cp_sparse
    import cupyx.scipy.sparse.linalg as cp_linalg
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    from pyamg import smoothed_aggregation_solver
    PYAMG_AVAILABLE = True
except ImportError:
    PYAMG_AVAILABLE = False

from scipy.interpolate import RegularGridInterpolator
from joblib import Parallel, delayed

SAMPLE_RANDOM_START_SEEDS = True  # Set to False to sample seeds uniformly on a circle around the start point (instead of random donut)
HIGH_GRAD_POINT_SAMPLE_AMOUNT = 500  # Number of high-gradient points to sample for multimodal path generation
KNN_FOR_GRAPH = 6  # Number of nearest neighbors to connect in the graph for multimodal path generation
PATH_NUMBER_FOR_PLOTTING = 50  # Number of multimodal paths to compute and plot (set to 1 for just one path, increase for more diversity but slower computation)
DIJKSTRA_COST_VAR = 1  # Multiplier for Dijkstra edge weights (higher values encourage shorter paths, lower values allow more exploration of longer paths)
CUBIC_SMOOTHING = False  # If True, apply cubic spline smoothing to multimodal paths for nicer visualization (set to False for raw paths)
ROBUST_THRESHOLD_PARAM_FOR_SAMPLING = 0.1  # Lambda parameter for robust thresholding (med + lambda*mad) when selecting high-gradient points for multimodal path generation (adjust based on how many points you want to select; higher values select fewer points)
MAX_DIST = 0.0625  # Maximum distance for connecting neighbors in the graph for multimodal path generation (adjust based on the scale of your environment; smaller values create sparser graphs)
def generate_potential_field_trajectories(
    start_point,
    obstacles,
    end_point,
    n_traj=10,
    N_waypoints=10,
    n_output_path=None,
    sample_random_start_seeds=SAMPLE_RANDOM_START_SEEDS,
    
):
    """
    Generate trajectories from start to end point using a WALL-aware harmonic potential.


    
    - Obstacles are treated as solid walls (no-penetration) by solving Laplace(phi)=0
      on the free space with Neumann wall behavior.
    - Start/goal are enforced as Dirichlet "wells" (small disks):
        phi=0 at start disk, phi=1 at goal disk
      Then v = grad(phi), and streamlines go from start to goal.

        Obstacle input supports two formats:
        - Obstacle centers: array-like of shape (n_obs, 2) or (n_obs, >=2), using x,y columns.
        - Binary mask: square array of shape (n, n) with 0/1 or bool values.
            In this mode, the mask is used directly as solid cells.
    """

    # =========================
    # Parameters
    # =========================
    VERBOSE= True
    USE_GPU= True  # GPU slower than PyAMG for Laplace on 2D grids. Set to True only if PYAMG unavailable.
    PRINT_INTEGRATION_INFO = False  # Print trace_to_sink iteration details
    
    MARGIN_FACTOR = 1
    STEP_SIZE = 0.02

    REMOVE_COLLIDING_TRAJS_OBS_RADIUS = True
    REMOVE_DIVERGING_TRAJS = True     
    REMOVE_NOT_SUIT_D3IL_TRAJS = False
    REMOVE_JITTERY_TRAJS = False
    D3IL_SETUP = False  # If True, use rectangular goal well; if False, use circular goal well
    OBSTACLE_RADIUS = 0.03  # collision check radius (your value)
    SAFETY_MARGIN = 0.00  # additional margin for collision checking 

    # Potential/PDE grid
    resolution = 200  # Reduced from 200 for faster training (still good accuracy). Change to 200 for higher accuracy if needed.
    epsilon = 1e-12

    # Detect whether obstacles are given as a square binary mask.
    obstacles_arr = np.asarray(obstacles)
    is_obstacle_mask_input = (
        obstacles_arr.ndim == 2
        and obstacles_arr.shape[0] == obstacles_arr.shape[1]
        and np.all(np.isin(obstacles_arr, [0, 1]))
    )
    if is_obstacle_mask_input:
        # Keep grid resolution consistent with provided mask.
        resolution = int(obstacles_arr.shape[0])

    # Dirichlet "well" radii (in world units)
    START_WELL_RADIUS = 0.01
    GOAL_WELL_RADIUS = 0.01
    DIRICHLET_START_POTENTIAL = 0.0
    DIRICHLET_GOAL_POTENTIAL = 1.0    

    # Sink arrival tolerance for tracer (use goal well radius)
    r_sink = max(GOAL_WELL_RADIUS * 1.5, 0.04)

    # If n_output_path not specified, return all trajectories
    if n_output_path is None:
        n_output_path = n_traj

    def sample_seeds(start_xy, start_well_radius, n_samples, sample_random=True):
        t0 = time.time()
        if not sample_random:
            radius = start_well_radius * 1.2
            theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
            seeds = np.column_stack(
                [start_xy[0] + radius * np.cos(theta), start_xy[1] + radius * np.sin(theta)]
            )
            if VERBOSE:
                print(f"sample_seeds (uniform circle) completed in {time.time() - t0:.4f}s")
            return seeds, radius, radius

        r_min = start_well_radius * 1.2
        r_max = start_well_radius * 1.6

        r_mean = 0.5 * (r_min + r_max)
        sigma = r_mean
        cov = [[sigma**2, 0], [0, sigma**2]]

        seeds_list = []
        remaining = n_samples
        while remaining > 0:
            batch = max(remaining * 5, 1000)
            samples = np.random.multivariate_normal(start_xy, cov, batch)
            distances = np.linalg.norm(samples - start_xy, axis=1)
            accept = samples[(distances >= r_min) & (distances <= r_max)]
            if accept.size == 0:
                continue
            seeds_list.append(accept)
            remaining -= accept.shape[0]

        seeds = np.vstack(seeds_list)
        if seeds.shape[0] > n_samples:
            seeds = seeds[:n_samples]

        if VERBOSE:
            print(f"sample_seeds (random donut) completed in {time.time() - t0:.4f}s")
        return seeds, r_min, r_max

    # =========================
    # Bounds from start/end
    # =========================
    start_x, start_y = float(start_point[0]), float(start_point[1])
    end_x, end_y = float(end_point[0]), float(end_point[1])

    distance = np.sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2)
    margin = distance * MARGIN_FACTOR

    x_min = -0.5 
    x_max = 0.5 
    y_min = -0.95 
    y_max = 0.05 

    # =========================
    # Grid
    # =========================
    start_time_grid = time.time()
    x_lin = np.linspace(x_min, x_max, resolution)
    y_lin = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x_lin, y_lin)  # X,Y shape (ny,nx)

    dx = (x_max - x_min) / (resolution - 1 + epsilon)
    dy = (y_max - y_min) / (resolution - 1 + epsilon)

    # =========================
    # Build masks: solid walls + Dirichlet wells
    # =========================
    if is_obstacle_mask_input:
        solid = obstacles_arr.astype(bool, copy=True)
        inflated_solid_mask = solid.copy()
    else:
        obstacles = np.asarray(obstacles, dtype=float)
        if obstacles.ndim == 1:
            obstacles = obstacles[None, :]

        solid = np.zeros_like(X, dtype=bool)
        inflated_solid_mask = np.zeros_like(X, dtype=bool)
        for obs in obstacles:
            ox, oy = float(obs[0]), float(obs[1])
            solid |= ((X - ox) ** 2 + (Y - oy) ** 2) <= (OBSTACLE_RADIUS ** 2)  # For each obstacle center (ox, oy), mark all grid points within OBSTACLE_RADIUS as solid. |= accumulates obstacles (union)
            #this creates a boolean mask of solid cells.
            inflated_solid_mask |= ((X - ox) ** 2 + (Y - oy) ** 2) <= ((OBSTACLE_RADIUS + SAFETY_MARGIN) ** 2)

    if is_obstacle_mask_input:
        from scipy.ndimage import binary_dilation

        inflation_radius_cells = int(np.ceil(SAFETY_MARGIN / max(dx, dy)))
        if inflation_radius_cells > 0:
            structure = np.ones((2 * inflation_radius_cells + 1, 2 * inflation_radius_cells + 1), dtype=bool)
            inflated_solid_mask = binary_dilation(inflated_solid_mask, structure=structure)
    end_time_grid = time.time()
    obstacle_mask=solid.copy()  # Save obstacle mask before modifying for wells
    inflated_obstacle_mask = inflated_solid_mask.copy()
    print(f"Grid creation and solid mask completed in {end_time_grid - start_time_grid:.4f}s")
    start_well = ((X - start_x) ** 2 + (Y - start_y) ** 2) <= (START_WELL_RADIUS ** 2)
    
    # Goal well: circular (default) or rectangular (D3IL_SETUP)
    if D3IL_SETUP:
        # Goal well as a rectangle (x: 0.27 to 0.71, y: 0.45 to 0.5)
        goal_well = (X >= 0.27) & (X <= 0.71) & (Y >= 0.45) & (Y <= 0.5)
    else:
        # Goal well as a circle with radius GOAL_WELL_RADIUS
        goal_well = ((X - end_x) ** 2 + (Y - end_y) ** 2) <= (GOAL_WELL_RADIUS ** 2)

    # Dirichlet cells cannot be solid (if overlap occurs, prioritize wells)
    solid = solid & (~start_well) & (~goal_well)  # If an obstacle overlaps start/goal wells,  force the well region to be non-solid.
    inflated_solid_mask = inflated_solid_mask & (~start_well) & (~goal_well)

    # =========================

    def solve_phi_sparse(phi_init=None): 
        """
        Function to solve for potential field phi on grid with walls.

        Solve Laplace(phi)=0 on free cells with:
        - Dirichlet: phi=0 on start_well, phi=1 on goal_well
        - Neumann (no-flux): on obstacle boundaries + outer box boundary
            implemented by mirroring (blocked neighbor treated as center value).

        This builds a sparse linear system A*phi=b and solves it with CG.
        """
        solver_start = time.time()

        ny, nx = X.shape  # grid shape (number of rows, number of columns)

        assembly_start = time.time()

        # Unknowns are all NON-SOLID cells
        free = ~solid # boolean mask of free cells (not solid)
        idx_map = -np.ones((ny, nx), dtype=np.int32) #2D integer array the same size as the grid, filled with -1
        free_ij = np.argwhere(free) # gets the (row, col) indices of every True cell in "free"
        n = free_ij.shape[0]
        idx_map[free] = np.arange(n, dtype=np.int32) #assigns each free cell a unique 1D index from 0 to n-1, while solid cells remain -1

        # Dirichlet sets among unknowns
        dir_s = start_well & free # 2D boolean mask of free cells that are in start well same size as grid
        dir_g = goal_well & free # 2D boolean mask of free cells that are in goal well same size as grid
        is_dir = dir_s | dir_g  # 2D boolean mask of free cells that are dirichlet (either start or goal)


        # Build sparse matrix (vectorized assembly, then CSR for solve)
        b = np.zeros(n, dtype=np.float64) # RHS vector of size n (number of free cells) for defining equations for dirichlet rows such as phi=0 or phi=1
        b[idx_map[dir_s]] = DIRICHLET_START_POTENTIAL # Enforce phi=0 at start well (dirichlet bc), 
        b[idx_map[dir_g]] = DIRICHLET_GOAL_POTENTIAL # Enforce phi=1 at goal well (dirichlet bc)

        # Neighbor availability masks (inside grid AND not solid). these masks are sizes of the grid
        unblocked_up = np.zeros((ny, nx), dtype=bool)
        unblocked_down = np.zeros((ny, nx), dtype=bool)
        unblocked_left = np.zeros((ny, nx), dtype=bool)
        unblocked_right = np.zeros((ny, nx), dtype=bool)

        unblocked_up[1:, :] = ~solid[:-1, :]
        unblocked_down[:-1, :] = ~solid[1:, :]
        unblocked_left[:, 1:] = ~solid[:, :-1]
        unblocked_right[:, :-1] = ~solid[:, 1:]

        # Only assemble off-diagonals for non-Dirichlet rows
        active = free & (~is_dir)  # boolean mask of free cells that are not dirichlet. These are cells where Laplace equation applies.

        rows = [] #to store row indices, column indices, and data values for sparse matrix construction in COO format
        cols = []
        data = []

        # Up neighbors (j-1, i)
        mask = active & unblocked_up
        if np.any(mask): #Check if there are any True values in the mask
            j_idx, i_idx = np.nonzero(mask) #Get the (row, col) indices where mask is True .Example: if mask is True at (1,0) and (2,3), then j_idx=[1,2], i_idx=[0,3]
            p = idx_map[j_idx, i_idx] #Looks up the 1D system index for each masked cell. p=array of equation row numbers in the sparse system
            q = idx_map[j_idx - 1, i_idx] #Looks up the 1D system index for each cell's upper neighbor. q=array of column numbers corresponding to upper neighbors
            rows.append(p)
            cols.append(q)
            data.append(np.full(p.shape, -1.0))

        # Down neighbors (j+1, i)
        mask = active & unblocked_down
        if np.any(mask):
            j_idx, i_idx = np.nonzero(mask)
            p = idx_map[j_idx, i_idx]
            q = idx_map[j_idx + 1, i_idx]
            rows.append(p)
            cols.append(q)
            data.append(np.full(p.shape, -1.0))

        # Left neighbors (j, i-1)
        mask = active & unblocked_left
        if np.any(mask):
            j_idx, i_idx = np.nonzero(mask)
            p = idx_map[j_idx, i_idx]
            q = idx_map[j_idx, i_idx - 1]
            rows.append(p)
            cols.append(q)
            data.append(np.full(p.shape, -1.0))

        # Right neighbors (j, i+1)
        mask = active & unblocked_right
        if np.any(mask):
            j_idx, i_idx = np.nonzero(mask)
            p = idx_map[j_idx, i_idx]
            q = idx_map[j_idx, i_idx + 1]
            rows.append(p)
            cols.append(q)
            data.append(np.full(p.shape, -1.0))

        # Diagonal entries
        count_unblocked = (
            unblocked_up.astype(np.int32)
            + unblocked_down.astype(np.int32)
            + unblocked_left.astype(np.int32)
            + unblocked_right.astype(np.int32)
        )

        # Non-Dirichlet rows: diag = number of unblocked neighbors
        diag_active = idx_map[active]
        rows.append(diag_active)
        cols.append(diag_active)
        data.append(count_unblocked[active].astype(np.float64))

        # Dirichlet rows: identity
        diag_dir = idx_map[is_dir]
        rows.append(diag_dir)
        cols.append(diag_dir)
        data.append(np.ones(diag_dir.shape, dtype=np.float64))

        rows = np.concatenate(rows) if rows else np.array([], dtype=np.int32)
        cols = np.concatenate(cols) if cols else np.array([], dtype=np.int32)
        data = np.concatenate(data) if data else np.array([], dtype=np.float64)

        A = sp.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.float64).tocsr()

        

        # Initial guess speeds CG up a lot (warms start)
        if phi_init is None:
            # linear ramp init (same idea as before)
            gx = end_x - start_x
            gy = end_y - start_y
            gnorm2 = gx * gx + gy * gy + 1e-12
            proj = ((X - start_x) * gx + (Y - start_y) * gy) / gnorm2
            phi0 = np.clip(proj, DIRICHLET_START_POTENTIAL, DIRICHLET_GOAL_POTENTIAL)
        else:
            phi0 = np.asarray(phi_init, dtype=np.float64)

        x0 = phi0[free].reshape(-1)

        if VERBOSE:
            assembly_time = time.time() - assembly_start
            print(f"Sparse system assembly completed in {assembly_time:.4f}s")

        # Solve with AMG (preferred for Laplace) or CG fallback
        if PYAMG_AVAILABLE:
            if VERBOSE:
                print("Using AMG solver (optimal for Laplace problems)...")
            ml = smoothed_aggregation_solver(A)
            sol = ml.solve(b, x0=x0, tol=1e-4, maxiter=2000)
            info = 0  # AMG returns solution directly
        elif USE_GPU and CUPY_AVAILABLE:
            if VERBOSE:
                print("Using GPU acceleration (CuPy)...")
            # Convert to GPU arrays
            A_gpu = cp_sparse.csr_matrix(A)
            b_gpu = cp.asarray(b)
            x0_gpu = cp.asarray(x0)
            
            # Solve on GPU (CuPy uses 'tol' instead of 'rtol')
            sol_gpu, info = cp_linalg.cg(A_gpu, b_gpu, x0=x0_gpu, tol=1e-4, maxiter=2000)
            
            # Convert back to CPU
            sol = cp.asnumpy(sol_gpu)
        else:
            if VERBOSE:
                if USE_GPU:
                    print("Warning: GPU requested but CuPy not available. Using CPU CG.")
                else:
                    print("Using CPU CG solver...")
            sol, info = spla.cg(A, b, x0=x0, rtol=1e-4, maxiter=2000)

        if info != 0 and not PYAMG_AVAILABLE:
            # info > 0: did not fully converge in maxiter, still usable
            # info < 0: breakdown
            print(f"Warning: CG did not fully converge (info={info}). Consider looser rtol or higher maxiter.")

        phi = np.zeros((ny, nx), dtype=np.float64)
        phi[free] = sol

        # Enforce Dirichlet exactly
        phi[start_well] = DIRICHLET_START_POTENTIAL
        phi[goal_well] = DIRICHLET_GOAL_POTENTIAL

        # Keep phi in [0,2] for stability
        np.clip(phi, DIRICHLET_START_POTENTIAL, DIRICHLET_GOAL_POTENTIAL, out=phi)

        solver_time = time.time() - solver_start
        if VERBOSE:
            print(f"solve_phi_sparse completed in {solver_time:.4f}s")

        return phi
    


    phi = solve_phi_sparse()

    # =========================
    # Velocity field v = grad(phi)
    # Streamlines follow increasing phi: start(0) -> goal(1)
    # =========================
    dphi_dy, dphi_dx = np.gradient(phi, dy, dx)  # shapes (ny,nx)
    vx = dphi_dx
    vy = dphi_dy

    # Mask velocity inside solids to avoid NaNs in interpolation
    vx = vx.copy()
    vy = vy.copy()
    vx[solid] = np.nan
    vy[solid] = np.nan

    # Pre-compute grid spacing for fast interpolation
    dx = (x_lin[-1] - x_lin[0]) / (len(x_lin) - 1)
    dy = (y_lin[-1] - y_lin[0]) / (len(y_lin) - 1)
    x_min_grid = x_lin[0]
    y_min_grid = y_lin[0]

    def v_at(x, y):
        """Fast bilinear interpolation (much faster than RegularGridInterpolator)"""
        # Clip to grid bounds
        x = np.clip(x, x_lin[0], x_lin[-1])
        y = np.clip(y, y_lin[0], y_lin[-1])
        
        # Find grid cell indices
        ix = int(np.clip((x - x_min_grid) / dx, 0, len(x_lin) - 2))
        iy = int(np.clip((y - y_min_grid) / dy, 0, len(y_lin) - 2))
        
        # Normalized coordinates within cell (0 to 1)
        tx = (x - x_lin[ix]) / dx if dx > 0 else 0.0
        ty = (y - y_lin[iy]) / dy if dy > 0 else 0.0
        tx = np.clip(tx, 0.0, 1.0)
        ty = np.clip(ty, 0.0, 1.0)
        
        # Bilinear interpolation for vx
        vx_00 = vx[iy, ix]
        vx_10 = vx[iy, ix + 1]
        vx_01 = vx[iy + 1, ix]
        vx_11 = vx[iy + 1, ix + 1]
        
        vxi = (1 - tx) * (1 - ty) * vx_00 + tx * (1 - ty) * vx_10 + \
              (1 - tx) * ty * vx_01 + tx * ty * vx_11
        
        # Bilinear interpolation for vy
        vy_00 = vy[iy, ix]
        vy_10 = vy[iy, ix + 1]
        vy_01 = vy[iy + 1, ix]
        vy_11 = vy[iy + 1, ix + 1]
        
        vyi = (1 - tx) * (1 - ty) * vy_00 + tx * (1 - ty) * vy_10 + \
              (1 - tx) * ty * vy_01 + tx * ty * vy_11
        
        return float(vxi), float(vyi)
    
    def v_at_batch(x_arr, y_arr):
        """Vectorized bilinear interpolation for arrays of points"""
        x_arr = np.asarray(x_arr)
        y_arr = np.asarray(y_arr)

        # Handle non-finite inputs safely
        finite = np.isfinite(x_arr) & np.isfinite(y_arr)
        vxi = np.full_like(x_arr, np.nan, dtype=float)
        vyi = np.full_like(y_arr, np.nan, dtype=float)
        if not np.any(finite):
            return vxi, vyi

        x_f = x_arr[finite]
        y_f = y_arr[finite]

        # Clip to grid bounds
        x_f = np.clip(x_f, x_lin[0], x_lin[-1])
        y_f = np.clip(y_f, y_lin[0], y_lin[-1])

        # Find grid cell indices (vectorized)
        ix = np.clip(((x_f - x_min_grid) / dx).astype(int), 0, len(x_lin) - 2)
        iy = np.clip(((y_f - y_min_grid) / dy).astype(int), 0, len(y_lin) - 2)

        # Normalized coordinates within cell
        tx = (x_f - x_lin[ix]) / dx
        ty = (y_f - y_lin[iy]) / dy
        tx = np.clip(tx, 0.0, 1.0)
        ty = np.clip(ty, 0.0, 1.0)

        # Bilinear interpolation for vx
        vx_00 = vx[iy, ix]
        vx_10 = vx[iy, np.clip(ix + 1, 0, len(x_lin) - 1)]
        vx_01 = vx[np.clip(iy + 1, 0, len(y_lin) - 1), ix]
        vx_11 = vx[np.clip(iy + 1, 0, len(y_lin) - 1), np.clip(ix + 1, 0, len(x_lin) - 1)]

        vxf = (1 - tx) * (1 - ty) * vx_00 + tx * (1 - ty) * vx_10 + \
              (1 - tx) * ty * vx_01 + tx * ty * vx_11

        # Bilinear interpolation for vy
        vy_00 = vy[iy, ix]
        vy_10 = vy[iy, np.clip(ix + 1, 0, len(x_lin) - 1)]
        vy_01 = vy[np.clip(iy + 1, 0, len(y_lin) - 1), ix]
        vy_11 = vy[np.clip(iy + 1, 0, len(y_lin) - 1), np.clip(ix + 1, 0, len(x_lin) - 1)]

        vyf = (1 - tx) * (1 - ty) * vy_00 + tx * (1 - ty) * vy_10 + \
              (1 - tx) * ty * vy_01 + tx * ty * vy_11

        vxi[finite] = vxf
        vyi[finite] = vyf
        return vxi, vyi

    def trace_to_sink(
        seed,
        x_sink,
        y_sink,
        r_sink,
        dt=STEP_SIZE,
        max_steps=2000,
        x_bounds=None,
        y_bounds=None,
    ):
        """
        Returns a polyline (M,2) from seed toward sink.
        Stops when within r_sink of sink.
        """
        if x_bounds is None:
            x_bounds = (-np.inf, np.inf)
        if y_bounds is None:
            y_bounds = (-np.inf, np.inf)

        x, y = float(seed[0]), float(seed[1])
        pts = [(x, y)]

        for step in range(max_steps):
            # stop if reached sink
            dist_to_sink = np.hypot(x - x_sink, y - y_sink)
            if dist_to_sink <= r_sink:
                pts.append((x_sink, y_sink))
                if PRINT_INTEGRATION_INFO:
                    print(f"Trace completed at iteration {step} (reached sink)")
                return np.asarray(pts, dtype=float)

            vx1, vy1 = v_at(x, y)
            if not np.isfinite(vx1) or not np.isfinite(vy1):
                if PRINT_INTEGRATION_INFO:
                    print(f"Trace stopped at iteration {step} (non-finite velocity)")
                return np.asarray(pts, dtype=float)

            # Normalize direction to keep step size meaningful
            spd = np.hypot(vx1, vy1)
            if spd < 1e-12:
                if PRINT_INTEGRATION_INFO:
                    print(f"Trace stopped at iteration {step} (zero speed)")
                return np.asarray(pts, dtype=float)
            vx1 /= spd
            vy1 /= spd

            # RK4 on direction field
            k1x, k1y = vx1, vy1

            v2x, v2y = v_at(x + 0.5 * dt * k1x, y + 0.5 * dt * k1y)
            if not np.isfinite(v2x) or not np.isfinite(v2y):
                if PRINT_INTEGRATION_INFO:
                    print(f"Trace stopped at iteration {step} (RK4 k2 non-finite)")
                return np.asarray(pts, dtype=float)
            s2 = np.hypot(v2x, v2y)
            if s2 < 1e-12:
                if PRINT_INTEGRATION_INFO:
                    print(f"Trace stopped at iteration {step} (RK4 k2 zero speed)")
                return np.asarray(pts, dtype=float)
            k2x, k2y = v2x / s2, v2y / s2

            v3x, v3y = v_at(x + 0.5 * dt * k2x, y + 0.5 * dt * k2y)
            if not np.isfinite(v3x) or not np.isfinite(v3y):
                if PRINT_INTEGRATION_INFO:
                    print(f"Trace stopped at iteration {step} (RK4 k3 non-finite)")
                return np.asarray(pts, dtype=float)
            s3 = np.hypot(v3x, v3y)
            if s3 < 1e-12:
                if PRINT_INTEGRATION_INFO:
                    print(f"Trace stopped at iteration {step} (RK4 k3 zero speed)")
                return np.asarray(pts, dtype=float)
            k3x, k3y = v3x / s3, v3y / s3

            v4x, v4y = v_at(x + dt * k3x, y + dt * k3y)
            if not np.isfinite(v4x) or not np.isfinite(v4y):
                if PRINT_INTEGRATION_INFO:
                    print(f"Trace stopped at iteration {step} (RK4 k4 non-finite)")
                return np.asarray(pts, dtype=float)
            s4 = np.hypot(v4x, v4y)
            if s4 < 1e-12:
                if PRINT_INTEGRATION_INFO:
                    print(f"Trace stopped at iteration {step} (RK4 k4 zero speed)")
                return np.asarray(pts, dtype=float)
            k4x, k4y = v4x / s4, v4y / s4

            x_new = x + (dt / 6.0) * (k1x + 2 * k2x + 2 * k3x + k4x)
            y_new = y + (dt / 6.0) * (k1y + 2 * k2y + 2 * k3y + k4y)

            # safety: bounds
            if not (x_bounds[0] <= x_new <= x_bounds[1] and y_bounds[0] <= y_new <= y_bounds[1]):
                if PRINT_INTEGRATION_INFO:
                    print(f"Trace stopped at iteration {step} (out of bounds)")
                return np.asarray(pts, dtype=float)

            x, y = x_new, y_new
            pts.append((x, y))

        if PRINT_INTEGRATION_INFO:
            print(f"Trace stopped at iteration {max_steps} (max_steps reached)")
        return np.asarray(pts, dtype=float)

    def resample_polyline(points, N=200):
        pts = np.asarray(points, dtype=float)
        if pts.shape[0] < 2:
            return np.repeat(pts[:1], N, axis=0)

        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.concatenate(([0.0], np.cumsum(seg)))
        total = s[-1]
        if total < 1e-12:
            return np.repeat(pts[:1], N, axis=0)

        s_target = np.linspace(0.0, total, N)
        x = np.interp(s_target, s, pts[:, 0])
        y = np.interp(s_target, s, pts[:, 1])
        return np.column_stack([x, y])

    # =========================
    # Seeds around start
    # =========================
    start_xy = np.array([start_x, start_y], dtype=float)
    seeds, r_min, r_max = sample_seeds(start_xy, START_WELL_RADIUS, n_traj, sample_random=sample_random_start_seeds)

    def trace_and_resample(seed):
        poly = trace_to_sink(
            seed,
            end_x,
            end_y,
            r_sink,
            dt=STEP_SIZE,
            max_steps=1000,
            x_bounds=(x_min, x_max),
            y_bounds=(y_min, y_max),
        )
        return resample_polyline(poly, N=N_waypoints)
    
    def trace_batch_fast(seeds_array):
        """
        Batch trajectory tracing using vectorized velocity interpolation.
        Much faster than sequential version while maintaining RK4 accuracy.
        Returns list of trajectories.
        """
        trajectories = []
        n_traj = seeds_array.shape[0]
        
        # Initialize position and status for each trajectory
        x = seeds_array[:, 0].copy()
        y = seeds_array[:, 1].copy()
        active = np.ones(n_traj, dtype=bool)
        pts_list = [seeds_array.copy()]  # Store initial positions
        
        for step in range(1000):
            if not active.any():
                break
            
            # Get indices of active trajectories
            active_idx = np.where(active)[0]
            x_active = x[active_idx]
            y_active = y[active_idx]
            
            # Check sink arrival for active trajectories
            dist_to_sink = np.hypot(x_active - end_x, y_active - end_y)
            reached = dist_to_sink <= r_sink
            active[active_idx[reached]] = False
            
            if not active.any():
                break
            
            # Recompute active indices after filtering
            active_idx = np.where(active)[0]
            x_active = x[active_idx]
            y_active = y[active_idx]
            
            # Vectorized RK4 integration
            # k1
            vx1, vy1 = v_at_batch(x_active, y_active)
            spd1 = np.hypot(vx1, vy1)
            spd1 = np.maximum(spd1, 1e-12)
            vx1 /= spd1
            vy1 /= spd1
            k1x, k1y = vx1, vy1
            
            # k2
            vx2, vy2 = v_at_batch(x_active + 0.5 * STEP_SIZE * k1x, y_active + 0.5 * STEP_SIZE * k1y)
            spd2 = np.hypot(vx2, vy2)
            spd2 = np.maximum(spd2, 1e-12)
            vx2 /= spd2
            vy2 /= spd2
            k2x, k2y = vx2, vy2
            
            # k3
            vx3, vy3 = v_at_batch(x_active + 0.5 * STEP_SIZE * k2x, y_active + 0.5 * STEP_SIZE * k2y)
            spd3 = np.hypot(vx3, vy3)
            spd3 = np.maximum(spd3, 1e-12)
            vx3 /= spd3
            vy3 /= spd3
            k3x, k3y = vx3, vy3
            
            # k4
            vx4, vy4 = v_at_batch(x_active + STEP_SIZE * k3x, y_active + STEP_SIZE * k3y)
            spd4 = np.hypot(vx4, vy4)
            spd4 = np.maximum(spd4, 1e-12)
            vx4 /= spd4
            vy4 /= spd4
            k4x, k4y = vx4, vy4
            
            # RK4 update
            x_new = x_active + (STEP_SIZE / 6.0) * (k1x + 2*k2x + 2*k3x + k4x)
            y_new = y_active + (STEP_SIZE / 6.0) * (k1y + 2*k2y + 2*k3y + k4y)
            
            # Check bounds
            in_bounds = (x_min <= x_new) & (x_new <= x_max) & (y_min <= y_new) & (y_new <= y_max)
            
            # Update positions
            x[active_idx[in_bounds]] = x_new[in_bounds]
            y[active_idx[in_bounds]] = y_new[in_bounds]
            active[active_idx[~in_bounds]] = False
            
            # Store current state
            current_pts = np.column_stack([x, y])
            pts_list.append(current_pts.copy())
        
        # Convert trajectory history to individual trajectories
        pts_array = np.array(pts_list)  # (timesteps, n_traj, 2)
        
        # Extract per-trajectory paths
        for i in range(n_traj):
            traj_points = pts_array[:, i, :]
            trajectories.append(traj_points)
        
        return trajectories

    # Use fast batch tracing if all seeds are available
    trajectories_raw = trace_batch_fast(seeds)
    
    # Resample each trajectory to fixed length
    trajectories = [resample_polyline(traj, N=N_waypoints) for traj in trajectories_raw]
    traj_mat = np.asarray(trajectories, dtype=float)

    # =========================
    # Filter + pick shortest
    # =========================
    def trajectory_collides_with_obstacles(trajectory, obstacle_mask_grid):
        # Collision check on the same occupancy grid used by the PDE.
        x = trajectory[:, 0]
        y = trajectory[:, 1]
        ix = np.clip(np.round((x - x_min) / (x_max - x_min + epsilon) * (resolution - 1)).astype(int), 0, resolution - 1)
        iy = np.clip(np.round((y - y_min) / (y_max - y_min + epsilon) * (resolution - 1)).astype(int), 0, resolution - 1)
        return np.any(obstacle_mask_grid[iy, ix])

    def trajectory_diverges(trajectory, end_pt_xy, max_error_percent=15.0):
        goal_pos = end_pt_xy[:2]
        final_pos = trajectory[-1]
        final_distance_to_goal = np.linalg.norm(final_pos - goal_pos)
        start_to_goal_distance = np.linalg.norm(goal_pos - np.array([start_x, start_y]))
        max_allowed_error = (max_error_percent / 100.0) * start_to_goal_distance
        return final_distance_to_goal > max_allowed_error

    def trajectory_suitable_for_d3il_setup(trajectory, rect_x_min=0.27, rect_x_max=0.71, rect_y_min=0.45, rect_y_max=0.5):
        """
        Check if trajectory is suitable for d3il setup:
        - Must reach the goal rectangle (x: 0.27-0.71, y: 0.45-0.5)
        - Must not overshoot beyond y=0.5
        Returns True if suitable, False otherwise.
        """
        # Check if trajectory overshoots y value (goes beyond y_max)
        if np.any(trajectory[:, 1] > rect_y_max):
            return False
        
        # Check if trajectory reaches the rectangle
        for point in trajectory:
            x, y = point[0], point[1]
            if rect_x_min <= x <= rect_x_max and rect_y_min <= y <= rect_y_max:
                return True
        
        return False
    
    def is_path_smooth(path, outlier_mad_threshold=3.0, min_step=1e-6):
        """
        Smoothness = no locally sharp curvature spikes (jitter).
        Uses robust stats (median + MAD).
        """
        p = np.asarray(path, dtype=float)
        if p.shape[0] < 3:
            return True

        # Compute segment vectors (first derivatives for actual segment lengths)
        dx = np.diff(p[:, 0], n=1)
        dy = np.diff(p[:, 1], n=1)
        
        # Segment lengths
        seg_len = np.sqrt(dx**2 + dy**2)
        
        # Protect against duplicate points / tiny steps
        good = seg_len > min_step
        if np.count_nonzero(good) < 2:
            return True
        
        # Point energies (curvature from second derivatives)
        ddx = np.diff(p[:, 0], n=2)
        ddy = np.diff(p[:, 1], n=2)
        point_energies = ddx**2 + ddy**2
        good_energies = point_energies[good[:-1]]  # Align indexing (2nd deriv has T-2 points)
        
        # robust outlier detection (MAD)
        med = np.median(good_energies)
        mad = np.median(np.abs(good_energies - med)) + 1e-12
        robust_z = 0.6745 * (good_energies - med) / mad  # ~z-score if normal

        return not np.any(np.abs(robust_z) > outlier_mad_threshold)

    avg_distances = []
    colliding_count = 0
    diverging_count = 0
    d3il_unsuitable_count = 0
    jittery_count = 0

    for i in range(traj_mat.shape[0]):
        if REMOVE_COLLIDING_TRAJS_OBS_RADIUS:
            if trajectory_collides_with_obstacles(traj_mat[i], inflated_obstacle_mask):
                colliding_count += 1
                continue

        if REMOVE_DIVERGING_TRAJS:
            if trajectory_diverges(traj_mat[i], end_point):
                diverging_count += 1
                continue

        if REMOVE_NOT_SUIT_D3IL_TRAJS:
            if not trajectory_suitable_for_d3il_setup(traj_mat[i]):
                d3il_unsuitable_count += 1
                continue

        if REMOVE_JITTERY_TRAJS:
            if not is_path_smooth(traj_mat[i], outlier_mad_threshold=50.0):
                jittery_count += 1
                continue

        dists = np.linalg.norm(np.diff(traj_mat[i], axis=0), axis=1)
        avg_dist = np.mean(dists) if dists.size else np.inf
        avg_distances.append((avg_dist, i))

    if colliding_count > 0:
        print(f"Warning: {colliding_count} trajectories collided with obstacles and were rejected")

    if diverging_count > 0:
        print(f"Warning: {diverging_count} trajectories diverged from goal and were rejected")

    if d3il_unsuitable_count > 0:
        print(f"Warning: {d3il_unsuitable_count} trajectories did not reach rectangle or overshot y and were rejected")

    if jittery_count > 0:
        print(f"Warning: {jittery_count} trajectories were jittery and were rejected")

    avg_distances.sort(key=lambda x: x[0])
    shortest_indices = [idx for _, idx in avg_distances[:n_output_path]]
    traj_mat = traj_mat[shortest_indices]

    if len(shortest_indices) < n_output_path:
        print(
            f"Warning: Only found {len(shortest_indices)} non-colliding trajectories "
            f"out of {n_output_path} requested"
        )

    # Return trajectories and phi field
    return traj_mat, phi, X, Y, seeds, r_min, r_max, START_WELL_RADIUS, obstacle_mask

def sample_jerky_potential_field_path(phi, X, Y, start_xy, goal_xy, n_paths):
    def sample_points_high_gradient_safe(
        phi: np.ndarray,
        X: np.ndarray,
        Y: np.ndarray,
        n_points: int = HIGH_GRAD_POINT_SAMPLE_AMOUNT,
        obstacle_value: float = 0.0,
        safety_radius_cells: int = 2,     # forbid sampling within this many cells of forbidden mask
        forbidden_mask: np.ndarray | None = None,  # optionally provide your own forbidden cells
        percentile: float = 95.0,
        abs_threshold: float | None = None,
        robust_threshold_param: float = ROBUST_THRESHOLD_PARAM_FOR_SAMPLING,  # lam parameter for robust thresholding (med + lam*mad)
        alpha: float = 1.0,
        rng: np.random.Generator | None = None,
    ):
        """
        Samples points in high-gradient regions while enforcing a safety radius
        around forbidden cells (obstacles + optional start/goal or any mask).

        Returns: tuple of (grid_indices, world_coordinates)
            * grid_indices: (n_points, 2) ints (row, col)
            * world_coordinates: (n_points, 2) floats (x, y)
        """
        if rng is None:
            rng = np.random.default_rng()

        if phi.ndim != 2 or phi.shape[0] != phi.shape[1]:
            raise ValueError(f"phi must be square 2D array, got shape {phi.shape}")

        # Base forbidden: obstacles (phi == obstacle_value)
        base_forbidden = (phi == obstacle_value)

        # Add optional forbidden cells (e.g., start/goal)
        if forbidden_mask is not None:
            if forbidden_mask.shape != phi.shape:
                raise ValueError("forbidden_mask must have same shape as phi")
            forbidden = base_forbidden | forbidden_mask.astype(bool)
        else:
            forbidden = base_forbidden

        # Build allowed region by applying safety radius
        allowed = ~forbidden

        if safety_radius_cells > 0:
            from scipy.ndimage import distance_transform_edt
            # distance (in cells) to nearest forbidden cell
            dist = distance_transform_edt(~forbidden)
            allowed = allowed & (dist >= float(safety_radius_cells))

        if not np.any(allowed):
            raise ValueError("Safety radius too large: no allowed cells remain.")

        # Gradient magnitude (vectorized)
        gy, gx = np.gradient(phi.astype(np.float64))
        grad_mag = np.hypot(gx, gy) # compute gradient magnitude at each cell (resolution x resolution array)

        # Do not consider forbidden/unsafe areas (mask in-place to avoid extra array)
        grad_mag[~allowed] = 0.0

        # Threshold “interesting” gradient regions over allowed cells only
        gvals = grad_mag[allowed] # extract gradient magnitudes at allowed cells to one-dimensional array
        if gvals.size == 0:
            raise ValueError("No allowed cells for gradient sampling.")

        if abs_threshold is not None:
            thr = float(abs_threshold)
        else:
            # Faster percentile via partial selection
            #k = int((percentile / 100.0) * (gvals.size - 1))
            #thr = np.partition(gvals, k)[k]
            lam = robust_threshold_param  # robust threshold parameter
            med = np.median(gvals)
            mad = np.median(np.abs(gvals - med)) + 1e-12  # avoid 0
            thr = med + lam * mad   # lam ~ 2..6

        candidate = allowed & (grad_mag >= thr)  # the main boolean grid which shows where sampling is allowed

        # Fallback if threshold too strict
        if not np.any(candidate):
            raise ValueError(
            "No candidate cells found. Threshold too strict or field too flat."
        )

        candidate_sample_vector = np.argwhere(candidate)  # (N_cand,2) array of (row,col) of candidate cells where N_cand is number of candidate cells
        w = grad_mag[candidate] # excract gradient magnitude values at candidate cells size (N_cand,)

        # If weights degenerate, sample uniformly
        if np.all(w <= 0):
            raise ValueError(
            "All candidate gradient magnitudes are zero; cannot sample based on gradient."
        )

        if alpha != 1.0:
            w = np.power(w, alpha)
        w_sum = w.sum()
        p = w / w_sum if w_sum > 0 else None

        idx = rng.choice(candidate_sample_vector.shape[0], size=n_points, replace=False, p=p)
        sampled_points = candidate_sample_vector[idx].astype(np.int64)

        reso = phi.shape[0]
        x_lin_recovered = X[0, :]
        y_lin_recovered = Y[:, 0]

        sampled_points_xy = np.column_stack(
            [
                np.interp(sampled_points[:, 1], np.arange(reso), x_lin_recovered),
                np.interp(sampled_points[:, 0], np.arange(reso), y_lin_recovered),
            ]
        )

        # Add jitter within cell to avoid grid artifacts
        Lx = x_lin_recovered.max() - x_lin_recovered.min()
        Ly = y_lin_recovered.max() - y_lin_recovered.min()

        sigma_x = 0.0025 * Lx
        sigma_y = 0.0025 * Ly

        sampled_points_xy += np.column_stack([  #add Gaussian noise to each sampled point
            np.random.normal(0.0, sigma_x, size=sampled_points_xy.shape[0]),
            np.random.normal(0.0, sigma_y, size=sampled_points_xy.shape[0]),
        ])

        return sampled_points_xy

    def convert_to_cubic_spline(path, num_points=104):
        """
        Smoothen a path using cubic spline interpolation.
        
        Args:
            path: (N, 2) array of waypoints
            num_points: number of points to sample along the spline
        
        Returns:
            (num_points, 2) array of smoothed path points
        """
        from scipy.interpolate import CubicSpline
        
        if path is None or len(path) < 2:
            return path
        
        # Compute cumulative distance along path
        distances = np.zeros(len(path))
        for i in range(1, len(path)):
            distances[i] = distances[i-1] + np.linalg.norm(path[i] - path[i-1])
        
        # Handle degenerate case (all points identical)
        if distances[-1] < 1e-12:
            return np.tile(path[0], (num_points, 1))
        
        # Create cubic spline for x and y separately
        cs_x = CubicSpline(distances, path[:, 0])
        cs_y = CubicSpline(distances, path[:, 1])
        
        # Sample along the spline
        s_new = np.linspace(0, distances[-1], num_points)
        x_new = cs_x(s_new)
        y_new = cs_y(s_new)
        
        return np.column_stack([x_new, y_new])

    def resample_path(path, num_points=104):
        """
        Resample a path to a fixed number of points using linear interpolation.
        
        Args:
            path: (N, 2) array of waypoints
            num_points: number of points to resample to (default 104)
        
        Returns:
            (num_points, 2) array of resampled path points
        """
        path = np.asarray(path, dtype=float)
        if path.shape[0] < 2:
            return np.repeat(path[:1], num_points, axis=0)
        
        # Compute cumulative distance along path
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        s = np.concatenate(([0.0], np.cumsum(seg)))
        total = s[-1]
        
        if total < 1e-12:
            return np.repeat(path[0], num_points, axis=0)
        
        # Linear interpolation along cumulative distance
        s_target = np.linspace(0.0, total, num_points)
        x = np.interp(s_target, s, path[:, 0])
        y = np.interp(s_target, s, path[:, 1])
        
        return np.column_stack([x, y])

    def get_multimodal_path_from_grad_sampled_points(
        sampled_points_xy, start_xy, goal_xy, phi, X, Y,
        k_neighbors=KNN_FOR_GRAPH, obstacle_value=0.0, use_edge_safety_check=False
    ):
        nodes = np.vstack([start_xy, sampled_points_xy, goal_xy])
        n_total = len(nodes)
        
        # Pre-calculate grid limits for O(1) coordinate mapping
        x_min, x_max = X.min(), X.max()
        y_min, y_max = Y.min(), Y.max()
        rows, cols = phi.shape

        tree = KDTree(nodes)
        distances, indices = tree.query(nodes, k=k_neighbors + 1,distance_upper_bound=MAX_DIST)

        G = nx.Graph()

        # Optimized safety check: No np.argmin calls
        # def is_edge_safe(p1, p2):
        #     for t in [0.2, 0.4, 0.6, 0.8]: # Check 4 points along edge
        #         pt = p1 + t * (p2 - p1)
        #         # Math-based mapping to indices
        #         c = int((pt[0] - x_min) / (x_max - x_min) * (cols - 1))
        #         r = int((pt[1] - y_min) / (y_max - y_min) * (rows - 1))
                
        #         if 0 <= r < rows and 0 <= c < cols:
        #             if phi[r, c] == obstacle_value: return False
        #         else: return False # Out of bounds
        #     return True

        # Build graph with randomized weights (Option A)
        for i in range(n_total):
            for j_idx, dist in zip(indices[i], distances[i]):

                # 1. SKIP THE GHOST NODES (This stops the IndexError)
                if j_idx == n_total: continue

                if i >= j_idx: continue # Avoid double-checking edges
                
                #p1, p2 = nodes[i], nodes[j_idx]
                
                # Only check safety if enabled
                # if use_edge_safety_check:
                #     if not is_edge_safe(p1, p2):
                #         continue
                
                # Apply the Option A randomization
                jittered_weight = dist + ( np.random.uniform(0, DIJKSTRA_COST_VAR))  # was before dist* (1+ np.random.uniform(0, DIJKSTRA_COST_VAR))
                G.add_edge(i, j_idx, weight=jittered_weight)

        try:
            path_idx = nx.shortest_path(G, 0, n_total - 1, weight='weight')
            return nodes[path_idx]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    paths = []
    attempts = 0
    max_attempts = max(n_paths * 50, 1000)
    last_sampled_points_xy = None
    while len(paths) < n_paths:
        attempts += 1
        if attempts > max_attempts:
            print(f"Warning: Failed to generate {n_paths} paths after {max_attempts} attempts. Returning empty array.")
            break
        sampled_points_xy = sample_points_high_gradient_safe(phi, X, Y)
        last_sampled_points_xy = sampled_points_xy
        path = get_multimodal_path_from_grad_sampled_points(
            sampled_points_xy, start_xy, goal_xy, phi, X, Y
        )
        if path is None:
            continue
        paths.append(resample_path(path, num_points=104))
    
    # If we didn't get enough paths, return empty
    if len(paths) < n_paths:
        if n_paths == 1:
            return np.array([]), last_sampled_points_xy
        return np.array([]), last_sampled_points_xy
    
    print(f"{attempts} attempts tried to reach desired number of paths.")

    if n_paths == 1:
        return paths[0], last_sampled_points_xy

    return paths, last_sampled_points_xy
    
    


# ================================================================================
# Script execution (only runs when script is executed directly)
# ================================================================================
if __name__ == "__main__":
    # Load all params once
    yaml_path = "params/potential_field_paths_params.yaml"
    params_main = {}
    if os.path.exists(yaml_path):
        with open(yaml_path, "r") as f:
            params_main = yaml.safe_load(f) or {}
    else:
        print(f"Warning: {yaml_path} not found. Using defaults.")
        params_main = {}

    # Parameters (Q values kept for compatibility, but NOT used for obstacles anymore)
    Q_val = float(params_main.get("Q_VAL", 0.5))
    starter_Q = float(params_main.get("STARTER_Q", 2.0))
    

    # Obstacles
    if "obstacles" in params_main:
        obstacles = np.array(params_main["obstacles"], dtype=float)
        print(f"Loaded obstacles from YAML: {obstacles.shape[0]} obstacles")
    else:
        obstacles = np.array(
            [
                [0.0, 0.4, Q_val],
                [0.1, 0.8, Q_val],
                [0.2, 0.6, Q_val],
                [-0.2, 0.6, Q_val],
            ],
            dtype=float,
        )
        print("Using default obstacles")

    sx, sy = params_main["start_point"][0]
    ex, ey = params_main["goal_point"][0]

    # Start and end points keep 3 entries for compatibility, but only x,y are used
    start_point = np.array([sx, sy, starter_Q], dtype=float)
    end_point = np.array([ex, ey, -4 * Q_val - starter_Q], dtype=float)

    # Generate trajectories
    n_traj = 100
    N_waypoints = 104
    n_output_path = 60


    print("Starting trajectory generation...")
    start_time = time.time()

    traj_mat, phi, X, Y, seeds, r_min, r_max, start_well_radius, obstacle_mask = generate_potential_field_trajectories(
        start_point, obstacles, end_point, n_traj, N_waypoints, n_output_path
    )

    end_time = time.time()
    print(f"Trajectory generation completed in {end_time - start_time:.2f} seconds")

    # Plot seeds with r_min, r_max, and start well radius
    plt.figure(figsize=(8, 8))
    plt.scatter(seeds[:, 0], seeds[:, 1], s=20, c="cyan", edgecolors="black", linewidths=0.5, label="Seeds")
    start_circle = plt.Circle((start_point[0], start_point[1]), start_well_radius, fill=False, color="green", linewidth=2, label="Start well")
    rmin_circle = plt.Circle((start_point[0], start_point[1]), r_min, fill=False, color="orange", linestyle="--", linewidth=2, label="r_min")
    rmax_circle = plt.Circle((start_point[0], start_point[1]), r_max, fill=False, color="red", linestyle="--", linewidth=2, label="r_max")
    ax = plt.gca()
    ax.add_patch(start_circle)
    ax.add_patch(rmin_circle)
    ax.add_patch(rmax_circle)
    plt.title("Seed distribution with start well and donut bounds")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.legend()
    plt.tight_layout()
    plt.show()

    sampled_points_start = time.time()
    
    # Compute multimodal paths using the consolidated helper
    multimodal_paths, sampled_points_xy = sample_jerky_potential_field_path(
        phi, X, Y, start_point[:2], end_point[:2], PATH_NUMBER_FOR_PLOTTING
    )
    if PATH_NUMBER_FOR_PLOTTING == 1:
        multimodal_paths = [multimodal_paths] if multimodal_paths is not None else []
    
    if multimodal_paths:
        print(f"Generated {len(multimodal_paths)} paths in {time.time() - sampled_points_start:.4f}s")
    else:
        print("Warning: No paths found from start to goal")

    # Save multimodal (jerky) paths for initialization
    os.makedirs("potential_field_paths", exist_ok=True)
    jerky_paths_array = np.asarray(multimodal_paths, dtype=float) if multimodal_paths else np.empty((0, 0, 2), dtype=float)
    np.save("potential_field_paths/jerky_potential_paths_for_initialization.npy", jerky_paths_array)
    print("Saved jerky paths to 'potential_field_paths/jerky_potential_paths_for_initialization.npy'")
    
    # Save trajectories to file
    os.makedirs("potential_field_paths", exist_ok=True)
    np.save("potential_field_paths/traj_matrix_from_potential_field_with_walls_play.npy", traj_mat)
    print("Saved trajectories to 'potential_field_paths/traj_matrix_from_potential_field_with_walls_play.npy'")
    print(f"Shape: {traj_mat.shape}")

    # Define obstacle radius for plotting
    obstacle_radius = 0.03

    # Plot phi field
    plt.figure(figsize=(10, 8))
    cf = plt.contourf(X, Y, phi, levels=50, cmap="viridis")
    plt.colorbar(cf, label="Phi")
    
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(
        sampled_points_xy[:, 0],
        sampled_points_xy[:, 1],
        s=12,
        c="white",
        edgecolors="black",
        linewidths=0.3,
        alpha=0.9,
        label="Sampled points",
    )
    
    # Plot all multimodal paths with same color (black)
    if multimodal_paths:
        for idx, path in enumerate(multimodal_paths):
            plt.plot(
                path[:, 0],
                path[:, 1],
                color="black",
                linewidth=1.0,
                alpha=0.7,
            )
            plt.plot(
                path[:, 0],
                path[:, 1],
                "o",
                color="black",
                markersize=2,
                alpha=0.5,
            )
    
    plt.title(f"Potential Field (Phi) with {len(multimodal_paths)} Multimodal Paths")
    plt.legend(loc="upper left", fontsize=8, ncol=2)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()

    # Plot computed trajectories
    plt.figure(figsize=(10, 8))
    for i in range(traj_mat.shape[0]):
        plt.plot(traj_mat[i, :, 0], traj_mat[i, :, 1], "b.", alpha=0.3, markersize=2)
        mid_idx = N_waypoints // 2
        mid_x, mid_y = traj_mat[i, mid_idx, 0], traj_mat[i, mid_idx, 1]
        #plt.text(mid_x, mid_y, str(i), fontsize=7, ha="center", va="center")

    plt.plot(start_point[0], start_point[1], "go", markersize=10, label="Start")
    plt.plot(end_point[0], end_point[1], "rs", markersize=10, label="Goal")

    # Mark obstacles + obstacle circles (collision radius)
    for obs in obstacles:
        plt.plot(obs[0], obs[1], "rx", markersize=8)
    obstacle_radius = 0.03
    for obs in obstacles:
        circle = plt.Circle(
            (obs[0], obs[1]),
            obstacle_radius,
            fill=False,
            color="orange",
            linewidth=1.5,
            linestyle="--",
        )
        plt.gca().add_patch(circle)

    distance = np.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2)
    margin = distance * 1
    x_min_plot = -1 #min(start_point[0], end_point[0]) - margin
    x_max_plot = 1 #max(start_point[0], end_point[0]) + margin
    y_min_plot = -1 #min(start_point[1], end_point[1]) - margin
    y_max_plot = 1.5 #max(start_point[1], end_point[1]) + margin

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Computed Trajectories with WALL Obstacles (n={n_traj})")
    plt.legend()
    plt.grid(False)
    plt.xlim([x_min_plot, x_max_plot])
    plt.ylim([y_min_plot, y_max_plot])
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()

    # Plot randomly sampled 100 trajectories (optional)
    PLOT_SAMPLED_TRAJ = False
    if PLOT_SAMPLED_TRAJ:
        n_sample = min(100, traj_mat.shape[0])
        sampled_indices = np.random.choice(traj_mat.shape[0], size=n_sample, replace=False)
        
        plt.figure(figsize=(10, 8))
        for idx in sampled_indices:
            plt.plot(traj_mat[idx, :, 0], traj_mat[idx, :, 1], "b-", alpha=0.5, linewidth=1.5)

        plt.plot(start_point[0], start_point[1], "go", markersize=10, label="Start")
        plt.plot(end_point[0], end_point[1], "rs", markersize=10, label="Goal")

        # Mark obstacles
        for obs in obstacles:
            plt.plot(obs[0], obs[1], "rx", markersize=8)
            circle = plt.Circle(
                (obs[0], obs[1]),
                obstacle_radius,
                fill=False,
                color="orange",
                linewidth=1.5,
                linestyle="--",
            )
            plt.gca().add_patch(circle)

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Randomly Sampled 100 Trajectories")
        plt.legend()
        plt.grid(False)
        plt.xlim([x_min_plot, x_max_plot])
        plt.ylim([y_min_plot, y_max_plot])
        plt.gca().set_aspect("equal", adjustable="box")
        plt.show()
