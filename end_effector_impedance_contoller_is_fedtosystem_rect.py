import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D


# ============================================================
# 0) Obstacles
# ============================================================

obstacles_xy_raw = np.array([
    [-0.32, -0.45], [-0.288, -0.45], [-0.256, -0.45], [-0.224, -0.45], [-0.192, -0.45],
    [-0.064, -0.45], [-0.032, -0.45], [0.0, -0.45], [0.032, -0.45], [0.064, -0.45],
    [-0.192, -0.322], [-0.16, -0.322], [-0.128, -0.322], [-0.096, -0.322], [-0.064, -0.322],
    [0.064, -0.322], [0.096, -0.322], [0.128, -0.322], [0.16, -0.322], [0.192, -0.322],
    [0.064, -0.706], [0.064, -0.674], [0.064, -0.642], [0.064, -0.61], [0.064, -0.578],
    [-0.192, -0.578], [-0.192, -0.546], [-0.192, -0.514], [-0.192, -0.482],
    [-0.064, -0.578], [-0.064, -0.546], [-0.064, -0.514], [-0.064, -0.482],
    [0.192, -0.578], [0.192, -0.546], [0.192, -0.514], [0.192, -0.482], [0.192, -0.45],
    [-0.064, -0.418], [-0.064, -0.386], [-0.064, -0.354],
    [0.192, -0.418], [0.192, -0.386], [0.192, -0.354],
    [0.064, -0.29], [0.064, -0.258], [0.064, -0.226], [0.064, -0.194],
], dtype=np.float64)

OBSTACLE_RADIUS = 0.03
SAFETY_MARGIN = 0.01

# Rectangular end-effector footprint
RECT_LENGTH = 0.1   # along local x-axis
RECT_WIDTH = 0.01    # along local y-axis
GHOST_MARGIN_LENGTH = 0.0  # extra margin added to rectangle dimensions during optimization to encourage more conservative solutions


# ============================================================
# 1) Helpers
# ============================================================

def format_pose_array(name, arr, precision=8):
    arr = np.asarray(arr, dtype=np.float64)
    lines = [f"{name} = np.array(["]
    for row in arr:
        vals = []
        for v in row:
            vals.append(f"{v:.{precision}f}".rstrip("0").rstrip("."))
        lines.append("    [" + ", ".join(vals) + "],")
    lines.append("])")
    return "\n".join(lines)


def compute_ref_velocity(path_ref, dt):
    path_ref = np.asarray(path_ref, dtype=np.float64)
    T = path_ref.shape[0]
    v_ref = np.zeros_like(path_ref)

    v_ref[0] = (path_ref[1] - path_ref[0]) / dt
    v_ref[-1] = (path_ref[-1] - path_ref[-2]) / dt

    for k in range(1, T - 1):
        v_ref[k] = (path_ref[k + 1] - path_ref[k - 1]) / (2.0 * dt)

    return v_ref


def compute_path_theta(path_ref):
    path_ref = np.asarray(path_ref, dtype=np.float64)
    T = path_ref.shape[0]
    theta = np.zeros(T)

    for k in range(T):
        if k == 0:
            tangent = path_ref[1] - path_ref[0]
        elif k == T - 1:
            tangent = path_ref[-1] - path_ref[-2]
        else:
            tangent = path_ref[k + 1] - path_ref[k - 1]

        theta[k] = np.arctan2(tangent[1], tangent[0])

    return np.unwrap(theta)


def compute_path_curvature(path):
    path = np.asarray(path, dtype=np.float64)
    T = path.shape[0]
    kappa = np.zeros(T)

    for i in range(1, T - 1):
        p_prev = path[i - 1]
        p = path[i]
        p_next = path[i + 1]

        dp = p_next - p
        ddp = p_next - 2.0 * p + p_prev

        denom = np.linalg.norm(dp) ** 2 + 1e-8
        kappa[i] = np.linalg.norm(ddp) / denom

    if T > 2:
        kappa[0] = kappa[1]
        kappa[-1] = kappa[-2]

    return kappa


def soft_bound_cost_grad_hess(z, zmax, w):
    cost = 0.0
    grad = np.zeros_like(z)
    hess = np.zeros((z.size, z.size))

    for i in range(z.size):
        violation = abs(z[i]) - zmax
        if violation > 0.0:
            s = 1.0 if z[i] >= 0 else -1.0
            cost += w * violation**2
            grad[i] += 2.0 * w * violation * s
            hess[i, i] += 2.0 * w

    return cost, grad, hess


def rectangle_circle_collision_cost_vec(q, obstacles, rect_length, rect_width, circle_radius, w_obs):
    """
    q = [px, py, theta]

    Rectangle is centered at p and rotated by theta.
    Obstacles are circles.
    Cost is quadratic penetration:
        w_obs * max(0, circle_radius - distance(circle_center, rectangle))^2

    distance is computed in rectangle local frame.
    """
    px, py, theta = q
    p = np.array([px, py])

    cth = np.cos(theta)
    sth = np.sin(theta)

    # world -> local rotation: R^T
    R_T = np.array([
        [cth, sth],
        [-sth, cth],
    ])

    hx = rect_length / 2.0
    hy = rect_width / 2.0

    rel_world = obstacles - p
    rel_local = rel_world @ R_T.T

    closest = np.empty_like(rel_local)
    closest[:, 0] = np.clip(rel_local[:, 0], -hx, hx)
    closest[:, 1] = np.clip(rel_local[:, 1], -hy, hy)

    delta = rel_local - closest
    dist = np.linalg.norm(delta, axis=1)

    # If obstacle center is inside rectangle, use negative clearance proxy
    inside = dist < 1e-9
    if np.any(inside):
        dx_inside = hx - np.abs(rel_local[inside, 0])
        dy_inside = hy - np.abs(rel_local[inside, 1])
        dist[inside] = -np.minimum(dx_inside, dy_inside)

    violation = circle_radius - dist
    active = violation > 0.0

    return w_obs * np.sum(violation[active] ** 2)


def rectangle_circle_collision_cost(q, obstacles, rect_length, rect_width, circle_radius, w_obs):
    return rectangle_circle_collision_cost_vec(
        q, obstacles, rect_length, rect_width, circle_radius, w_obs
    )


def rectangle_obstacle_cost_grad_hess(
    q,
    obstacles,
    rect_length,
    rect_width,
    circle_radius,
    w_obs,
    eps_fd=1e-5,
    hess_scale=1.0,
):
    """
    Numerical gradient of rectangle-circle obstacle cost wrt [px, py, theta].
    Hessian uses a cheap Gauss-Newton-style PSD approximation.
    """
    q = np.asarray(q, dtype=np.float64)
    n = 3

    def f(qq):
        return rectangle_circle_collision_cost_vec(
            qq, obstacles, rect_length, rect_width, circle_radius, w_obs
        )

    f0 = f(q)

    grad = np.zeros(n)
    hess = np.zeros((n, n))

    for i in range(n):
        dq = np.zeros(n)
        dq[i] = eps_fd
        fp = f(q + dq)
        fm = f(q - dq)
        grad[i] = (fp - fm) / (2.0 * eps_fd)

    grad_norm_sq = float(np.dot(grad, grad))
    hess_psd = hess_scale * np.outer(grad, grad) / (grad_norm_sq + 1e-8)
    hess_psd += 1e-8 * np.eye(n)

    return f0, grad, hess_psd


# ============================================================
# 2) Dynamics: 2D rectangle pose with jerk input
# ============================================================

def dynamics_matrices_pose(dt):
    """
    State:
        x = [
            px, py, theta,
            vx, vy, omega,
            ax, ay, alpha
        ]

    Control:
        u = [jx, jy, jtheta]

    Triple-integrator dynamics for x, y, theta:
        pose_next = pose + dt*vel + 0.5*dt^2*acc + 1/6*dt^3*jerk
        vel_next  = vel  + dt*acc + 0.5*dt^2*jerk
        acc_next  = acc  + dt*jerk
    """
    A = np.eye(9)
    B = np.zeros((9, 3))

    A[0:3, 3:6] = dt * np.eye(3)
    A[0:3, 6:9] = 0.5 * dt**2 * np.eye(3)

    A[3:6, 6:9] = dt * np.eye(3)

    B[0:3, :] = (1.0 / 6.0) * dt**3 * np.eye(3)
    B[3:6, :] = 0.5 * dt**2 * np.eye(3)
    B[6:9, :] = dt * np.eye(3)

    return A, B


def rollout_pose(x0, us, A, B):
    Tm1 = us.shape[0]
    xs = np.zeros((Tm1 + 1, 9))
    xs[0] = x0

    for k in range(Tm1):
        xs[k + 1] = A @ xs[k] + B @ us[k]

    return xs


# ============================================================
# 3) iLQR optimizer: rectangle + orientation
# ============================================================

def optimize_rectangle_pose_ilqr(
    path_ref,
    obstacles_xy,
    obstacle_radius=0.03,
    safety_margin=0.01,
    rect_length=0.075,
    rect_width=0.035,
    # ghost margins applied only during optimization (not used for plotting)
    rect_margin_length= GHOST_MARGIN_LENGTH,
    rect_margin_width= GHOST_MARGIN_LENGTH,
    dt=0.06,

    # impedance-inspired parameters for translational task-space
    m_eff=5.0,
    k_imp=250.0,
    d_imp=30.0,

    # bounds
    v_max=0.5,
    omega_max=2.0,
    a_max=0.5,
    alpha_max=5.0,
    j_max=2.0,
    jtheta_max=20.0,

    # costs
    w_track=800000.0,
    w_theta_ref=10000.0,

    w_vel=1.0,
    w_omega=1.0,
    w_acc=5.0,
    w_alpha=1.0,
    w_jerk=50.0,
    w_jtheta=10.0,

    w_impedance_acc=100.0,
    w_terminal_impedance_acc=1e9,

    w_curve_vel=100.0,

    w_obs=5e8,
    w_bound=1e8,

    w_terminal_track=1e8,
    w_terminal_theta=1e4,
    w_terminal_vel=1e10,
    w_terminal_omega=1e8,
    w_terminal_acc=1e9,
    w_terminal_alpha=1e8,

    boundary_window=30,
    boundary_gain=20.0,

    max_iter=100,
    reg=1e-6,
):
    path_ref = np.asarray(path_ref, dtype=np.float64)
    obstacles_xy = np.asarray(obstacles_xy, dtype=np.float64)

    T = path_ref.shape[0]
    nx, nu = 9, 3

    inflated_obstacle_radius = obstacle_radius + safety_margin

    # Use a ghost (inflated) rectangle during optimization collision checks
    used_rect_length = rect_length + rect_margin_length
    used_rect_width = rect_width + rect_margin_width

    v_ref = compute_ref_velocity(path_ref, dt)
    v_ref[-3:] = 0.0

    theta_ref = compute_path_theta(path_ref)
    kappa_ref = compute_path_curvature(path_ref)

    A, B = dynamics_matrices_pose(dt)

    alpha_k = k_imp / m_eff
    alpha_d = d_imp / m_eff

    x0 = np.zeros(nx)
    x0[0:2] = path_ref[0]
    x0[2] = theta_ref[0]
    x0[3:6] = 0.0
    x0[6:9] = 0.0

    us = np.zeros((T - 1, nu))

    def stage_cost_only(x, u, idx):
        """
        Pure cost computation: no derivatives, fast for line search.
        """
        p = x[0:2]
        theta = x[2]
        v = x[3:5]
        omega = x[5]
        a = x[6:8]
        alpha = x[8]
        j = u[0:2]
        jtheta = u[2]

        pref = path_ref[idx]
        vref = v_ref[idx]
        thref = theta_ref[idx]
        kappa = kappa_ref[idx]

        edge_dist = min(idx, (T - 1) - idx)
        if boundary_window > 0 and edge_dist < boundary_window:
            phase = 1.0 - edge_dist / float(boundary_window)
            edge_scale = 1.0 + boundary_gain * phase**2
        else:
            edge_scale = 1.0

        cost = 0.0

        # Position tracking
        e = p - pref
        cost += w_track * np.dot(e, e)
        e_theta = theta - thref
        cost += w_theta_ref * e_theta**2

        # Smoothness
        cost += (w_vel * edge_scale) * np.dot(v, v)
        cost += (w_omega * edge_scale) * omega**2
        cost += (w_acc * edge_scale) * np.dot(a, a)
        cost += (w_alpha * edge_scale) * alpha**2
        cost += (w_jerk * edge_scale) * np.dot(j, j)
        cost += (w_jtheta * edge_scale) * jtheta**2

        # Impedance consistency for translation only
        pdd_imp = a + alpha_k * (pref - p) + alpha_d * (vref - v)
        cost += w_impedance_acc * np.dot(pdd_imp, pdd_imp)

        # Curvature-aware slowing
        curve_weight = w_curve_vel * kappa
        cost += curve_weight * np.dot(v, v)

        # Rectangle-circle obstacle penalty (direct cost, no FD gradients)
        q_rect = np.array([p[0], p[1], theta])
        c_obs = rectangle_circle_collision_cost_vec(
            q_rect,
            obstacles_xy,
            used_rect_length,
            used_rect_width,
            inflated_obstacle_radius,
            w_obs,
        )
        cost += c_obs

        # Soft bounds (cost only)
        c_v, _, _ = soft_bound_cost_grad_hess(v, v_max, w_bound)
        c_a, _, _ = soft_bound_cost_grad_hess(a, a_max, w_bound)
        c_j, _, _ = soft_bound_cost_grad_hess(j, j_max, w_bound)
        c_om, _, _ = soft_bound_cost_grad_hess(np.array([omega]), omega_max, w_bound)
        c_al, _, _ = soft_bound_cost_grad_hess(np.array([alpha]), alpha_max, w_bound)
        c_jt, _, _ = soft_bound_cost_grad_hess(np.array([jtheta]), jtheta_max, w_bound)

        cost += c_v + c_a + c_j + c_om + c_al + c_jt

        return cost

    def stage_cost_derivatives(x, u, idx):
        p = x[0:2]
        theta = x[2]

        v = x[3:5]
        omega = x[5]

        a = x[6:8]
        alpha = x[8]

        j = u[0:2]
        jtheta = u[2]

        pref = path_ref[idx]
        vref = v_ref[idx]
        thref = theta_ref[idx]
        kappa = kappa_ref[idx]

        edge_dist = min(idx, (T - 1) - idx)
        if boundary_window > 0 and edge_dist < boundary_window:
            phase = 1.0 - edge_dist / float(boundary_window)
            edge_scale = 1.0 + boundary_gain * phase**2
        else:
            edge_scale = 1.0

        lx = np.zeros(nx)
        lu = np.zeros(nu)
        lxx = np.zeros((nx, nx))
        luu = np.zeros((nu, nu))
        lux = np.zeros((nu, nx))

        cost = 0.0

        # ----------------------------------------------------
        # Position tracking
        # ----------------------------------------------------
        e = p - pref
        cost += w_track * np.dot(e, e)
        lx[0:2] += 2.0 * w_track * e
        lxx[0:2, 0:2] += 2.0 * w_track * np.eye(2)

        # Weak orientation reference along path tangent
        e_theta = theta - thref
        cost += w_theta_ref * e_theta**2
        lx[2] += 2.0 * w_theta_ref * e_theta
        lxx[2, 2] += 2.0 * w_theta_ref

        # ----------------------------------------------------
        # Smoothness
        # ----------------------------------------------------
        cost += (w_vel * edge_scale) * np.dot(v, v)
        lx[3:5] += 2.0 * w_vel * edge_scale * v
        lxx[3:5, 3:5] += 2.0 * w_vel * edge_scale * np.eye(2)

        cost += (w_omega * edge_scale) * omega**2
        lx[5] += 2.0 * w_omega * edge_scale * omega
        lxx[5, 5] += 2.0 * w_omega * edge_scale

        cost += (w_acc * edge_scale) * np.dot(a, a)
        lx[6:8] += 2.0 * w_acc * edge_scale * a
        lxx[6:8, 6:8] += 2.0 * w_acc * edge_scale * np.eye(2)

        cost += (w_alpha * edge_scale) * alpha**2
        lx[8] += 2.0 * w_alpha * edge_scale * alpha
        lxx[8, 8] += 2.0 * w_alpha * edge_scale

        cost += (w_jerk * edge_scale) * np.dot(j, j)
        lu[0:2] += 2.0 * w_jerk * edge_scale * j
        luu[0:2, 0:2] += 2.0 * w_jerk * edge_scale * np.eye(2)

        cost += (w_jtheta * edge_scale) * jtheta**2
        lu[2] += 2.0 * w_jtheta * edge_scale * jtheta
        luu[2, 2] += 2.0 * w_jtheta * edge_scale

        # ----------------------------------------------------
        # Impedance consistency for translation only
        # ----------------------------------------------------
        pdd_imp = a + alpha_k * (pref - p) + alpha_d * (vref - v)

        cost += w_impedance_acc * np.dot(pdd_imp, pdd_imp)

        C = np.zeros((2, nx))
        C[:, 0:2] = -alpha_k * np.eye(2)
        C[:, 3:5] = -alpha_d * np.eye(2)
        C[:, 6:8] = np.eye(2)

        lx += 2.0 * w_impedance_acc * C.T @ pdd_imp
        lxx += 2.0 * w_impedance_acc * C.T @ C

        # ----------------------------------------------------
        # Curvature-aware slowing
        # ----------------------------------------------------
        curve_weight = w_curve_vel * kappa
        cost += curve_weight * np.dot(v, v)
        lx[3:5] += 2.0 * curve_weight * v
        lxx[3:5, 3:5] += 2.0 * curve_weight * np.eye(2)

        # ----------------------------------------------------
        # Rectangle-circle obstacle penalty wrt [px, py, theta]
        # ----------------------------------------------------
        q_rect = np.array([p[0], p[1], theta])

        c_obs, g_obs, H_obs = rectangle_obstacle_cost_grad_hess(
            q=q_rect,
            obstacles=obstacles_xy,
            rect_length=used_rect_length,
            rect_width=used_rect_width,
            circle_radius=inflated_obstacle_radius,
            w_obs=w_obs,
        )

        cost += c_obs

        lx[0] += g_obs[0]
        lx[1] += g_obs[1]
        lx[2] += g_obs[2]

        idxs = np.array([0, 1, 2])
        for ii in range(3):
            for jj in range(3):
                lxx[idxs[ii], idxs[jj]] += H_obs[ii, jj]

        # ----------------------------------------------------
        # Soft bounds
        # ----------------------------------------------------
        c_v, g_v, H_v = soft_bound_cost_grad_hess(v, v_max, w_bound)
        c_a, g_a, H_a = soft_bound_cost_grad_hess(a, a_max, w_bound)
        c_j, g_j, H_j = soft_bound_cost_grad_hess(j, j_max, w_bound)

        c_om, g_om, H_om = soft_bound_cost_grad_hess(
            np.array([omega]), omega_max, w_bound
        )
        c_al, g_al, H_al = soft_bound_cost_grad_hess(
            np.array([alpha]), alpha_max, w_bound
        )
        c_jt, g_jt, H_jt = soft_bound_cost_grad_hess(
            np.array([jtheta]), jtheta_max, w_bound
        )

        cost += c_v + c_a + c_j + c_om + c_al + c_jt

        lx[3:5] += g_v
        lx[6:8] += g_a
        lu[0:2] += g_j

        lxx[3:5, 3:5] += H_v
        lxx[6:8, 6:8] += H_a
        luu[0:2, 0:2] += H_j

        lx[5] += g_om[0]
        lx[8] += g_al[0]
        lu[2] += g_jt[0]

        lxx[5, 5] += H_om[0, 0]
        lxx[8, 8] += H_al[0, 0]
        luu[2, 2] += H_jt[0, 0]

        return cost, lx, lu, lxx, luu, lux

    def terminal_cost_only(x):
        """
        Pure terminal cost computation: no derivatives, fast for line search.
        """
        p = x[0:2]
        theta = x[2]
        v = x[3:5]
        omega = x[5]
        a = x[6:8]
        alpha = x[8]

        pref = path_ref[-1]
        thref = theta_ref[-1]
        vref = np.zeros(2)

        cost = 0.0

        e = p - pref
        cost += w_terminal_track * np.dot(e, e)

        e_theta = theta - thref
        cost += w_terminal_theta * e_theta**2

        cost += w_terminal_vel * np.dot(v, v)
        cost += w_terminal_omega * omega**2
        cost += w_terminal_acc * np.dot(a, a)
        cost += w_terminal_alpha * alpha**2

        # Terminal impedance acceleration consistency
        pdd_T = a + alpha_k * (pref - p) + alpha_d * (vref - v)
        cost += w_terminal_impedance_acc * np.dot(pdd_T, pdd_T)

        # Terminal obstacle penalty (direct cost, no FD gradients)
        q_rect = np.array([p[0], p[1], theta])
        c_obs = rectangle_circle_collision_cost_vec(
            q_rect,
            obstacles_xy,
            used_rect_length,
            used_rect_width,
            inflated_obstacle_radius,
            w_obs,
        )
        cost += c_obs

        return cost

    def terminal_cost_derivatives(x):
        p = x[0:2]
        theta = x[2]
        v = x[3:5]
        omega = x[5]
        a = x[6:8]
        alpha = x[8]

        pref = path_ref[-1]
        thref = theta_ref[-1]
        vref = np.zeros(2)

        lx = np.zeros(nx)
        lxx = np.zeros((nx, nx))
        cost = 0.0

        e = p - pref
        cost += w_terminal_track * np.dot(e, e)
        lx[0:2] += 2.0 * w_terminal_track * e
        lxx[0:2, 0:2] += 2.0 * w_terminal_track * np.eye(2)

        e_theta = theta - thref
        cost += w_terminal_theta * e_theta**2
        lx[2] += 2.0 * w_terminal_theta * e_theta
        lxx[2, 2] += 2.0 * w_terminal_theta

        cost += w_terminal_vel * np.dot(v, v)
        lx[3:5] += 2.0 * w_terminal_vel * v
        lxx[3:5, 3:5] += 2.0 * w_terminal_vel * np.eye(2)

        cost += w_terminal_omega * omega**2
        lx[5] += 2.0 * w_terminal_omega * omega
        lxx[5, 5] += 2.0 * w_terminal_omega

        cost += w_terminal_acc * np.dot(a, a)
        lx[6:8] += 2.0 * w_terminal_acc * a
        lxx[6:8, 6:8] += 2.0 * w_terminal_acc * np.eye(2)

        cost += w_terminal_alpha * alpha**2
        lx[8] += 2.0 * w_terminal_alpha * alpha
        lxx[8, 8] += 2.0 * w_terminal_alpha

        # Terminal impedance acceleration consistency
        pdd_T = a + alpha_k * (pref - p) + alpha_d * (vref - v)

        cost += w_terminal_impedance_acc * np.dot(pdd_T, pdd_T)

        C = np.zeros((2, nx))
        C[:, 0:2] = -alpha_k * np.eye(2)
        C[:, 3:5] = -alpha_d * np.eye(2)
        C[:, 6:8] = np.eye(2)

        lx += 2.0 * w_terminal_impedance_acc * C.T @ pdd_T
        lxx += 2.0 * w_terminal_impedance_acc * C.T @ C

        # Terminal obstacle penalty
        q_rect = np.array([p[0], p[1], theta])
        c_obs, g_obs, H_obs = rectangle_obstacle_cost_grad_hess(
            q=q_rect,
            obstacles=obstacles_xy,
            rect_length=used_rect_length,
            rect_width=used_rect_width,
            circle_radius=inflated_obstacle_radius,
            w_obs=w_obs,
        )

        cost += c_obs

        lx[0] += g_obs[0]
        lx[1] += g_obs[1]
        lx[2] += g_obs[2]

        idxs = np.array([0, 1, 2])
        for ii in range(3):
            for jj in range(3):
                lxx[idxs[ii], idxs[jj]] += H_obs[ii, jj]

        return cost, lx, lxx

    def total_cost(xs, us):
        """Fast cost evaluation using cost-only functions (no derivatives)."""
        total = 0.0
        for i in range(T - 1):
            total += stage_cost_only(xs[i], us[i], i)
        total += terminal_cost_only(xs[-1])
        return total

    cost_trace = []

    for it in range(max_iter):
        xs = rollout_pose(x0, us, A, B)
        old_cost = total_cost(xs, us)
        cost_trace.append(old_cost)

        k_ff = np.zeros((T - 1, nu))
        K_fb = np.zeros((T - 1, nu, nx))

        _, Vx, Vxx = terminal_cost_derivatives(xs[-1])
        Vxx = 0.5 * (Vxx + Vxx.T)

        diverged = False

        for t in reversed(range(T - 1)):
            _, lx, lu, lxx, luu, lux = stage_cost_derivatives(xs[t], us[t], t)

            Qx = lx + A.T @ Vx
            Qu = lu + B.T @ Vx
            Qxx = lxx + A.T @ Vxx @ A
            Quu = luu + B.T @ Vxx @ B
            Qux = lux + B.T @ Vxx @ A

            Quu = 0.5 * (Quu + Quu.T)
            Quu_reg = Quu + reg * np.eye(nu)

            # try:
            #     Quu_inv = np.linalg.inv(Quu_reg)
            # except np.linalg.LinAlgError:
            #     diverged = True
            #     break

            k_t = -np.linalg.solve(Quu_reg, Qu)
            K_t = -np.linalg.solve(Quu_reg, Qux)

            k_ff[t] = k_t
            K_fb[t] = K_t

            Vx = Qx + K_t.T @ Quu @ k_t + K_t.T @ Qu + Qux.T @ k_t
            Vxx = Qxx + K_t.T @ Quu @ K_t + K_t.T @ Qux + Qux.T @ K_t
            Vxx = 0.5 * (Vxx + Vxx.T)

        if diverged:
            reg *= 10.0
            continue

        accepted = False
        new_cost = old_cost

        for alpha_ls in [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]:
            xs_new = np.zeros_like(xs)
            us_new = np.zeros_like(us)
            xs_new[0] = x0

            for t in range(T - 1):
                du = alpha_ls * k_ff[t] + K_fb[t] @ (xs_new[t] - xs[t])
                us_new[t] = us[t] + du

                us_new[t, 0:2] = np.clip(us_new[t, 0:2], -j_max, j_max)
                us_new[t, 2] = np.clip(us_new[t, 2], -jtheta_max, jtheta_max)

                xs_new[t + 1] = A @ xs_new[t] + B @ us_new[t]

            new_cost = total_cost(xs_new, us_new)

            if new_cost < old_cost:
                us = us_new
                accepted = True
                break

        print(f"iter {it:03d} | cost {old_cost:.6e} | reg {reg:.1e} | accepted {accepted}")

        if not accepted:
            reg *= 10.0
        else:
            reg = max(reg * 0.5, 1e-8)

        if accepted and abs(old_cost - new_cost) / max(1.0, old_cost) < 1e-6:
            break

    xs = rollout_pose(x0, us, A, B)

    return {
        "pose": xs[:, 0:3],
        "pos": xs[:, 0:2],
        "theta": xs[:, 2],
        "vel": xs[:, 3:5],
        "omega": xs[:, 5],
        "acc": xs[:, 6:8],
        "alpha": xs[:, 8],
        "jerk": us[:, 0:2],
        "jtheta": us[:, 2],
        "xs": xs,
        "us": us,
        "cost": total_cost(xs, us),
        "cost_trace": np.array(cost_trace),
        "theta_ref": theta_ref,
        "kappa_ref": kappa_ref,
        "v_ref": v_ref,
        "safe_radius": inflated_obstacle_radius,
        "rect_length": rect_length,
        "rect_width": rect_width,
        "m_eff": m_eff,
        "k_imp": k_imp,
        "d_imp": d_imp,
    }


# ============================================================
# 4) Reference path
# ============================================================

EXAMPLE_TRAJECTORY_XY = np.array([
    [-0.29119676, -0.27565762],
    [-0.29016447, -0.27445644],
    [-0.28805077, -0.27227855],
    [-0.2848965, -0.269503],
    [-0.28075832, -0.2661789],
    [-0.2752894, -0.2626621],
    [-0.2682139, -0.25883627],
    [-0.2592516, -0.25499722],
    [-0.25371802, -0.25395337],
    [-0.24765292, -0.25308686],
    [-0.24129054, -0.25214285],
    [-0.2344853, -0.2512745],
    [-0.22720361, -0.25041404],
    [-0.2195526, -0.24973239],
    [-0.21146232, -0.248965],
    [-0.20337416, -0.24819876],
    [-0.19511485, -0.24726456],
    [-0.18683574, -0.24640988],
    [-0.17836812, -0.24592923],
    [-0.16992289, -0.24564722],
    [-0.16152483, -0.24530722],
    [-0.15338063, -0.24531212],
    [-0.1452163, -0.24541995],
    [-0.13724992, -0.2457852],
    [-0.1291886, -0.24635944],
    [-0.12155505, -0.24681357],
    [-0.11382836, -0.24702723],
    [-0.10659543, -0.24723458],
    [-0.09936537, -0.24725881],
    [-0.09218173, -0.24743614],
    [-0.08475301, -0.24792118],
    [-0.07760504, -0.248298],
    [-0.07055832, -0.24856451],
    [-0.06312077, -0.24920523],
    [-0.05558784, -0.2501558],
    [-0.04805115, -0.25162247],
    [-0.04064866, -0.2536475],
    [-0.03322272, -0.25675878],
    [-0.0262367, -0.26109707],
    [-0.01957487, -0.26723194],
    [-0.01337626, -0.2754313],
    [-0.0074517, -0.2859977],
    [-0.00247295, -0.2979443],
    [0.00131624, -0.3106825],
    [0.00480328, -0.3235507],
    [0.00829062, -0.33661062],
    [0.01240013, -0.348449],
    [0.01635449, -0.35864216],
    [0.01980917, -0.366175],
    [0.02307153, -0.3713117],
    [0.02612028, -0.3748054],
    [0.02924444, -0.37733257],
    [0.03246243, -0.37889814],
    [0.03672069, -0.38017145],
    [0.04146875, -0.38151467],
    [0.04702609, -0.3825743],
    [0.05350593, -0.38354948],
    [0.06101975, -0.3851176],
    [0.06978492, -0.38699734],
    [0.07938413, -0.3896914],
    [0.08920829, -0.39302868],
    [0.09952284, -0.39803335],
    [0.10884078, -0.40460557],
    [0.11680527, -0.4125609],
    [0.1216211, -0.4208489],
    [0.12452709, -0.42934185],
    [0.12607664, -0.4379493],
    [0.12737289, -0.44774276],
    [0.12816021, -0.45765692],
    [0.12847973, -0.46799362],
    [0.12881973, -0.47825757],
    [0.12901768, -0.4886583],
    [0.1291916, -0.49854016],
    [0.12951545, -0.50837517],
    [0.12971678, -0.51778924],
    [0.12997793, -0.5269898],
    [0.13015062, -0.5360936],
    [0.13071328, -0.5454973],
    [0.13134839, -0.55511695],
    [0.13281393, -0.5654652],
    [0.13465947, -0.5760185],
    [0.13743262, -0.58698094],
    [0.14053163, -0.59722584],
    [0.14388299, -0.60666347],
    [0.1476335, -0.6143729],
    [0.1511468, -0.62080395],
    [0.15423256, -0.62591654],
    [0.15785629, -0.6299536],
    [0.16147041, -0.6330896],
    [0.16555803, -0.6358491],
    [0.16929734, -0.63802564],
    [0.17348385, -0.6399515],
    [0.17689334, -0.6415247],
    [0.18022582, -0.6431261],
    [0.18282768, -0.64459944],
    [0.18523267, -0.6462174],
    [0.18714842, -0.6479199],
    [0.19047982, -0.6553862],
    [0.19318269, -0.6624361],
    [0.19546825, -0.669055],
    [0.19720872, -0.67510366],
    [0.19862151, -0.6805643],
    [0.19950354, -0.685045],
    [0.19989581, -0.6881308],
])


# ============================================================
# 5) Run
# ============================================================

if __name__ == "__main__":
    path_ref = EXAMPLE_TRAJECTORY_XY.copy()
    dt = 0.08

    start_time = time.perf_counter()

    out = optimize_rectangle_pose_ilqr(
        path_ref=path_ref,
        obstacles_xy=obstacles_xy_raw,
        obstacle_radius=OBSTACLE_RADIUS,
        safety_margin=SAFETY_MARGIN,
        rect_length=RECT_LENGTH,
        rect_width=RECT_WIDTH,
        dt=dt,

        m_eff=5.0,
        k_imp=250.0,
        d_imp=30.0,

        v_max=0.5,
        omega_max=2.0,
        a_max=0.5,
        alpha_max=5.0,
        j_max=2.0,
        jtheta_max=20.0,

        w_track=80.0,
        w_theta_ref=10000.0,

        w_vel=1.0,
        w_omega=1.0,
        w_acc=5.0,
        w_alpha=1.0,
        w_jerk=50.0,
        w_jtheta=5.0,

        w_impedance_acc=100.0,
        w_terminal_impedance_acc=1e9,

        w_curve_vel=100.0,

        w_obs=5e12,
        w_bound=1e12,

        w_terminal_track=1e8,
        w_terminal_theta=1e4,
        w_terminal_vel=1e10,
        w_terminal_omega=1e8,
        w_terminal_acc=1e9,
        w_terminal_alpha=1e8,

        boundary_window=30,
        boundary_gain=20.0,

        max_iter=100,
    )

    elapsed_s = time.perf_counter() - start_time

    print("Final cost:", out["cost"])
    print("execution time [s]:", elapsed_s)
    print("safe radius:", out["safe_radius"])
    print("rect length:", out["rect_length"])
    print("rect width:", out["rect_width"])

    print("pose:", out["pose"].shape)
    print("pos:", out["pos"].shape)
    print("theta:", out["theta"].shape)
    print("vel:", out["vel"].shape)
    print("omega:", out["omega"].shape)
    print("acc:", out["acc"].shape)
    print("alpha:", out["alpha"].shape)
    print("jerk:", out["jerk"].shape)
    print("jtheta:", out["jtheta"].shape)

    print("start vel:", out["vel"][0], "end vel:", out["vel"][-1])
    print("start omega:", out["omega"][0], "end omega:", out["omega"][-1])
    print("start acc:", out["acc"][0], "end acc:", out["acc"][-1])
    print("start alpha:", out["alpha"][0], "end alpha:", out["alpha"][-1])

    pose_opt = out["pose"]
    pos_opt = out["pos"]
    theta_opt = out["theta"]
    vel_opt = out["vel"]
    omega_opt = out["omega"]
    acc_opt = out["acc"]
    alpha_opt = out["alpha"]
    jerk_opt = out["jerk"]
    jtheta_opt = out["jtheta"]

    print("\n" + format_pose_array("OPTIMIZED_TRAJECTORY_XYTHETA", pose_opt))

    t_series = np.arange(pos_opt.shape[0]) * dt
    t_jerk = np.arange(jerk_opt.shape[0]) * dt

    fig_geo, ax_geo = plt.subplots(figsize=(8, 8))

    ax_geo.plot(path_ref[:, 0], path_ref[:, 1], "k--", label="reference path")
    ax_geo.plot(pos_opt[:, 0], pos_opt[:, 1], "b-", label="optimized rectangle center")
    ax_geo.scatter(pos_opt[:, 0], pos_opt[:, 1], s=20, c="tab:blue")
    ax_geo.scatter(obstacles_xy_raw[:, 0], obstacles_xy_raw[:, 1], s=30, c="red", label="obstacles")

    for c in obstacles_xy_raw:
        circle = plt.Circle(c, OBSTACLE_RADIUS + SAFETY_MARGIN, color="red", alpha=0.15)
        ax_geo.add_patch(circle)

    draw_every = max(1, len(pos_opt) // 20)
    for k in range(0, len(pos_opt)):
        rect = Rectangle(
            (-RECT_LENGTH / 2, -RECT_WIDTH / 2),
            RECT_LENGTH,
            RECT_WIDTH,
            fill=False,
            edgecolor="blue",
            alpha=0.5,
            linewidth=1.0,
        )
        trans = (
            Affine2D()
            .rotate(theta_opt[k])
            .translate(pos_opt[k, 0], pos_opt[k, 1])
            + ax_geo.transData
        )
        rect.set_transform(trans)
        ax_geo.add_patch(rect)

    ax_geo.set_title("Optimized Trajectory, Obstacles, and Rectangle Footprint")
    ax_geo.set_xlabel("x")
    ax_geo.set_ylabel("y")
    ax_geo.axis("equal")
    ax_geo.grid(True, alpha=0.3)
    ax_geo.legend()

    # Separate figure: reference path with reference rectangle orientations
    fig_ref, ax_ref = plt.subplots(figsize=(8, 8))
    ax_ref.plot(path_ref[:, 0], path_ref[:, 1], "k-", label="reference path")
    ax_ref.scatter(path_ref[:, 0], path_ref[:, 1], s=20, c="tab:gray")
    ax_ref.scatter(obstacles_xy_raw[:, 0], obstacles_xy_raw[:, 1], s=30, c="red", label="obstacles")

    for c in obstacles_xy_raw:
        circle = plt.Circle(c, OBSTACLE_RADIUS + SAFETY_MARGIN, color="red", alpha=0.15)
        ax_ref.add_patch(circle)

    theta_ref_plot = out["theta_ref"]
    draw_every_ref = max(1, len(path_ref) // 20)
    for k in range(0, len(path_ref)):
        rect = Rectangle(
            (-RECT_LENGTH / 2, -RECT_WIDTH / 2),
            RECT_LENGTH,
            RECT_WIDTH,
            fill=False,
            edgecolor="tab:green",
            alpha=0.6,
            linewidth=1.0,
        )
        trans = (
            Affine2D()
            .rotate(theta_ref_plot[k])
            .translate(path_ref[k, 0], path_ref[k, 1])
            + ax_ref.transData
        )
        rect.set_transform(trans)
        ax_ref.add_patch(rect)

    ax_ref.set_title("Reference Path and Reference Rectangle Footprint")
    ax_ref.set_xlabel("x")
    ax_ref.set_ylabel("y")
    ax_ref.axis("equal")
    ax_ref.grid(True, alpha=0.3)
    ax_ref.legend()

    fig, axes = plt.subplots(7, 1, figsize=(10, 22), sharex=False)

    axes[0].plot(path_ref[:, 0], path_ref[:, 1], "k--", label="reference path")
    axes[0].plot(pos_opt[:, 0], pos_opt[:, 1], "b-", label="optimized rectangle center")
    axes[0].scatter(pos_opt[:, 0], pos_opt[:, 1], s=20, c="tab:blue")
    axes[0].scatter(obstacles_xy_raw[:, 0], obstacles_xy_raw[:, 1], s=30, c="red", label="obstacles")

    for c in obstacles_xy_raw:
        circle = plt.Circle(c, OBSTACLE_RADIUS + SAFETY_MARGIN, color="red", alpha=0.15)
        axes[0].add_patch(circle)

    # Draw rectangles every few waypoints
    draw_every = max(1, len(pos_opt) // 20)
    for k in range(0, len(pos_opt), draw_every):
        rect = Rectangle(
            (-RECT_LENGTH / 2, -RECT_WIDTH / 2),
            RECT_LENGTH,
            RECT_WIDTH,
            fill=False,
            edgecolor="blue",
            alpha=0.5,
            linewidth=1.0,
        )
        trans = (
            Affine2D()
            .rotate(theta_opt[k])
            .translate(pos_opt[k, 0], pos_opt[k, 1])
            + axes[0].transData
        )
        rect.set_transform(trans)
        axes[0].add_patch(rect)

    axes[0].set_title("Rectangle Pose Trajectory")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t_series, theta_opt, label="theta")
    axes[1].plot(t_series, out["theta_ref"], "--", label="theta_ref")
    axes[1].set_title("Orientation")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(t_series, vel_opt[:, 0], label="vx")
    axes[2].plot(t_series, vel_opt[:, 1], label="vy")
    axes[2].plot(t_series, omega_opt, label="omega")
    axes[2].set_title("Velocity / Angular Velocity")
    axes[2].grid(True)
    axes[2].legend()

    axes[3].plot(t_series, acc_opt[:, 0], label="ax")
    axes[3].plot(t_series, acc_opt[:, 1], label="ay")
    axes[3].plot(t_series, alpha_opt, label="angular acceleration")
    axes[3].set_title("Acceleration")
    axes[3].grid(True)
    axes[3].legend()

    axes[4].plot(t_jerk, jerk_opt[:, 0], label="jx")
    axes[4].plot(t_jerk, jerk_opt[:, 1], label="jy")
    axes[4].plot(t_jerk, jtheta_opt, label="jtheta")
    axes[4].set_title("Jerk")
    axes[4].grid(True)
    axes[4].legend()

    axes[5].plot(t_series, out["kappa_ref"], label="reference curvature")
    axes[5].set_title("Reference Curvature Proxy")
    axes[5].grid(True)
    axes[5].legend()

    axes[6].plot(out["cost_trace"])
    axes[6].set_yscale("log")
    axes[6].set_title("iLQR Cost Trace")
    axes[6].grid(True)

    fig.tight_layout()
    plt.show()