from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.sparse import spmatrix

from .grid import CorticalGrid


def order_parameter(theta: np.ndarray) -> tuple[float, float]:
    """Compute the order parameter. R measures global coherence (0 = incoherent, 1 = fully synchronised).
    """
    z = np.mean(np.exp(1j * theta))
    return float(np.abs(z)), float(np.angle(z))

def local_order(
    theta: np.ndarray,
    grid: CorticalGrid,
    radius: float = 2.0,
) -> np.ndarray:
    """
    Local order parameter for each node (coherence within neighborhood).
    Vectorized via box convolution of exp(1j*theta); boundaries use mode='reflect'.

    Returns:
        2D array (n_rows, n_cols) of local R values.
    """
    theta_2d = grid.unflatten(theta)
    z = np.exp(1j * theta_2d)
    size = (int(2 * radius) + 1, int(2 * radius) + 1)
    z_mean_real = uniform_filter(z.real, size=size, mode="reflect")
    z_mean_imag = uniform_filter(z.imag, size=size, mode="reflect")
    local_R = np.abs(z_mean_real + 1j * z_mean_imag)
    return local_R


def coupling_term(
    theta: np.ndarray,
    K: spmatrix,
    theta_source: np.ndarray | None = None,
) -> np.ndarray:
    """Coupling term for sparse K: cos(theta)*(K@sin(src)) - sin(theta)*(K@cos(src)).
    Single evaluation for reuse in theta_dot and coupling_tension."""
    src = theta if theta_source is None else theta_source
    return np.cos(theta) * (K @ np.sin(src)) - np.sin(theta) * (K @ np.cos(src))


def coupling_term_uniform(
    theta: np.ndarray,
    strength: float,
    theta_source: np.ndarray | None = None,
) -> np.ndarray:
    """Coupling term for uniform all-to-all: strength * (cos(theta)*S - sin(theta)*C)."""
    src = theta if theta_source is None else theta_source
    S, C = np.sum(np.sin(src)), np.sum(np.cos(src))
    return strength * (np.cos(theta) * S - np.sin(theta) * C)


def coupling_tension(
    theta: np.ndarray,
    omega: np.ndarray,
    K: spmatrix,
    theta_source: np.ndarray | None = None,
) -> np.ndarray:
    """Coupling tension F = omega - coupling term"""
    src = theta if theta_source is None else theta_source
    coupling = np.cos(theta) * (K @ np.sin(src)) - np.sin(theta) * (K @ np.cos(src))
    return omega - coupling


# ---------------------------------------------------------------------------
# Phase-field gradient (traveling-wave / spatial structure)
# ---------------------------------------------------------------------------
def angle_diff(b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Circular difference (b - a) wrapped to (-π, π], in radians."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    return np.angle(np.exp(1j * d))


def phase_gradient_2d(
    theta_2d: np.ndarray,
    periodic_bc: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Spatial gradient of phase on a 2D grid (rad per grid step).

    Uses central differences with circular (angle) difference. Respects
    periodic boundaries when periodic_bc is True.

    Returns:
        dtheta_dr: gradient along rows (axis 0), shape = theta_2d.shape
        dtheta_dc: gradient along columns (axis 1), shape = theta_2d.shape
    """
    theta_2d = np.asarray(theta_2d)
    nr, nc = theta_2d.shape
    dtheta_dr = np.zeros_like(theta_2d)
    dtheta_dc = np.zeros_like(theta_2d)

    if periodic_bc:
        for i in range(nr):
            i_prev = (i - 1) % nr
            i_next = (i + 1) % nr
            dtheta_dr[i, :] = angle_diff(theta_2d[i_next, :], theta_2d[i_prev, :]) / 2.0
        for j in range(nc):
            j_prev = (j - 1) % nc
            j_next = (j + 1) % nc
            dtheta_dc[:, j] = angle_diff(theta_2d[:, j_next], theta_2d[:, j_prev]) / 2.0
    else:
        dtheta_dr[1 : nr - 1, :] = (
            angle_diff(theta_2d[2:, :], theta_2d[:-2, :]) / 2.0
        )
        dtheta_dc[:, 1 : nc - 1] = (
            angle_diff(theta_2d[:, 2:], theta_2d[:, :-2]) / 2.0
        )
        dtheta_dr[0, :] = angle_diff(theta_2d[1, :], theta_2d[0, :])
        dtheta_dr[nr - 1, :] = angle_diff(theta_2d[nr - 1, :], theta_2d[nr - 2, :])
        dtheta_dc[:, 0] = angle_diff(theta_2d[:, 1], theta_2d[:, 0])
        dtheta_dc[:, nc - 1] = angle_diff(theta_2d[:, nc - 1], theta_2d[:, nc - 2])

    return dtheta_dr, dtheta_dc


def gradient_magnitude_2d(dtheta_dr: np.ndarray, dtheta_dc: np.ndarray) -> np.ndarray:
    """|∇θ| from row and column gradients (rad per grid step)."""
    return np.sqrt(dtheta_dr**2 + dtheta_dc**2)


def phase_time_derivative(
    theta_prev: np.ndarray,
    theta_next: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Time derivative ∂θ/∂t using circular difference. rad per time unit.

    theta_prev, theta_next: flat or 2D phase at t_prev and t_next.
    dt: t_next - t_prev (time span). Returns angle_diff(theta_next, theta_prev) / dt.
    """
    d = angle_diff(np.asarray(theta_next), np.asarray(theta_prev))
    return d / float(dt)


def gradient_from_state(
    theta_flat: np.ndarray,
    grid: CorticalGrid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute spatial gradient of theta from flat state vector.

    Returns:
        dtheta_dr: gradient along rows (2D)
        dtheta_dc: gradient along columns (2D)
        magnitude: |∇θ| (2D)
    """
    theta_2d = grid.unflatten(np.asarray(theta_flat))
    dtheta_dr, dtheta_dc = phase_gradient_2d(theta_2d, periodic_bc=grid.periodic_bc)
    mag = gradient_magnitude_2d(dtheta_dr, dtheta_dc)
    return dtheta_dr, dtheta_dc, mag


def gradient_and_time_derivative(
    theta_prev: np.ndarray,
    theta_curr: np.ndarray,
    theta_next: np.ndarray,
    t_prev: float,
    t_next: float,
    grid: CorticalGrid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Spatial gradient and time derivative at current snapshot (central difference in time).

    Returns:
        dtheta_dr: gradient along rows (2D)
        dtheta_dc: gradient along columns (2D)
        magnitude: |∇θ| (2D)
        dtheta_dt: ∂θ/∂t (2D), rad per time unit
    """
    dtheta_dr, dtheta_dc, mag = gradient_from_state(theta_curr, grid)
    dt = t_next - t_prev
    dtheta_dt_flat = phase_time_derivative(theta_prev, theta_next, dt)
    dtheta_dt = grid.unflatten(dtheta_dt_flat)
    return dtheta_dr, dtheta_dc, mag, dtheta_dt


def phase_velocity_2d(
    dtheta_dt_2d: np.ndarray,
    dtheta_dr: np.ndarray,
    dtheta_dc: np.ndarray,
    grad_mag: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Velocity v (grid-step per time unit) such that v·∇θ = −∂θ/∂t (co-moving frame Dθ/Dt = 0).

    When |∇θ| = 0, returns zero velocity. Uses v = −(∂θ/∂t) ∇θ / |∇θ|².

    Returns:
        v_r: velocity component along rows (2D)
        v_c: velocity component along columns (2D)
    """
    if grad_mag is None:
        grad_mag = gradient_magnitude_2d(dtheta_dr, dtheta_dc)
    sq = np.maximum(grad_mag**2, 1e-20)
    v_r = -dtheta_dt_2d * dtheta_dr / sq
    v_c = -dtheta_dt_2d * dtheta_dc / sq
    return v_r, v_c


def material_derivative_2d(
    dtheta_dt_2d: np.ndarray,
    dtheta_dr: np.ndarray,
    dtheta_dc: np.ndarray,
    v_r: np.ndarray,
    v_c: np.ndarray,
) -> np.ndarray:
    """Material derivative Dθ/Dt = ∂θ/∂t + v·∇θ (2D). v in grid-step per time unit."""
    return dtheta_dt_2d + v_r * dtheta_dr + v_c * dtheta_dc


def bulk_gradient_metric(
    theta_flat: np.ndarray,
    grid: CorticalGrid,
    metric: str = "mean_magnitude",
) -> float:
    """Single scalar summarizing the theta gradient field.

    metric:
        'mean_magnitude': mean of |∇θ| over the grid (rad per grid step)
        'std_magnitude': std of |∇θ|
        'max_magnitude': max of |∇θ|
    """
    _, _, mag = gradient_from_state(theta_flat, grid)
    if metric == "mean_magnitude":
        return float(np.nanmean(mag))
    if metric == "std_magnitude":
        return float(np.nanstd(mag))
    if metric == "max_magnitude":
        return float(np.nanmax(mag))
    raise ValueError(f"Unknown metric: {metric!r}")


def gradient_time_series(
    state_list: list[dict],
    grid: CorticalGrid,
    metric: str = "mean_magnitude",
) -> list[float]:
    """Bulk gradient metric at each time step (post-processing).

    state_list: from Simulation.run() or storage snapshots (each dict has 'theta').
    """
    return [
        bulk_gradient_metric(s["theta"], grid, metric=metric)
        for s in state_list
    ]


def gradient_maps(
    state_list: list[dict],
    grid: CorticalGrid,
    indices: list[int] | None = None,
    t_list: list[float] | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Gradient maps at selected time indices (post-processing).

    If t_list is None: returns list of (dtheta_dr, dtheta_dc, magnitude) per index.
    If t_list is given: returns list of (dtheta_dr, dtheta_dc, magnitude, dtheta_dt) per index.
    Time derivative uses central difference where possible, one-sided at endpoints.

    If indices is None, uses every state (can be large).
    """
    if indices is None:
        indices = list(range(len(state_list)))
    n = len(state_list)
    have_time = t_list is not None and len(t_list) == n and n >= 2
    out = []
    for i in indices:
        s = state_list[i]
        dr, dc, mag = gradient_from_state(s["theta"], grid)
        if not have_time:
            out.append((dr, dc, mag))
            continue
        # ∂θ/∂t at index i
        if i == 0:
            dt = t_list[1] - t_list[0]
            dtheta_dt_flat = phase_time_derivative(
                state_list[0]["theta"], state_list[1]["theta"], dt
            )
        elif i == n - 1:
            dt = t_list[-1] - t_list[-2]
            dtheta_dt_flat = phase_time_derivative(
                state_list[-2]["theta"], state_list[-1]["theta"], dt
            )
        else:
            dt = t_list[i + 1] - t_list[i - 1]
            dtheta_dt_flat = phase_time_derivative(
                state_list[i - 1]["theta"], state_list[i + 1]["theta"], dt
            )
        dtheta_dt = grid.unflatten(dtheta_dt_flat)
        out.append((dr, dc, mag, dtheta_dt))
    return out


def gradient_and_material_maps(
    state_list: list[dict],
    t_list: list[float],
    grid: CorticalGrid,
    indices: list[int] | None = None,
    v_r: np.ndarray | None = None,
    v_c: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Gradient, time derivative, and material derivative Dθ/Dt = ∂θ/∂t + v·∇θ.

    Returns list of (dtheta_dr, dtheta_dc, grad_mag, dtheta_dt, material_dt) 2D arrays
    for each index. material_dt is Dθ/Dt. If v_r, v_c are not provided, v is the
    phase velocity (v·∇θ = −∂θ/∂t), so Dθ/Dt ≈ 0. Otherwise pass v_r, v_c (2D per
    snapshot or fixed field) in grid-step per time unit.
    """
    maps = gradient_maps(state_list, grid, indices=indices, t_list=t_list)
    result = []
    for idx, item in enumerate(maps):
        if len(item) != 4:
            raise ValueError("gradient_maps must be called with t_list for material derivative")
        dr, dc, mag, dtheta_dt = item
        if v_r is None or v_c is None:
            v_r_i, v_c_i = phase_velocity_2d(dtheta_dt, dr, dc, grad_mag=mag)
        else:
            v_r_i, v_c_i = v_r, v_c
        Dtheta_Dt = material_derivative_2d(dtheta_dt, dr, dc, v_r_i, v_c_i)
        result.append((dr, dc, mag, dtheta_dt, v_r_i, v_c_i, Dtheta_Dt))
    return result