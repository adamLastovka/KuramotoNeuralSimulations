from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from scipy.ndimage import uniform_filter

from .grid import CorticalGrid

#TODO: unify handling of jax vs numpy functions

def get_R(theta: np.ndarray) -> tuple[np.ndarray | float, np.ndarray | float]:
    """Compute the R parameter.

    Accepts:
      - ``theta`` with shape ``(N,)`` -> returns ``(R, psi)`` as floats
      - ``theta`` with shape ``(T, N)`` -> returns ``(R_t, psi_t)`` as arrays
      - ``theta`` as a simulation-like object with ``.results`` -> uses ``.results["theta"]``
    """
    if hasattr(theta, "results"):
        theta = theta.results["theta"]
    x = np.asarray(theta)
    if x.ndim == 1:
        z = np.mean(np.exp(1j * x))
        return float(np.abs(z)), float(np.angle(z))
    if x.ndim == 2:
        z = np.mean(np.exp(1j * x), axis=1)  # (T,)
        return np.abs(z), np.angle(z)
    raise ValueError(f"get_R expects (N,) or (T,N); got shape={x.shape}")


def get_R_jax(theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Get R in JAX Handles both (N,) and (T, N) shapes. Returns (R, psi).

    Args:
        theta: phases. Shape (N,) or (T, N).

    Returns:
        (R, psi) if (N,) input; (R_t, psi_t) if (T, N) input.
    """
    theta = jnp.asarray(theta)
    if theta.ndim == 1:
        z = jnp.mean(jnp.exp(1j * theta))
        return jnp.abs(z)
    elif theta.ndim == 2:
        z = jnp.mean(jnp.exp(1j * theta), axis=1)  # (T,)
        return jnp.abs(z)
    else:
        raise ValueError(f"get_R_link_jax expects (N,) or (T,N); got shape={theta.shape}")

def functional_connectivity(theta: jnp.ndarray, dt: float) -> jnp.ndarray:
    """Functional connectivity matrix in JAX.

    C_ij = |mean_t exp(i (theta_i(t) - theta_j(t)))| (phase locking over time).

    Args:
        theta: Phase angles over time ``(T, N)``.
        dt: Time step deprecated

    Returns:
        Matrix ``C`` of shape ``(N, N)``.
    """
    return jnp.abs(jnp.mean(jnp.exp(1j * (theta[:, None, :] - theta[:, :, None])), axis=0))


def get_R_link_jax(theta: jnp.ndarray, dt: float) -> jnp.ndarray:
    """Scalar pairwise phase-locking measure R_link (JAX).

    Schmidt et al. 2015 (used as the canonical definition in this package)
    Let C_ij = |mean_t exp(i*(theta_i(t)-theta_j(t)))|
    be the functional connectivity matrix from functional_connectivity.
    Then

        R_link = (1/(N*(N-1))) * sum_{i=1}^N sum_{j=1}^N C_ij

    The double sum includes the diagonal. For phases, C_ii = 1.

    Alternative (off-diagonal-only) expression (not used for now):
    Since C_ii = 1, the sum over all entries is N + sum for i != j of C_ij.
    Defining

        R_link_prime = (1/(N*(N-1))) * (sum_{i,j} C_ij - N)
                    = (1/(N*(N-1))) * sum_{i != j} C_ij

    This equals R_link - 1/(N-1); it is not the same as the paper form above.
    This module implements the paper definition via jnp.sum(C) / (N * (N - 1)).

    Args:
        theta: Phases (N,) (single snapshot, promoted to (1, N)) or
            trajectory (T, N).
        dt: Passed to functional_connectivity.

    Returns:
        Scalar R_link.
   
    """
    theta = jnp.asarray(theta)
    if theta.ndim == 1:
        theta = theta[None, :]
    if theta.ndim != 2:
        raise ValueError(f"get_R_link_jax expects theta of shape (N,) or (T, N); got {theta.shape}")
    n = theta.shape[1]
    if n < 2:
        raise ValueError(f"get_R_link_jax requires N >= 2; got N={n}")
    C = functional_connectivity(theta, dt)
    return jnp.sum(C) / (n * (n - 1))


def get_R_link(theta: jnp.ndarray | np.ndarray, dt: float = 1.0) -> jnp.ndarray:
    """Pairwise phase locking (Schmidt et al. 2015); wraps :func:`get_R_link_jax`.

    Args:
        theta: Phase angles ``(N,)`` or ``(T, N)``.
        dt: Time step for functional connectivity (default ``1.0`` if unknown).

    Returns:
        Scalar :math:`R_{\\mathrm{link}}`.
    """
    if not isinstance(theta, (np.ndarray, jnp.ndarray)):
        raise ValueError(f"Expected theta to be a numpy or jnp array, got {type(theta)}")
    return get_R_link_jax(jnp.asarray(theta), float(dt))

def compute_effective_coupling(theta: jnp.ndarray, K: jnp.ndarray) -> jnp.ndarray:
    """Compute the effective coupling matrix. K_eff_ij = K_ij * cos(theta_j - theta_i)
    Args:
        theta: Phase angles at a single time step. [N,]
        K: Coupling matrix. [N,N]
    Returns:
        Effective coupling matrix. [N,N]
    """
    delta = theta[None, :] - theta[:, None]  # shape (N, N)
    return K * jnp.cos(delta)

def avg_effective_coupling(theta_list: jnp.ndarray, K: jnp.ndarray) -> jnp.ndarray:
    """Compute the time averaged effective coupling matrix. K_eff_ij = K_ij * cos(theta_j - theta_i)
    Args:
        theta_list: Array of phase angles. [T,N]
        K: Coupling matrix. [N,N]
    Returns:
        Time averaged effective coupling matrix. [N,N]
    """
    delta = theta_list[:, None, :] - theta_list[:, :, None]  # shape (T, N, N)
    K_eff = K * jnp.cos(delta)
    return jnp.mean(K_eff, axis=0)

def local_order(
    theta: np.ndarray,
    grid: CorticalGrid,
    radius: float = 2.0,
) -> np.ndarray:
    """
    Local order parameter for each node
    Vectorized via box convolution of exp(1j*theta);

    Accepts:
        - ``theta`` with shape ``(N,)`` -> returns ``R`` as a float
        - ``theta`` with shape ``(T, N)`` -> returns ``R_t`` as an array
        - ``theta`` as a simulation-like object with ``.results`` -> returns ``R_t`` as an array or float
    """
    if hasattr(theta, "results"):
        theta = theta.results["theta"]


    x = np.asarray(theta)
    if x.ndim == 1: 
        theta_2d = grid.unflatten(x)
        z = np.exp(1j * theta_2d)
        size = (int(2 * radius) + 1, int(2 * radius) + 1)
        z_mean_real = uniform_filter(z.real, size=size, mode="reflect")
        z_mean_imag = uniform_filter(z.imag, size=size, mode="reflect")
        return np.abs(z_mean_real + 1j * z_mean_imag)

    if x.ndim == 2:
        # Return (T, n_rows, n_cols); loop over time because scipy uniform_filter
        # operates on single 2D arrays.
        return np.stack([local_order(xi, grid, radius=radius) for xi in x], axis=0)

    raise ValueError(f"local_order expects (N,) or (T,N); got shape={x.shape}")


def coupling_term(
    theta: np.ndarray,
    K: np.ndarray,
    theta_source: np.ndarray | None = None,
) -> np.ndarray:
    """Coupling term: cos(theta)*(K@sin(src)) - sin(theta)*(K@cos(src))."""
    src = theta if theta_source is None else theta_source
    return np.cos(theta) * (K @ np.sin(src)) - np.sin(theta) * (K @ np.cos(src))


def coupling_tension(
    theta: np.ndarray,
    omega: np.ndarray,
    K: np.ndarray,
    theta_source: np.ndarray | None = None,
) -> np.ndarray:
    """Coupling tension F = omega - coupling term."""
    src = theta if theta_source is None else theta_source
    coupling = np.cos(theta) * (K @ np.sin(src)) - np.sin(theta) * (K @ np.cos(src))
    return omega - coupling

# ---------------------------------------------------------------------------
# Phase-field gradient (traveling-wave / spatial structure) - deprecated for now
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
    state_list: list[dict] | np.ndarray,
    grid: CorticalGrid,
    metric: str = "mean_magnitude",
) -> list[float]:
    """Bulk gradient metric at each time step (post-processing).

    Accepts:
      - ``state_list`` as a list of dicts with ``{"theta": ...}`` (legacy)
      - ``theta_series`` as an array with shape ``(T, N)``
      - a simulation-like object with ``.results["theta"]``
    """
    if hasattr(state_list, "results"):
        theta_series = state_list.results["theta"]
        return [bulk_gradient_metric(theta_series[i], grid, metric=metric) for i in range(theta_series.shape[0])]

    x = np.asarray(state_list)
    if isinstance(state_list, np.ndarray) and x.ndim == 1:
        return [bulk_gradient_metric(x, grid, metric=metric)]
    if isinstance(state_list, np.ndarray) and x.ndim == 2:
        return [bulk_gradient_metric(x[i], grid, metric=metric) for i in range(x.shape[0])]

    # Legacy path: list[dict]
    return [bulk_gradient_metric(s["theta"], grid, metric=metric) for s in state_list]


def gradient_maps(
    state_list: list[dict] | np.ndarray,
    grid: CorticalGrid,
    indices: list[int] | None = None,
    t_list: list[float] | np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Gradient maps at selected time indices (post-processing).

    If t_list is None: returns list of (dtheta_dr, dtheta_dc, magnitude) per index.
    If t_list is given: returns list of (dtheta_dr, dtheta_dc, magnitude, dtheta_dt) per index.
    Time derivative uses central difference where possible, one-sided at endpoints.

    If indices is None, uses every state (can be large).
    """
    theta_series: np.ndarray | None = None
    ts: np.ndarray | None = None
    legacy_state_list: list[dict] | None = None

    if hasattr(state_list, "results"):
        theta_series = state_list.results["theta"]
        ts = np.asarray(state_list.results.get("ts")) if "ts" in state_list.results else None
        legacy_state_list = None
    elif isinstance(state_list, np.ndarray):
        theta_series = state_list
        legacy_state_list = None
    else:
        legacy_state_list = state_list  # type: ignore[assignment]

    if theta_series is not None and t_list is None and ts is not None:
        t_list = ts

    if legacy_state_list is not None:
        if indices is None:
            indices = list(range(len(legacy_state_list)))
        n = len(legacy_state_list)
    else:
        if indices is None:
            indices = list(range(theta_series.shape[0]))  # type: ignore[union-attr]
        n = theta_series.shape[0]  # type: ignore[union-attr]

    t_arr = np.asarray(t_list) if t_list is not None else None
    have_time = t_arr is not None and t_arr.shape[0] == n and n >= 2

    out = []
    for i in indices:
        if legacy_state_list is not None:
            s = legacy_state_list[i]
            theta_i = s["theta"]
        else:
            theta_i = theta_series[i]  # type: ignore[index]

        dr, dc, mag = gradient_from_state(theta_i, grid)
        if not have_time:
            out.append((dr, dc, mag))
            continue
        # ∂θ/∂t at index i
        if i == 0:
            dt = float(t_arr[1] - t_arr[0])  # type: ignore[index]
            if legacy_state_list is not None:
                th_prev = legacy_state_list[0]["theta"]
                th_next = legacy_state_list[1]["theta"]
            else:
                th_prev = theta_series[0]  # type: ignore[index]
                th_next = theta_series[1]  # type: ignore[index]
            dtheta_dt_flat = phase_time_derivative(th_prev, th_next, dt)
        elif i == n - 1:
            dt = float(t_arr[-1] - t_arr[-2])  # type: ignore[index]
            if legacy_state_list is not None:
                th_prev = legacy_state_list[-2]["theta"]
                th_next = legacy_state_list[-1]["theta"]
            else:
                th_prev = theta_series[-2]  # type: ignore[index]
                th_next = theta_series[-1]  # type: ignore[index]
            dtheta_dt_flat = phase_time_derivative(th_prev, th_next, dt)
        else:
            dt = float(t_arr[i + 1] - t_arr[i - 1])  # type: ignore[index]
            if legacy_state_list is not None:
                th_prev = legacy_state_list[i - 1]["theta"]
                th_next = legacy_state_list[i + 1]["theta"]
            else:
                th_prev = theta_series[i - 1]  # type: ignore[index]
                th_next = theta_series[i + 1]  # type: ignore[index]
            dtheta_dt_flat = phase_time_derivative(th_prev, th_next, dt)
        dtheta_dt = grid.unflatten(dtheta_dt_flat)
        out.append((dr, dc, mag, dtheta_dt))
    return out


def gradient_and_material_maps(
    state_list: list[dict] | np.ndarray,
    t_list: list[float] | np.ndarray | None,
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
    if t_list is None and hasattr(state_list, "results"):
        t_list = state_list.results.get("ts")  # type: ignore[assignment]

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