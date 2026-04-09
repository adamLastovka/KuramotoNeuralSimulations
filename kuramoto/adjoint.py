from __future__ import annotations

import jax
import jax.numpy as jnp

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, CenteredNorm, TwoSlopeNorm

from .analysis import order_parameter_jax, r_link_jax
from .simulation import Simulation, KuramotoParams, solve_forward
from .coupling import apply_node_lesions

# --- Objectives ---
def final_order_parameter(
    params: KuramotoParams,
    theta0: jnp.ndarray,
    t0: float,
    t1: float,
    dt: float,
    ts: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """R(T) -- order parameter at the final saved time."""
    sol = solve_forward(params, theta0, t0=t0, t1=t1, dt=dt, ts=ts)
    return order_parameter_jax(sol.ys[-1])


def mean_order_parameter(
    params: KuramotoParams,
    theta0: jnp.ndarray,
    t0: float,
    t1: float,
    dt: float,
    ts: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Time-averaged R -- mean of R(t) over all saved times."""
    sol = solve_forward(params, theta0, t0=t0, t1=t1, dt=dt, ts=ts)
    Rs = jax.vmap(order_parameter_jax)(sol.ys)
    return jnp.mean(Rs)


def final_order_parameter_lesioned(
    params: KuramotoParams,
    alpha: jnp.ndarray,
    theta0: jnp.ndarray,
    t0: float,
    t1: float,
    dt: float,
    ts: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """R(T) with continuous node lesions applied to params.K."""
    params_eff = KuramotoParams(omega=params.omega, K=apply_node_lesions(params.K, alpha))
    return final_order_parameter(params_eff, theta0, t0, t1, dt, ts)


def mean_order_parameter_lesioned(
    params: KuramotoParams,
    alpha: jnp.ndarray,
    theta0: jnp.ndarray,
    t0: float,
    t1: float,
    dt: float,
    ts: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Mean R with continuous node lesions applied to params.K."""
    params_eff = KuramotoParams(omega=params.omega, K=apply_node_lesions(params.K, alpha))
    return mean_order_parameter(params_eff, theta0, t0, t1, dt, ts)


def mean_r_link(
    params: KuramotoParams,
    theta0: jnp.ndarray,
    t0: float,
    t1: float,
    dt: float,
    ts: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """mean R_link"""
    sol = solve_forward(params, theta0, t0=t0, t1=t1, dt=dt, ts=ts)
    return r_link_jax(sol.ys, dt)


def mean_r_link_lesioned(
    params: KuramotoParams,
    alpha: jnp.ndarray,
    theta0: jnp.ndarray,
    t0: float,
    t1: float,
    dt: float,
    ts: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """R_link with continuous node lesions applied to params.K."""
    params_eff = KuramotoParams(omega=params.omega, K=apply_node_lesions(params.K, alpha))
    return mean_r_link(params_eff, theta0, t0, t1, dt, ts)


# --- Adjoint functions ---
def grads_final_R(
    params: KuramotoParams,
    theta0: jnp.ndarray,
    t0: float,
    t1: float,
    dt: float,
    ts: jnp.ndarray | None = None,
) -> KuramotoParams:
    """Gradients of R(T) w.r.t. coupling weights K and natural frequencies omega."""
    return jax.grad(final_order_parameter)(params, theta0, t0, t1, dt, ts)


def grads_mean_R(
    params: KuramotoParams,
    theta0: jnp.ndarray,
    t0: float,
    t1: float,
    dt: float,
    ts: jnp.ndarray | None = None,
) -> KuramotoParams:
    """Gradients of mean R w.r.t. coupling weights K and natural frequencies omega."""
    return jax.grad(mean_order_parameter)(params, theta0, t0, t1, dt, ts)


def grads_final_R_alpha(
    params: KuramotoParams,
    alpha: jnp.ndarray,
    theta0: jnp.ndarray,
    t0: float,
    t1: float,
    dt: float,
    ts: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Gradient of R(T) w.r.t. node lesion parameters alpha."""
    return jax.grad(final_order_parameter_lesioned, argnums=1)(params, alpha, theta0, t0, t1, dt, ts)


def grads_mean_R_alpha(
    params: KuramotoParams,
    alpha: jnp.ndarray,
    theta0: jnp.ndarray,
    t0: float,
    t1: float,
    dt: float,
    ts: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Gradient of mean R w.r.t. node lesion parameters alpha."""
    return jax.grad(mean_order_parameter_lesioned, argnums=1)(params, alpha, theta0, t0, t1, dt, ts)


def grads_mean_r_link_alpha(
    params: KuramotoParams,
    alpha: jnp.ndarray,
    theta0: jnp.ndarray,
    t0: float,
    t1: float,
    dt: float,
    ts: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Gradient of R_link w.r.t. node lesion parameters alpha."""
    return jax.grad(mean_r_link_lesioned, argnums=1)(params, alpha, theta0, t0, t1, dt, ts)


def integrated_grads_alpha(
    obj_lesioned,
    params: KuramotoParams,
    alpha_target: jnp.ndarray,
    theta0: jnp.ndarray,
    t0: float,
    t1: float,
    dt: float,
    ts: jnp.ndarray | None = None,
    n_steps: int = 20,
) -> jnp.ndarray:
    """Integrated Gradients w.r.t. node lesion params along uniform path 0 -> alpha_target.
    Reference: (Sundararajan et al. 2017). 

    Integrates dJ/dalpha along the straight-line path alpha(s) = s * alpha_target,
    s in [0, 1], via the trapezoid rule.  

    This captures the full nonlinear response of J to lesioning, unlike the first-order
    approximation dJ/dalpha|_{alpha=0}.

    Args:
        obj_lesioned: Differentiable objective with signature
            (params, alpha, theta0, t0, t1, dt, ts) -> scalar.
        params: KuramotoParams for the intact network.
        alpha_target: Target lesion vector (N,);
        theta0: Initial phases (N,).
        t0, t1, dt: Time span and step.
        ts: Saved time points (passed through to the ODE solver).
        n_steps: Number of quadrature steps along the path (default 20).

    Returns:
        IG attribution vector of shape (N,).
    """
    scales = jnp.linspace(0.0, 1.0, n_steps)

    def grad_at_scale(s: jnp.ndarray) -> jnp.ndarray:
        alpha = s * alpha_target
        return jax.grad(obj_lesioned, argnums=1)(params, alpha, theta0, t0, t1, dt, ts)

    grads_path = jax.vmap(grad_at_scale)(scales)  # (n_steps, N)
    return jnp.trapezoid(grads_path, scales, axis=0) * alpha_target  # (N,)


def ig_mean_R_alpha(
    params: KuramotoParams,
    alpha_target: jnp.ndarray,
    theta0: jnp.ndarray,
    t0: float,
    t1: float,
    dt: float,
    ts: jnp.ndarray | None = None,
    n_steps: int = 20,
) -> jnp.ndarray:
    """Integrated Gradients of mean R w.r.t. node lesion params."""
    return integrated_grads_alpha(
        mean_order_parameter_lesioned, params, alpha_target, theta0, t0, t1, dt, ts, n_steps
    )


def ig_mean_r_link_alpha(
    params: KuramotoParams,
    alpha_target: jnp.ndarray,
    theta0: jnp.ndarray,
    t0: float,
    t1: float,
    dt: float,
    ts: jnp.ndarray | None = None,
    n_steps: int = 20,
) -> jnp.ndarray:
    """Integrated Gradients of mean R_link w.r.t. node lesion params."""
    return integrated_grads_alpha(
        mean_r_link_lesioned, params, alpha_target, theta0, t0, t1, dt, ts, n_steps
    )


def node_importance_from_gradK(K: jnp.ndarray, dJ_dK: jnp.ndarray) -> jnp.ndarray:
    """Aggregate edge sensitivities into a node-importance score.

    Implements:
      I_i = abs(sum_j (dJ/dK[i,j] * K[i,j] + dJ/dK[j,i] * K[j,i]))
    """
    K = jnp.asarray(K)
    dJ_dK = jnp.asarray(dJ_dK)
    term_out = jnp.sum(dJ_dK * K, axis=1)  # sum_j dJ/dK[i,j]*K[i,j]
    term_in = jnp.sum(dJ_dK * K, axis=0)   # sum_j dJ/dK[j,i]*K[j,i]
    return term_out + term_in # NOTE: leave signed for now


def get_adjoint_grads(
    sim: Simulation,
    *,
    T_END: float,
    dt: float,
    n_ig_steps: int = 20,
) -> dict[str, np.ndarray]:
    """Compute linearized and integrated-gradient node-lesion sensitivities.

    Args:
        sim: Simulation object (must already have run, so sim.theta0 is set).
        T_END: End time used for the adjoint solve.
        dt: Timestep.
        n_ig_steps: Number of quadrature steps for Integrated Gradients.

    Returns:
        Dictionary of sensitivity vectors, each (N,):
        - "dRf_dalpha"        : dRf/dα  (linearized, final-R objective)
        - "dRm_dalpha"        : dRm/dα  (linearized, mean-R objective)
        - "dRlink_dalpha"     : d(r_link)/dα (linearized, link-R objective)
        - "IG_dRm_dalpha"     : Integrated Gradients for mean-R
        - "IG_dRlink_dalpha"  : Integrated Gradients for link-R
    """
    t0, t1 = 0.0, T_END
    ts = jnp.arange(t0 + dt, t1 + dt / 2, dt)
    ts = ts[ts <= t1]

    alpha0 = jnp.zeros((sim.grid.N,), dtype=sim.params.K.dtype)
    alpha_target = jnp.ones((sim.grid.N,), dtype=sim.params.K.dtype)

    dRf_dalpha = grads_final_R_alpha(sim.params, alpha0, sim.theta0, t0, t1, dt, ts=jnp.array([t1]))
    dRm_dalpha = grads_mean_R_alpha(sim.params, alpha0, sim.theta0, t0, t1, dt, ts=ts)
    dRlink_dalpha = grads_mean_r_link_alpha(sim.params, alpha0, sim.theta0, t0, t1, dt, ts=ts)

    IG_dRm_dalpha = ig_mean_R_alpha(sim.params, alpha_target, sim.theta0, t0, t1, dt, ts=ts, n_steps=n_ig_steps)
    IG_dRlink_dalpha = ig_mean_r_link_alpha(sim.params, alpha_target, sim.theta0, t0, t1, dt, ts=ts, n_steps=n_ig_steps)

    return {
        "dRf_dalpha": np.asarray(dRf_dalpha),
        "dRm_dalpha": np.asarray(dRm_dalpha),
        "dRlink_dalpha": np.asarray(dRlink_dalpha),
        "IG_dRm_dalpha": np.asarray(IG_dRm_dalpha),
        "IG_dRlink_dalpha": np.asarray(IG_dRlink_dalpha),
    }

def get_adjoint_metrics(
    sim: Simulation | None = None,
    grads: dict[str, np.ndarray] | None = None,
    *,
    T_END: float | None = None,
    dt: float | None = None,
    n_ig_steps: int = 20,
) -> dict[str, np.ndarray]:
    """Compute linearized and integrated-gradient node-lesion sensitivities.

    Args:
        sim: Simulation object (must already have run, so sim.theta0 is set).
        grads: Dictionary of sensitivity vectors, each (N,):
        - "dRm_dalpha"        : dRm/dα  (linearized, mean-R objective)
        - "dRlink_dalpha"     : d(r_link)/dα (linearized, link-R objective)
        - "IG_dRm_dalpha"     : Integrated Gradients for mean-R
        - "IG_dRlink_dalpha"  : Integrated Gradients for link-R
        T_END: End time used for the adjoint solve.
        dt: Timestep.
        n_ig_steps: Number of quadrature steps for Integrated Gradients.

    Returns:
        Dictionary of sensitivity vectors, each (N,):
        - "IRm_a"        : -dRm/dα  (linearized, mean-R objective)
        - "IRlink_a"     : -dRlink/dα (linearized, link-R objective)
        - "IG_IRm_a"     : -IG dRm/dα (Integrated Gradients for mean-R)
        - "IG_IRlink_a"  : -IG dRlink/dα (Integrated Gradients for link-R)
    """
    if sim is not None:
        if T_END is None or dt is None:
            raise ValueError("Provide T_END and dt when sim is provided.")
        if grads is not None:
            raise ValueError("Provide either sim or grads, not both.")
        t0, t1 = 0.0, T_END
        ts = jnp.arange(t0 + dt, t1 + dt / 2, dt)
        ts = ts[ts <= t1]

        alpha0 = jnp.zeros((sim.grid.N,), dtype=sim.params.K.dtype)
        alpha_target = jnp.ones((sim.grid.N,), dtype=sim.params.K.dtype)

        dRm_dalpha = grads_mean_R_alpha(sim.params, alpha0, sim.theta0, t0, t1, dt, ts=ts)
        dRlink_dalpha = grads_mean_r_link_alpha(sim.params, alpha0, sim.theta0, t0, t1, dt, ts=ts)

        IG_dRm_dalpha = ig_mean_R_alpha(sim.params, alpha_target, sim.theta0, t0, t1, dt, ts=ts, n_steps=n_ig_steps)
        IG_dRlink_dalpha = ig_mean_r_link_alpha(sim.params, alpha_target, sim.theta0, t0, t1, dt, ts=ts, n_steps=n_ig_steps)
    elif grads is not None:
        dRm_dalpha = grads["dRm_dalpha"]
        dRlink_dalpha = grads["dRlink_dalpha"]
        IG_dRm_dalpha = grads["IG_dRm_dalpha"]
        IG_dRlink_dalpha = grads["IG_dRlink_dalpha"]
    else:
        raise ValueError("Provide either sim or grads.")

    # NOTE: negative sign since increasing alpha results in negative grad if node is important
    IRm_a = -dRm_dalpha
    IRlink_a = -dRlink_dalpha
    IG_IRm_a = -IG_dRm_dalpha
    IG_IRlink_a = -IG_dRlink_dalpha

    return {
        "IRm_a": np.asarray(IRm_a),
        "IRlink_a": np.asarray(IRlink_a),
        "IG_IRm_a": np.asarray(IG_IRm_a),
        "IG_IRlink_a": np.asarray(IG_IRlink_a),
    }

# --- Finite difference ---
def finite_diff_dJ_dalpha(sim, obj_fn, node_idx, t0, t1, dt, ts, eps=1e-2):
    """Central finite-difference for dJ/dalpha_i."""
    alpha_plus = jnp.zeros(sim.grid.N).at[node_idx].set(eps)
    alpha_minus = jnp.zeros(sim.grid.N).at[node_idx].set(-eps)

    K_plus = apply_node_lesions(sim.params.K, alpha_plus)
    K_minus = apply_node_lesions(sim.params.K, alpha_minus)

    J_plus = obj_fn(
        KuramotoParams(omega=sim.params.omega, K=K_plus),
        sim.theta0, t0, t1, dt, ts,
    )
    J_minus = obj_fn(
        KuramotoParams(omega=sim.params.omega, K=K_minus),
        sim.theta0, t0, t1, dt, ts,
    )
    return float((J_plus - J_minus) / (2 * eps))

# --- Visualization --- # TODO: move to plotting.py
def get_norm_cmap(array: jnp.ndarray) -> tuple[Normalize, str]:
    if np.any(array < 0) and np.any(array > 0):
        norm = TwoSlopeNorm(vmin=np.min(array), vcenter=0.0, vmax=np.max(array))
        cmap = "bwr"
    elif np.all(array >= 0):
        norm = Normalize(vmin=0, vmax=np.max(array))
        cmap = "Reds"
    else: # np.all(array <= 0)
        norm = Normalize(vmin=np.min(array), vmax=0.0)
        cmap = "Blues"
    return norm, cmap

def plot_basic_grads(g: KuramotoParams, grid_shape: tuple[int, int], title: str = "Basic gradients") -> None:
    dR_dK = np.asarray(g.K)
    dR_domega = np.asarray(g.omega)
    dR_domega_2d = dR_domega.reshape(grid_shape)

    N_edges = dR_dK.shape[0]
    fig, axs = plt.subplots(1,2,figsize=(12,5))
    axs = axs.ravel()

    norm, cmap = get_norm_cmap(dR_dK)

    im = axs[0].imshow(dR_dK, cmap=cmap, norm=norm)
    axs[0].set_title("dJ/dK")
    if N_edges < 20:
        axs[0].set_xticks(np.arange(0, N_edges, 1))
        axs[0].set_yticks(np.arange(0, N_edges, 1))
    elif N_edges < 100:
        axs[0].set_xticks(np.arange(0, N_edges, 10))
        axs[0].set_yticks(np.arange(0, N_edges, 10))
    else:
        axs[0].set_xticks(np.arange(0, N_edges, 20))
        axs[0].set_yticks(np.arange(0, N_edges, 20))
    axs[0].tick_params(axis='x', labelrotation=45)
    fig.colorbar(im,ax=axs[0],fraction=0.046, pad=0.04)

    norm, cmap = get_norm_cmap(dR_domega_2d)

    im = axs[1].imshow(dR_domega_2d, cmap=cmap, norm=norm)
    axs[1].set_title("dJ/domega0")
    axs[1].set_xticks(np.arange(0, dR_domega_2d.shape[1], 1))
    axs[1].set_yticks(np.arange(0, dR_domega_2d.shape[0], 1))
    axs[1].tick_params(axis='x', labelrotation=45) 
    fig.suptitle(title)

def plot_advanced_grads(dR_dalpha: jnp.ndarray, I_node: jnp.ndarray, grid_shape: tuple[int, int], title: str = "Node importance") -> None:
    fig,ax = plt.subplots(1,2,figsize=(10,5),constrained_layout=True)
    
    norm, cmap = get_norm_cmap(dR_dalpha)
    im = ax[0].imshow(dR_dalpha.reshape(grid_shape), norm=norm, cmap=cmap)
    ax[0].set_title("dJ/dalpha")
    ax[0].set_xticks(np.arange(0, grid_shape[1], 1))
    ax[0].set_yticks(np.arange(0, grid_shape[0], 1))
    fig.colorbar(im,ax=ax[0],fraction=0.046, pad=0.04, norm=norm)


    norm, cmap = get_norm_cmap(I_node)
    im = ax[1].imshow(I_node.reshape(grid_shape), cmap=cmap, norm=norm)
    ax[1].set_title("I_node")
    ax[1].set_xticks(np.arange(0, grid_shape[1], 1))
    ax[1].set_yticks(np.arange(0, grid_shape[0], 1))
    fig.colorbar(im,ax=ax[1],fraction=0.046, pad=0.04, norm=norm)

    fig.suptitle(title)

# Grouped plotting 
def plot_adjoint_grads(
    grads: dict[str, np.ndarray] | None = None,
    sim: "Simulation | None" = None,
    grid_shape: tuple[int, int] | None = None,
    *,
    T_END: float | None = None,
    dt: float | None = None,
    n_ig_steps: int = 20,
    title: str | None = None,
    axs: list[plt.Axes] | None = None,
    cmap: str = "bwr",
) -> tuple[plt.Figure, np.ndarray]:
    """Plot linearized (top row) and Integrated Gradients (bottom row) adjoint sensitivities.

    Args:
        metrics: Pre-computed dict from ``get_adjoint_metrics``. If None, ``sim``,
                 ``T_END``, and ``dt`` must be provided.
        sim: Simulation object (used when metrics is None).
        grid_shape: 2-tuple (rows, cols) for reshaping the (N,) vectors to 2D.
                    Inferred from sim.grid if not given.
        T_END, dt, n_ig_steps: Passed to ``get_adjoint_metrics`` when needed.
        title: Figure suptitle.
        axs: Pre-existing 2×3 axes array. Created internally if None.
        cmap: Diverging colormap; default "bwr".

    Returns:
        (fig, axs)  — axs is shape (2, 3).
    """
    if grads is None:
        if sim is None or T_END is None or dt is None:
            raise ValueError("Provide either grads, or (sim, T_END, dt).")
        grads = get_adjoint_grads(sim, T_END=T_END, dt=dt, n_ig_steps=n_ig_steps)

    if grid_shape is None:
        if sim is None:
            raise ValueError("grid_shape must be provided when sim is None.")
        grid_shape = sim.grid.shape

    if axs is None:
        fig, axs = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    else:
        fig = axs[0, 0].get_figure()

    panels = [
        # (row, col, key,             display_title)
        (0, 0, "dRf_dalpha",       "dRf/dα (linear)"),
        (0, 1, "dRm_dalpha",       "dRm/dα (linear)"),
        (0, 2, "dRlink_dalpha",    "dRlink/dα (linear)"),
        (1, 1, "IG_dRm_dalpha",    f"IG dRm/dα ({n_ig_steps} steps)"),
        (1, 2, "IG_dRlink_dalpha", f"IG dRlink/dα ({n_ig_steps} steps)"),
    ]

    axs[1, 0].set_visible(False)  # no IG equivalent for Rf

    for row, col, key, panel_title in panels:
        ax = axs[row, col]
        data = grads[key].reshape(grid_shape)
        im = ax.imshow(data, norm=CenteredNorm(), cmap=cmap)
        ax.set_title(panel_title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title is not None:
        fig.suptitle(title)

    return fig, axs

def plot_adjoint_metrics(
    metrics: dict[str, np.ndarray] | None = None,
    sim: "Simulation | None" = None,
    grid_shape: tuple[int, int] | None = None,
    *,
    T_END: float | None = None,
    dt: float | None = None,
    n_ig_steps: int = 20,
    title: str | None = None,
    axs: list[plt.Axes] | None = None,
    cmap: str = "bwr",
) -> tuple[plt.Figure, np.ndarray]:
    """Plot linearized (top row) and Integrated Gradients (bottom row) adjoint sensitivities.

    Args:
        metrics: Pre-computed dict from ``get_adjoint_metrics``. If None, ``sim``,
                 ``T_END``, and ``dt`` must be provided.
        sim: Simulation object (used when metrics is None).
        grid_shape: 2-tuple (rows, cols) for reshaping the (N,) vectors to 2D.
                    Inferred from sim.grid if not given.
        T_END, dt, n_ig_steps: Passed to ``get_adjoint_metrics`` when needed.
        title: Figure suptitle.
        axs: Pre-existing 2×2 axes array. Created internally if None.
        cmap: Diverging colormap; default "bwr".

    Returns:
        (fig, axs)  — axs is shape (2, 2).
    """
    if metrics is None:
        if sim is None or T_END is None or dt is None:
            raise ValueError("Provide either metrics, or (sim, T_END, dt).")
        metrics = get_adjoint_metrics(sim, T_END=T_END, dt=dt, n_ig_steps=n_ig_steps)

    if grid_shape is None:
        if sim is None:
            raise ValueError("grid_shape must be provided when sim is None.")
        grid_shape = sim.grid.shape

    if axs is None:
        fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    else:
        fig = axs[0, 0].get_figure()

    panels = [
        # (row, col, key,             display_title)
        (0, 0, "IRm_a",       "Mean R (−d⟨R⟩/dα, Linear)"),
        (0, 1, "IRlink_a",    "Mean R_link (−d⟨R_link⟩/dα, Linear)"),
        (1, 0, "IG_IRm_a",    f"Mean R (−IG d⟨R⟩/dα, {n_ig_steps} steps)"),
        (1, 1, "IG_IRlink_a", f"Mean R_link (−IG d⟨R_link⟩/dα, {n_ig_steps} steps)"),
   
    ]

    for row, col, key, panel_title in panels:
        ax = axs[row, col]
        data = metrics[key].reshape(grid_shape)
        im = ax.imshow(data, norm=CenteredNorm(), cmap=cmap)
        ax.set_title(panel_title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title is not None:
        fig.suptitle(title)

    return fig, axs