from __future__ import annotations

import jax
import jax.numpy as jnp

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, CenteredNorm, TwoSlopeNorm

from .analysis import order_parameter_jax, r_link_jax
from .simulation import KuramotoParams, solve_forward
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

# --- Visualization ---
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