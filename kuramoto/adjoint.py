from __future__ import annotations

import jax
import jax.numpy as jnp

from .analysis import order_parameter_jax
from .simulation import KuramotoParams, solve_forward


# --- Lesion parametrization utilities ---
def apply_node_lesions(K: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
    """Apply continuous node lesions to a coupling matrix.

    alpha_i in [0, 1] scales all edges incident to node i:
      K[i, j] -> (1 - alpha_i) * K[i, j]
      K[j, i] -> (1 - alpha_i) * K[j, i]

    Vectorized form implements row then column scaling, yielding:
      K_eff[i, j] = (1 - alpha_i) * (1 - alpha_j) * K[i, j]
    """
    alpha = jnp.asarray(alpha)
    s = 1.0 - alpha  # (N,)
    return (s[:, None] * K) * s[None, :]


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


def node_importance_from_gradK(K: jnp.ndarray, dJ_dK: jnp.ndarray) -> jnp.ndarray:
    """Aggregate edge sensitivities into a node-importance score.

    Implements:
      I_i = abs(sum_j (dJ/dK[i,j] * K[i,j] + dJ/dK[j,i] * K[j,i]))
    """
    K = jnp.asarray(K)
    dJ_dK = jnp.asarray(dJ_dK)
    term_out = jnp.sum(dJ_dK * K, axis=1)  # sum_j dJ/dK[i,j]*K[i,j]
    term_in = jnp.sum(dJ_dK * K, axis=0)   # sum_j dJ/dK[j,i]*K[j,i]
    return jnp.abs(term_out + term_in)
