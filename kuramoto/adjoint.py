from __future__ import annotations

import jax
import jax.numpy as jnp

from .analysis import order_parameter_jax
from .simulation import KuramotoParams, solve_forward


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


# ---------------------------------------------------------------------------
# Gradient helpers (convenience wrappers around jax.grad)
# ---------------------------------------------------------------------------

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
