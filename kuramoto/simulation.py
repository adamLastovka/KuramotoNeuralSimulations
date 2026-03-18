from __future__ import annotations

from typing import NamedTuple

import numpy as np
import jax.numpy as jnp
import diffrax

from .coupling import CouplingMatrix
from .grid import CorticalGrid

class KuramotoParams(NamedTuple):
    """Differentiable parameter pytree for the Kuramoto model."""
    omega: jnp.ndarray  # (N,) natural frequencies
    K: jnp.ndarray       # (N, N) coupling weight matrix


def kuramoto_rhs(t, theta, params: KuramotoParams) -> jnp.ndarray:
    """Kuramoto RHS in JAX

    dtheta_i/dt = omega_i + sum_j K_ij sin(theta_j - theta_i)
    """
    sin_theta = jnp.sin(theta)
    cos_theta = jnp.cos(theta)
    coupling = cos_theta * (params.K @ jnp.sin(theta)) - sin_theta * (params.K @ jnp.cos(theta))
    return params.omega + coupling

def solve_forward(
    params: KuramotoParams,
    theta0: jnp.ndarray,
    t0: float,
    t1: float,
    dt: float,
    ts: jnp.ndarray | None = None,
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> diffrax.Solution:
    """Forward solve for the kuramoto model"""
    term = diffrax.ODETerm(kuramoto_rhs)
    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

    if ts is not None:
        saveat = diffrax.SaveAt(ts=ts)
    else:
        saveat = diffrax.SaveAt(t1=True)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=t0,
        t1=t1,
        dt0=dt,
        y0=theta0,
        args=params,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
    )
    return sol


class Simulation:
    """Kuramoto simulation object.
    Runs ``solve_forward()`` and collects results into dicts.
    """

    def __init__(
        self,
        grid: CorticalGrid,
        coupling: CouplingMatrix,
        omega0: jnp.ndarray,
        theta0: jnp.ndarray,
        config=None,
    ):
        self.grid = grid
        self.coupling = coupling
        self.omega0 = jnp.asarray(omega0)
        self.theta0 = jnp.asarray(theta0)
        self.config = config

        self.params = KuramotoParams(omega=self.omega0, K=self.coupling.K)
        self.results: dict[str, np.ndarray] | None = None

    def run(
        self,
        t_span: tuple[float, float],
        dt: float,
        rng=None,
    ) -> dict[str, np.ndarray]:
        """Integrate the Kuramoto model.

        Populates and returns ``self.results`` with keys:
          - ``ts``: (T,)
          - ``theta``: (T, N)
          - ``theta_dot``: (T, N)
          - ``omega``: (N,)
          - ``K``: (N, N)
        """
        t_eval = jnp.arange(t_span[0] + dt, t_span[1] + dt / 2, dt)
        t_eval = t_eval[t_eval <= t_span[1]]

        sol = solve_forward(
            self.params,
            self.theta0,
            t0=t_span[0],
            t1=t_span[1],
            dt=dt,
            ts=t_eval,
        )

        # Convert to numpy arrays
        ts = np.asarray(sol.ts)  # (T,)
        theta = np.asarray(sol.ys)  # (T, N) diffrax convention for y0=(N,)

        omega_np = np.asarray(self.omega0)  # (N,)
        K_np = np.asarray(self.coupling.K)  # (N, N)

        # Calculate theta_dot
        sin_theta = np.sin(theta)  # (T, N)
        cos_theta = np.cos(theta)  # (T, N)

        K_sin = K_np @ sin_theta.T  # (N, T)
        K_cos = K_np @ cos_theta.T  # (N, T)

        coupling = cos_theta * K_sin.T - sin_theta * K_cos.T  # (T, N)
        theta_dot = omega_np + coupling  # (T, N)

        self.results = {
            "ts": ts,
            "theta": theta,
            "theta_dot": theta_dot,
            "omega": omega_np,
            "K": K_np,
        }
        return self.results
