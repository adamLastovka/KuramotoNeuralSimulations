from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from kuramoto.config import (
    SimulationConfig,
    GridConfig,
    CouplingConfig,
    InitThetaConfig,
    InitOmegaConfig,
    build_simulation,
)
from kuramoto.adjoint import (
    final_order_parameter,
    mean_order_parameter,
    grads_final_R,
    grads_mean_R,
)


def main():
    cfg = SimulationConfig(
        grid=GridConfig(shape=(16, 16)),
        coupling=CouplingConfig(
            mode="spatial",
            kernel="gaussian",
            base_strength=1.0,
            radius=5.0,
            kernel_params={"sigma": 2.0},
        ),
        initial_theta=InitThetaConfig(mode="uniform"),
        initial_omega=InitOmegaConfig(mode="normal", mu=0.0, sigma=0.3),
        seed=42,
    )

    rng = np.random.default_rng(42)
    sim = build_simulation(config=cfg, rng=rng)

    t0, t1, dt = 0.0, 5.0, 0.05
    ts = jnp.arange(t0 + dt, t1 + dt / 2, dt)
    ts = ts[ts <= t1]

    # Optional: JIT compile objectives and gradient wrappers.
    final_R_jit = jax.jit(final_order_parameter)
    mean_R_jit = jax.jit(mean_order_parameter)
    grads_final_jit = jax.jit(grads_final_R)
    grads_mean_jit = jax.jit(grads_mean_R)

    print("=== Adjoint demo: spatial coupling ===")
    print(f"N = {sim.grid.n_total} ({cfg.grid.shape[0]}x{cfg.grid.shape[1]})")
    print(f"Steps saved: {len(ts)}   dt={dt}   T={t1}")

    # Warmup compile
    _ = float(final_R_jit(sim.params, sim.theta0, t0, t1, dt, ts=ts))
    _ = float(mean_R_jit(sim.params, sim.theta0, t0, t1, dt, ts=ts))
    _ = grads_final_jit(sim.params, sim.theta0, t0, t1, dt, ts=ts)
    _ = grads_mean_jit(sim.params, sim.theta0, t0, t1, dt, ts=ts)

    # Timed run
    t_start = time.perf_counter()
    Rf = final_R_jit(sim.params, sim.theta0, t0, t1, dt, ts=ts)
    Rm = mean_R_jit(sim.params, sim.theta0, t0, t1, dt, ts=ts)
    gRf = grads_final_jit(sim.params, sim.theta0, t0, t1, dt, ts=ts)
    gRm = grads_mean_jit(sim.params, sim.theta0, t0, t1, dt, ts=ts)
    elapsed = time.perf_counter() - t_start

    # Force computation to finish before printing timings
    Rf_v = float(Rf)
    Rm_v = float(Rm)
    gRfK_max = float(jnp.max(jnp.abs(gRf.K)))
    gRfo_max = float(jnp.max(jnp.abs(gRf.omega)))
    gRmK_max = float(jnp.max(jnp.abs(gRm.K)))
    gRmo_max = float(jnp.max(jnp.abs(gRm.omega)))

    print(f"final R: {Rf_v:.6f}")
    print(f"mean  R: {Rm_v:.6f}")
    print(f"max |dR_final/dK|:   {gRfK_max:.6e}")
    print(f"max |dR_final/domega|: {gRfo_max:.6e}")
    print(f"max |dR_mean/dK|:    {gRmK_max:.6e}")
    print(f"max |dR_mean/domega|:  {gRmo_max:.6e}")
    print(f"Elapsed (objectives+grads, post-warmup): {elapsed:.3f}s")


if __name__ == "__main__":
    main()

