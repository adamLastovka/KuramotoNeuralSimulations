from __future__ import annotations

import time
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# REPO_ROOT = Path(__file__).resolve().parents[1]
# if str(REPO_ROOT) not in sys.path:
#     sys.path.insert(0, str(REPO_ROOT))

from kuramoto.adjoint import (
    grads_final_R,
    grads_final_R_alpha,
    grads_mean_R,
    grads_mean_R_alpha,
    final_order_parameter,
    final_order_parameter_lesioned,
    mean_order_parameter,
    mean_order_parameter_lesioned,
    node_importance_from_gradK,
)
from kuramoto.config import (
    CouplingConfig,
    GridConfig,
    InitOmegaConfig,
    InitThetaConfig,
    SimulationConfig,
    build_simulation,
)


def topk(I: jnp.ndarray, k: int = 10) -> tuple[np.ndarray, np.ndarray]:
    I_np = np.asarray(I)
    k = int(min(k, I_np.shape[0]))
    idx = np.argsort(-I_np)[:k]
    return idx, I_np[idx]


def finite_diff_alpha(
    objective_lesioned,
    params,
    alpha: jnp.ndarray,
    theta0: jnp.ndarray,
    t0: float,
    t1: float,
    dt: float,
    ts: jnp.ndarray,
    i: int,
    eps: float = 1e-3,
) -> float:
    ei = jnp.zeros_like(alpha).at[i].set(1.0)
    fp = objective_lesioned(params, alpha + eps * ei, theta0, t0, t1, dt, ts)
    fm = objective_lesioned(params, alpha - eps * ei, theta0, t0, t1, dt, ts)
    return float((fp - fm) / (2 * eps))


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

    N = int(sim.grid.n_total)
    alpha0 = jnp.zeros((N,), dtype=sim.params.K.dtype)

    # JIT wrappers
    final_R_jit = jax.jit(final_order_parameter)
    mean_R_jit = jax.jit(mean_order_parameter)
    grads_final_jit = jax.jit(grads_final_R)
    grads_mean_jit = jax.jit(grads_mean_R)

    final_R_les_jit = jax.jit(final_order_parameter_lesioned)
    mean_R_les_jit = jax.jit(mean_order_parameter_lesioned)
    grads_final_alpha_jit = jax.jit(grads_final_R_alpha)
    grads_mean_alpha_jit = jax.jit(grads_mean_R_alpha)

    print("=== Node lesion importance demo ===")
    print(f"N = {N} ({cfg.grid.shape[0]}x{cfg.grid.shape[1]})")
    print(f"Steps saved: {len(ts)}   dt={dt}   T={t1}")

    # Warmup compile
    _ = float(final_R_jit(sim.params, sim.theta0, t0, t1, dt, ts=ts))
    _ = float(mean_R_jit(sim.params, sim.theta0, t0, t1, dt, ts=ts))
    _ = grads_final_jit(sim.params, sim.theta0, t0, t1, dt, ts=ts)
    _ = grads_mean_jit(sim.params, sim.theta0, t0, t1, dt, ts=ts)
    _ = float(final_R_les_jit(sim.params, alpha0, sim.theta0, t0, t1, dt, ts=ts))
    _ = float(mean_R_les_jit(sim.params, alpha0, sim.theta0, t0, t1, dt, ts=ts))
    _ = grads_final_alpha_jit(sim.params, alpha0, sim.theta0, t0, t1, dt, ts=ts)
    _ = grads_mean_alpha_jit(sim.params, alpha0, sim.theta0, t0, t1, dt, ts=ts)

    # Timed run
    t_start = time.perf_counter()

    Rf = final_R_jit(sim.params, sim.theta0, t0, t1, dt, ts=ts)
    Rm = mean_R_jit(sim.params, sim.theta0, t0, t1, dt, ts=ts)
    gRf = grads_final_jit(sim.params, sim.theta0, t0, t1, dt, ts=ts)
    gRm = grads_mean_jit(sim.params, sim.theta0, t0, t1, dt, ts=ts)

    I_final = node_importance_from_gradK(sim.params.K, gRf.K)
    I_mean = node_importance_from_gradK(sim.params.K, gRm.K)

    dRf_dalpha = grads_final_alpha_jit(sim.params, alpha0, sim.theta0, t0, t1, dt, ts=ts)
    dRm_dalpha = grads_mean_alpha_jit(sim.params, alpha0, sim.theta0, t0, t1, dt, ts=ts)

    elapsed = time.perf_counter() - t_start

    print(f"final R: {float(Rf):.6f}")
    print(f"mean  R: {float(Rm):.6f}")
    print(f"Elapsed (objectives+grads+importance, post-warmup): {elapsed:.3f}s")

    k = 10
    idx_f, val_f = topk(I_final, k=k)
    idx_m, val_m = topk(I_mean, k=k)

    print(f"\nTop-{k} nodes by importance for final R(T):")
    for rank, (i, v) in enumerate(zip(idx_f.tolist(), val_f.tolist()), start=1):
        print(f"  {rank:2d}. node {i:4d}   I={v:.6e}   dR/dalpha={float(dRf_dalpha[i]):+.6e}")

    print(f"\nTop-{k} nodes by importance for mean R:")
    for rank, (i, v) in enumerate(zip(idx_m.tolist(), val_m.tolist()), start=1):
        print(f"  {rank:2d}. node {i:4d}   I={v:.6e}   dR/dalpha={float(dRm_dalpha[i]):+.6e}")

    # Finite-difference spot check for a high-importance node (final R)
    i0 = int(idx_f[0])
    fd = finite_diff_alpha(
        final_order_parameter_lesioned,
        sim.params,
        alpha0,
        sim.theta0,
        t0,
        t1,
        dt,
        ts,
        i=i0,
        eps=1e-3,
    )
    ad = float(dRf_dalpha[i0])
    print("\nFinite-difference check (final R):")
    print(f"  node {i0}: fd dR/dalpha={fd:+.6e}   adjoint dR/dalpha={ad:+.6e}")


if __name__ == "__main__":
    main()

