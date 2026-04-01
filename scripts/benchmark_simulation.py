"""Benchmark forward and adjoint performance.

Run with:
  - `py -3 -m scripts.benchmark_simulation`
  - profiling: `py -3 -m cProfile -s cumtime scripts/benchmark_simulation.py`

Benchmarks:
  - forward solve (uniform + spatial)
  - forward solve + storage (optional)
  - adjoint gradients (final R and mean R) for spatial coupling
"""
from __future__ import annotations

import cProfile
import io
import pstats
import time
from pathlib import Path

# Add project root for imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import jax
import jax.numpy as jnp
import numpy as np

from kuramoto.analysis import order_parameter, local_order
from kuramoto.config import (
    SimulationConfig,
    GridConfig,
    CouplingConfig,
    build_simulation,
)
from kuramoto.adjoint import grads_final_R, grads_mean_R


def run_benchmark(
    grid_shape: tuple[int, int] = (32, 32),
    t_span: tuple[float, float] = (0.0, 5.0),
    dt: float = 0.1,
    with_postprocess: bool = False,
    profile: bool = False,
    coupling_mode: str = "spatial",
) -> dict[str, float]:
    """Run simulation and return timing breakdown."""
    config = SimulationConfig(
        grid=GridConfig(shape=grid_shape, periodic=False),
        coupling=CouplingConfig(
            kernel="gaussian",
            base_strength=1.0,
            radius=5.0,
            kernel_params={"sigma": 2.0},
            mode=coupling_mode,
        ),
        seed=42,
    )
    sim = build_simulation(config, rng=np.random.default_rng(42))

    if profile:
        pr = cProfile.Profile()
        pr.enable()

    t0 = time.perf_counter()
    results = sim.run(t_span=t_span, dt=dt)
    if with_postprocess:
        # Simple postprocessing to include NumPy/SciPy cost.
        theta_final = results["theta"][-1]
        R_final, _ = order_parameter(theta_final)
        _ = local_order(theta_final, sim.grid)
        # Force evaluation of R_final.
        _ = float(R_final)
    elapsed = time.perf_counter() - t0

    if profile:
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
        ps.print_stats(25)
        print(s.getvalue())

    return {
        "elapsed_s": elapsed,
        "n_steps": int(results["ts"].shape[0]),
        "n_oscillators": sim.grid.N,
    }


def run_adjoint_benchmark(
    grid_shape: tuple[int, int] = (32, 32),
    t_span: tuple[float, float] = (0.0, 5.0),
    dt: float = 0.1,
) -> dict[str, float]:
    config = SimulationConfig(
        grid=GridConfig(shape=grid_shape, periodic=False),
        coupling=CouplingConfig(
            kernel="gaussian",
            base_strength=1.0,
            radius=5.0,
            kernel_params={"sigma": 2.0},
            mode="spatial",
        ),
        seed=42,
    )
    sim = build_simulation(config, rng=np.random.default_rng(42))

    ts = jnp.arange(t_span[0] + dt, t_span[1] + dt / 2, dt)
    ts = ts[ts <= t_span[1]]

    grads_final_jit = jax.jit(grads_final_R)
    grads_mean_jit = jax.jit(grads_mean_R)

    # compile
    t0 = time.perf_counter()
    _ = grads_final_jit(sim.params, sim.theta0, t_span[0], t_span[1], dt, ts=ts)
    _ = grads_mean_jit(sim.params, sim.theta0, t_span[0], t_span[1], dt, ts=ts)
    T_compile = time.perf_counter() - t0

    # run
    t0 = time.perf_counter()
    g1 = grads_final_jit(sim.params, sim.theta0, t_span[0], t_span[1], dt, ts=ts)
    g2 = grads_mean_jit(sim.params, sim.theta0, t_span[0], t_span[1], dt, ts=ts)
    _ = float(jnp.max(jnp.abs(g1.K))) + float(jnp.max(jnp.abs(g2.K)))
    T_run = time.perf_counter() - t0

    return {
        "T_compile": T_compile,
        "T_run": T_run,
        "n_steps": int(ts.shape[0]),
        "n_oscillators": sim.grid.N,
    }


def main():
    print("=== Simulation benchmark (32x32, t=0..5, dt=0.1) ===\n")

    r = run_benchmark(with_postprocess=False, coupling_mode="uniform")
    print(
        f"Uniform, no storage: {r['elapsed_s']:.3f}s "
        f"({r['n_steps']} steps, N={r['n_oscillators']})"
    )

    r = run_benchmark(with_postprocess=False, coupling_mode="spatial")
    print(
        f"Spatial, no storage: {r['elapsed_s']:.3f}s "
        f"({r['n_steps']} steps, N={r['n_oscillators']})"
    )

    r = run_benchmark(with_postprocess=True, coupling_mode="spatial")
    print(f"Spatial, with postprocess: {r['elapsed_s']:.3f}s")


    print("\n=== Adjoint benchmark (spatial) ===\n")
    r = run_adjoint_benchmark()
    print(
        f"Adjoint grads (Compile, Run, Total): {r['T_compile']:.3f}s, {r['T_run']:.3f}s, {r['T_compile'] + r['T_run']:.3f}s "
        f"({r['n_steps']} saved steps, N={r['n_oscillators']})"
    )

    # print("\n=== Profile (spatial, with storage) ===")
    # run_benchmark(with_postprocess=True, coupling_mode="spatial", profile=True)


if __name__ == "__main__":
    main()
