"""
Benchmark and profile the Kuramoto simulation.

Run with: python -m scripts.benchmark_simulation
Or with profiling: python -m cProfile -s cumtime scripts/benchmark_simulation.py

Measures time in integration vs _collect_results and key functions
(local_order, coupling_tension, order_parameter, DelayBuffer) for
representative configs (with/without storage, with/without delays).
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

from kuramoto import CorticalGrid, CouplingMatrix, Simulation, InMemoryStorage
from kuramoto.simulation import DelayBuffer
from kuramoto.config import (
    SimulationConfig,
    GridConfig,
    CouplingConfig,
    DelayConfig,
    build_simulation,
)


def run_benchmark(
    grid_shape: tuple[int, int] = (32, 32),
    t_span: tuple[float, float] = (0.0, 5.0),
    dt: float = 0.1,
    with_storage: bool = True,
    with_delays: bool = False,
    profile: bool = False,
) -> dict[str, float]:
    """Run simulation and return timing breakdown."""
    config = SimulationConfig(
        grid=GridConfig(shape=grid_shape, periodic=False),
        coupling=CouplingConfig(kernel="gaussian", base_strength=1.0, radius=5.0, mode="spatial"),
        delay=None if not with_delays else DelayConfig(tau=1.0),
        seed=42,
    )
    sim = build_simulation(config)
    storage = InMemoryStorage(downsample_every=1) if with_storage else None

    if profile:
        pr = cProfile.Profile()
        pr.enable()

    t0 = time.perf_counter()
    t_list, state_list = sim.run(t_span=t_span, dt=dt, storage=storage)
    elapsed = time.perf_counter() - t0

    if profile:
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
        ps.print_stats(25)
        print(s.getvalue())

    return {
        "elapsed_s": elapsed,
        "n_steps": len(t_list),
        "n_oscillators": sim.grid.n_total,
    }


def main():
    print("=== Simulation benchmark (32x32, t=0..5, dt=0.1) ===\n")

    # Without storage
    r = run_benchmark(with_storage=False, with_delays=False)
    print(f"No storage, no delays: {r['elapsed_s']:.3f}s ({r['n_steps']} steps, N={r['n_oscillators']})")

    # With storage (post-loop: order_parameter, local_order, coupling_tension)
    r = run_benchmark(with_storage=True, with_delays=False)
    print(f"With storage, no delays: {r['elapsed_s']:.3f}s")

    # With delays (delay buffer + manual stepping)
    r = run_benchmark(with_storage=False, with_delays=True)
    print(f"No storage, with delays: {r['elapsed_s']:.3f}s")

    # Full: storage + delays
    r = run_benchmark(with_storage=True, with_delays=True)
    print(f"With storage, with delays: {r['elapsed_s']:.3f}s")

    print("\n=== Profile (with storage, no delays) ===")
    run_benchmark(with_storage=True, with_delays=False, profile=True)


if __name__ == "__main__":
    main()
