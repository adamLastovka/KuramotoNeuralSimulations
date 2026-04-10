from __future__ import annotations

import numpy as np

from kuramoto.config import (
    SimulationConfig,
    GridConfig,
    CouplingConfig,
    InitThetaConfig,
    InitOmegaConfig,
    build_simulation,
)
from kuramoto.analysis import get_R


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

    results = sim.run((0.0, 10.0), dt=0.05)

    R0, _ = get_R(results["theta"][0])
    Rf, _ = get_R(results["theta"][-1])

    print("=== Forward demo: spatial coupling ===")
    print(f"N = {sim.grid.N} ({cfg.grid.shape[0]}x{cfg.grid.shape[1]})")
    print(f"Steps: {results['ts'].shape[0]}   dt={0.05}   T={results['ts'][-1]:.3f}")
    print(f"R(t0)={R0:.4f}   R(tf)={Rf:.4f}")

if __name__ == "__main__":
    main()

