from __future__ import annotations

import numpy as np

from kuramoto.config import (
    SimulationConfig,
    GridConfig,
    CouplingConfig,
    InitThetaConfig,
    InitOmegaConfig,
    KernelComponentConfig,
    build_simulation,
)
from kuramoto.analysis import order_parameter


def make_group_ids(shape: tuple[int, int]) -> list[int]:
    """Split grid into two row-based groups (top vs bottom)."""
    n_rows, n_cols = shape
    group_ids = np.zeros((n_rows, n_cols), dtype=int)
    group_ids[n_rows // 2 :, :] = 1
    return group_ids.ravel().tolist()


def main():
    grid_shape = (8, 8)
    group_ids = make_group_ids(grid_shape)

    components = [
        KernelComponentConfig(
            kernel="gaussian",
            base_strength=1.0,
            kernel_params={"sigma": 1.0},
            radius=4.0,
            node_groups=[0],
            edge_mode="within",
        ),
        KernelComponentConfig(
            kernel="gaussian",
            base_strength=0.8,
            kernel_params={"sigma": 3.0},
            radius=4.0,
            node_groups=[1],
            edge_mode="within",
        ),
        KernelComponentConfig(
            kernel="gaussian",
            base_strength=0.4,
            kernel_params={"sigma": 2.0},
            radius=4.0,
            node_groups=[1],
            edge_mode="custom",
            to_node_groups=[0],
        )#,
        # KernelComponentConfig(
        #     kernel="gaussian",
        #     base_strength=0.4,
        #     kernel_params={"sigma": 2.0},
        #     radius=4.0,
        #     node_groups=[0],
        #     edge_mode="custom",
        #     to_node_groups=[1],
        # ),
    ]

    cfg = SimulationConfig(
        grid=GridConfig(shape=grid_shape, periodic=False),
        coupling=CouplingConfig(
            # legacy fields still required by dataclass; hetero uses `components`
            kernel="gaussian",
            base_strength=1.0,
            radius=4.0,
            mode="spatial",
            components=components,
            group_ids=group_ids,
        ),
        initial_theta=InitThetaConfig(mode="uniform"),
        initial_omega=InitOmegaConfig(mode="normal", mu=0.0, sigma=0.3),
        seed=42,
    )

    sim = build_simulation(config=cfg, rng=np.random.default_rng(42))
    results = sim.run((0.0, 2.0), dt=0.05)

    # Plot coupling matrix
    import matplotlib.pyplot as plt
    from kuramoto.plotting import plot_coupling_matrix
    plot_coupling_matrix(sim.params.K, title="Coupling matrix K")
    plt.show()

    R0, _ = order_parameter(results["theta"][0])
    Rf, _ = order_parameter(results["theta"][-1])

    print("=== Heterogeneous coupling demo ===")
    print(f"N={sim.grid.n_total}, grid={grid_shape}, groups={{0,1}}")
    print(f"R(t0)={R0:.4f}, R(tf)={Rf:.4f}")

    K = np.asarray(sim.params.K)
    print("K stats:")
    print(f"  min={K.min():.3e} max={K.max():.3e} mean={K.mean():.3e}")

if __name__ == "__main__":
    main()

