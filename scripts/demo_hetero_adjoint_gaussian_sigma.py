from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
from kuramoto.plotting import plot_2d, plot_coupling_matrix

from kuramoto.config import (
    GridConfig,
    CouplingConfig,
    InitThetaConfig,
    InitOmegaConfig,
    KernelComponentConfig,
    SimulationConfig,
    build_simulation,
)
from kuramoto.analysis import order_parameter_jax
from kuramoto.simulation import KuramotoParams, solve_forward


def main():
    seed = 42
    grid_shape = (6, 6)  # keep small for gradient smoke test

    # Initiallization configs
    theta_init_cfg = InitThetaConfig(mode="uniform")
    omega_init_cfg = InitOmegaConfig(mode="normal", mu=1.0, sigma=0.3)

    # Two groups (top vs bottom rows)
    n_rows, n_cols = grid_shape
    group_ids = np.zeros((n_rows, n_cols), dtype=int)
    group_ids[n_rows // 2 :, :] = 1
    group_ids = group_ids.ravel().tolist()

    # Fixed initial conditions and omega
    cfg = SimulationConfig(
        grid=GridConfig(shape=grid_shape, periodic=False),
        coupling=CouplingConfig(
            kernel="gaussian",
            base_strength=1.0,
            radius=4.0,
            mode="spatial",
            group_ids=group_ids,
        ),
        initial_theta=theta_init_cfg,
        initial_omega=omega_init_cfg,
        seed=seed,
    )
    sim = build_simulation(config=cfg, rng=np.random.default_rng(seed))

    t0, t1, dt = 0.0, 2.0, 0.1
    ts = jnp.arange(t0 + dt, t1 + dt / 2, dt)
    ts = ts[ts <= t1]

    theta0 = sim.theta0
    omega0 = sim.omega0

    # Build objective: final R as a function of sigma for the (0->0) component.
    def objective(sigma_00: jnp.ndarray) -> jnp.ndarray:
        components = [
            KernelComponentConfig(
                kernel="gaussian",
                base_strength=1.0,
                kernel_params={"sigma": sigma_00},
                radius=4.0,
                node_groups=[0],
                edge_mode="within",
            ),
            KernelComponentConfig(
                kernel="gaussian",
                base_strength=0.8,
                kernel_params={"sigma": 2.5},
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
            ),
            KernelComponentConfig(
                kernel="gaussian",
                base_strength=0.4,
                kernel_params={"sigma": 2.0},
                radius=4.0,
                node_groups=[0],
                edge_mode="custom",
                to_node_groups=[1],
            ),
        ]

        # Construct a coupling matrix inside the objective so sigma affects K and gradients.
        from kuramoto.coupling import CouplingMatrix

        grid = sim.grid
        coupling = CouplingMatrix(
            grid=grid,
            mode="spatial",
            components=components,
            group_ids=group_ids,
        )
        params = KuramotoParams(omega=omega0, K=coupling.K)
        sol = solve_forward(params, theta0, t0=t0, t1=t1, dt=dt, ts=ts)
        return order_parameter_jax(sol.ys[-1])

    sigma0 = jnp.array(1.2)
    dR_dsigma = jax.grad(objective)(sigma0)

    print("=== Hetero coupling sigma gradient ===")
    print(f"sigma_00={float(sigma0):.3f}")
    print(f"dR_final/dsigma_00={float(dR_dsigma):.6e}")

    # Adjoint gradients w.r.t. K and omega
    sim_het = build_simulation(
        config=SimulationConfig(
            grid=GridConfig(shape=grid_shape, periodic=False),
            coupling=CouplingConfig(
                kernel="gaussian",
                base_strength=1.0,
                radius=4.0,
                mode="spatial",
                components=[
                    KernelComponentConfig(
                        kernel="gaussian",
                        base_strength=1.0,
                        kernel_params={"sigma": float(sigma0)},
                        radius=4.0,
                        node_groups=[0],
                        edge_mode="within",
                    ),
                    KernelComponentConfig(
                        kernel="gaussian",
                        base_strength=0.8,
                        kernel_params={"sigma": 2.5},
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
                    ),
                    KernelComponentConfig(
                        kernel="gaussian",
                        base_strength=0.4,
                        kernel_params={"sigma": 2.0},
                        radius=4.0,
                        node_groups=[0],
                        edge_mode="custom",
                        to_node_groups=[1],
                    ),
                ],
                group_ids=group_ids,
            ),
            initial_theta=theta_init_cfg,
            initial_omega=omega_init_cfg,
            seed=seed,
        ),
        rng=np.random.default_rng(seed),
    )

    from kuramoto.adjoint import grads_final_R

    g = grads_final_R(sim_het.params, sim_het.theta0, t0=t0, t1=t1, dt=dt, ts=ts)

    dK = np.asarray(g.K)
    domega = np.asarray(g.omega)

    domega_2d = sim_het.grid.unflatten(domega)

    print("Adjoint grad sanity:")
    print(f"  max |dR/dK| = {float(jnp.max(jnp.abs(g.K))):.6e}")
    print(f"  max |dR/domega| = {float(jnp.max(jnp.abs(g.omega))):.6e}")

    # Plot coupling matrix
    plot_coupling_matrix(sim_het.params.K, title="Coupling matrix K")
    plot_2d(sim_het.grid.unflatten(sim_het.params.omega), variable="omega", title="Natural frequencies omega")

    # Plot sensitivity matrix
    fig,ax = plt.subplots(1, 2, figsize=(10, 5))
    img1 = ax[0].imshow(dK, cmap="viridis", aspect="equal")
    ax[0].set_title("Sensitivity matrix dR/dK")
    img2 = ax[1].imshow(domega_2d, cmap="viridis", aspect="equal")
    ax[1].set_title("Sensitivity matrix dR/domega0")
    plt.colorbar(img1, ax=ax[0])
    plt.colorbar(img2, ax=ax[1])
    plt.show()

if __name__ == "__main__":
    main()