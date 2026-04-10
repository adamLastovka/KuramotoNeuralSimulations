from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import yaml

from .coupling import CouplingMatrix, closest_square_shape
from .grid import CorticalGrid
from .simulation import Simulation


# --- config dataclasses ---
@dataclass
class GridConfig:
    shape: tuple[int, int] = (32, 32)
    periodic: bool = False


@dataclass
class KernelComponentConfig:
    kernel: str = "gaussian"
    base_strength: float = 1.0
    radius: float | None = None
    kernel_params: dict | None = None
    # Node selection/grouping for directed edge masks.
    #
    # Interpretation of directed coupling in this codebase:
    #   K[i,j] = strength * weight(i,j)  corresponds to oscillator j -> oscillator i.
    #
    # For each component, we build an edge mask M[i,j] from:
    # - `node_groups`: groups that the component "applies to"
    # - `edge_mode`: how `node_groups` maps onto directed edges
    # - optional `to_node_groups` for `edge_mode="custom"`
    node_groups: list[int] | None = None
    # Allowed: within/outgoing/incoming/custom
    # - within: receiver and sender are both in node_groups
    # - outgoing: sender (j) in node_groups, receiver (i) is any node
    # - incoming: receiver (i) in node_groups, sender (j) is any node
    # - custom: sender in node_groups, receiver in to_node_groups
    edge_mode: str = "outgoing"
    # Receiver groups used only for edge_mode="custom".
    to_node_groups: list[int] | None = None
    seed: int | None = None # For reproducibility of dropout kernel


@dataclass
class CouplingConfig:
    kernel: str = "gaussian"
    base_strength: float = 1.0
    radius: float | None = 5.0
    kernel_params: dict | None = None
    mode: str = "spatial"
    # Heterogeneous coupling (optional):
    components: list[KernelComponentConfig] | None = None
    # Group membership per node, used to build pairwise masks for heterogeneous components.
    # Length must equal grid.N.
    group_ids: list[int] | None = None
    # Preset coupling matrix (N, N).  When provided, kernel/mode/components are
    # ignored and this matrix is used directly as K.
    K_matrix: np.ndarray | None = field(default=None, repr=False)


@dataclass
class NeuronConfig:
    omega_min: float = 0.9
    omega_max: float = 1.1


@dataclass
class InitThetaConfig:
    mode: str = "vonmises"
    mu: float = np.pi
    sigma: float = 1.0
    gamma: float = 0.3

@dataclass
class InitOmegaConfig:
    mode: str = "normal"
    mu: float = 0
    sigma: float = 0.5
    gamma: float = 0.5


@dataclass
class SimulationConfig:
    grid: GridConfig = field(default_factory=GridConfig)
    coupling: CouplingConfig = field(default_factory=CouplingConfig)
    neurons: NeuronConfig | None = None
    initial_theta: InitThetaConfig = field(default_factory=InitThetaConfig)
    initial_omega: InitOmegaConfig = field(default_factory=InitOmegaConfig)
    seed: int | None = 42

# --- Distributions ---
def get_distribution(rng: np.random.Generator, n: int, config: InitOmegaConfig | InitThetaConfig) -> np.ndarray:
    if config.mode == "normal":
        return rng.normal(config.mu, config.sigma, size=n)
    elif config.mode == "uniform":
        return rng.uniform(0, 2 * np.pi, size=n)
    elif config.mode == "laplace":
        return rng.laplace(config.mu, config.gamma, size=n)
    elif config.mode == "cauchy":
        return rng.standard_cauchy(size=n)
    elif config.mode == "exponential":
        return rng.exponential(scale=config.gamma, size=n)
    elif config.mode == "gamma":
        return rng.gamma(shape=config.gamma, scale=config.mu, size=n)
    elif config.mode == "lognormal":
        return rng.lognormal(config.mu, config.sigma, size=n)
    elif config.mode == "vonmises":
        return rng.vonmises(config.mu, config.gamma, size=n)
    else:
        raise ValueError(f"Invalid distribution mode: {config.mode}")


# --- YAML Configuration loading ---
def load_config(path: str | Path) -> SimulationConfig:
    """Read a YAML file into a SimulationConfig."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    raw_grid = data.get("grid", {})
    grid = GridConfig(
        shape=tuple(raw_grid.get("shape", [32, 32])),
        periodic=bool(raw_grid.get("periodic", False)),
    )

    raw_coupling = data.get("coupling", {})
    raw_components = raw_coupling.get("components")
    raw_group_ids = raw_coupling.get("group_ids")

    components = None
    if raw_components is not None:
        if not isinstance(raw_components, list):
            raise ValueError("coupling.components must be a list")
        components = []
        for c in raw_components:
            components.append(
                KernelComponentConfig(
                    kernel=str(c.get("kernel", "gaussian")),
                    base_strength=float(c.get("base_strength", 1.0)),
                    radius=c.get("radius"),
                    kernel_params=dict(c.get("kernel_params", {}) or {}),
                    node_groups=list(c["node_groups"])
                    if c.get("node_groups") is not None
                    else None,
                    edge_mode=str(c.get("edge_mode", "outgoing")),
                    to_node_groups=list(c["to_node_groups"])
                    if c.get("to_node_groups") is not None
                    else None,
                )
            )

    coupling = CouplingConfig(
        kernel=str(raw_coupling.get("kernel", "gaussian")),
        base_strength=float(raw_coupling.get("base_strength", 1.0)),
        radius=raw_coupling.get("radius"),
        kernel_params=dict(raw_coupling.get("kernel_params", {})),
        mode=str(raw_coupling.get("mode", "spatial")),
        components=components,
        group_ids=list(raw_group_ids) if raw_group_ids is not None else None,
    )

    raw_neurons = data.get("neurons", {})
    neurons = NeuronConfig(
        omega_min=float(raw_neurons.get("omega_min", 0.9)),
        omega_max=float(raw_neurons.get("omega_max", 1.1)),
    )

    raw_theta = data.get("initial_theta", "uniform")
    if isinstance(raw_theta, str):
        init_theta = InitThetaConfig(mode=raw_theta)
    else:
        init_theta = InitThetaConfig(
            mode=str(raw_theta.get("mode", "uniform")),
            mu=float(raw_theta.get("mu", np.pi)),
            sigma=float(raw_theta.get("sigma", 1.0)),
            gamma=float(raw_theta.get("gamma", 0.5)),
        )

    raw_omega = data.get("initial_omega", "normal")
    if isinstance(raw_omega, str):
        init_omega = InitOmegaConfig(mode=raw_omega)
    else:
        init_omega = InitOmegaConfig(
            mode=str(raw_omega.get("mode", "normal")),
            mu=float(raw_omega.get("mu", 0)),
            sigma=float(raw_omega.get("sigma", 0.5)),
            gamma=float(raw_omega.get("gamma", 0.5)),
        )

    return SimulationConfig(
        grid=grid,
        coupling=coupling,
        neurons=neurons,
        initial_theta=init_theta,
        initial_omega=init_omega,
        seed=data.get("seed"),
    )


# --- building simulation from config ---
_GRID_DEFAULT_SHAPE = GridConfig().shape


def build_simulation(
    config: SimulationConfig,
    rng: np.random.Generator | None = None,
) -> Simulation:
    if rng is None:
        rng = np.random.default_rng(config.seed)

    cc = config.coupling

    if cc.K_matrix is not None:
        K_mat = np.asarray(cc.K_matrix)
        if K_mat.ndim != 2 or K_mat.shape[0] != K_mat.shape[1]:
            raise ValueError(
                f"K_matrix must be a square 2-D array, got shape {K_mat.shape}"
            )
        N = K_mat.shape[0]

        grid_shape = config.grid.shape
        if grid_shape == _GRID_DEFAULT_SHAPE and grid_shape[0] * grid_shape[1] != N:
            grid_shape = closest_square_shape(N)

        grid = CorticalGrid(shape=grid_shape, periodic_bc=config.grid.periodic)
        if grid.N != N:
            raise ValueError(
                f"grid.N={grid.N} (from shape {grid_shape}) does not match "
                f"K_matrix size {N}"
            )
        coupling = CouplingMatrix(grid=grid, K_preset=K_mat, group_ids=cc.group_ids)
    else:
        grid = CorticalGrid(shape=config.grid.shape, periodic_bc=config.grid.periodic)
        coupling = CouplingMatrix(
            grid=grid,
            kernel=cc.kernel,
            base_strength=cc.base_strength,
            kernel_params=cc.kernel_params,
            radius=cc.radius,
            mode=cc.mode,
            components=cc.components,
            group_ids=cc.group_ids,
        )

    omega0 = jnp.array(get_distribution(rng, grid.N, config.initial_omega))
    theta0 = jnp.array(get_distribution(rng, grid.N, config.initial_theta))

    return Simulation(
        grid=grid,
        coupling=coupling,
        omega0=omega0,
        theta0=theta0,
        config=config,
    )


# --- Predefined network configurations ---
def get_groupids_bottleneck(grid_shape: tuple[int, int]) -> list[int]:
    n_rows, n_cols = grid_shape
    group_ids = np.zeros((n_rows, n_cols), dtype=int)
    group_ids[n_rows // 2 :, :] = 1  # bottom half
    group_ids[n_rows // 2 - 2 : n_rows // 2 + 2, n_cols // 2 - 2 : n_cols // 2 + 2] = 2  # central block
    return group_ids.ravel().tolist()

def get_groupids_unstructured(grid_shape: tuple[int, int], n_groups: int, rng: np.random.Generator) -> list[int]:
    n = int(np.prod(grid_shape))
    return rng.integers(0, n_groups, size=n, endpoint=False).astype(int).tolist()

def get_components_bottleneck(k_factor: float = 1.0, dropout_frac: float = 0.3, seed: int = 42) -> list[KernelComponentConfig]:
    return [
        KernelComponentConfig(
            kernel="gaussian",
            base_strength=0.5*k_factor,
            kernel_params={"sigma": 1.0},
            radius=2.0,
            node_groups=[0],
            edge_mode="within",
        ),
        KernelComponentConfig(
            kernel="gaussian",
            base_strength=0.5*k_factor,
            kernel_params={"sigma": 1.0},
            radius=2.0,
            node_groups=[1],
            edge_mode="within",
        ),
        # bottleneck-like hub (group 2) coupling to/from everyone
        KernelComponentConfig(
            kernel="gaussian",
            base_strength=0.4*k_factor,
            kernel_params={"sigma": 4.0},
            radius=4.0,
            node_groups=[2],
            edge_mode="outgoing",
        ),
        KernelComponentConfig(
            kernel="gaussian",
            base_strength=0.4*k_factor,
            kernel_params={"sigma": 4.0},
            radius=4.0,
            node_groups=[2],
            edge_mode="incoming",
        ),
        # weak one-way coupling from group 1 to group 0
        KernelComponentConfig(
            kernel="gaussian",
            base_strength=0.1*k_factor,
            kernel_params={"sigma": 2.0},
            radius=2.0,
            node_groups=[1],
            edge_mode="custom",
            to_node_groups=[0],
        ),
        # Apply dropout to all groups
        KernelComponentConfig(
        kernel="dropout",
        kernel_params={"dropout_frac": dropout_frac},
        node_groups=[0,1,2],
        edge_mode="within",
        seed=seed,
        )
    ]

def get_components_unstructured(n_groups: int = 6, k_factor: float = 2.0, seed: int = 42) -> list[KernelComponentConfig]:
    if n_groups % 2 == 1:
        raise ValueError("n_groups must be even")

    # Size bounds
    SIZE_LB = 0.5
    SIZE_UB = 2.0

    # Size factors
    size_factors = np.random.uniform(SIZE_LB, SIZE_UB, size=n_groups-1)

    comps: list[KernelComponentConfig] = []
    for g, sf in enumerate(size_factors):
        comps.append(
            KernelComponentConfig(
                kernel="gaussian",
                base_strength=1.0*k_factor,
                kernel_params={"sigma": 1.0*sf},
                radius=3.0*sf,
                node_groups=[g],
                edge_mode="within",
            )
        )

    # Only one mexican hat component
    mh_params = {"sigma_e": 1.0, "sigma_i": 3.0, "a_e": 1.0, "a_i": 0.8}
    comps.append(
        KernelComponentConfig(
            kernel="mexican_hat",
            base_strength=0.3,
            kernel_params=mh_params,
            radius=4.0,
            node_groups=[n_groups-1],
            edge_mode="within",
        )
    )

    # Apply dropout to all groups
    comps.append(KernelComponentConfig(
        kernel="dropout",
        kernel_params={"dropout_frac": 0.3},
        node_groups=list(range(n_groups)),
        edge_mode="within",
        seed=seed,
    ))

    return comps