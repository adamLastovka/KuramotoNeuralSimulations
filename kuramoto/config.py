from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import yaml

from .coupling import CouplingMatrix
from .grid import CorticalGrid
from .simulation import Simulation


# --- config dataclasses ---
@dataclass
class GridConfig:
    shape: tuple[int, int] = (32, 32)
    periodic: bool = False


@dataclass
class CouplingConfig:
    kernel: str = "gaussian"
    base_strength: float = 1.0
    radius: float | None = 5.0
    kernel_params: dict | None = None
    mode: str = "spatial"


@dataclass
class NeuronConfig:
    omega_min: float = 0.9
    omega_max: float = 1.1


@dataclass
class InitThetaConfig:
    mode: str = "uniform"
    mu: float = np.pi
    sigma: float = 1.0
    gamma: float = 0.5

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
    coupling = CouplingConfig(
        kernel=str(raw_coupling.get("kernel", "gaussian")),
        base_strength=float(raw_coupling.get("base_strength", 1.0)),
        radius=raw_coupling.get("radius"),
        kernel_params=dict(raw_coupling.get("kernel_params", {})),
        mode=str(raw_coupling.get("mode", "spatial")),
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
def build_simulation(
    config: SimulationConfig,
    rng: np.random.Generator | None = None,
) -> Simulation:
    if rng is None:
        rng = np.random.default_rng(config.seed)

    grid = CorticalGrid(shape=config.grid.shape, periodic_bc=config.grid.periodic)

    cc = config.coupling
    coupling = CouplingMatrix(
        grid=grid,
        kernel=cc.kernel,
        base_strength=cc.base_strength,
        kernel_params=cc.kernel_params,
        radius=cc.radius,
        mode=cc.mode,
    )

    omega0 = jnp.array(get_distribution(rng, grid.n_total, config.initial_omega))
    theta0 = jnp.array(get_distribution(rng, grid.n_total, config.initial_theta))

    return Simulation(
        grid=grid,
        coupling=coupling,
        omega0=omega0,
        theta0=theta0,
        config=config,
    )
