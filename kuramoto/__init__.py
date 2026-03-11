"""Minimal Kuramoto oscillator package for 2D cortical sheets.

Supports uniform and spatial coupling, optional global time delays,
and standard analysis / visualisation helpers.
"""

from .grid import CorticalGrid
from .kernels import (
    gaussian_kernel,
    exponential_kernel,
    step_kernel,
    mexican_hat_kernel,
    gabor_kernel,
    elongated_gaussian_kernel,
    lesioned_wedge_kernel,
    apply_kernel,
)
from .coupling import CouplingMatrix
from .config import (
    GridConfig,
    CouplingConfig,
    NeuronConfig,
    InitThetaConfig,
    InitOmegaConfig,
    DelayConfig,
    SimulationConfig,
    load_config,
    build_simulation,
)
from .simulation import (
    Simulation,
    DelayBuffer,
    kuramoto_rhs,
    kuramoto_rhs_uniform,
)
from .analysis import (
    order_parameter,
    gradient_from_state,
    gradient_time_series,
    gradient_maps,
    bulk_gradient_metric,
    phase_gradient_2d,
    gradient_magnitude_2d,
    phase_time_derivative,
    gradient_and_time_derivative,
    phase_velocity_2d,
    material_derivative_2d,
    gradient_and_material_maps,
)
from .storage import InMemoryStorage
from .plotting import (
    plot_2d,
    plot_coupling_matrix,
    plot_gradient_map,
    coupling_to_2d,
    animate_2d,
    animate_from_run,
)

__all__ = [
    "CorticalGrid",
    "gaussian_kernel",
    "exponential_kernel",
    "step_kernel",
    "mexican_hat_kernel",
    "gabor_kernel",
    "elongated_gaussian_kernel",
    "lesioned_wedge_kernel",
    "apply_kernel",
    "CouplingMatrix",
    "GridConfig",
    "CouplingConfig",
    "NeuronConfig",
    "InitThetaConfig",
    "InitOmegaConfig",
    "DelayConfig",
    "SimulationConfig",
    "build_simulation",
    "load_config",
    "Simulation",
    "DelayBuffer",
    "kuramoto_rhs",
    "kuramoto_rhs_uniform",
    "order_parameter",
    "InMemoryStorage",
    "gradient_from_state",
    "gradient_time_series",
    "gradient_maps",
    "bulk_gradient_metric",
    "phase_gradient_2d",
    "gradient_magnitude_2d",
    "phase_time_derivative",
    "gradient_and_time_derivative",
    "phase_velocity_2d",
    "material_derivative_2d",
    "gradient_and_material_maps",
    "plot_2d",
    "plot_coupling_matrix",
    "plot_gradient_map",
    "coupling_to_2d",
    "animate_2d",
    "animate_from_run",
]
