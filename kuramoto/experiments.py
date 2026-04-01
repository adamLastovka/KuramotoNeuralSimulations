from jax import numpy as jnp
import numpy as np

from kuramoto.coupling import apply_node_lesions
from .simulation import Simulation

def run_lesion_study(sim: Simulation, metric: jnp.ndarray | str, lesion_frac: float, lesion_strength: float = 1.0, T_END: float = 10.0, dt: float = 0.01, RNG: np.random.Generator = None) -> None:
    """Run a lesion study for a given metric and lesion fraction.
    
    Args:
        sim: Simulation object
        metric: Metric vector to use for lesion study (N,) or 'random' for random lesioning
        lesion_frac: Fraction of nodes to lesion
        lesion_strength: Strength of lesion
        T_END: End time
        dt: Time step
        RNG: Random number generator
    
    Returns:
        results_lesioned: Results of lesioned simulation
        alpha: Lesion mask
    """
    N = sim.grid.N
    n_lesion = max(1, int(round(lesion_frac * N)))
    
    if isinstance(metric, str):
        if metric == "random":
            lesion_idx = RNG.choice(N, size=n_lesion, replace=False) # random lesioning
        else:
            raise ValueError(f"Invalid metric: {metric}")
    else:
        ranked_nodes = np.argsort(-np.asarray(metric, dtype=float)) # ranked lesioning
        lesion_idx = jnp.asarray(ranked_nodes[:n_lesion], dtype=jnp.int32)
    
    # Create lesion mask
    alpha = jnp.zeros((N,), dtype=jnp.float32)
    alpha = alpha.at[lesion_idx].set(lesion_strength) 

    # Run lesioned simulation
    results_lesioned = sim.run_with_lesions(alpha, (0, T_END), dt, rng=RNG) # run lesioned simulation
    return results_lesioned, alpha, apply_node_lesions(sim.coupling.K, alpha)