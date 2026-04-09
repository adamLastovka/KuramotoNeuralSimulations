from jax import numpy as jnp
import numpy as np

from kuramoto.coupling import apply_node_lesions
from .simulation import Simulation

from kuramoto.network import create_cortical_graph, get_graph_metrics
from kuramoto.analysis import avg_effective_coupling, functional_connectivity
from kuramoto.adjoint import grads_final_R, grads_mean_R, node_importance_from_gradK, grads_final_R_alpha, grads_mean_R_alpha, grads_mean_r_link_alpha
from kuramoto.analysis import order_parameter

def run_lesion_study(sim: Simulation, metric: jnp.ndarray | str, lesion_frac: float, lesion_strength: float = 1.0, T_END: float = 10.0, dt: float = 0.01, SEED: int = 42) -> None:
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
    RNG = np.random.default_rng(SEED)
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

def evaluate_metric_scores(sim: Simulation, T_END: float = 10.0, dt: float = 0.01, RNG: np.random.Generator = None, n_random_repeats: int = 10, lesion_fracs: np.ndarray = np.arange(0, 0.3, 0.02), lesion_strength: float = 1.0,verbose: bool = True) -> dict[str, float]:
    """Evaluate metric scores for a given simulation.
    
    Args:
        sim: Simulation object
        T_END: End time
        dt: Time step
        RNG: Random number generator
        n_random_repeats: Number of random repeats
        lesion_fracs: Lesion fractions
        lesion_strength: Lesion strength
    Returns:
        metric_scores: Metric scores
    """
    # Run base simulation
    if verbose:
        print("Running base simulation...")
    res_base = sim.run((0, T_END), dt, rng=RNG)
    
    # Create graphs
    if verbose:
        print("Creating graphs...")
    G = create_cortical_graph(sim)

    K_eff_avg = avg_effective_coupling(sim.results["theta"], sim.coupling.K)
    G_eff = create_cortical_graph(K_eff_avg, omega=sim.params.omega)

    C_avg = functional_connectivity(sim.results["theta"], dt=dt)
    G_C_avg = create_cortical_graph(C_avg, omega=sim.params.omega)

    # Graph metrics
    if verbose:
        print("Calculating graph metrics...")
    graph_metrics = get_graph_metrics(G)
    graph_metrics_eff = get_graph_metrics(G_eff)
    graph_metrics_C_avg = get_graph_metrics(G_C_avg)

    # Adjoint metrics
    if verbose:
        print("Calculating adjoint metrics...")
    t0, t1 = 0.0, T_END
    ts = jnp.arange(t0+dt, t1 + dt / 2, dt)
    ts = ts[ts <= t1]

    g = grads_final_R(sim.params, sim.theta0, t0=0.0, t1=T_END, dt=dt, ts=[T_END])
    g_avg = grads_mean_R(sim.params, sim.theta0, t0=0.0, t1=T_END, dt=dt, ts=ts)

    # equivalent to IRf_a and IRm_a so not included
    # IRf_k = node_importance_from_gradK(sim.params.K, g.K)
    # IRm_k = node_importance_from_gradK(sim.params.K, g_avg.K)

    alpha0 = jnp.zeros((sim.grid.N,), dtype=sim.params.K.dtype)
    dRf_dalpha = grads_final_R_alpha(sim.params, alpha0, sim.theta0, t0, t1, dt, ts=jnp.array([t1]))
    dRm_dalpha = grads_mean_R_alpha(sim.params, alpha0, sim.theta0, t0, t1, dt, ts=ts)
    dr_link_dalpha = grads_mean_r_link_alpha(sim.params, alpha0, sim.theta0, t0, t1, dt, ts=ts)

    IRf_a = -dRf_dalpha
    IRm_a = -dRm_dalpha
    IRlink_a = -dr_link_dalpha

    metrics = {
        "deg_base": graph_metrics["deg_cent"],
        "deg_eff": graph_metrics_eff["deg_cent"],
        "deg_C_avg": graph_metrics_C_avg["deg_cent"],
        "closeness_base": graph_metrics["closeness"],
        "closeness_eff": graph_metrics_eff["closeness"],
        "closeness_C_avg": graph_metrics_C_avg["closeness"],
        "betweenness_base": graph_metrics["betweenness"],
        "betweenness_eff": graph_metrics_eff["betweenness"],
        "betweenness_C_avg": graph_metrics_C_avg["betweenness"],
        "eigenvector_base": graph_metrics["eigenvector"],
        "eigenvector_eff": graph_metrics_eff["eigenvector"],
        "eigenvector_C_avg": graph_metrics_C_avg["eigenvector"],
        # "I_final_base": I_final,
        # "IRm_k_base": IRm_k,
        # "IRf_a_base": IRf_a, # not including Rf as performance metric
        "IRm_a_base": IRm_a,
        "IRlink_a_base": IRlink_a,
    }

    # Run lesion study for each metric
    if verbose:
        print("Running lesion study for each metric...")
    metric_scores = {}
    for metric_name, metric in metrics.items():
        print(f"Evaluating {metric_name}...")

        R_final_ranked = []
        R_avg_ranked = []
        R_final_random = []
        R_avg_random = []
        for lesion_frac in lesion_fracs:
            res_ranked_lesion, alpha_ranked, K_lesioned_ranked = run_lesion_study(sim, metric, lesion_frac, lesion_strength, T_END, dt, RNG)
            R_ranked_lesion, _ = order_parameter(res_ranked_lesion["theta"])

            R_random_lesion = np.zeros((len(ts), n_random_repeats))
            for i in range(n_random_repeats):
                res_random_lesion, alpha_random, K_lesioned_random = run_lesion_study(sim, "random", lesion_frac, lesion_strength, T_END, dt, RNG)
                R_random_lesion[:, i], _ = order_parameter(res_random_lesion["theta"])
            R_random_lesion = np.mean(R_random_lesion, axis=1)

            R_final_ranked.append(R_ranked_lesion[-1])
            R_avg_ranked.append(np.mean(R_ranked_lesion))
            R_final_random.append(R_random_lesion[-1])
            R_avg_random.append(np.mean(R_random_lesion))

        # Metric scores
        AUC_ranked = np.trapezoid(R_avg_ranked, lesion_fracs)
        AUC_random = np.trapezoid(R_avg_random, lesion_fracs)
        ABC = AUC_random - AUC_ranked

        metric_scores[metric_name] = {
            "AUC_ranked": AUC_ranked,
            "AUC_random": AUC_random,
            "ABC": ABC,
            "R_final_ranked": R_final_ranked,
            "R_avg_ranked": R_avg_ranked,
            "R_final_random": R_final_random,
            "R_avg_random": R_avg_random,
        }
    return metric_scores