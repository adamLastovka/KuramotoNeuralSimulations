from __future__ import annotations

from typing import Any

from jax import numpy as jnp
import numpy as np

from kuramoto.coupling import apply_node_lesions
from .simulation import Simulation

from kuramoto.network import create_cortical_graph, get_graph_metrics
from kuramoto.analysis import avg_effective_coupling, functional_connectivity
from kuramoto.adjoint import grads_final_R_alpha, grads_mean_R_alpha, grads_mean_r_link_alpha, ig_mean_R_alpha, ig_mean_r_link_alpha
from kuramoto.analysis import get_R_jax

# Spacing between per-fraction blocks in deterministic random-lesion seeds (need n_repeats < this).
_RANDOM_REPEAT_STRIDE = 1_000


def evaluate_single_lesion(
    sim: Simulation,
    metric: jnp.ndarray | str,
    lesion_frac: float,
    lesion_strength: float = 1.0,
    T_END: float = 10.0,
    dt: float = 0.01,
    rng: np.random.Generator | int | None = None,
    *,
    SEED: int | None = None,
) -> tuple[dict, jnp.ndarray, jnp.ndarray]:
    """Run one lesioned forward solve (ranked by ``metric`` or random).

    Does not re-run the base simulation or rebuild graphs.

    Args:
        sim: Simulation with coupling and initial state already configured.
        metric: Per-node scores (N,) for ranked lesions, or the string ``"random"``.
        lesion_frac: Fraction of nodes to lesion.
        lesion_strength: Continuous lesion strength on selected nodes.
        T_END, dt: Horizon and step for ``run_with_lesions``.
        rng: Optional RNG, or an ``int`` seed (wrapped with ``default_rng``).
        SEED: Keyword-only alternative to passing an int as ``rng`` (backward compatible).

    Returns:
        ``results_lesioned``, lesion mask ``alpha``, effective coupling ``K_lesioned``.
    """
    N = sim.grid.N
    if isinstance(rng, np.random.Generator):
        RNG = rng
    elif rng is not None:
        RNG = np.random.default_rng(int(rng))
    elif SEED is not None:
        RNG = np.random.default_rng(SEED)
    else:
        RNG = np.random.default_rng(42)
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
    results_lesioned = sim.run_with_lesions(alpha, (0, T_END), dt, rng=RNG)
    return results_lesioned, alpha, apply_node_lesions(sim.coupling.K, alpha)


def run_lesion_study(
    sim: Simulation,
    metric: jnp.ndarray | str,
    *,
    T_END: float = 10.0,
    dt: float = 0.01,
    base_seed: int = 42,
    n_random_repeats: int = 10,
    lesion_fracs: np.ndarray | None = None,
    lesion_strength: float = 1.0,
) -> dict[str, Any]:
    """Lesion sweep for one ranking vector on an existing simulation (no base re-run, no graphs).

    Only ``run_with_lesions`` is called for each fraction. Random baselines use **deterministic**
    seeds so the same ``(lesion_frac index, repeat index)`` always draws the same random node set,
    independent of which metric vector is ranked. That makes random curves comparable across
    metrics and fully repeatable. Requires ``n_random_repeats < _RANDOM_REPEAT_STRIDE`` (10_000).

    Returns:
        Same structure as one entry from ``evaluate_metric_scores``:
        ``AUC_ranked``, ``AUC_random``, ``ABC``, and the four R curve lists.
    """
    if n_random_repeats >= _RANDOM_REPEAT_STRIDE:
        raise ValueError(f"n_random_repeats must be < {_RANDOM_REPEAT_STRIDE}")

    if lesion_fracs is None:
        lesion_fracs = np.arange(0, 0.3, 0.02)
    lesion_fracs = np.asarray(lesion_fracs, dtype=float)

    R_final_ranked: list[float] = []
    R_avg_ranked: list[float] = []
    R_final_random: list[float] = []
    R_avg_random: list[float] = []

    n_t: int | None = None
    for fi, lesion_frac in enumerate(lesion_fracs):
        res_ranked, _, _ = evaluate_single_lesion(
            sim,
            metric,
            float(lesion_frac),
            lesion_strength,
            T_END,
            dt,
            base_seed,
        )
        R_ranked = get_R_jax(res_ranked["theta"])
        if n_t is None:
            n_t = int(R_ranked.shape[0])

        R_random = np.zeros((n_t, n_random_repeats))
        rng = np.random.default_rng(base_seed + _RANDOM_REPEAT_STRIDE * fi)
        random_seeds = rng.integers(low=0, high=2**32 - 1, size=n_random_repeats, dtype=np.uint32)
        for i, repeat_seed in enumerate(random_seeds):
            res_rand, _, _ = evaluate_single_lesion(
                sim, "random", float(lesion_frac), lesion_strength, T_END, dt, repeat_seed
            )
            R_random[:, i] = get_R_jax(res_rand["theta"])
        R_random = np.mean(R_random, axis=1)

        R_final_ranked.append(float(R_ranked[-1]))
        R_avg_ranked.append(float(np.mean(R_ranked)))
        R_final_random.append(float(R_random[-1]))
        R_avg_random.append(float(np.mean(R_random)))

    AUC_ranked = float(np.trapezoid(R_avg_ranked, lesion_fracs))
    AUC_random = float(np.trapezoid(R_avg_random, lesion_fracs))
    ABC = AUC_random - AUC_ranked

    return {
        "AUC_ranked": AUC_ranked,
        "AUC_random": AUC_random,
        "ABC": ABC,
        "R_final_ranked": R_final_ranked,
        "R_avg_ranked": R_avg_ranked,
        "R_final_random": R_final_random,
        "R_avg_random": R_avg_random,
    }


def evaluate_metric_scores(
    sim: Simulation,
    T_END: float = 10.0,
    dt: float = 0.01,
    RNG: np.random.Generator | None = None,
    n_random_repeats: int = 10,
    lesion_fracs: np.ndarray | None = None,
    lesion_strength: float = 1.0,
    n_ig_steps: int = 20,
    verbose: bool = True,
    base_seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Evaluate metric scores for a given simulation.
    
    Args:
        sim: Simulation object
        T_END: End time
        dt: Time step
        RNG: Random number generator (base forward run and coupling dropout, etc.)
        n_random_repeats: Number of random repeats
        lesion_fracs: Lesion fractions
        lesion_strength: Lesion strength
        base_seed: Base integer for deterministic random-lesion draws in
            ``run_lesion_study`` (same across all metrics; vary per ensemble seed in multi-seed studies).
    Returns:
        metric_scores: Metric scores
    """
    if RNG is None:
        RNG = np.random.default_rng()
    if lesion_fracs is None:
        lesion_fracs = np.arange(0, 0.3, 0.02)

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

    alpha0 = jnp.zeros((sim.grid.N,), dtype=sim.params.K.dtype)
    dRf_dalpha = grads_final_R_alpha(sim.params, alpha0, sim.theta0, t0, t1, dt, ts=jnp.array([t1]))
    dRm_dalpha = grads_mean_R_alpha(sim.params, alpha0, sim.theta0, t0, t1, dt, ts=ts)
    dr_link_dalpha = grads_mean_r_link_alpha(sim.params, alpha0, sim.theta0, t0, t1, dt, ts=ts)

    IRm_a = -dRm_dalpha
    IRlink_a = -dr_link_dalpha

    # Integrated Gradient metric
    if verbose:
        print("Calculating Integrated Gradients metrics...")
    alpha_target = jnp.ones(sim.grid.N, dtype=sim.params.K.dtype) * lesion_strength
    IG_Rm = -ig_mean_R_alpha(sim.params, alpha_target, sim.theta0, t0, t1, dt, ts, n_ig_steps)
    IG_link = -ig_mean_r_link_alpha(sim.params, alpha_target, sim.theta0, t0, t1, dt, ts, n_ig_steps)

    # NOTE: ommitting C_avg metrics except for eigenvector because they are not effective
    metrics = {
        "deg_base": graph_metrics["deg_cent"],
        "deg_eff": graph_metrics_eff["deg_cent"],
        # "deg_C_avg": graph_metrics_C_avg["deg_cent"], 
        "closeness_base": graph_metrics["closeness"],
        "closeness_eff": graph_metrics_eff["closeness"],
        # "closeness_C_avg": graph_metrics_C_avg["closeness"],
        "betweenness_base": graph_metrics["betweenness"],
        "betweenness_eff": graph_metrics_eff["betweenness"],
        # "betweenness_C_avg": graph_metrics_C_avg["betweenness"],
        "eigenvector_base": graph_metrics["eigenvector"],
        "eigenvector_eff": graph_metrics_eff["eigenvector"],
        "eigenvector_C_avg": graph_metrics_C_avg["eigenvector"],
        # "I_final_base": I_final,
        # "IRm_k_base": IRm_k,
        # "IRf_a_base": IRf_a, # not including Rf as performance metric
        "IRm_a_base": IRm_a,
        "IRlink_a_base": IRlink_a,
        "IG_IRm_a": IG_Rm,
        "IG_IRlink_a": IG_link,
    }

    # Run lesion study for each metric
    if verbose:
        print("Running lesion study for each metric...")
    metric_scores = {}
    for metric_name, metric in metrics.items():
        if verbose:
            print(f"Evaluating {metric_name}...")
        metric_scores[metric_name] = run_lesion_study(
            sim,
            metric,
            T_END=T_END,
            dt=dt,
            base_seed=base_seed,
            n_random_repeats=n_random_repeats,
            lesion_fracs=lesion_fracs,
            lesion_strength=lesion_strength,
        )
    return metric_scores

# --- Multi-seed aggregation helpers ---
def list_metrics(results_for_case: dict[int, dict]) -> list[str]:
    """Return sorted metric names from a dict keyed by seed."""
    first_seed = next(iter(results_for_case))
    return sorted(results_for_case[first_seed].keys())


def aggregate_scores(results: dict[str, dict[int, dict]]) -> dict:
    """Aggregate per-seed metric scores across seeds for each case.

    Args:
        results: Nested dict: results[case_name][seed][metric_name] = score_dict.
                 Each score_dict must have keys "ABC", "AUC_ranked", "AUC_random",
                 "R_avg_ranked" (array), "R_avg_random" (array).

    Returns:
        Aggregated dict with mean/std over seeds for each statistic.
    """
    out = {}
    for case_name, by_seed in results.items():
        metrics = list_metrics(by_seed)

        out[case_name] = {
            "metrics": metrics,
            "ABC_mean": {},
            "ABC_std": {},
            "AUC_ranked_mean": {},
            "AUC_ranked_std": {},
            "AUC_random_mean": {},
            "AUC_random_std": {},
            "R_avg_ranked_mean": {},
            "R_avg_ranked_std": {},
            "R_avg_random_mean": {},
            "R_avg_random_std": {},
        }

        for m in metrics:
            ABC = np.array([by_seed[s][m]["ABC"] for s in by_seed])
            AUC_r = np.array([by_seed[s][m]["AUC_ranked"] for s in by_seed])
            AUC_rand = np.array([by_seed[s][m]["AUC_random"] for s in by_seed])
            R_ranked = np.array([by_seed[s][m]["R_avg_ranked"] for s in by_seed], dtype=float)
            R_random = np.array([by_seed[s][m]["R_avg_random"] for s in by_seed], dtype=float)

            out[case_name]["ABC_mean"][m] = float(np.mean(ABC))
            out[case_name]["ABC_std"][m] = float(np.std(ABC))
            out[case_name]["AUC_ranked_mean"][m] = float(np.mean(AUC_r))
            out[case_name]["AUC_ranked_std"][m] = float(np.std(AUC_r))
            out[case_name]["AUC_random_mean"][m] = float(np.mean(AUC_rand))
            out[case_name]["AUC_random_std"][m] = float(np.std(AUC_rand))
            out[case_name]["R_avg_ranked_mean"][m] = np.mean(R_ranked, axis=0)
            out[case_name]["R_avg_ranked_std"][m] = np.std(R_ranked, axis=0)
            out[case_name]["R_avg_random_mean"][m] = np.mean(R_random, axis=0)
            out[case_name]["R_avg_random_std"][m] = np.std(R_random, axis=0)

    return out