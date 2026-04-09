"""Plotting helpers for phases, coupling footprints, and animations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from scipy.stats import spearmanr
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import colormaps
from matplotlib.colors import to_hex, Normalize, CenteredNorm, TwoSlopeNorm
import numpy as np
from scipy.sparse import spmatrix

from .grid import CorticalGrid

def set_plot_settings():
    plt.rcParams.update({
        "font.size": 14,
        "figure.titlesize": 20,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })


def _cmap_hex(name: str, t: float) -> str:
    cmap = colormaps[name]
    return to_hex(cmap(float(np.clip(t, 0.0, 1.0))))

def color_for_metric(name: str, fallback: str = "#757575") -> str:
    METRIC_COLORS: dict[str, str] = {
        # Degree — blues
        "deg_base": _cmap_hex("Blues", 0.95),
        "deg_eff": _cmap_hex("Blues", 0.7),
        # Closeness — greens
        "closeness_base": _cmap_hex("Purples", 0.95),
        "closeness_eff": _cmap_hex("Purples", 0.7),
        # Betweenness — greys
        "betweenness_base": _cmap_hex("Greys", 0.95),
        "betweenness_eff": _cmap_hex("Greys", 0.7),
        # Eigenvector — purples
        "eigenvector_base": _cmap_hex("Greens", 0.98),
        "eigenvector_eff": _cmap_hex("Greens", 0.8),
        "eigenvector_C_avg": _cmap_hex("Greens", 0.6),
        # Adjoint gradient — oranges
        "IRm_a_base": _cmap_hex("Oranges", 0.8),
        "IRlink_a_base": _cmap_hex("Oranges", 0.6),
        # Integrated gradient — reds
        "IG_IRm_a": _cmap_hex("Reds", 0.95),
        "IG_IRlink_a": _cmap_hex("Reds", 0.8),
    }
    return METRIC_COLORS.get(name, fallback)


def coupling_to_2d(
    K: spmatrix,
    grid: CorticalGrid,
    mode: str = "in",
    node: int | None = None,
) -> np.ndarray:
    """Convert coupling matrix onto the 2D grid for visualisation.

    Modes: "in" (row sums), "out" (col sums), "from" (column slice), "to" (row slice).
    """
    if mode == "in":
        v = np.asarray(K.sum(axis=1)).ravel()
    elif mode == "out":
        v = np.asarray(K.sum(axis=0)).ravel()
    elif mode == "from":
        if node is None:
            raise ValueError("mode='from' requires a node index")
        v = K[:, node].toarray().ravel()
    elif mode == "to":
        if node is None:
            raise ValueError("mode='to' requires a node index")
        v = K[node, :].toarray().ravel()
    else:
        raise ValueError(f"Unknown mode: {mode!r}")
    return grid.unflatten(v)


# --- static plots ---
def plot_2d(
    data_2d: np.ndarray,
    variable: str = "phase",
    ax=None,
    title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cbar: bool = True,
    **kwargs: Any,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    is_phase = variable == "phase"
    cmap = kwargs.pop("cmap", "hsv" if is_phase else "viridis")

    if is_phase:
        data = np.mod(data_2d, 2 * np.pi)
        vmin = vmin if vmin is not None else 0
        vmax = vmax if vmax is not None else 2 * np.pi
    else:
        data = data_2d
        if vmin is None:
            vmin = float(np.nanmin(data))
        if vmax is None:
            vmax = float(np.nanmax(data))

    im = ax.imshow(
        data, cmap=cmap, vmin=vmin, vmax=vmax,
        aspect="equal", interpolation="nearest", **kwargs,
    )
    ax.set_title(title or variable)
    label = "Phase (rad)" if is_phase else variable
    cbar = plt.colorbar(im, ax=ax, label=label)

    if is_phase:
        cbar.set_ticks([0, np.pi, 2 * np.pi])
        cbar.set_ticklabels(["0", r"$\pi$", r"$2\pi$"])


def plot_gradient_map(
    magnitude_2d: np.ndarray,
    dtheta_dr: np.ndarray | None = None,
    dtheta_dc: np.ndarray | None = None,
    ax=None,
    title: str = r"$|\nabla\theta|$",
    subsample_quiver: int | None = None,
    **kwargs: Any,
):
    """Plot gradient magnitude as heatmap; optionally overlay direction as quiver.

    magnitude_2d: |∇θ| from analysis.gradient_from_state or gradient_magnitude_2d.
    If dtheta_dr and dtheta_dc are provided, direction is drawn (subsample_quiver
    e.g. 4 to avoid clutter).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(
        magnitude_2d,
        cmap=kwargs.pop("cmap", "viridis"),
        aspect="equal",
        interpolation="nearest",
        **kwargs,
    )
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label=r"$|\nabla\theta|$ (rad / grid step)")

    if dtheta_dr is not None and dtheta_dc is not None and subsample_quiver is not None:
        nr, nc = magnitude_2d.shape
        step = max(1, subsample_quiver)
        jj = np.arange(0, nc, step)
        ii = np.arange(0, nr, step)
        J, I = np.meshgrid(jj, ii)
        U = dtheta_dc[np.ix_(ii, jj)]
        V = dtheta_dr[np.ix_(ii, jj)]
        ax.quiver(J, I, U, V, color="white", scale=None, alpha=0.8)
    return im


def plot_coupling_matrix(
    K: spmatrix,
    ax=None,
    title: str = "Coupling matrix K",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    **kwargs: Any,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    dense = K.toarray() if hasattr(K, "toarray") else np.asarray(K)
    im = ax.imshow(
        dense, cmap=cmap, aspect="equal", interpolation="nearest",
        vmin=vmin, vmax=vmax, **kwargs,
    )
    ax.set_xlabel("Source (j)")
    ax.set_ylabel("Target (i)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="K[i,j]")
    return im


# --- animations ---
def animate_2d(
    snapshots: list[np.ndarray],
    variable: str = "phase",
    t_list: list[float] | None = None,
    interval: int = 50,
    figsize: tuple[int, int] = (6, 5),
    save_path: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> FuncAnimation:
    is_phase = variable == "phase"
    cmap = "hsv" if is_phase else "viridis"

    if is_phase:
        display = np.mod(snapshots[0], 2 * np.pi)
        vmin = vmin if vmin is not None else 0
        vmax = vmax if vmax is not None else 2 * np.pi
    else:
        display = snapshots[0].copy()
        if vmin is None:
            vmin = float(np.nanmin([np.nanmin(s) for s in snapshots]))
        if vmax is None:
            vmax = float(np.nanmax([np.nanmax(s) for s in snapshots]))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(display, cmap=cmap, vmin=vmin, vmax=vmax,
                   aspect="equal", interpolation="nearest")
    label = "Phase (rad)" if is_phase else variable
    cbar = plt.colorbar(im, ax=ax, label=label)
    if is_phase:
        cbar.set_ticks([0, np.pi, 2 * np.pi])
        cbar.set_ticklabels(["0", r"$\pi$", r"$2\pi$"])
    title_obj = ax.set_title("")

    def update(frame: int):
        s = snapshots[frame]
        im.set_data(np.mod(s, 2 * np.pi) if is_phase else s)
        if t_list is not None and frame < len(t_list):
            title_obj.set_text(f"t = {t_list[frame]:.2f}")
        else:
            title_obj.set_text(f"Frame {frame}")
        return [im, title_obj]

    anim = FuncAnimation(fig, update, frames=len(snapshots), interval=interval, blit=True)
    if save_path:
        anim.save(save_path, writer=PillowWriter(fps=1000 // interval))
    return anim


def animate_from_run(
    source: Any,
    grid: CorticalGrid,
    t_list: list[float] | np.ndarray | None = None,
    variable: str = "phase",
    downsample: int = 1,
    interval: int = 50,
    save_path: str | None = None,
    **kwargs: Any,
) -> FuncAnimation:
    """Animate phase/fields from either a sim object, a theta array, or legacy state list.

    Supported inputs:
      - ``sim``: object with ``sim.results["theta"]`` and optionally ``sim.results["ts"]``
      - ``theta_series``: ndarray with shape ``(T, N)`` or snapshot ``(N,)``
      - legacy ``state_list``: list[dict] with ``state["theta"]``
    """
    theta_series: np.ndarray | None = None
    if hasattr(source, "results"):
        theta_series = np.asarray(source.results["theta"])
        if t_list is None and "ts" in source.results:
            t_list = np.asarray(source.results["ts"])
    elif isinstance(source, np.ndarray):
        theta_series = np.asarray(source)
    else:
        # Legacy: list[dict]
        state_list = source
        indices = list(range(0, len(state_list), downsample))
        states = [state_list[i] for i in indices]
        times = [t_list[i] for i in indices] if t_list is not None else None
        snapshots = [grid.unflatten(s["theta"]) for s in states]
        return animate_2d(
            snapshots,
            variable=variable,
            t_list=times,
            interval=interval,
            save_path=save_path,
            **kwargs,
        )

    x = np.asarray(theta_series)
    if x.ndim == 1:
        snapshots = [grid.unflatten(x)]
        times = [t_list[0]] if t_list is not None and np.asarray(t_list).size > 0 else None
    elif x.ndim == 2:
        indices = list(range(0, x.shape[0], downsample))
        snapshots = [grid.unflatten(x[i]) for i in indices]
        times = [float(np.asarray(t_list)[i]) for i in indices] if t_list is not None else None
    else:
        raise ValueError(f"animate_from_run expects (N,) or (T,N); got shape={x.shape}")

    return animate_2d(
        snapshots,
        variable=variable,
        t_list=times,
        interval=interval,
        save_path=save_path,
        **kwargs,
    )

# --- Lesion Study and ensemble plots ---
def plot_rt_traces_per_case(
    rt_by_case: Mapping[str, np.ndarray],
    dt: float,
    case_labels: Mapping[str, str],
    *,
    suptitle: str | None = None,
    ylim: tuple[float, float] = (0.0, 1.05),
    xlabel: str = "t",
    ylabel: str = "R(t)",
    figsize_per_col: float = 4.0,
    fig_height: float = 3.0,
    suptitle_y: float = 1.06,
):
    """One R(t) trace per case (e.g. one chosen seed per θ-sigma / ω-sigma)."""
    case_names = list(rt_by_case.keys())
    n = len(case_names)
    if n == 0:
        raise ValueError("rt_by_case must be non-empty")

    fig, axs = plt.subplots(1, n, figsize=(figsize_per_col * n, fig_height), constrained_layout=True)
    if n == 1:
        axs = np.array([axs])

    for ax, case_name in zip(axs.flat, case_names):
        Rt = np.asarray(rt_by_case[case_name])
        ax.plot(np.arange(len(Rt)) * dt, Rt)
        ax.set_title(case_labels[case_name])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)

    if suptitle is not None:
        fig.suptitle(suptitle, y=suptitle_y)
    return fig, axs


def plot_abc_mean_heatmap(
    agg: Mapping[str, Mapping],
    case_names: Sequence[str],
    metrics: Sequence[str],
    case_labels: Mapping[str, str],
    *,
    title: str,
    mean_key: str = "ABC_mean",
    cmap: str = "viridis",
    cbar_label: str = "ABC",
    fig_width_per_case: float = 1.2,
    fig_width_base: float = 6.0,
    fig_height_per_metric: float = 0.25,
    fig_height_base: float = 3.0,
    xtick_rotation: float = 30.0,
    ha: str = "right",
):
    """Rows = metrics, columns = cases; values from ``agg[case][mean_key][metric]``."""
    metrics = list(metrics)
    case_names = list(case_names)
    if not metrics or not case_names:
        raise ValueError("metrics and case_names must be non-empty")

    heat = np.array(
        [[agg[c][mean_key][m] for c in case_names] for m in metrics],
        dtype=float,
    )
    fig_w = fig_width_per_case * len(case_names) + fig_width_base
    fig_h = fig_height_per_metric * len(metrics) + fig_height_base
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    im = ax.imshow(heat, aspect="auto", cmap=cmap)
    ax.set_xticks(np.arange(len(case_names)))
    ax.set_xticklabels([case_labels[c] for c in case_names], rotation=xtick_rotation, ha=ha)
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels(metrics)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    return fig, ax


def abc_rank_correlation_matrix(
    agg: Mapping[str, Mapping],
    case_names: Sequence[str],
) -> np.ndarray:
    """Pairwise Spearman ρ between mean-ABC vectors across cases (same metric ordering)."""
    case_names = list(case_names)
    mlist = agg[case_names[0]]["metrics"]
    abc_vecs = np.array([[agg[c]["ABC_mean"][m] for m in mlist] for c in case_names])
    n = len(case_names)
    corr = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = spearmanr(abc_vecs[i], abc_vecs[j])
            corr[i, j] = corr[j, i] = rho
    return corr


def plot_abc_rank_correlation_heatmap(
    corr: np.ndarray,
    case_names: Sequence[str],
    case_labels: Mapping[str, str],
    *,
    title: str,
    vmin: float = -1.0,
    vmax: float = 1.0,
    cmap: str = "RdBu_r",
    cbar_label: str = "Spearman ρ",
    figsize: tuple[float, float] = (8.0, 5.0),
    xtick_rotation: float = 45.0,
    tick_fontsize: float = 8.0,
    ann_fontsize: float = 7.0,
    text_fmt: str = "{:.2f}",
    ha: str = "right",
):
    """Annotated heatmap for a symmetric correlation matrix (e.g. from ``abc_rank_correlation_matrix``)."""
    case_names = list(case_names)
    n = len(case_names)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    im = ax.imshow(corr, vmin=vmin, vmax=vmax, cmap=cmap)
    ticks = np.arange(n)
    ax.set_xticks(ticks)
    ax.set_xticklabels(
        [case_labels[c] for c in case_names], rotation=xtick_rotation, ha=ha, fontsize=tick_fontsize
    )
    ax.set_yticks(ticks)
    ax.set_yticklabels([case_labels[c] for c in case_names], fontsize=tick_fontsize)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, text_fmt.format(corr[i, j]), ha="center", va="center", fontsize=ann_fontsize)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    ax.set_title(title)
    return fig, ax

def plot_lesion_r_avg_ranked_overlay(
    lesion_fracs: np.ndarray,
    case_names: Sequence[str],
    case_labels: Mapping[str, str],
    agg: Mapping[str, Mapping],
    metrics: Sequence[str],
    *,
    suptitle: str | None = None,
    ylabel: str = r"R$_{avg}$ (ranked metric curves)",
    xlabel: str = "lesion_frac",
    cmap: str = "tab10",
    figsize_per_col: float = 4,
    fig_height: float = 4,
    line_lw: float = 2.0,
    fill_alpha: float = 0.18,
    random_color: str = "tab:gray",
    random_lw: float = 1.7,
    random_fill_alpha: float = 0.15,
    legend_ncol_max: int = 5,
    legend_bbox: tuple[float, float] = (0.5, -0.08),
    legend_fontsize: float = 9,
    title_fontsize: float = 11,
    suptitle_y: float = 1.06,
    sharex: bool = True,
    sharey: bool = True,
):
    """Overlay R_avg (ranked) vs lesion_frac for several metrics; optional random baseline.

    ``agg`` must match ``aggregate_scores`` output: per case, keys
    ``R_avg_ranked_mean``, ``R_avg_ranked_std``, ``R_avg_random_mean``,
    ``R_avg_random_std`` indexed by metric name.

    The random baseline is taken from ``metrics[0]`` (same random curve for all metrics).
    """
    metrics = list(metrics)
    if not metrics:
        raise ValueError("metrics must be non-empty")

    ncols = len(case_names)
    if ncols == 0:
        raise ValueError("case_names must be non-empty")

    colormap = plt.get_cmap(cmap)
    fig, axs = plt.subplots(
        1,
        ncols,
        figsize=(figsize_per_col * ncols, fig_height),
        constrained_layout=True,
        sharex=sharex,
        sharey=sharey,
    )
    if ncols == 1:
        axs = np.array([axs])

    x = np.asarray(lesion_fracs)

    for c, case_name in enumerate(case_names):
        ax = axs.flat[c]
        for i, metric in enumerate(metrics):
            y = agg[case_name]["R_avg_ranked_mean"][metric]
            yerr = agg[case_name]["R_avg_ranked_std"][metric]
            color = colormap(i)
            ax.plot(x, y, label=metric, color=color, lw=line_lw)
            ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=fill_alpha)

        y0 = agg[case_name]["R_avg_random_mean"][metrics[0]]
        y0err = agg[case_name]["R_avg_random_std"][metrics[0]]
        ax.plot(x, y0, label="random", color=random_color, linestyle="--", lw=random_lw)
        ax.fill_between(x, y0 - y0err, y0 + y0err, color=random_color, alpha=random_fill_alpha)

        ax.set_title(case_labels[case_name], fontsize=title_fontsize)
        if c == 0:
            ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

    handles, labels = axs.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(handles), legend_ncol_max),
        bbox_to_anchor=legend_bbox,
        fontsize=legend_fontsize,
        frameon=False,
    )
    if suptitle is not None:
        fig.suptitle(suptitle, y=suptitle_y)

    return fig, axs

def _style_metric_colored_horizontal_boxplot(
    box_dict: dict,
    metric_names: Sequence[str],
    *,
    patch_alpha: float = 0.8,
    median_lw: float = 2.0,
) -> None:
    """Shared look: per-metric facecolor, median matches metric, 0.5 whiskers/caps, small fliers."""
    names = list(metric_names)

    for i, patch in enumerate(box_dict["boxes"]):
        mc = color_for_metric(names[i])
        patch.set_facecolor(mc)
        patch.set_alpha(patch_alpha)
        patch.set_edgecolor("0.1")
        patch.set_linewidth(0.6)
    for i, med in enumerate(box_dict["medians"]):
        mc = color_for_metric(names[i])
        med.set_color(mc)
        med.set_linewidth(median_lw)
    for w in box_dict["whiskers"]:
        w.set_color("0.1")
    for cap in box_dict["caps"]:
        cap.set_color("0.1")
    for flier in box_dict["fliers"]:
        flier.set_marker("o")
        flier.set_markersize(4)
        flier.set_alpha(0.8)
        flier.set_markeredgecolor("0.1")
        flier.set_markerfacecolor("white")


def plot_abc_spread_boxplot_h(
    abc_by_metric: Mapping[str, Sequence[float]],
    *,
    title: str,
    xlabel: str = "ABC",
    sort_ascending: bool = True,
    fig_width: float = 10.0,
    row_height: float = 0.35,
    min_fig_height: float = 8.0,
    vline_x: float | None = 0.0,
    vline_kwargs: Mapping[str, Any] | None = None,
    grid_axis_x: bool = True,
    grid_alpha: float = 0.4,
    patch_alpha: float = 0.8,
    use_tight_layout: bool = True,
    ax: Any | None = None,
):
    """Horizontal boxplots of ABC (or similar) across seeds, one row per metric, sorted by median.

    ``abc_by_metric`` maps metric name -> list of values (e.g. one ABC per seed).
    """
    metrics = list(abc_by_metric.keys())
    if not metrics:
        raise ValueError("abc_by_metric must be non-empty")

    def _median(m: str) -> float:
        return float(np.median(np.asarray(abc_by_metric[m], dtype=float)))

    order = sorted(metrics, key=_median, reverse=not sort_ascending)
    data = [np.asarray(abc_by_metric[m], dtype=float).ravel() for m in order]

    if ax is None:
        fig_h = max(min_fig_height, row_height * len(order))
        fig, ax = plt.subplots(figsize=(fig_width, fig_h))
    else:
        fig = ax.figure

    bp = ax.boxplot(data, vert=False, patch_artist=True)
    ax.set_yticks(np.arange(1, len(order) + 1))
    ax.set_yticklabels(order)
    _style_metric_colored_horizontal_boxplot(bp, order, patch_alpha=patch_alpha)

    ax.set_xlabel(xlabel)
    ax.set_title(title)

    vline_kw = dict(color="k", lw=0.8, alpha=0.4, zorder=0) | dict(vline_kwargs or {})
    if vline_x is not None:
        ax.axvline(vline_x, **vline_kw)
    if grid_axis_x:
        ax.grid(axis="x", alpha=grid_alpha)
    if use_tight_layout:
        fig.tight_layout()
    return fig, ax


def plot_topk_abc_boxplots_h(
    results: Mapping[str, Mapping[int, Mapping[str, Mapping]]],
    agg: Mapping[str, Mapping],
    seeds: Sequence[int],
    case_names: Sequence[str],
    metrics: Sequence[str],
    case_labels: Mapping[str, str],
    *,
    top_k: int = 8,
    score_key: str = "ABC",
    mean_key: str = "ABC_mean",
    suptitle: str | None = None,
    xlabel: str = "ABC",
    figsize_per_col: float = 5,
    fig_height: float = 4.0,
    title_fontsize: float = 9.0,
    suptitle_y: float = 1.06,
    vline_x: float | None = 0.0,
    vline_kwargs: Mapping[str, Any] | None = None,
    grid_axis_x: bool = True,
    grid_alpha: float = 0.3,
    patch_alpha: float = 0.6,
):
    """Horizontal boxplots of raw per-seed scores for top-K metrics (ranked by agg means)."""
    metrics = list(metrics)
    case_names = list(case_names)
    if not metrics:
        raise ValueError("metrics must be non-empty")

    n = len(case_names)
    fig, axs_flat = plt.subplots(
        1, n, figsize=(figsize_per_col * n, fig_height), constrained_layout=True, sharex=False
    )
    if n == 1:
        axs_flat = np.array([axs_flat])
    else:
        axs_flat = np.asarray(axs_flat).ravel()

    vline_kw = dict(color="k", lw=0.8, alpha=0.4, zorder=0) | dict(vline_kwargs or {})

    for ax, case_name in zip(axs_flat, case_names):
        means = {m: agg[case_name][mean_key][m] for m in metrics}
        top_m = sorted(means, key=means.__getitem__, reverse=True)[:top_k][::-1]
        data = [[results[case_name][s][m][score_key] for s in seeds] for m in top_m]

        box = ax.boxplot(data, vert=False, labels=top_m, patch_artist=True)
        _style_metric_colored_horizontal_boxplot(box, top_m, patch_alpha=patch_alpha)

        if vline_x is not None:
            ax.axvline(vline_x, **vline_kw)
        if grid_axis_x:
            ax.grid(axis="x", alpha=grid_alpha)
        ax.set_title(case_labels[case_name], fontsize=title_fontsize)
        ax.set_xlabel(xlabel)

    if suptitle is not None:
        fig.suptitle(suptitle, y=suptitle_y)
    return fig, axs_flat

# --- Adjoint / gradient visualization ---

def get_norm_cmap(array: np.ndarray) -> tuple[Normalize, str]:
    """Choose a norm and colormap based on the sign of array values."""
    if np.any(array < 0) and np.any(array > 0):
        norm = TwoSlopeNorm(vmin=float(np.min(array)), vcenter=0.0, vmax=float(np.max(array)))
        cmap = "bwr"
    elif np.all(array >= 0):
        norm = Normalize(vmin=0, vmax=float(np.max(array)))
        cmap = "Reds"
    else:
        norm = Normalize(vmin=float(np.min(array)), vmax=0.0)
        cmap = "Blues"
    return norm, cmap


def plot_basic_grads(
    g: Any,
    grid_shape: tuple[int, int],
    title: str = "Basic gradients",
) -> tuple[plt.Figure, np.ndarray]:
    """Plot dJ/dK matrix and dJ/dω spatial map for a KuramotoParams gradient struct."""
    dR_dK = np.asarray(g.K)
    dR_domega = np.asarray(g.omega)
    dR_domega_2d = dR_domega.reshape(grid_shape)

    N_edges = dR_dK.shape[0]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs = axs.ravel()

    norm, cmap = get_norm_cmap(dR_dK)
    im = axs[0].imshow(dR_dK, cmap=cmap, norm=norm)
    axs[0].set_title("dJ/dK")
    if N_edges < 20:
        axs[0].set_xticks(np.arange(0, N_edges, 1))
        axs[0].set_yticks(np.arange(0, N_edges, 1))
    elif N_edges < 100:
        axs[0].set_xticks(np.arange(0, N_edges, 10))
        axs[0].set_yticks(np.arange(0, N_edges, 10))
    else:
        axs[0].set_xticks(np.arange(0, N_edges, 20))
        axs[0].set_yticks(np.arange(0, N_edges, 20))
    axs[0].tick_params(axis="x", labelrotation=45)
    fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)

    norm, cmap = get_norm_cmap(dR_domega_2d)
    im = axs[1].imshow(dR_domega_2d, cmap=cmap, norm=norm)
    axs[1].set_title("dJ/domega0")
    axs[1].set_xticks(np.arange(0, dR_domega_2d.shape[1], 1))
    axs[1].set_yticks(np.arange(0, dR_domega_2d.shape[0], 1))
    axs[1].tick_params(axis="x", labelrotation=45)
    fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)
    fig.suptitle(title)
    return fig, axs


def plot_advanced_grads(
    dR_dalpha: np.ndarray,
    I_node: np.ndarray,
    grid_shape: tuple[int, int],
    title: str = "Node importance",
) -> tuple[plt.Figure, np.ndarray]:
    """Plot dJ/dα and node-importance maps side-by-side."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

    norm, cmap = get_norm_cmap(np.asarray(dR_dalpha))
    im = ax[0].imshow(np.asarray(dR_dalpha).reshape(grid_shape), norm=norm, cmap=cmap)
    ax[0].set_title("dJ/dalpha")
    ax[0].set_xticks(np.arange(0, grid_shape[1], 1))
    ax[0].set_yticks(np.arange(0, grid_shape[0], 1))
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)

    norm, cmap = get_norm_cmap(np.asarray(I_node))
    im = ax[1].imshow(np.asarray(I_node).reshape(grid_shape), cmap=cmap, norm=norm)
    ax[1].set_title("I_node")
    ax[1].set_xticks(np.arange(0, grid_shape[1], 1))
    ax[1].set_yticks(np.arange(0, grid_shape[0], 1))
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)

    fig.suptitle(title)
    return fig, ax


def plot_adjoint_grads(
    grads: dict[str, np.ndarray] | None = None,
    sim: Any | None = None,
    grid_shape: tuple[int, int] | None = None,
    *,
    T_END: float | None = None,
    dt: float | None = None,
    n_ig_steps: int = 20,
    title: str | None = None,
    axs: Any | None = None,
    cmap: str = "bwr",
) -> tuple[plt.Figure, np.ndarray]:
    """Plot raw adjoint gradient maps (dRf/dα, dRm/dα, dRlink/dα + IG variants)."""
    from .adjoint import get_adjoint_grads  # local import avoids circular dependency

    if grads is None:
        if sim is None or T_END is None or dt is None:
            raise ValueError("Provide either grads, or (sim, T_END, dt).")
        grads = get_adjoint_grads(sim, T_END=T_END, dt=dt, n_ig_steps=n_ig_steps)

    if grid_shape is None:
        if sim is None:
            raise ValueError("grid_shape must be provided when sim is None.")
        grid_shape = sim.grid.shape

    if axs is None:
        fig, axs = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    else:
        fig = axs[0, 0].get_figure()

    panels = [
        (0, 0, "dRf_dalpha",       "dRf/dα (linear)"),
        (0, 1, "dRm_dalpha",       "dRm/dα (linear)"),
        (0, 2, "dRlink_dalpha",    "dRlink/dα (linear)"),
        (1, 1, "IG_dRm_dalpha",    f"IG dRm/dα ({n_ig_steps} steps)"),
        (1, 2, "IG_dRlink_dalpha", f"IG dRlink/dα ({n_ig_steps} steps)"),
    ]
    axs[1, 0].set_visible(False)

    for row, col, key, panel_title in panels:
        ax = axs[row, col]
        data = grads[key].reshape(grid_shape)
        im = ax.imshow(data, norm=CenteredNorm(), cmap=cmap)
        ax.set_title(panel_title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title is not None:
        fig.suptitle(title)
    return fig, axs


def plot_adjoint_metrics(
    metrics: dict[str, np.ndarray] | None = None,
    sim: Any | None = None,
    grid_shape: tuple[int, int] | None = None,
    *,
    T_END: float | None = None,
    dt: float | None = None,
    n_ig_steps: int = 20,
    title: str | None = None,
    axs: Any | None = None,
    cmap: str = "bwr",
) -> tuple[plt.Figure, np.ndarray]:
    """Plot node-importance maps: IRm_a, IRlink_a, IG_IRm_a, IG_IRlink_a (2×2 grid)."""
    from .adjoint import get_adjoint_metrics  # local import avoids circular dependency

    if metrics is None:
        if sim is None or T_END is None or dt is None:
            raise ValueError("Provide either metrics, or (sim, T_END, dt).")
        metrics = get_adjoint_metrics(sim, T_END=T_END, dt=dt, n_ig_steps=n_ig_steps)

    if grid_shape is None:
        if sim is None:
            raise ValueError("grid_shape must be provided when sim is None.")
        grid_shape = sim.grid.shape

    if axs is None:
        fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    else:
        fig = axs[0, 0].get_figure()

    panels = [
        (0, 0, "IRm_a",       "Mean R (−d⟨R⟩/dα, Linear)"),
        (0, 1, "IRlink_a",    "Mean R_link (−d⟨R_link⟩/dα, Linear)"),
        (1, 0, "IG_IRm_a",    f"Mean R (−IG d⟨R⟩/dα, {n_ig_steps} steps)"),
        (1, 1, "IG_IRlink_a", f"Mean R_link (−IG d⟨R_link⟩/dα, {n_ig_steps} steps)"),
    ]

    for row, col, key, panel_title in panels:
        ax = axs[row, col]
        data = metrics[key].reshape(grid_shape)
        im = ax.imshow(data, norm=CenteredNorm(), cmap=cmap)
        ax.set_title(panel_title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title is not None:
        fig.suptitle(title)
    return fig, axs


# --- Lesion-study level: single-seed ---
def plot_single_metric_rt_comparison(
    ts: np.ndarray,
    R_base: np.ndarray,
    R_ranked: np.ndarray,
    R_random_mean: np.ndarray,
    R_random_std: np.ndarray,
    *,
    lesion_frac: float,
    metric_name: str = "",
    title: str | None = None,
    ax: Any | None = None,
) -> tuple[plt.Figure, Any]:
    """R(t) traces: base vs ranked-lesion vs random mean±std (single metric, single seed).

    Args:
        ts: Time vector (T,).
        R_base: Baseline R(t) (T,).
        R_ranked: R(t) after ranked lesioning (T,).
        R_random_mean: Mean R(t) for random lesions, shape (T,).
        R_random_std: Std of R(t) for random lesions, shape (T,).
        lesion_frac: Fraction of nodes lesioned (used in legend label).
        metric_name: Name of the metric (used in legend label).
        title: Axes title.
        ax: Existing axes; created if None.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    ax.plot(ts, R_base, label="base", color="k", lw=2)

    mc = color_for_metric(metric_name)
    label = f"ranked ({lesion_frac:.0%}" + (f", {metric_name}" if metric_name else "") + ")"
    ax.plot(ts, R_ranked, label=label, color=mc, lw=2)

    # Plot random mean and fill between mean ± std
    random_label = f"random ({lesion_frac:.0%})"
    ax.plot(ts, R_random_mean, linestyle="--", color="tab:gray", alpha=0.9, label=random_label)
    ax.fill_between(
        ts,
        R_random_mean - R_random_std,
        R_random_mean + R_random_std,
        color="tab:gray",
        alpha=0.3,
        zorder=0,
        label=None
    )

    ax.set_xlabel("t")
    ax.set_ylabel("R(t)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.set_title(title or "R(t): ranked vs random lesioning")
    fig.tight_layout()
    return fig, ax


def plot_r_vs_lesion_frac(
    lesion_fracs: np.ndarray,
    R_final_ranked: Sequence[float],
    R_avg_ranked: Sequence[float],
    R_final_random: Sequence[float],
    R_avg_random: Sequence[float],
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (14, 4),
) -> tuple[plt.Figure, np.ndarray]:
    """Two-panel plot: R_final and R_avg vs lesion fraction (single metric, single seed)."""
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)
    x = np.asarray(lesion_fracs) * 100

    axes[0].plot(x, R_final_ranked, label="Ranked", color="#0072B2", lw=2)
    axes[0].plot(x, R_final_random, label="Random", color="#E69F00", lw=2, ls="--")
    axes[0].set_xlabel("Lesion fraction (%)")
    axes[0].set_ylabel("Final R")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].plot(x, R_avg_ranked, label="Ranked", color="#0072B2", lw=2)
    axes[1].plot(x, R_avg_random, label="Random", color="#E69F00", lw=2, ls="--")
    axes[1].set_xlabel("Lesion fraction (%)")
    axes[1].set_ylabel("Mean R(t)")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    if title is not None:
        fig.suptitle(title)
    return fig, axes


def plot_auc_abc_diagram(
    lesion_fracs: np.ndarray,
    R_avg_ranked: Sequence[float],
    R_avg_random: Sequence[float],
    *,
    AUC_ranked: float,
    AUC_random: float,
    ABC: float,
    title: str | None = None,
    ax: Any | None = None,
) -> tuple[plt.Figure, Any]:
    """Geometric illustration: AUC fills to axis + ABC gap between curves (single metric)."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
    else:
        fig = ax.get_figure()

    x = np.asarray(lesion_fracs) * 100
    Rr = np.asarray(R_avg_ranked, dtype=float)
    Rm = np.asarray(R_avg_random, dtype=float)

    ax.fill_between(x, 0, Rr, alpha=0.35, color="#0072B2",
                    label=rf"AUC (ranked) $\approx$ {AUC_ranked:.4f}")
    ax.fill_between(x, 0, Rm, alpha=0.35, color="#E69F00",
                    label=rf"AUC (random) $\approx$ {AUC_random:.4f}")
    ax.fill_between(x, Rr, Rm, alpha=0.4, color="#6B4C9A",
                    label=rf"Between curves (ABC $\approx$ {ABC:+.4f})")
    ax.plot(x, Rr, color="#0072B2", lw=2, zorder=5)
    ax.plot(x, Rm, color="#E69F00", lw=2, ls="--", zorder=5)

    ax.set_xlabel("Lesioned fraction (%)")
    ax.set_ylabel(r"Mean $R$")
    ax.set_title(title or r"Mean $R$ degradation: AUCs (to axis) and ABC (gap)")
    ax.grid(True, axis="y", alpha=0.3, ls=":")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig, ax


def plot_auc_dotplot(
    metric_scores: Mapping[str, Mapping],
    *,
    title: str | None = None,
    xlabel: str = "AUC (smaller = faster degradation)",
    fig_width: float = 9.0,
    row_height: float = 0.35,
    min_fig_height: float = 6.0,
    ax: Any | None = None,
) -> tuple[plt.Figure, Any]:
    """Ranked vs random AUC dot-chart for all metrics, colored by metric family."""
    metrics = list(metric_scores.keys())
    order = sorted(metrics, key=lambda m: metric_scores[m]["ABC"])
    y = np.arange(len(order))
    auc_r = np.array([metric_scores[m]["AUC_ranked"] for m in order])
    auc_rand = np.array([metric_scores[m]["AUC_random"] for m in order])

    if ax is None:
        fig_h = max(min_fig_height, row_height * len(order))
        fig, ax = plt.subplots(figsize=(fig_width, fig_h))
    else:
        fig = ax.get_figure()

    ax.hlines(y, np.minimum(auc_r, auc_rand), np.maximum(auc_r, auc_rand),
              color="0.85", lw=2, zorder=0)
    for i, m in enumerate(order):
        mc = color_for_metric(m)
        ax.plot(auc_r[i], y[i], "o", ms=7, color=mc, zorder=2, label="ranked" if i == 0 else None)
        ax.plot(auc_rand[i], y[i], "s", ms=6, color=mc, alpha=0.55, zorder=2,
                label="random" if i == 0 else None)

    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel(xlabel)
    ax.set_title(title or "R AUC: importance-ranked vs random node lesions")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_abc_lollipop(
    metric_scores: Mapping[str, Mapping],
    *,
    title: str | None = None,
    xlabel: str = "ABC",
    fig_width: float = 10.0,
    row_height: float = 0.35,
    min_fig_height: float = 8.0,
    ax: Any | None = None,
) -> tuple[plt.Figure, Any]:
    """ABC lollipop chart: all metrics sorted by ABC, colored by metric family."""
    metrics = list(metric_scores.keys())
    order = sorted(metrics, key=lambda m: metric_scores[m]["ABC"])
    abc = np.array([metric_scores[m]["ABC"] for m in order])
    y = np.arange(len(order))

    if ax is None:
        fig_h = max(min_fig_height, row_height * len(order))
        fig, ax = plt.subplots(figsize=(fig_width, fig_h))
    else:
        fig = ax.get_figure()

    ax.hlines(y, 0, abc, color="0.75", lw=2, zorder=1)
    for i, m in enumerate(order):
        ax.plot(abc[i], y[i], "o", ms=8, color=color_for_metric(m), zorder=2, clip_on=False)

    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel(xlabel)
    ax.set_title(title or "Area Between Curves (higher = better targeting)")
    ax.axvline(0, color="k", lw=0.8, alpha=0.4)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig, ax