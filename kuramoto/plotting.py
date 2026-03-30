"""Plotting helpers for phases, coupling footprints, and animations."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from scipy.sparse import spmatrix

from .grid import CorticalGrid


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
