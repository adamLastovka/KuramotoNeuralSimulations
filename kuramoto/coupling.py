from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from .grid import CorticalGrid
from .kernels import apply_kernel, apply_kernel_jax, dropout_kernel

if TYPE_CHECKING:
    from .config import KernelComponentConfig

def closest_square_shape(N):
    for nrows in range(int(np.ceil(np.sqrt(N))), 0, -1):
        if N % nrows == 0:
            ncols = N // nrows
            return (nrows, ncols)
    return (1, N)

class CouplingMatrix:
    """Coupling weights K[i,j] = strength from oscillator j to oscillator i.

    Supports:
    - Legacy single-kernel construction (numpy kernels) for backwards compatibility
    - Heterogeneous component construction by summing masked kernel components into a dense K

    Heterogeneous mode:
    - Provide ``components`` and ``group_ids``.
    - For now, gaussian components are differentiable in their params (sigma).
    """

    def __init__(
        self,
        grid: CorticalGrid,
        kernel: str = "gaussian",
        base_strength: float = 1.0,
        kernel_params: dict | None = None,
        radius: float | None = None,
        mode: str = "spatial",
        components: list[KernelComponentConfig] | None = None,
        group_ids: list[int] | None = None,
        K_preset: np.ndarray | jnp.ndarray | None = None,
    ):
        self.grid = grid
        self.kernel_name = kernel
        self.base_strength = base_strength
        self.kernel_params = kernel_params or {}
        self.radius = radius
        self.mode = mode

        self.components = components
        self.group_ids = group_ids

        if K_preset is not None:
            K_preset = jnp.asarray(K_preset, dtype=jnp.float32)
            if K_preset.shape != (grid.N, grid.N):
                raise ValueError(
                    f"K_preset shape {K_preset.shape} does not match grid N={grid.N}; "
                    f"expected ({grid.N}, {grid.N})"
                )
            self.K = K_preset
        elif self.components is not None:
            if not self.components:
                raise ValueError("components provided but empty.")
            self.K = self._build_from_components(self.components)
        else:
            components_to_build = self._expand_legacy_mode_to_components()
            self.K = self._build_from_components(components_to_build)

    def _expand_legacy_mode_to_components(self) -> list[dict]: # NOTE: Temporary - remove evenually
        """Expand `mode`/legacy fields into the component-list representation."""
        n = self.grid.N
        if self.mode == "uniform":
            # Match previous behavior: include diagonal; all entries equal base_strength/n.
            return [
                {
                    "kernel": "constant",
                    "base_strength": self.base_strength / n,
                    "kernel_params": {},
                    "radius": None,
                    "node_groups": None,
                    "edge_mode": "outgoing",
                    "to_node_groups": None,
                }
            ]
        if self.mode == "spatial":
            # Match previous behavior: distance-kernel with optional hard cutoff, exclude self.
            return [
                {
                    "kernel": self.kernel_name,
                    "base_strength": self.base_strength,
                    "kernel_params": self.kernel_params,
                    "radius": self.radius,
                    "node_groups": None,
                    "edge_mode": "within",
                    "to_node_groups": None,
                }
            ]

        raise ValueError(f"Unknown coupling mode: {self.mode!r}")

    @staticmethod
    def _component_get(comp: object, key: str, default=None): # NOTE: enforce dataclasses eventually
        """Read fields from either dataclass-like objects or dict specs."""
        if isinstance(comp, dict):
            return comp.get(key, default)
        return getattr(comp, key, default)

    def _build_from_components(self, components: list[object]) -> jnp.ndarray:
        """Build heterogeneous/directed coupling by summing masked kernel components.

        Additive components (gaussian, constant, etc.) are summed into K first.
        Dropout components are collected and applied multiplicatively afterwards,
        scoped by their edge mask so only the targeted edges are randomly zeroed.
        """
        n = self.grid.N

        dist_np, dr_np, dc_np = self.grid.pairwise_distances()
        dist = jnp.asarray(dist_np, dtype=jnp.float32)

        group_ids_np = None
        if self.group_ids is not None:
            group_ids_np = np.asarray(self.group_ids, dtype=int)
            if group_ids_np.shape[0] != n:
                raise ValueError(
                    f"group_ids length must equal N={n}, got {group_ids_np.shape[0]}"
                )

        K = jnp.zeros((n, n), dtype=jnp.float32)
        deferred_dropout: list[tuple[jnp.ndarray, jnp.ndarray]] = []

        for comp_idx, comp in enumerate(components):
            kernel_name = self._component_get(comp, "kernel", "gaussian")
            strength = self._component_get(comp, "base_strength", 1.0)
            kernel_params = self._component_get(comp, "kernel_params", {})
            cutoff_radius = self._component_get(comp, "radius", None)
            seed = self._component_get(comp, "seed", 0)
            if cutoff_radius is None:
                cutoff_radius = self.radius

            edge_mode = self._component_get(comp, "edge_mode", "outgoing")
            node_groups = self._component_get(comp, "node_groups", None)
            to_node_groups = self._component_get(comp, "to_node_groups", None)

            if edge_mode not in {"within", "outgoing", "incoming", "custom"}:
                raise ValueError(
                    f"Component {comp_idx} has invalid edge_mode={edge_mode!r}. "
                    "Expected one of: within/outgoing/incoming/custom."
                )

            # Sender selection
            if node_groups is None:
                sender_sel = np.ones((n,), dtype=bool)
            else:
                if group_ids_np is None:
                    raise ValueError(
                        f"Component {comp_idx} specifies `node_groups` but `group_ids` is not provided."
                    )
                sender_sel = np.isin(group_ids_np, node_groups)

            # Receiver selection
            if edge_mode == "custom":
                if to_node_groups is None:
                    raise ValueError(
                        f"Component {comp_idx} has edge_mode='custom' but no `to_node_groups` provided."
                    )
                if group_ids_np is None:
                    raise ValueError(
                        f"Component {comp_idx} specifies `to_node_groups` but `group_ids` is not provided."
                    )
                receiver_sel = np.isin(group_ids_np, to_node_groups)
            elif edge_mode == "within":
                receiver_sel = sender_sel
            elif edge_mode == "incoming":
                receiver_sel = sender_sel
            elif edge_mode == "outgoing":
                receiver_sel = None

            # Directed edge mask M[i,j]
            if edge_mode == "within":
                mask_np = sender_sel[:, None] & sender_sel[None, :]
            elif edge_mode == "outgoing":
                mask_np = np.broadcast_to(sender_sel[None, :], (n, n))
            elif edge_mode == "incoming":
                mask_np = np.broadcast_to(sender_sel[:, None], (n, n))
            else:  # custom
                mask_np = receiver_sel[:, None] & sender_sel[None, :]

            mask = jnp.asarray(mask_np.astype(np.float32))

            # Bufffer dropout mask for multiplicative application
            if kernel_name == "dropout":
                binary_mask = dropout_kernel(
                    d=dist,
                    dropout_frac=kernel_params.get("dropout_frac", 0.5),
                    seed=seed,
                )
                deferred_dropout.append((binary_mask, mask))
                continue

            # Additive kernels
            if kernel_name == "constant":
                weights = apply_kernel_jax(
                    d=dist, name=kernel_name, params=kernel_params, radius=None,
                )
            elif kernel_name == "gaussian":
                weights = apply_kernel_jax(
                    d=dist, name=kernel_name, params=kernel_params,
                    radius=cutoff_radius,
                )
            else:
                if cutoff_radius is not None:
                    dist_mask = (dist_np <= cutoff_radius) & (dist_np > 0)
                else:
                    dist_mask = dist_np > 0

                weights_np = apply_kernel(
                    d=dist_np, name=kernel_name, params=kernel_params,
                    radius=cutoff_radius, dx=dr_np, dy=dc_np,
                )
                weights_np = weights_np * dist_mask.astype(weights_np.dtype)
                weights = jnp.asarray(weights_np, dtype=jnp.float32)

            K = K + jnp.asarray(strength, dtype=jnp.float32) * weights * mask

        # Apply dropout masks multiplicatively
        for binary_mask, edge_mask in deferred_dropout:
            K = K * jnp.where(edge_mask > 0, binary_mask, 1.0)

        return K

# --- Lesion parametrization utilities ---
def apply_node_lesions(K: jnp.ndarray, alpha: jnp.ndarray) -> jnp.ndarray:
    """Apply continuous node lesions to a coupling matrix.

    alpha_i in [0, 1] scales all edges incident to node i:
      K[i, j] -> (1 - alpha_i) * K[i, j]
      K[j, i] -> (1 - alpha_i) * K[j, i]

    Vectorized form implements row then column scaling, yielding:
      K_lesioned[i, j] = (1 - alpha_i) * (1 - alpha_j) * K[i, j]
    """
    alpha = jnp.asarray(alpha)
    s = 1.0 - alpha  # (N,)
    return (s[:, None] * K) * s[None, :]

# --- Visualization utilities ---
def plot_lesioned_coupling(alpha: jnp.ndarray, K_base: jnp.ndarray, K_lesioned: jnp.ndarray, grid_shape: tuple[int, int], title: str | None = None, axs: list[plt.Axes] | None = None) -> None:
    """Visualize the lesioned coupling matrix.

    alpha: lesion mask
    K_base: original coupling matrix
    K_lesioned: lesioned coupling matrix
    """
    K0 = np.asarray(K_base)
    K1 = np.asarray(K_lesioned)
    removed = np.clip(K0 - K1, 0, None)  # only drops where lesion zeros weight

    if axs is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    else:
        fig = axs[0].get_figure()
        axes = axs.ravel()

    im_a = axes[0].imshow(alpha.reshape(grid_shape), aspect="equal", cmap="Reds", vmin=0, vmax=1)
    axes[0].set_title("Lesion Mask")
    # axes[0].set_xlabel("j")
    # axes[0].set_ylabel("i")
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(im_a, ax=axes[0], fraction=0.046, pad=0.04)

    im_b = axes[1].imshow(removed, aspect="equal", cmap="magma")
    axes[1].set_title("Removed coupling (K - K_lesioned)")
    # axes[1].set_xlabel("j")
    # axes[1].set_ylabel("i")
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im_b, ax=axes[1], fraction=0.046, pad=0.04)
    
    if title is not None:
        fig.suptitle(title)

    return fig, axes