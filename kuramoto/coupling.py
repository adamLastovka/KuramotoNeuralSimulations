from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import jax.numpy as jnp

from .grid import CorticalGrid
from .kernels import apply_kernel, apply_kernel_jax

if TYPE_CHECKING:
    from .config import KernelComponentConfig


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
    ):
        self.grid = grid
        self.kernel_name = kernel
        self.base_strength = base_strength
        self.kernel_params = kernel_params or {}
        self.radius = radius
        self.mode = mode

        self.components = components
        self.group_ids = group_ids

        if self.components is not None:
            if not self.components:
                raise ValueError("components provided but empty.")
            components_to_build = self.components
        else:
            components_to_build = self._expand_legacy_mode_to_components()

        self.K = self._build_from_components(components_to_build)

    def _expand_legacy_mode_to_components(self) -> list[dict]: # NOTE: Temporary - remove evenually
        """Expand `mode`/legacy fields into the component-list representation."""
        n = self.grid.n_total
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
        """Build heterogeneous/directed coupling by summing masked kernel components."""
        n = self.grid.n_total

        # Distances are constants w.r.t. kernel params; keep them in JAX once.
        dist_np, dr_np, dc_np = self.grid.pairwise_distances()
        dist = jnp.asarray(dist_np, dtype=jnp.float32)

        # Only required if any component references group_ids (directed node selectors).
        group_ids_np = None
        if self.group_ids is not None:
            group_ids_np = np.asarray(self.group_ids, dtype=int)
            if group_ids_np.shape[0] != n:
                raise ValueError(
                    f"group_ids length must equal n_total={n}, got {group_ids_np.shape[0]}"
                )

        K = jnp.zeros((n, n), dtype=jnp.float32)

        for comp_idx, comp in enumerate(components):
            kernel_name = self._component_get(comp, "kernel", "gaussian")
            strength = self._component_get(comp, "base_strength", 1.0)
            kernel_params = self._component_get(comp, "kernel_params", {})
            cutoff_radius = self._component_get(comp, "radius", None)
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

            # Sender selection: group membership for column j.
            if node_groups is None:
                sender_sel = np.ones((n,), dtype=bool)
            else:
                if group_ids_np is None:
                    raise ValueError(
                        f"Component {comp_idx} specifies `node_groups` but `group_ids` is not provided."
                    )
                sender_sel = np.isin(group_ids_np, node_groups)

            # Receiver selection: group membership for row i.
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
                receiver_sel = None  # any receiver

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

            # Kernel weights W[i,j]
            if kernel_name == "constant":
                weights = apply_kernel_jax(
                    d=dist,
                    name=kernel_name,
                    params=kernel_params,
                    radius=None,
                )
            elif kernel_name == "gaussian":
                weights = apply_kernel_jax(
                    d=dist,
                    name=kernel_name,
                    params=kernel_params,
                    radius=cutoff_radius,
                )
            else:
                # Non-JAX kernels: compute via NumPy and apply the same spatial self/threshold rules.
                if cutoff_radius is not None:
                    dist_mask = (dist_np <= cutoff_radius) & (dist_np > 0)
                else:
                    dist_mask = dist_np > 0

                weights_np = apply_kernel(
                    d=dist_np,
                    name=kernel_name,
                    params=kernel_params,
                    radius=cutoff_radius,
                    dx=dr_np,
                    dy=dc_np,
                )
                weights_np = weights_np * dist_mask.astype(weights_np.dtype)
                weights = jnp.asarray(weights_np, dtype=jnp.float32)

            K = K + jnp.asarray(strength, dtype=jnp.float32) * weights * mask

        return K
