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
            if mode == "uniform":
                raise NotImplementedError(
                    "Heterogeneous coupling components are currently supported only for spatial coupling."
                )
            if not self.components:
                raise ValueError("components provided but empty.")
            self.K = self._build_heterogeneous()
        elif mode == "uniform":
            self._uniform_strength = base_strength / grid.n_total
            n = grid.n_total
            self.K = jnp.full((n, n), self._uniform_strength)
        else:
            self.K = self._build_single_kernel()

    def _build_single_kernel(self) -> jnp.ndarray:
        dist, dr, dc = self.grid.pairwise_distances()
        n = self.grid.n_total

        if self.radius is not None:
            mask = (dist <= self.radius) & (dist > 0)
        else:
            mask = dist > 0

        rows, cols = np.where(mask)
        d = dist[rows, cols]
        dx = dr[rows, cols]
        dy = dc[rows, cols]

        weights = self.base_strength * apply_kernel(
            d,
            self.kernel_name,
            self.kernel_params,
            self.radius,
            dx=dx,
            dy=dy,
        )

        K_np = np.zeros((n, n), dtype=np.float32)
        K_np[rows, cols] = weights
        return jnp.asarray(K_np, dtype=jnp.float32)

    def _build_heterogeneous(self) -> jnp.ndarray:
        """Build heterogeneous dense K by summing masked kernel components."""
        if self.group_ids is None:
            raise ValueError("Heterogeneous coupling requires `group_ids`.")

        n = self.grid.n_total
        group_ids_np = np.asarray(self.group_ids, dtype=int)
        if group_ids_np.shape[0] != n:
            raise ValueError(
                f"group_ids length must equal n_total={n}, got {group_ids_np.shape[0]}"
            )

        # Distances are constants w.r.t. kernel params; keep them in JAX once.
        dist_np, dr_np, dc_np = self.grid.pairwise_distances()

        dist = jnp.asarray(dist_np, dtype=jnp.float32)
        _dr = jnp.asarray(dr_np)
        _dc = jnp.asarray(dc_np)

        K = jnp.zeros((n, n), dtype=jnp.float32)

        for comp in self.components or []:
            apply_to = getattr(comp, "apply_to", None)
            if apply_to is None:
                raise ValueError("Each heterogeneous component must have `apply_to`.")

            target_groups = apply_to.get("target_groups")
            source_groups = apply_to.get("source_groups")
            if target_groups is None or source_groups is None:
                raise ValueError(
                    "apply_to must contain `target_groups` and `source_groups`."
                )

            target_mask = np.isin(group_ids_np, target_groups)  # (n,)
            source_mask = np.isin(group_ids_np, source_groups)  # (n,)
            mask_np = target_mask[:, None] & source_mask[None, :]  # (n,n)
            mask = jnp.asarray(mask_np.astype(np.float64))

            kernel_name = comp.kernel

            cutoff_radius = comp.radius if comp.radius is not None else self.radius

            strength = jnp.asarray(comp.base_strength)
            kernel_params = getattr(comp, "kernel_params", None) or {}

            weights = apply_kernel_jax(
                d=dist,
                name=kernel_name,
                params=kernel_params,
                radius=cutoff_radius,
            )

            K = K + strength * weights * mask

        return K
