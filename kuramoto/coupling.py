from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .grid import CorticalGrid
from .kernels import apply_kernel


class CouplingMatrix:
    """Coupling weights K[i,j] = strength from oscillator j to oscillator i.

    Uniform: K[i,j] = base_strength / N
    Spatial: K[i,j] = base_strength * kernel(distance(i,j))
    """

    def __init__(
        self,
        grid: CorticalGrid,
        kernel: str = "gaussian",
        base_strength: float = 1.0,
        kernel_params: dict | None = None,
        radius: float | None = None,
        mode: str = "spatial",
    ):
        self.grid = grid
        self.kernel_name = kernel
        self.base_strength = base_strength
        self.kernel_params = kernel_params or {}
        self.radius = radius
        self.mode = mode


        if mode == "uniform":
            self._uniform_strength = base_strength / grid.n_total

            n = grid.n_total
            self.K = jnp.full((n, n), self._uniform_strength)

        else:
            self.K = self._build()

    def _build(self) -> jnp.ndarray:
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

        # Get Coupling weights
        weights = self.base_strength * apply_kernel(
            d,
            self.kernel_name,
            self.kernel_params,
            self.radius,
            dx=dx,
            dy=dy,
        )

        K_np = np.zeros((n, n), dtype=np.float64)
        K_np[rows, cols] = weights
        return jnp.array(K_np)
