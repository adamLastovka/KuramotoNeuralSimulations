from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

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

        self._K: csr_matrix | None = None
        self._uniform_strength = base_strength / grid.n_total if mode == "uniform" else 0.0

        if mode != "uniform":
            self._K = self._build()
        # Uniform mode: no matrix built; RHS uses uniform_strength only

    def _build(self) -> csr_matrix:
        dist,dr,dc = self.grid.pairwise_distances()
        n = self.grid.n_total

        if self.radius is not None:
            mask = (dist <= self.radius) & (dist > 0)
        else:
            mask = dist > 0

        rows, cols = np.where(mask)

        # Get distanecs and displacements for slice of the matrix
        d = dist[rows, cols]
        dx = dr[rows, cols]
        dy = dc[rows, cols]

        # Get coupling weights from kernel and K
        weights = self.base_strength * apply_kernel(
            d,
            self.kernel_name,
            self.kernel_params,
            self.radius,
            dx=dx,
            dy=dy,
        )
        return csr_matrix((weights, (rows, cols)), shape=(n, n))

    @property
    def K(self) -> csr_matrix | None:
        return self._K

    @property
    def is_uniform(self) -> bool:
        return self.mode == "uniform"

    @property
    def uniform_strength(self) -> float:
        return self._uniform_strength
