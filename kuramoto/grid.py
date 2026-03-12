from __future__ import annotations

import numpy as np


class CorticalGrid:
    def __init__(self, shape: tuple[int, int], periodic_bc: bool = False):
        self.shape = tuple(shape)
        self.n_rows, self.n_cols = self.shape
        self.n_total = self.n_rows * self.n_cols
        self.periodic_bc = periodic_bc

    def flat_to_2d(self, idx: int) -> tuple[int, int]:
        """Flat index -> (row, col)."""
        return divmod(idx, self.n_cols)

    def idx_2d_to_flat(self, row: int, col: int) -> int:
        """(row, col) -> flat index."""
        return row * self.n_cols + col

    def flatten(self, arr_2d: np.ndarray) -> np.ndarray:
        """Reshape (n_rows, n_cols) array to (n_total,)."""
        return np.asarray(arr_2d).ravel()

    def unflatten(self, arr_flat: np.ndarray) -> np.ndarray:
        """Reshape (n_total,) array to (n_rows, n_cols)."""
        return np.asarray(arr_flat).reshape(self.shape)

    def pairwise_distances(self) -> np.ndarray:
        """Euclidean distance and displacement matrices (n_total x n_total)."""
        r = np.arange(self.n_total, dtype=float) // self.n_cols
        c = np.arange(self.n_total, dtype=float) % self.n_cols

        dr = r[:, None] - r[None, :]
        dc = c[:, None] - c[None, :]

        if self.periodic_bc:
            dr = np.minimum(np.abs(dr), self.n_rows - np.abs(dr))
            dc = np.minimum(np.abs(dc), self.n_cols - np.abs(dc))

        return np.sqrt(dr**2 + dc**2), dr, dc
