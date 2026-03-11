from __future__ import annotations

from scipy.sparse import spmatrix

class InMemoryStorage:
    """Container for simulation data and analysis results."""

    def __init__(self, downsample_every: int = 1):
        self.downsample_every = downsample_every
        self.t_list: list[float] = []
        self.theta_list: list[list[float] | None] = []
        self.K_list: list = []
        self.scalars: dict[str, list[tuple[float, float]]] = {}
        self._step = 0

    def write_snapshot(self, t: float, state: dict, K: "spmatrix | None" = None):
        self._step += 1
        if self._step % self.downsample_every != 0:
            return

        self.t_list.append(t)
        theta = state.get("theta")
        self.theta_list.append(theta.tolist())
        if K is not None:
            self.K_list.append(K.copy())

    def write_scalar(self, t: float, name: str, value: float):
        self.scalars.setdefault(name, []).append(value) # (t, value)

    def finalize(self):
        # NOTE: Add save function
        pass
