from __future__ import annotations

from collections import deque

import numpy as np
from scipy.integrate import RK45, solve_ivp
from scipy.sparse import spmatrix

from .analysis import order_parameter, local_order, coupling_tension
from .coupling import CouplingMatrix
from .grid import CorticalGrid

from typing import TYPE_CHECKING 
if TYPE_CHECKING:
    from .config import SimulationConfig # Only import for type check
    from .storage import InMemoryStorage

# --- Dynamics: Kuramoto right-hand sides ---

def kuramoto_rhs(
    theta: np.ndarray,
    omega: np.ndarray,
    K: spmatrix,
    theta_source: np.ndarray | None = None,
) -> np.ndarray:
    """Sparse-coupling:  dtheta_i/dt = omega_i + sum_j K_ij sin(theta_j - theta_i)."""
    src = theta if theta_source is None else theta_source
    coupling = np.cos(theta) * (K @ np.sin(src)) - np.sin(theta) * (K @ np.cos(src))
    return omega + coupling


def kuramoto_rhs_uniform(
    theta: np.ndarray,
    omega: np.ndarray,
    strength: float,
    theta_source: np.ndarray | None = None,
) -> np.ndarray:
    """all-to-all coupling with uniform coupling strength:
    dtheta_i/dt = omega_i + sum_j K sin(theta_j - theta_i)"""
    
    src = theta if theta_source is None else theta_source
    S, C = np.sum(np.sin(src)), np.sum(np.cos(src))
    coupling = strength * (np.cos(theta) * S - np.sin(theta) * C)
    return omega + coupling


# --- Delay buffer ---
class DelayBuffer:
    """Ring buffer that stores recent theta snapshots for time-delayed coupling."""

    def __init__(self, tau: float, dt: float, n_total: int):
        self.tau = tau
        self.dt = dt
        self.n_total = n_total
        self._buf: deque[tuple[float, np.ndarray]] = deque()
        self._maxlen = int(np.ceil(tau / dt)) + 2

    def initialize(self, t: float, theta: np.ndarray, dt: float):
        """Fill buffer assuming constant phase before t_start."""
        self.dt = dt
        self._maxlen = int(np.ceil(self.tau / dt)) + 2
        self._buf.clear()
        for _ in range(self._maxlen):
            self._buf.append((t, np.copy(theta)))
            t -= dt

    def push(self, t: float, theta: np.ndarray):
        """Add new theta snapshot to buffer, remove oldest if buffer is full"""
        self._buf.append((t, np.copy(theta)))
        while len(self._buf) > self._maxlen:
            self._buf.popleft()

    def get_delayed(self, t: float) -> np.ndarray:
        """Return theta at time (t - tau), linearly interpolattion between entries"""
        t_want = t - self.tau
        if not self._buf:
            return np.zeros(self.n_total)

        times = [b[0] for b in self._buf]
        if t_want <= times[0]:
            return np.copy(self._buf[0][1])
        if t_want >= times[-1]:
            return np.copy(self._buf[-1][1])

        for i in range(len(self._buf) - 1):
            t0, th0 = self._buf[i]
            t1, th1 = self._buf[i + 1]
            if t0 <= t_want <= t1:
                if t1 == t0:
                    return np.copy(th0)
                alpha = (t_want - t0) / (t1 - t0)
                return (1 - alpha) * th0 + alpha * th1

        return np.copy(self._buf[-1][1])


# --- Simulation ---
class Simulation:
    def __init__(
        self,
        grid: CorticalGrid,
        coupling: CouplingMatrix,
        omega0: np.ndarray,
        theta0: np.ndarray,
        delays: DelayBuffer | None = None,
        config: SimulationConfig | None = None,
    ):
        self.grid = grid
        self.coupling = coupling
        self.omega0 = omega0
        self.theta0 = theta0
        self.delays = delays
        self.config = config
    
    def run(
        self,
        t_span: tuple[float, float],
        dt: float,
        rng: np.random.Generator | None = None,
        storage=None,
    ) -> tuple[list[float], list[dict]]:
        """Integrate kuramoto model, returns (t_list, state_list) where each state dict has 'theta' and 'omega'.
        """
        if rng is None:
            rng = np.random.default_rng()

        theta0 = self.theta0
        rhs = self._make_rhs()

        t_eval = np.arange(t_span[0] + dt, t_span[1] + dt / 2, dt)
        t_eval = t_eval[t_eval <= t_span[1]]

        if self.delays is not None:
            self.delays.initialize(t_span[0], theta0, dt)
            t_out, y_out = self._step_with_delays(rhs, theta0, t_span, dt)
        else:
            sol = solve_ivp(
                rhs, t_span, theta0,
                method="RK45", t_eval=t_eval, max_step=dt,
                rtol=1e-6, atol=1e-6,
            )
            if sol.status != 0:
                raise RuntimeError(f"solve_ivp failed: {sol.message}")
            t_out, y_out = sol.t, sol.y

        return self._collect_results(t_out, y_out, storage)

    def _make_rhs(self):
        """Select appropriate rhs function."""
        omega = self.omega0
        coupling = self.coupling
        delays = self.delays

        def rhs(t: float, theta: np.ndarray) -> np.ndarray:
            theta_src = delays.get_delayed(t) if delays is not None else None # pointer to delays object persists between calls

            if coupling.is_uniform:
                return kuramoto_rhs_uniform(theta, omega, coupling.uniform_strength, theta_src)
            return kuramoto_rhs(theta, omega, coupling.K, theta_src)

        return rhs

    def _step_with_delays(self, rhs, theta0, t_span, dt):
        """Manual RK45 stepping so we can feed the delay buffer after each step."""
        t_list, y_list = [], []
        next_save = t_span[0] + dt

        stepper = RK45(rhs, t_span[0], theta0, t_span[1],
                       max_step=dt, rtol=1e-6, atol=1e-6)

        while stepper.status == "running":
            stepper.step()
            self.delays.push(stepper.t, stepper.y)

            if stepper.t >= next_save - 1e-12:
                t_list.append(stepper.t)
                y_list.append(stepper.y.copy())
                next_save += dt

        t_out = np.array(t_list)
        y_out = np.column_stack(y_list) if y_list else np.empty((len(theta0), 0))
        return t_out, y_out

    def _collect_results(self, t_out, y_out, storage: InMemoryStorage | None):
        """Pack solver output into the (t_list, state_list) format."""
        t_list: list[float] = []
        state_list: list[dict] = []

        rhs = self._make_rhs() # for evaluating omega at each step

        for i in range(len(t_out)):
            t = float(t_out[i])
            theta = y_out[:, i].copy()

            theta_dot = rhs(t, theta)
            omega = self.omega0 # omega constant for now

            state = {"theta": theta, "theta_dot": theta_dot, "omega": omega} 
            t_list.append(t)
            state_list.append(state)

            K = self.coupling.K

            if storage is not None:
                storage.write_snapshot(t, state, K=K)
                R, _ = order_parameter(theta)
                storage.write_scalar(t, "order_param", R)
                local_R = local_order(theta, self.grid)
                storage.write_scalar(t, "local_order", local_R)
                coupling_t= coupling_tension(theta, omega, K)
                storage.write_scalar(t, "coupling_tension", coupling_t)

        if storage is not None:
            storage.finalize()

        return t_list, state_list
