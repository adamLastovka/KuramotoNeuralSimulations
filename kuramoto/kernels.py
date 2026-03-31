import numpy as np
import jax.numpy as jnp

def gaussian_kernel(d: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian falloff: exp(-d^2 / 2sigma^2)."""
    return np.exp(-(d**2) / (2 * sigma**2))


def exponential_kernel(d: np.ndarray, tau: float) -> np.ndarray:
    """Exponential falloff: exp(-d / tau)."""
    return np.exp(-d / tau)


def step_kernel(d: np.ndarray, radius: float) -> np.ndarray:
    """Hard cutoff: 1 inside radius, 0 outside."""
    return (d <= radius).astype(float)


def gabor_kernel(
    d: np.ndarray,
    sigma: float,
    theta_pref: float,
    freq: float,
    gamma: float = 1.0,
    dx: np.ndarray | None = None,
    dy: np.ndarray | None = None,
) -> np.ndarray:
    """
    Preferred direction is theta_pref (radians, 0 = horizontal).
    """
    xp = dx * np.cos(theta_pref) + dy * np.sin(theta_pref)
    yp = -dx * np.sin(theta_pref) + dy * np.cos(theta_pref)
    envelope = np.exp(-(xp**2 + (yp / gamma) ** 2) / (2 * sigma**2))
    carrier = np.cos(2 * np.pi * freq * xp)

    return envelope * np.maximum(carrier, 0.0)


def elongated_gaussian_kernel(
    d: np.ndarray,
    sigma_par: float,
    sigma_perp: float,
    theta: float,
    dx: np.ndarray | None = None,
    dy: np.ndarray | None = None,
) -> np.ndarray:
    """Anisotropic Gaussian: stronger coupling along direction theta.
    """
    xp = dx * np.cos(theta) + dy * np.sin(theta)
    yp = -dx * np.sin(theta) + dy * np.cos(theta)
    return np.exp(-(xp**2 / (2 * sigma_par**2) + yp**2 / (2 * sigma_perp**2)))


def lesioned_wedge_kernel(
    d: np.ndarray,
    theta_center: float,
    wedge_width: float,
    sigma: float = 2.0,
    dx: np.ndarray | None = None,
    dy: np.ndarray | None = None,
) -> np.ndarray:
    """Gaussian coupling only inside an angular wedge

    Wedge is centered at theta_center (radians) with total width wedge_width
    (radians). Outside the wedge coupling is zero. 
    """
    base = np.exp(-(d**2) / (2 * sigma**2))
    angle = np.arctan2(dy, dx)
    half = wedge_width / 2
    lo = theta_center - half
    hi = theta_center + half
    mask = (angle >= lo) & (angle <= hi)
    return base * mask.astype(float)

def mexican_hat_kernel(
    d: np.ndarray,
    sigma_e: float = 1.0,
    sigma_i: float = 3.0,
    a_e: float = 1.0,
    a_i: float = 0.5,
) -> np.ndarray:
    """Difference-of-Gaussians: local excitation, surround inhibition."""
    exc = a_e * np.exp(-(d**2) / (2 * sigma_e**2))
    inh = a_i * np.exp(-(d**2) / (2 * sigma_i**2))
    return exc - inh


_KERNELS = {
    "gaussian": lambda d, p, r, dx=None, dy=None: gaussian_kernel(
        d, p.get("sigma", 2.0)
    ),
    "exponential": lambda d, p, r, dx=None, dy=None: exponential_kernel(
        d, p.get("tau", 2.0)
    ),
    "step": lambda d, p, r, dx=None, dy=None: step_kernel(
        d, p.get("radius", r or 3.0)
    ),
    "mexican_hat": lambda d, p, r, dx=None, dy=None: mexican_hat_kernel(
        d,
        sigma_e=p.get("sigma_e", 1.0),
        sigma_i=p.get("sigma_i", 3.0),
        a_e=p.get("a_e", 1.0),
        a_i=p.get("a_i", 0.5),
    ),
    "gabor": lambda d, p, r, dx=None, dy=None: gabor_kernel(
        d,
        sigma=p.get("sigma", 2.0),
        theta_pref=p.get("theta_pref", 0.0),
        freq=p.get("freq", 0.2),
        gamma=p.get("gamma", 1.0),
        dx=dx,
        dy=dy,
    ),
    "elongated_gaussian": lambda d, p, r, dx=None, dy=None: elongated_gaussian_kernel(
        d,
        sigma_par=p.get("sigma_par", 3.0),
        sigma_perp=p.get("sigma_perp", 1.0),
        theta=p.get("theta", 0.0),
        dx=dx,
        dy=dy,
    ),
    "lesioned_wedge": lambda d, p, r, dx=None, dy=None: lesioned_wedge_kernel(
        d,
        theta_center=p.get("theta_center", 0.0),
        wedge_width=p.get("wedge_width", np.pi / 2),
        sigma=p.get("sigma", 2.0),
        dx=dx,
        dy=dy,
    ),
}


def apply_kernel(
    d: np.ndarray,
    name: str,
    params: dict,
    radius: float | None = None,
    dx: np.ndarray | None = None,
    dy: np.ndarray | None = None,
) -> np.ndarray:
    """Apply kernel to get weights for coupling matrix"""
    fn = _KERNELS.get(name)
    if fn is None:
        raise ValueError(f"Unknown kernel: {name!r}")
    return fn(d, params, radius, dx, dy)


# --- JAX versions ---
def gaussian_kernel_jax(d: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """Differentiable Gaussian falloff: exp(-d^2 / (2*sigma^2))."""
    sigma = jnp.asarray(sigma)
    return jnp.exp(-(d**2) / (2.0 * sigma**2))


def apply_kernel_jax(
    d: jnp.ndarray,
    name: str,
    params: dict,
    radius: float | None = None,
) -> jnp.ndarray:
    """JAX kernel dispatcher.

    Notes
    - For `name="gaussian"`, diagonal entries are forced to 0 (matching the
      existing spatial coupling behavior).
    - For `name="constant"`, returns ones everywhere (diagonal included),
      and `radius` is ignored.
    """
    if name == "constant":
        return jnp.ones_like(d, dtype=jnp.float32)

    if name != "gaussian":
        raise NotImplementedError(
            "apply_kernel_jax currently supports kernel='gaussian' and kernel='constant'."
        )

    sigma = params.get("sigma", 2.0)
    out = gaussian_kernel_jax(d, sigma)

    if radius is not None:
        # Hard cutoff: not differentiable w.r.t radius, but OK for now.
        out = jnp.where((d <= radius) & (d > 0), out, 0.0)
    else:
        # Spatial semantics: exclude self-coupling.
        out = jnp.where(d > 0, out, 0.0)

    return out.astype(jnp.float32)
