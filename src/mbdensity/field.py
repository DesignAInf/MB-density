import numpy as np

def make_random_fourier_field(n: int, length_scale: float = 0.2, modes: int = 8, rng=None):
    """Simple stationary random Fourier series on [0,1]^2."""
    if rng is None:
        rng = np.random
    x = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    field = np.zeros_like(X, dtype=float)
    for _ in range(modes):
        ax, ay = rng.uniform(1.0, 6.0, size=2)
        px, py = rng.uniform(0, 2*np.pi, size=2)
        amp = rng.normal(scale=1.0)
        field += amp * np.cos(ax*X + px) * np.cos(ay*Y + py)
    # normalize
    field = (field - field.mean()) / (field.std() + 1e-12)
    # smooth-ish by gaussian-like scaling
    return np.exp(-length_scale) * field
