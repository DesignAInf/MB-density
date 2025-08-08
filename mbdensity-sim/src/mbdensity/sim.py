import numpy as np
from .config import DEFAULTS
from .rng import set_seed
from .field import make_random_fourier_field
from .ksg_estimator import mi_ksg, rho_normalized

def potential_from_field(F_raw):
    # simple rescaling to [0,1] then multiply
    F = F_raw - F_raw.min()
    F = F / (F.max() + 1e-12)
    return F

def grad(F):
    gx = np.gradient(F, axis=0)
    gy = np.gradient(F, axis=1)
    return gx, gy

def run_demo(n=64, steps=250, alpha=0.5, cfg=DEFAULTS, rng=None):
    if rng is None:
        rng = np.random
    set_seed(cfg.seed)
    # build fields
    F_raw = make_random_fourier_field(n, rng=rng)
    R_raw = make_random_fourier_field(n, rng=rng)
    F = potential_from_field(F_raw)
    # toy MI estimates at two adjacent windows (illustrative only)
    # we treat 2D positions as variables for illustration
    grid = np.stack(np.meshgrid(np.linspace(0,1,n), np.linspace(0,1,n), indexing="ij"), axis=-1).reshape(-1,2)
    # split to fake I(I;E) and I(I;E|B) proxies via neighborhood subsamples
    X = grid.copy()
    Y = np.roll(grid, 1, axis=0)
    # fake "blanket" variable via second roll
    Z = np.roll(grid, 2, axis=0)
    I_IE = mi_ksg(X, Y, k=cfg.k)
    I_IE_given_B = max(0.0, I_IE - mi_ksg(X, Z, k=cfg.k))  # crude proxy
    rho_val = float(np.clip(1.0 - I_IE_given_B / max(cfg.epsilon, I_IE), 0.0, 1.0))
    # spatially uniform rho for the demo, scaled by R field to create texture
    R = (R_raw - R_raw.min()) / (R_raw.max() + 1e-12)
    rho = np.clip(0.5*rho_val + 0.5*R, 0.0, 1.0)
    # integrate dynamics: x_{t+1} = x_t - alpha * (1 - rho) * grad F
    gx, gy = grad(F)
    pos = np.array([n//2, n//2], dtype=float)
    traj = [pos.copy()]
    for t in range(steps):
        i = int(np.clip(round(pos[0]), 0, n-1))
        j = int(np.clip(round(pos[1]), 0, n-1))
        g = np.array([gx[i,j], gy[i,j]])
        step = -alpha * (1.0 - rho[i,j]) * g
        pos = np.clip(pos + step, 0, n-1)
        traj.append(pos.copy())
    return {
        "F": F,
        "rho": rho,
        "traj": np.array(traj),
        "I_IE": I_IE,
        "I_IE_given_B": I_IE_given_B,
        "rho_scalar": rho_val,
        "defaults": dict(r1=cfg.r1, r2=cfg.r2, k=cfg.k, epsilon=cfg.epsilon, delta=cfg.delta, seed=cfg.seed),
    }
