import numpy as np

def gaussian_blob(x, c, Sigma):
    # x: (...,3) or (...,2)
    d = x - c
    if np.ndim(Sigma)==1:
        inv = np.diag(1.0/(np.array(Sigma)**2))
    else:
        inv = np.linalg.inv(Sigma)
    q = np.einsum("...i,ij,...j->...", d, inv, d)
    return np.exp(-0.5*q)

def anisotropic_gaussian_sum(xgrid, centers, sigmas, rotations=None, axes="xyz"):
    # Build anisotropic Gaussians with optional simple rotations around axes
    vals = np.zeros(xgrid.shape[:-1], dtype=float)
    for idx, c in enumerate(centers):
        Sigma = np.array(sigmas[idx])
        if rotations is not None and rotations[idx] is not None:
            # simple axial rotation angles tuple (ax, ay, az) in radians -> rotation matrix
            ax, ay, az = rotations[idx]
            Rx = np.array([[1,0,0],[0,np.cos(ax),-np.sin(ax)],[0,np.sin(ax),np.cos(ax)]])
            Ry = np.array([[np.cos(ay),0,np.sin(ay)],[0,1,0],[-np.sin(ay),0,np.cos(ay)]])
            Rz = np.array([[np.cos(az),-np.sin(az),0],[np.sin(az),np.cos(az),0],[0,0,1]])
            R = Rz @ Ry @ Rx
            inv = R @ np.diag(1.0/(Sigma**2)) @ R.T
            d = xgrid - c
            q = np.einsum("...i,ij,...j->...", d, inv, d)
            vals += np.exp(-0.5*q)
        else:
            vals += gaussian_blob(xgrid, c, Sigma)
    return vals

def random_fourier_field_3d(xgrid, K=15, w_std=1.5, w_range=(0.005,0.02), seed=None):
    rng = np.random.default_rng(seed)
    X = xgrid[...,0]; Y = xgrid[...,1]; Z = xgrid[...,2]
    acc = np.zeros_like(X, dtype=float)
    for _ in range(K):
        wk = rng.normal(0, w_std, size=3)
        omega = rng.uniform(w_range[0], w_range[1])
        phi = rng.uniform(0, 2*np.pi)
        acc += np.sin(wk[0]*X + wk[1]*Y + wk[2]*Z + phi)  # time will be added externally
    # normalize to [0,1]
    acc = (acc - acc.min()) / (acc.max() - acc.min() + 1e-12)
    return acc

def random_fourier_field_3d_time(xgrid, t, K=15, w_std=1.5, w_range=(0.005,0.02), seed=None):
    rng = np.random.default_rng(seed)
    X = xgrid[...,0]; Y = xgrid[...,1]; Z = xgrid[...,2]
    acc = np.zeros_like(X, dtype=float)
    for _ in range(K):
        wk = rng.normal(0, w_std, size=3)
        omega = rng.uniform(w_range[0], w_range[1])
        phi = rng.uniform(0, 2*np.pi)
        acc += np.sin(wk[0]*X + wk[1]*Y + wk[2]*Z + omega*t + phi)
    acc = (acc - acc.min()) / (acc.max() - acc.min() + 1e-12)
    return acc

def ar1_step(x, phi, sigma, rng):
    return phi*x + rng.normal(0.0, sigma, size=x.shape)
