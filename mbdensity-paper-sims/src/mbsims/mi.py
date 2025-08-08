import numpy as np
from sklearn.neighbors import NearestNeighbors

def _entropy_knn(X, k):
    n, d = X.shape
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(X)
    dist, _ = nbrs.kneighbors(X)
    # Use k-th neighbor distance
    eps = dist[:, -1] + 1e-12
    c_d = np.pi**(d/2) / np.math.gamma(d/2 + 1)
    H = np.log(eps).mean() + (d*np.log(n-1) - np.log(c_d)) / n
    return H

def mi_ksg(X, Y, k=5):
    XY = np.concatenate([X, Y], axis=1)
    Hx = _entropy_knn(X, k)
    Hy = _entropy_knn(Y, k)
    Hxy = _entropy_knn(XY, k)
    return max(0.0, Hx + Hy - Hxy)

def rho_from_mi(I_IE, I_IE_given_B, epsilon=1e-6):
    denom = max(epsilon, I_IE)
    return 1.0 - (I_IE_given_B / denom)
