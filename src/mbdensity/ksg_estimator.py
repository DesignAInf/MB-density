import numpy as np
from sklearn.neighbors import NearestNeighbors

def _entropy_knn(X, k):
    # Kozachenko-Leonenko entropy (very rough, for demo purposes)
    n, d = X.shape
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm="auto").fit(X)
    distances, _ = nbrs.kneighbors(X)
    # exclude self (distance zero)
    eps = distances[:, -1] + 1e-12
    c_d = np.pi**(d/2) / np.math.gamma(d/2 + 1)
    H = np.log(eps).mean() + np.log(2) + (d*np.log(n-1) - np.log(c_d)) / n
    return H

def mi_ksg(X, Y, k=5):
    # Very lightweight MI approximation using entropies
    XY = np.concatenate([X, Y], axis=1)
    Hx = _entropy_knn(X, k)
    Hy = _entropy_knn(Y, k)
    Hxy = _entropy_knn(XY, k)
    return max(0.0, Hx + Hy - Hxy)

def cmi_via_chain_rule(I_E, I_E_given_B):
    # ρ = 1 - I(I;E|B)/I(I;E)
    return I_E, I_E_given_B

def rho_normalized(I_E, I_E_given_B, epsilon=1e-6):
    denom = max(epsilon, I_E)
    return 1.0 - (I_E_given_B / denom)
