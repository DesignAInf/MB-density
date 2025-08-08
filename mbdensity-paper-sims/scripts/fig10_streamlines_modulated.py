import argparse, os, numpy as np, matplotlib.pyplot as plt
from src.mbsims.rng import set_seed

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="outputs/fig10")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=100)
    args = p.parse_args(); set_seed(args.seed); os.makedirs(args.out, exist_ok=True)
    n=args.n; x=np.linspace(-2,2,n); X,Y=np.meshgrid(x,x, indexing="ij")
    F = (X-0.5)**2 + (Y+0.2)**2
    rho = 0.2 + 0.6*np.exp(-((X+0.5)**2+(Y-0.3)**2)/0.5)
    gx, gy = np.gradient(F, axis=0), np.gradient(F, axis=1)
    U = -(1.0 - rho)*gx; V = -(1.0 - rho)*gy

    plt.figure()
    plt.contour(X,Y,F, levels=20)
    plt.imshow(rho, origin="lower", extent=[x.min(), x.max(), x.min(), x.max()], alpha=0.5)
    strm = plt.streamplot(X, Y, U, V, density=1.2)
    # sample trajectories from a few starts
    starts = np.array([[ -1.5, -1.5],[ -1.0, 1.6],[ 1.6, -1.2],[ 0.0, 1.8],[ 1.5, 1.5],[ -1.6, 0.0]])
    for s in starts:
        pos = np.array([(s[0]-x.min())/(x.max()-x.min())*(n-1), (s[1]-x.min())/(x.max()-x.min())*(n-1)], dtype=float)
        traj=[pos.copy()]
        for _ in range(200):
            i = int(np.clip(round(pos[0]),0,n-1)); j=int(np.clip(round(pos[1]),0,n-1))
            step = np.array([U[i,j], V[i,j]])
            pos = np.clip(pos + 0.05*step, 0, n-1); traj.append(pos.copy())
        traj=np.array(traj)
        xs = traj[:,1]/(n-1)*(x.max()-x.min())+x.min(); ys=traj[:,0]/(n-1)*(x.max()-x.min())+x.min()
        plt.plot(xs, ys, linewidth=2)
    plt.title("Figure 10 — Modulated Free-Energy Descent")
    plt.tight_layout(); plt.savefig(os.path.join(args.out,"fig10.png")); plt.close()

if __name__=="__main__":
    main()
