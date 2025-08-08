import argparse, os, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from src.mbsims.rng import set_seed

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="outputs/fig2")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--start", type=float, nargs=2, default=[0.8, 0.8])
    p.add_argument("--rho_low", type=float, default=0.2)
    p.add_argument("--rho_high", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    n = args.n
    # F(x,y)=x^2+y^2 on [-1,1]^2 grid
    x = np.linspace(-1,1,n); X,Y = np.meshgrid(x,x, indexing="ij")
    F = X**2 + Y**2

    def simulate(rho_const):
        pos = np.array([(args.start[0]+1)/2*n, (args.start[1]+1)/2*n], dtype=float)
        traj = [pos.copy()]
        gx, gy = np.gradient(F, axis=0), np.gradient(F, axis=1)
        for _ in range(args.steps):
            i = int(np.clip(round(pos[0]),0,n-1)); j=int(np.clip(round(pos[1]),0,n-1))
            step = -(1.0 - rho_const)*np.array([gx[i,j], gy[i,j]])
            pos = np.clip(pos + args.dt*step, 0, n-1); traj.append(pos.copy())
        return np.array(traj)

    traj_low = simulate(args.rho_low)
    traj_high = simulate(args.rho_high)

    # 2D plot with equal aspect
    plt.figure()
    plt.contour(X,Y,F,levels=20)
    plt.plot((traj_low[:,1]/(n-1))*2-1, (traj_low[:,0]/(n-1))*2-1, label=f"rho={args.rho_low}")
    plt.plot((traj_high[:,1]/(n-1))*2-1, (traj_high[:,0]/(n-1))*2-1, label=f"rho={args.rho_high}")
    plt.gca().set_aspect("equal")
    plt.legend(); plt.title("Figure 2 — Agent trajectories shaped by MB density")
    plt.tight_layout(); plt.savefig(os.path.join(args.out,"fig2.png")); plt.close()

if __name__ == "__main__":
    main()
