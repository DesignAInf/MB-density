import argparse, os, numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from src.mbsims.rng import set_seed

def run_panel(rho_expr, n=100, steps=80, dt=0.02, start=(0.8,-0.8)):
    x = np.linspace(-1,1,n); X,Y=np.meshgrid(x,x, indexing="ij")
    F = X**2 + Y**2
    rho = rho_expr(X,Y)
    pos = np.array([(start[0]+1)/2*n, (start[1]+1)/2*n], dtype=float)
    traj=[pos.copy()]
    gx, gy = np.gradient(F, axis=0), np.gradient(F, axis=1)
    rho_bar = rho.mean()
    for _ in range(steps):
        i = int(np.clip(round(pos[0]),0,n-1)); j=int(np.clip(round(pos[1]),0,n-1))
        step = -(1.0 - rho_bar)*np.array([gx[i,j], gy[i,j]])
        pos = np.clip(pos + dt*step, 0, n-1); traj.append(pos.copy())
    return X,Y,F,np.array(traj)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="outputs/fig5")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(); set_seed(args.seed); os.makedirs(args.out, exist_ok=True)

    rhoA = lambda X,Y: 0.2 + 0.3*( (X+1)/2 )
    rhoB = lambda X,Y: 0.8 + 0.2*( (X+1)/2 )
    X,Y,F,trA = run_panel(rhoA)
    X,Y,F,trB = run_panel(rhoB)

    # Render two panels
    for name, traj in [("A",trA),("B",trB)]:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, F, alpha=0.7, rstride=4, cstride=4)
        ax.plot(traj[:,1]*2/100-1, traj[:,0]*2/100-1, F[traj[:,0].astype(int),traj[:,1].astype(int)], linewidth=2)
        ax.set_title(f"Figure 5 — Panel {name}")
        plt.tight_layout(); plt.savefig(os.path.join(args.out, f"fig5_{name}.png")); plt.close()

if __name__ == "__main__":
    main()
