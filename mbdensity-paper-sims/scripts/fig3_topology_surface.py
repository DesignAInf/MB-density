import argparse, os, numpy as np, matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from src.mbsims.rng import set_seed

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="outputs/fig3")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(); set_seed(args.seed); os.makedirs(args.out, exist_ok=True)

    n=args.n; x=np.linspace(-1,1,n); X,Y=np.meshgrid(x,x, indexing="ij")
    # Synthetic high-density topology: ring + bump
    R = np.sqrt(X**2 + Y**2)
    rho = 0.8 + 0.2*np.exp(-((R-0.5)**2)/0.02) + 0.1*np.exp(-((X+0.3)**2+(Y-0.2)**2)/0.02)
    fig = plt.figure(); ax=fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, rho, rstride=2, cstride=2)
    ax.set_title("Figure 3 — MB density as informational topology")
    plt.tight_layout(); plt.savefig(os.path.join(args.out,"fig3.png")); plt.close()

if __name__ == "__main__":
    main()
