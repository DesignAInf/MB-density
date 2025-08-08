import argparse, os, numpy as np, matplotlib.pyplot as plt
from src.mbsims.rng import set_seed

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="outputs/fig4")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(); set_seed(args.seed); os.makedirs(args.out, exist_ok=True)

    n=args.n; x=np.linspace(-1,1,n); X,Y=np.meshgrid(x,x, indexing="ij")
    F = X**2 + Y**2
    # High density barrier circle
    rho = 0.05 + 0.9*(np.sqrt((X-0.3)**2+(Y+0.2)**2) < 0.25).astype(float)
    plt.figure()
    cs = plt.contour(X,Y,F, levels=20)
    plt.contour(X,Y,rho, levels=[0.5], linestyles="dashed")  # dashed barrier
    plt.scatter([0.2],[0.2], c="red")  # start
    plt.scatter([0.5],[0.5], c="black")  # target
    plt.gca().set_aspect("equal"); plt.title("Figure 4 — Obstructed minimization by high ρ")
    plt.tight_layout(); plt.savefig(os.path.join(args.out,"fig4.png")); plt.close()

if __name__ == "__main__":
    main()
