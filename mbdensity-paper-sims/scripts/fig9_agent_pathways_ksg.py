import argparse, os, numpy as np, json
from src.mbsims.rng import set_seed
from src.mbsims.mi import mi_ksg, rho_from_mi

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="outputs/fig9")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n", type=int, default=80)
    p.add_argument("--paths", type=int, default=20)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args(); set_seed(args.seed); os.makedirs(args.out, exist_ok=True)

    n=args.n; x=np.linspace(-2,2,n); X,Y=np.meshgrid(x,x, indexing="ij")
    # Synthetic MB density landscape with corridors
    rho = 0.2 + 0.6*(np.exp(-((X+0.8)**2+(Y+0.5)**2)/0.3)
                     + np.exp(-((X-0.9)**2+(Y-0.6)**2)/0.3))
    rho = np.clip(rho, 0, 1)
    F = (X**2 + Y**2)  # simple free energy

    # Online KSG: we create pseudo (I,E,B) samples from local neighborhoods
    rng = np.random.default_rng(args.seed)

    def step(pos):
        i = int(np.clip(round(pos[0]),0,n-1)); j=int(np.clip(round(pos[1]),0,n-1))
        # local cloud
        ii = np.clip(np.arange(i-2,i+3),0,n-1); jj=np.clip(np.arange(j-2,j+3),0,n-1)
        pts = np.array([(a,b) for a in ii for b in jj], dtype=float)
        I = pts + rng.normal(0,0.05, size=pts.shape)
        E = pts + rng.normal(0,0.05, size=pts.shape)
        B = pts + rng.normal(0,0.05, size=pts.shape)
        IIE = mi_ksg(I,E,k=args.k); IIEgB = max(0.0, IIE - mi_ksg(I,B,k=args.k))
        rho_hat = rho_from_mi(IIE, IIEgB)
        # descend
        gx, gy = np.gradient(F, axis=0), np.gradient(F, axis=1)
        step_v = -(1.0 - rho_hat)*np.array([gx[i,j], gy[i,j]])
        return np.clip(pos + 0.02*step_v, 0, n-1), rho_hat

    # sample many start points, keep best (shortest to min)
    starts = rng.integers(low=0, high=n, size=(args.paths, 2))
    best_paths = []
    for s in starts:
        pos = s.astype(float)
        path=[pos.copy()]; errs=0
        for _ in range(args.steps):
            pos, rh = step(pos); path.append(pos.copy())
            if rh>0.95: errs += 1
        best_paths.append((errs, np.array(path)))
    best_paths.sort(key=lambda t:t[0])
    selected = [p for _,p in best_paths[:6]]

    np.save(os.path.join(args.out,"rho.npy"), rho)
    for idx, tr in enumerate(selected):
        np.save(os.path.join(args.out,f"path_{idx}.npy"), tr)

if __name__=="__main__":
    main()
