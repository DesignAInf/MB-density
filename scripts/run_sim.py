import argparse, os, json, numpy as np
from pathlib import Path
from src.mbdensity.sim import run_demo
from src.mbdensity.config import DEFAULTS

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="./outputs")
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--alpha", type=float, default=0.5)
    # allow overrides
    p.add_argument("--r1", type=int, default=DEFAULTS.r1)
    p.add_argument("--r2", type=int, default=DEFAULTS.r2)
    p.add_argument("--k", type=int, default=DEFAULTS.k)
    p.add_argument("--epsilon", type=float, default=DEFAULTS.epsilon)
    p.add_argument("--delta", type=float, default=DEFAULTS.delta)
    p.add_argument("--seed", type=int, default=DEFAULTS.seed)
    args = p.parse_args()

    # update defaults on the fly
    from dataclasses import replace
    cfg = replace(DEFAULTS, r1=args.r1, r2=args.r2, k=args.k, epsilon=args.epsilon, delta=args.delta, seed=args.seed)

    os.makedirs(args.out, exist_ok=True)
    out = run_demo(n=args.n, steps=args.steps, alpha=args.alpha, cfg=cfg)

    # Save arrays and a small JSON
    np.save(os.path.join(args.out, "F.npy"), out["F"])
    np.save(os.path.join(args.out, "rho.npy"), out["rho"])
    np.save(os.path.join(args.out, "traj.npy"), out["traj"])
    meta = {k: float(v) if isinstance(v, (np.floating,)) else v for k,v in out.items() if k not in ["F","rho","traj"]}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved outputs to", args.out)

if __name__ == "__main__":
    main()
