import argparse, os, numpy as np, json
from pathlib import Path
from src.mbsims.rng import set_seed
from src.mbsims.fields import anisotropic_gaussian_sum, random_fourier_field_3d_time
from src.mbsims.dynamics import inertial_step_3d

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="outputs/fig6")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grid", type=int, default=35)  # 35^3 over [-3,3]^3
    p.add_argument("--tmax", type=int, default=400)
    p.add_argument("--dt", type=float, default=0.03)
    args = p.parse_args(); set_seed(args.seed); os.makedirs(args.out, exist_ok=True)

    n=args.grid; lin=np.linspace(-3,3,n); X,Y,Z=np.meshgrid(lin,lin,lin, indexing="ij")
    grid = np.stack([X,Y,Z], axis=-1)

    def c1(t): return np.array([1.5*np.cos(0.02*t), 1.5*np.sin(0.02*t), 0.0])
    def c2(t): return np.array([1.0-0.01*t, -1.0+0.01*t, 0.0])

    # Build a(x,t) from two Gaussians + (optional) RF
    def a_field(t):
        centers = [c1(t), c2(t)]
        sigmas = [np.array([0.7,0.7,0.7]), np.array([0.9,0.9,0.9])]
        val = anisotropic_gaussian_sum(grid, centers, sigmas)
        return val

    pos = np.array([0,0,0], dtype=float) + (n-1)/6  # near center
    vel = np.zeros(3, dtype=float)
    F = ( (X-2)**2 + (Y-2)**2 + (Z-2)**2 )  # target (2,2,2)
    traj=[]
    rho_slices = []
    for t in range(args.tmax):
        a = a_field(t)
        rho_corr = (a*a)/ (a*a + 0.9**2)
        # normalize I(x,t) and map to rho in [0,1]
        I = -0.5*np.log(np.clip(1 - rho_corr**2, 1e-9, 1.0))
        I = (I - I.min())/(I.max()-I.min()+1e-12)
        rho = I
        # phases: noisy then deterministic
        if t < 200:
            # noisy inertial update
            pos, vel = inertial_step_3d(pos, vel, F, rho, args.dt, m=1.0, gamma=0.6)
            pos = pos + np.random.normal(0, 0.9, size=3)
        else:
            pos, vel = inertial_step_3d(pos, vel, F, rho, args.dt, m=1.0, gamma=0.6)
        pos = np.clip(pos, 0, n-1)
        traj.append(pos.copy())
        if t in [0,200,399]:
            # store z-slices near -1,0,1 (approx indices)
            for zval in [-1.0, 0.0, 1.0]:
                k = int(np.argmin(np.abs(lin - zval)))
                rho_slices.append(rho[:,:,k])

    np.save(os.path.join(args.out,"rho_slices.npy"), np.array(rho_slices))
    np.save(os.path.join(args.out,"traj.npy"), np.array(traj))
    np.save(os.path.join(args.out,"F.npy"), F)
    with open(os.path.join(args.out,"meta.json"),"w") as f:
        json.dump({"note":"Figure 6 dynamic environment with two Gaussians"}, f, indent=2)

if __name__=="__main__":
    main()
