import argparse, os, numpy as np, json
from src.mbsims.rng import set_seed
from src.mbsims.fields import anisotropic_gaussian_sum, random_fourier_field_3d_time, ar1_step
from src.mbsims.dynamics import inertial_step_3d

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="outputs/fig7")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grid", type=int, default=40)
    p.add_argument("--tmax", type=int, default=700)
    p.add_argument("--dt", type=float, default=0.02)
    args = p.parse_args(); set_seed(args.seed); os.makedirs(args.out, exist_ok=True)

    n=args.grid; lin=np.linspace(-3,3,n); X,Y,Z=np.meshgrid(lin,lin,lin, indexing="ij"); grid=np.stack([X,Y,Z],-1)
    rng = np.random.default_rng(args.seed)

    def centers_sigmas_rot(t):
        c1 = np.array([1.5*np.cos(0.02*t), 1.5*np.sin(0.02*t), 0.0]); s1 = np.array([0.7,0.4,0.3]); r1=(0,0,0.01*t)
        c2 = np.array([1-0.01*t, -1+0.01*t, 0.5*np.sin(0.015*t)]); s2 = np.array([0.5,0.5,0.6]); r2=(0,0.015*t,0)
        c3 = np.array([0.5*np.cos(0.03*t), 0.5*np.sin(0.03*t), np.cos(0.02*t)]); s3 = np.array([0.6,0.3,0.7]); r3=(0.012*t,0,0)
        c4 = np.array([2*np.cos(0.01*t), 2*np.sin(0.01*t), -1+0.005*t]); s4 = np.array([0.4,0.6,0.5]); r4=(0,0,0.02*t)
        return [c1,c2,c3,c4],[s1,s2,s3,s4],[r1,r2,r3,r4]

    # Micro obstacles AR(1)
    micro = rng.uniform(-2.5, 2.5, size=(10,3))
    phi_micro=0.9; sigma_micro=0.3

    def a_field(t):
        centers, sigmas, rots = centers_sigmas_rot(t)
        blobs = anisotropic_gaussian_sum(grid, centers, sigmas, rotations=rots)
        micro_field = 0.0
        for i in range(micro.shape[0]):
            micro[i] = phi_micro*micro[i] + rng.normal(0, sigma_micro, size=3)
            d = grid - micro[i]
            q = np.sum(d*d, axis=-1)
            micro_field += 0.3*np.exp(-q/(2*(0.3**2)))
        rf = 0.5*random_fourier_field_3d_time(grid, t, K=15, w_std=1.5)
        return blobs + micro_field + rf

    # ρ from a(x,t)
    def rho_from_a(a):
        rho_corr = (a*a)/(a*a + 1.0**2)
        I = -0.5*np.log(np.clip(1 - rho_corr**2, 1e-9, 1.0))
        I = (I - I.min())/(I.max()-I.min()+1e-12)
        return I

    # inertial agent toward helix target
    def g(t): return np.array([2*np.cos(0.005*t), 2*np.sin(0.005*t), 2-0.002*t])

    pos = np.array([ (n-1)/3 ]*3, dtype=float); vel=np.zeros(3)
    F = ( (grid[...,0]-2.0)**2 + (grid[...,1]-2.0)**2 + (grid[...,2]-2.0)**2 )
    traj=[]
    # AR(1) noises
    phi_move=0.7; sigma_move=0.7
    phi_perc=0.6; sigma_perc=0.15
    for t in range(args.tmax):
        a = a_field(t)
        rho = rho_from_a(a)
        # perceived rho via neighborhood average + AR(1) perception noise
        # (approximate by local cell value + AR1)
        perc = phi_perc*(rho[int(pos[0]),int(pos[1]),int(pos[2])]) + np.random.normal(0,sigma_perc)
        rho_eff = np.clip(perc, 0, 1)
        if t < 400:
            vel = phi_move*vel + np.random.normal(0, sigma_move, size=3)
        pos, vel = inertial_step_3d(pos, vel, F, rho, args.dt, m=1.2, gamma=0.8)
        pos = np.clip(pos, 0, n-1); traj.append(pos.copy())

    np.save(os.path.join(args.out,"traj.npy"), np.array(traj))

if __name__=="__main__":
    main()
