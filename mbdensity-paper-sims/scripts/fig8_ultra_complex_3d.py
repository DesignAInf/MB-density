import argparse, os, numpy as np, json
from src.mbsims.rng import set_seed
from src.mbsims.fields import anisotropic_gaussian_sum, random_fourier_field_3d_time
from src.mbsims.dynamics import inertial_step_3d

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="outputs/fig8")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grid", type=int, default=30)
    p.add_argument("--tmax", type=int, default=600)
    p.add_argument("--dt", type=float, default=0.02)
    args = p.parse_args(); set_seed(args.seed); os.makedirs(args.out, exist_ok=True)

    n=args.grid; lin=np.linspace(-3,3,n); X,Y,Z=np.meshgrid(lin,lin,lin, indexing="ij"); grid=np.stack([X,Y,Z],-1)
    rng = np.random.default_rng(args.seed)

    def centers_sigmas_rot(t):
        c1 = np.array([1.5*np.cos(0.015*t), 1.5*np.sin(0.015*t), 0.5*np.sin(0.01*t)]); s1=np.array([0.8,0.5,0.3]); r1=(0,0,0.02*t)
        c2 = np.array([-1.2*np.cos(0.018*t), 1.2*np.sin(0.018*t), -0.5*np.cos(0.012*t)]); s2=np.array([0.6,0.4,0.7]); r2=(0.017*t,0,0)
        c3 = np.array([0.5*np.cos(0.02*t), -0.5*np.sin(0.02*t), 1.5*np.sin(0.015*t)]); s3=np.array([0.5,0.6,0.4]); r3=(0,0.013*t,0)
        return [c1,c2,c3],[s1,s2,s3],[r1,r2,r3]

    micro = rng.uniform(-2.0, 2.0, size=(8,3))
    phi_micro=0.85; sigma_micro=0.3

    def a_field(t):
        centers, sigmas, rots = centers_sigmas_rot(t)
        blobs = anisotropic_gaussian_sum(grid, centers, sigmas, rotations=rots)
        micro_field = 0.0
        for i in range(micro.shape[0]):
            micro[i] = phi_micro*micro[i] + rng.normal(0, sigma_micro, size=3)
            d = grid - micro[i]; q = np.sum(d*d, axis=-1)
            micro_field += 0.25*np.exp(-q/(2*(0.3**2)))
        rf = 0.5*random_fourier_field_3d_time(grid, t, K=15, w_std=1.5)
        return blobs + micro_field + rf

    def rho_from_a(a):
        rho_corr = (a*a)/(a*a + 1.0**2)
        I = -0.5*np.log(np.clip(1 - rho_corr**2, 1e-9, 1.0))
        I = (I - I.min())/(I.max()-I.min()+1e-12)
        return I

    def g(t): return np.array([2*np.cos(0.008*t), 2*np.sin(0.008*t), 2-0.0015*t])

    pos = np.array([ (n-1)/3 ]*3, dtype=float); vel=np.zeros(3)
    F = ( (grid[...,0]-2.0)**2 + (grid[...,1]-2.0)**2 + (grid[...,2]-2.0)**2 )
    traj=[]
    phi_move=0.75; sigma_move=0.7
    phi_perc=0.65; sigma_perc=0.12
    for t in range(args.tmax):
        a = a_field(t)
        rho = rho_from_a(a)
        perc = phi_perc*(rho[int(pos[0]),int(pos[1]),int(pos[2])]) + np.random.normal(0,sigma_perc)
        rho_eff = np.clip(perc,0,1)
        if t < 350:
            vel = phi_move*vel + np.random.normal(0, sigma_move, size=3)
        pos, vel = inertial_step_3d(pos, vel, F, rho, args.dt, m=1.0, gamma=0.6)
        pos = np.clip(pos, 0, n-1); traj.append(pos.copy())

    np.save(os.path.join(args.out,"traj.npy"), np.array(traj))

if __name__=="__main__":
    main()
