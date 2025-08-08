import numpy as np, matplotlib.pyplot as plt

def save_heatmap(img, path, title=None, contours=None):
    plt.figure()
    plt.imshow(img, origin="lower")
    if contours is not None:
        for level, style in contours:
            cs = plt.contour(img, levels=[level], origin="lower", linewidths=1.0)
            for c in cs.collections: c.set_linestyle(style)
    if title: plt.title(title)
    plt.tight_layout(); plt.savefig(path); plt.close()

def save_surface_with_traj(F, traj, path, title=None):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    n = F.shape[0]
    x = np.linspace(-1,1,n); X,Y = np.meshgrid(x,x, indexing="ij")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, F, alpha=0.7, rstride=4, cstride=4)
    ax.plot(traj[:,1]*2/n-1, traj[:,0]*2/n-1, F[traj[:,0].astype(int), traj[:,1].astype(int)], linewidth=2)
    if title: ax.set_title(title)
    plt.tight_layout(); plt.savefig(path); plt.close()

def save_slices_grid(rho_tz_list, pts_by_slice, path, titles):
    fig, axes = plt.subplots(3,3, figsize=(10,10))
    for r in range(3):
        for c in range(3):
            idx = r*3 + c
            rho = rho_tz_list[idx]
            axes[r,c].imshow(rho, origin="lower")
            if pts_by_slice[idx].size>0:
                axes[r,c].scatter(pts_by_slice[idx][:,1], pts_by_slice[idx][:,0], s=4)
            axes[r,c].set_title(titles[idx])
    plt.tight_layout(); plt.savefig(path); plt.close()
