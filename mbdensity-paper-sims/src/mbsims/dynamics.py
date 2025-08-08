import numpy as np

def grad_2d(F):
    gx = np.gradient(F, axis=0); gy = np.gradient(F, axis=1)
    return gx, gy

def grad_3d(F):
    gx = np.gradient(F, axis=0); gy = np.gradient(F, axis=1); gz = np.gradient(F, axis=2)
    return gx, gy, gz

def euler_step_2d(pos, F, rho, dt, alpha=1.0):
    n = F.shape[0]
    i = int(np.clip(round(pos[0]), 0, n-1))
    j = int(np.clip(round(pos[1]), 0, n-1))
    gx, gy = grad_2d(F)
    step = -alpha * (1.0 - rho[i,j]) * np.array([gx[i,j], gy[i,j]])
    return np.clip(pos + dt*step, 0, n-1)

def inertial_step_3d(pos, vel, F, rho, dt, m=1.0, gamma=0.6):
    n = F.shape[0]
    i = int(np.clip(round(pos[0]), 0, n-1))
    j = int(np.clip(round(pos[1]), 0, n-1))
    k = int(np.clip(round(pos[2]), 0, n-1))
    gx, gy, gz = grad_3d(F)
    force = -(1.0 - rho[i,j,k]) * np.array([gx[i,j,k], gy[i,j,k], gz[i,j,k]])
    # damped inertial update: m * dv/dt = force - gamma * v
    acc = (force - gamma*vel)/m
    vel_new = vel + dt*acc
    pos_new = np.clip(pos + dt*vel_new, 0, n-1)
    return pos_new, vel_new
