#!/usr/bin/env python3
"""
ellipsoid_plot_and_export.py

Esegue:
 - calcolo autovalori/autovettori da Q (o ricostruisce Q da C)
 - disegna i punti misurati, gli assi del sensore e gli assi principali dell'ellissoide
 - salva file JSON con risultati
Output:
 - ellipsoid_plot.png
 - ellipsoid_results.json
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
from scipy.optimize import least_squares
import os

# --------------------------
# Se hai già Q e b, puoi inserirli qui.
# Altrimenti lo script calcola Q,p,f con il fit non-lineare L,b come prima.
# Qui uso il L,b nonlineare approach (robusto) to get C and b.
# --------------------------

# --- normalize for numerical stability (same approach as before)
def read_elab_file_numpy(filename):
    # Legge il numero di record
    with open(filename, "rb") as f:
        # Legge primo uint32
        num = int(np.fromfile(f, dtype=np.uint32, count=1)[0])
        print(num)
        # Legge tutto il resto come float32
        data = np.fromfile(f, dtype=np.float32)

    if data.size % 12 != 0:
        raise ValueError("File non allineato: numero di float non multiplo di 12")

    n = data.size // 12
    data = data.reshape(n, 12)

    # prime 9 colonne → covarianze
    cov = data[:, 0:9].reshape(n, 3, 3).astype(np.float64)

    # ultime 3 colonne → medie
    avg = data[:, 9:12].astype(np.float64)

    #normalizza con media delle norme
    avg_mean = np.mean(np.linalg.norm(avg, axis=1))

    return cov, avg, avg_mean

cov_raw, samples, scale = read_elab_file_numpy("datiLIS3DH.bin")
cov = cov_raw/(scale**2)
mean_samples = samples.mean(axis=0)
Xn = (samples - mean_samples) / scale

#scale=256
#Xn = samples/ scale
# Simple initial guess
b0 = Xn.mean(axis=0)
L0 = np.eye(3)
# pack/unpack
def pack_params(b, L):
    return np.hstack([b, L[0,0], L[1,0], L[1,1], L[2,0], L[2,1], L[2,2]])
def unpack_params(params):
    b = params[:3]
    l11, l21, l22, l31, l32, l33 = params[3:]
    L = np.array([[l11, 0.0, 0.0],
                  [l21, l22, 0.0],
                  [l31, l32, l33]])
    return b, L

def residuals_Lb(params, X, cov):
    n = X.shape[0]
    b,L = unpack_params(params)
    Ys = (L @ (X - b).T).T
    norms = np.linalg.norm(Ys, axis=1)
    inv_norms = 1.0 / np.maximum(norms, 1e-12)

    resid = np.empty(n)
    for i in range(n):
        Yi = Ys[i]                 # (3,)
        # jacobiana del residuo scalare rispetto a Yi: 1x3
        d_r_d_Y = Yi * inv_norms[i]   # (3,) == Y/||Y||
        # jacobiana rispetto a X_i: 1x3 = (d_r_d_Y)^T @ L
        Jx = d_r_d_Y @ L            # (3,) row vector
        # varianza scalare del residuo: Jx @ CovData[i] @ Jx^T
        s2 = float(Jx @ cov[i] @ Jx.T)
        # regolarizzo nel caso s2 sia quasi zero / numericamente negativa
        if s2 <= 0 or not np.isfinite(s2):
            s2 = eps_regularize
        resid[i] = (norms[i] - 1.0) / np.sqrt(s2)

    return resid

def gfun(x, Q, p, f):
    return x @ Q @ x + 2 * p @ x + f

params0 = pack_params(b0, L0)
res = least_squares(residuals_Lb, params0, args=(Xn, cov), method='trf', loss='soft_l1', max_nfev=20000)
b_opt, L_opt = unpack_params(res.x)

# map back to raw units
b_cal = mean_samples + scale * b_opt
#b_cal = scale * b_opt

C_cal = L_opt / scale   # so that y_cal = C_cal @ (y_raw - b_cal) has norm ~1

# reconstruct Q, p, f
Q_cal = C_cal.T @ C_cal
p_cal = - Q_cal @ b_cal
f_cal = b_cal.T @ Q_cal @ b_cal - 1.0

# eigen-decomposition
evals, evecs = np.linalg.eigh(Q_cal)
axes = 1.0 / np.sqrt(evals)   # semi-axes lengths in raw units

# order by descending axis length
order = np.argsort(-axes)
evals_s = evals[order]
axes_s = axes[order]
evecs_s = evecs[:, order]

# compute Euler angles (ZYX) that rotate principal axes into sensor frame
def rot_to_euler_zyx(R):
    # R columns are principal axes in sensor frame
    # returns yaw(Z), pitch(Y), roll(X) in degrees
    if abs(R[2,0]) < 0.999999:
        pitch = -np.arcsin(R[2,0])
        yaw = np.arctan2(R[1,0]/np.cos(pitch), R[0,0]/np.cos(pitch))
        roll = np.arctan2(R[2,1]/np.cos(pitch), R[2,2]/np.cos(pitch))
    else:
        # gimbal lock
        pitch = np.pi/2 if R[2,0] < 0 else -np.pi/2
        yaw = np.arctan2(-R[0,1], R[1,1])
        roll = 0.0
    return np.degrees([yaw, pitch, roll])

euler_z_y_x = rot_to_euler_zyx(evecs_s)

# --------------------------
# Create ellipsoid surface for plotting
# --------------------------
u = np.linspace(0, 2*np.pi, 60)
v = np.linspace(0, np.pi, 30)
u, v = np.meshgrid(u, v)
# points on unit sphere
xs = np.cos(u)*np.sin(v)
ys = np.sin(u)*np.sin(v)
zs = np.cos(v)
pts = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=0)  # 3 x M
# scale by axes and rotate
ell_pts = (evecs_s @ (np.diag(axes_s) @ pts)).T  # M x 3 (in sensor frame, centered at origin)
ell_pts = ell_pts + b_cal.reshape(1,3)          # move to center b_cal

#calcola osservazioni dentro e fuori fitvals = np.array([gfun(x) for x in samples])
vals = np.array([gfun(x, Q_cal, p_cal, f_cal) for x in samples])
eps = 0.0002  # tolleranza superficie

inside = samples[vals < -eps]
surface = samples[np.abs(vals) <= eps]
outside = samples[vals > eps]

print("Dentro:", inside.shape[0])
print("Sulla superficie:", surface.shape[0])
print("Fuori:", outside.shape[0])

# --------------------------
# Plot
# --------------------------
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
# scatter samples

if len(inside)>0:
    ax.scatter(inside[:,0], inside[:,1], inside[:,2], c='magenta', s=10, label='Inside')

if len(surface)>0:
    ax.scatter(surface[:,0], surface[:,1], surface[:,2], c='green', s=20, label='On surface')

if len(outside)>0:
    ax.scatter(outside[:,0], outside[:,1], outside[:,2], c='yellow', s=10, label='Outside')


#ax.scatter(samples[:,0], samples[:,1], samples[:,2], s=20, alpha=0.6, label='samples')

# draw ellipsoid surface (as wire / transparent)
X_el = ell_pts[:,0].reshape(xs.shape)
Y_el = ell_pts[:,1].reshape(xs.shape)
Z_el = ell_pts[:,2].reshape(xs.shape)
ax.plot_surface(X_el, Y_el, Z_el, rstride=2, cstride=2, color='deepskyblue', alpha=0.1, linewidth=0)

# draw sensor axes (origin at 0)
origin = np.array([0.0, 0.0, 0.0])
ax.quiver(origin[0], origin[1], origin[2], 200, 0, 0, color='r', length=12200, normalize=True, label='sensor X', linewidth=1)
ax.quiver(origin[0], origin[1], origin[2], 0, 200, 0, color='g', length=12200, normalize=True, label='sensor Y', linewidth=1)
ax.quiver(origin[0], origin[1], origin[2], 0, 0, 200, color='b', length=12200, normalize=True, label='sensor Z', linewidth=1)

# draw principal axes (from center)
scale_vec = axes_s  # draw vectors equal to semi-axis lengths
for i in range(3):
    v = evecs_s[:,i]
    ax.plot([b_cal[0], b_cal[0]+v[0]*scale_vec[i]],
            [b_cal[1], b_cal[1]+v[1]*scale_vec[i]],
            [b_cal[2], b_cal[2]+v[2]*scale_vec[i]],
            color='k', linewidth=1)
    ax.plot([b_cal[0], b_cal[0]-v[0]*scale_vec[i]],
            [b_cal[1], b_cal[1]-v[1]*scale_vec[i]],
            [b_cal[2], b_cal[2]-v[2]*scale_vec[i]],
            color='k', linewidth=1)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Ellipsoid fit: samples (points), ellipsoid surface, principal axes')
ax.legend(loc='upper left', bbox_to_anchor=(1.1, 1))
ax.set_box_aspect([1,1,1])

# autoscale equal
max_range = np.array([samples[:,0].max()-samples[:,0].min(),
                      samples[:,1].max()-samples[:,1].min(),
                      samples[:,2].max()-samples[:,2].min()]).max() / 2.0
mid_x = (samples[:,0].max()+samples[:,0].min()) * 0.5
mid_y = (samples[:,1].max()+samples[:,1].min()) * 0.5
mid_z = (samples[:,2].max()+samples[:,2].min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.tight_layout()
pngfile = "ellipsoid_plot.png"
plt.savefig(pngfile, dpi=300)
print("Saved plot to", pngfile)

# --------------------------
# Save JSON results
# --------------------------
results = {
    "b": b_cal.tolist(),
    "C": C_cal.tolist(),
    "Q": Q_cal.tolist(),
    "p": p_cal.tolist(),
    "f": float(f_cal),
    "eigenvalues": evals_s.tolist(),
    "semi_axes": axes_s.tolist(),
    "eigenvectors": evecs_s.tolist(),
    "euler_zyx_deg": euler_z_y_x.tolist()
}

jsonfile = "ellipsoid_results.json"
with open(jsonfile, "w") as f:
    json.dump(results, f, indent=2)
print("Saved JSON to", jsonfile)

# show plot interactively if possible
try:
    plt.show()
except Exception:
    pass
