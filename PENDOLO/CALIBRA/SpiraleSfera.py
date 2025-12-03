import numpy as np
from scipy.interpolate import interp1
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parametri della spirale
R = 1.0      # raggio

M =1500        # numero di punti equispaziati da ottenere
N =8          # numero di giri
N= np.sqrt(np.pi*(M-1))/2 # numero di giri per equidistanza punti

# ==========================================
# 1) Definizione della curva parametrica
# ==========================================
def theta(t):
    return np.pi * t

def phi(t):
    return 2 * np.pi * N * t

def speed(t):
    # velocità istantanea |r'(t)|
    dtheta = np.pi
    dphi   = 2 * np.pi * N
    return R * np.sqrt(dtheta**2 + (np.sin(theta(t))**2) * dphi**2)

# ==========================================
# 2) Costruzione griglia fine di t per inversione
# ==========================================
n_samples = 5000
t_grid = np.linspace(0, 1, n_samples)
v_grid = speed(t_grid)

# Integrale cumulativo s(t) tramite trapezi
s_grid = np.concatenate(([0], np.cumsum(0.5*(v_grid[1:]+v_grid[:-1])*(t_grid[1]-t_grid[0]))))
S = s_grid[-1]   # lunghezza totale

# ==========================================
# 3) Inversione t(s)
# ==========================================
inv_t = interp1d(s_grid, t_grid)

# punti equispaziati in s
s_unif = np.linspace(0, S, M)

# parametri t_k corrispondenti
t_k = inv_t(s_unif)

# differenze Δt_k
delta_t = np.diff(t_k)

# ==========================================
# 4) Coordinate finali della spirale
# ==========================================
#asse nord-sud
x = R * np.sin(theta(t_k)) * np.cos(phi(t_k))
y = R * np.sin(theta(t_k)) * np.sin(phi(t_k))
z = R * np.cos(theta(t_k))
#asse est-ovest
'''
x = R * np.cos(theta(t_k))
y = R * np.sin(theta(t_k)) * np.cos(phi(t_k))
z = R * np.sin(theta(t_k)) * np.sin(phi(t_k))
'''

# Output di controllo
print("Lungheax.set_box_aspect([6, 6, 6])zza totale S =", S)
print("Primi 10 valori di t_k:", t_k[:10])
print("Prime 10 Δt_k:", delta_t[:10])

xx=t_grid*np.pi
yy=(1-np.cos(xx))*S/2
fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='green', s=10, label='Inside')
ax.set_box_aspect([2, 2, 2])

fig1= plt.figure(figsize=(10,8))
ax1 = fig1.add_subplot()
ax1.scatter(t_grid, s_grid, c='green', s=10, label='Inside')
ax1.scatter(t_grid, yy, c='red', s=1, label='Inside')

plt.show()


