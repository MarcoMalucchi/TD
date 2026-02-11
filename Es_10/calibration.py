import numpy as np
from scipy.interpolate import interp1d, griddata
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d
import math as mt
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy.linalg import eig, inv, schur
from scipy.spatial import Delaunay, cKDTree
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def sphere_model(coords, Xc, Yc, Zc, R):
    x, y, z = coords
    return np.sqrt((x-Xc)**2 + (y-Yc)**2 + (z-Zc)**2) - R

def setLimitAxes3D(ax, *data):
    D = []
    for M in data:
        D.extend(M)
    D = np.array(D)

    xmax=D[:,0].max()
    xmin=D[:,0].min()
    ymax=D[:,1].max()
    ymin=D[:,1].min()
    zmax=D[:,2].max()
    zmin=D[:,2].min()

    # Calcolo dei range per ogni asse
    x_range = xmax - xmin
    y_range = ymax - ymin
    z_range = zmax - zmin

    # Troviamo il range massimo
    max_range = max(x_range, y_range, z_range)*1.05

    # Calcoliamo i punti medi
    x_mid = (xmin + xmax) / 2
    y_mid = (ymin + ymax) / 2
    z_mid = (zmin + zmax) / 2

    ax.set_xlim3d([x_mid - max_range/2, x_mid + max_range/2])
    ax.set_ylim3d([y_mid - max_range/2, y_mid + max_range/2])
    ax.set_zlim3d([z_mid - max_range/2, z_mid + max_range/2])
    ax.set_box_aspect([1, 1, 1])


def compute_theta_phi(means):
    g = means / np.linalg.norm(means, axis=1, keepdims=True)

    theta = np.arccos(g[:, 2])              # [0, pi]
    phi = np.arctan2(g[:, 1], g[:, 0])      # [-pi, pi]

    return theta, phi, g

plt.close('all')
tiPrec = 0.0
lostRecord = 0
colori = ['#1f77b4', '#ff7f0e', '#2ca02c','#1A237E', '#2979FF', '#40C4FF']
colori1 = np.array(['#1A237E', '#2979FF', '#40C4FF', '#1f77b4', '#ff7f0e', '#2ca02c'])
colori2 = ['#ea5545', '#27aeef', '#87bc45', '#b33dc6']

#path = "/home/marco/Documents/SCUOLA/LAB3/"
#baseDir = "MPU6050/S0028/" #25
#fileList = "file_list.txt"

path = "/home/marco/Desktop/Uni_anno3/TD/Es_10/"
#baseDir = "MPU6050/" #25
ElabFile = "DataMPU6050.npz"
#+baseDir
# Tentativo di caricamento
data = np.load(path+ElabFile)
# Estrazione dei dati
accl = data['accl']


means = np.mean(accl, axis=1) # Media accelerazioni -> Risultato nx3
stds = np.std(accl, axis=1, ddof=1) # Deviazione standard -> Risultato nx3
covs = np.array([np.cov(accl[i], rowvar=False) for i in range(accl.shape[0])]) #matrici covarianze nx[3x3]


theta, phi, g = compute_theta_phi(means)
phi += np.pi

#troviamo il centro della sfera ed il raggio
X = means[:,0]
Y = means[:,1]
Z = means[:,2]
coords = np.vstack((X, Y, Z)) # forma 3 x n
ydata = np.zeros(X.shape)      # target = 0 distanza dal raggio

# stime iniziali
C0 = np.mean(means, axis=0)
R0 = np.mean(np.linalg.norm(means - C0, axis=1))
p0 = [C0[0], C0[1], C0[2], R0]

# curve_fit con cov=True per ottenere la covarianza dei parametri
#sipotrà usare anche l'incertezza dei dati, ma non so come fare
params, pcov = curve_fit(sphere_model, coords, ydata, p0=p0)
Xc, Yc, Zc, R = params

print("Centro fit:", Xc, Yc, Zc)
print("Raggio fit:", R)

Dist = np.linalg.norm(means - np.array([Xc, Yc, Zc]), axis=1)
Scale = Dist/R

Group = np.zeros(Z.shape[0], dtype=int)
Group[Z < -7500] = 1
Group[X > 7500] = 2
Group[Y > 7500] = 3
Group[X < -7500] = 4
Group[Y < -7500] = 5

mz1 = (Z < -7500)
mz2 = (Z >  7500)
mx1 = (X < -7500)
mx2 = (X >  7500)
my1 = (Y < -7500)
my2 = (Y >  7500)

phi[mz1] = 0
phi[mz2] = 0
phi[mx1] = 0

MeanDistGroup  = np.array([Dist[Group == g].mean() for g in range(6)]) #sono 6 gruppi da 0 a 5
MeanThetaGroup = np.array([theta[Group == g].mean() for g in range(6)]) #sono 6 gruppi da 0 a 5
MeanPhiGroup   = np.array([phi[Group == g].mean() for g in range(6)]) #sono 6 gruppi da 0 a 5

print('Farrore di scala z+: ', MeanDistGroup[0]/R)
print('Farrore di scala z-: ', MeanDistGroup[1]/R)
print('Farrore di scala x+: ', MeanDistGroup[2]/R)
print('Farrore di scala y+: ', MeanDistGroup[3]/R)
print('Farrore di scala x-: ', MeanDistGroup[4]/R)
print('Farrore di scala y-: ', MeanDistGroup[5]/R)

#garfico 3d dei punti osservati
fig = plt.figure(figsize=(15,9), dpi=100)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(means[:,0], means[:,1], means[:,2], c=colori1[Group], s=1, label='Punti osservati')
ax.scatter(Xc, Yc, Zc, c=colori1[1], s=10, label='Punti osservati')

u = np.linspace(0, 2 * np.pi, 30) # 30 meridiani
v = np.linspace(0, np.pi, 20)     # 20 paralleli

x4 = R*np.outer(np.cos(u), np.sin(v))+Xc
y4 = R*np.outer(np.sin(u), np.sin(v))+Yc
z4 = R*np.outer(np.ones(np.size(u)), np.cos(v))+Zc

# Disegno del wireframe (solo le linee)
# rstride e cstride controllano la densità dei meridiani e paralleli
ax.plot_wireframe(x4, y4, z4, color='gray', linewidth=0.4, rstride=1, cstride=1)

setLimitAxes3D(ax, means)
ax.set_box_aspect([2, 2, 2])

#garfico 3d delle scale
fig1 = plt.figure(figsize=(15,9), dpi=100)
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(theta, phi, Scale, c=colori1[2], s=1, label='Punti osservati')
ax1.scatter(MeanThetaGroup, MeanPhiGroup, MeanDistGroup/R, c=colori1[0], s=20, label='Punti osservati')

plt.show()