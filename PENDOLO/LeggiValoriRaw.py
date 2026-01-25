import numpy as np
#from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math as mt
from scipy.optimize import curve_fit
import scipy.stats as stats


def read_file(filename):
    # Legge il numero di record
    with open(filename, "rb") as f:
        # Legge primo uint32
        #num = int(np.fromfile(f, dtype=np.uint32, count=1)[0])
        #print(num)
        # Legge tutto il resto come float32
        data = np.fromfile(f, dtype=np.int16)
        print(data.size)

    if data.size % 3 != 0:
        raise ValueError("File non allineato: numero di float non multiplo di 9")

    n = data.size // 3
    data = data.reshape(n, 3)
    return data
path = "/home/marco/Desktop/Uni_anno3/TD/PENDOLO/MPU6050_11/"
fileName = "MPU6050_rawData00250.bin"
raw = read_file(path + fileName)
avg = np.mean(raw, axis=0)
std = np.std(raw, axis=0)
nrm = np.linalg.norm(raw, axis=1)
aa=(raw[:]-avg)/std
bb=np.linalg.norm(aa, axis=1)
cc=np.linalg.norm(raw/2**14, axis=1)
# the histogram of the data
#serie di dati
i=0


Ris = 300
fig1=plt.figure('Distribuzione norma', figsize=(10, 6), dpi=Ris)
ax1 = fig1.subplots()
Ds1= 300
n1, bins1, patches1 = ax1.hist(x=cc, bins=Ds1, color = 'darkgrey', edgecolor = 'white', density=True)

fig2=plt.figure('Distribuzione1', figsize=(10, 6), dpi=Ris)
ax2 = fig2.subplots()
Ds= int(np.max(raw[:,i])- np.min(raw[:,i])/2)
Val = raw[:,i]
n, bins, patches = ax2.hist(x=raw[:,i], bins=Ds, color = 'black', edgecolor = 'white', density=True)

ax2.plot(bins, stats.norm.pdf(bins, avg[i], std[i]), '--', color='orange')
ax2.plot(bins, stats.cauchy.pdf(bins, loc=avg[i], scale=std[i]),'--', color='green')
beta_fit, loc_fit, scale_fit = stats.gennorm.fit(raw[i])
ax2.plot(bins, stats.gennorm.pdf(bins, beta=2, loc=avg[i], scale=std[i]), '--', color ='red')







plt.show()
